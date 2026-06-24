#!/usr/bin/env python3
"""Compare WiLoR/Phase-C hand skeleton overlays stage by stage.

This diagnostic answers a narrow question: when does the image-space hand offset
start to appear?  For every stage candidate, it compares that stage's displayed
UV skeleton against the raw WiLoR `joints_uv` stored in the same NPZ row.

The output is intentionally visual and numeric:

- CSV/JSON: per-stage UV RMS, wrist delta, MCP delta, hand-size delta.
- Contact sheets: same source frame, columns are pipeline stages.  Magenta is
  raw WiLoR 2D `joints_uv`; green is the stage UV being diagnosed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np


HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

MCP_IDS = [5, 9, 13, 17]
RAW_COLOR = (255, 0, 255)      # magenta, BGR
STAGE_COLOR = (60, 255, 60)    # green, BGR
LEFT_COLOR = (255, 170, 60)
RIGHT_COLOR = (80, 220, 120)
WHITE = (245, 245, 245)
BLACK = (0, 0, 0)


@dataclass(frozen=True)
class StageSpec:
    name: str
    rel_npz: str
    mode: str
    key: str
    label: str


STAGES = [
    StageSpec("raw_uv", "raw_wilor_handresults/wilor_handresults_raw.npz", "uv", "joints_uv", "Raw WiLoR 2D"),
    StageSpec("phase_b_uv", "phase_b_track_postprocess/wilor_handresults_phase_b.npz", "uv", "joints_uv", "Phase-B kept 2D"),
    StageSpec("phase_c_depth_project", "phase_c_depth_align/wilor_handresults_phase_c_depth_aligned.npz", "project", "joints_cam_depth", "C depth project"),
    StageSpec("phase_c1b_depth_smooth_project", "phase_c_depth_smooth/wilor_handresults_phase_c1b_depth_smooth.npz", "project", "joints_cam_depth_smooth", "C1b depth smooth"),
    StageSpec("phase_c1c_motion_project", "phase_c_motion_infiller/wilor_handresults_phase_c1c_motion_infilled.npz", "project", "joints_cam_depth_smooth", "C1c motion"),
    StageSpec("phase_c2_mano_smooth_project", "phase_c_mano_smooth/wilor_handresults_phase_c2_mano_smooth.npz", "uv", "joints_uv_smooth_depth_camera", "C2 MANO smooth"),
    StageSpec("phase_c3_mesh_visibility_project", "phase_c_mesh_visibility/wilor_handresults_phase_c3_mesh_visibility.npz", "uv", "joints_uv_smooth_depth_camera", "C3 mesh visibility"),
    StageSpec("phase_c3_locked_fixed", "phase_c_mesh_visibility/wilor_handresults_phase_c3_mesh_visibility.npz", "locked_fixed", "joints_uv", "C3 locked fixed"),
    StageSpec("phase_c4_visibility_realign_project", "phase_c_visibility_depth_realign/wilor_handresults_phase_c4_visibility_depth_realign.npz", "project", "joints_cam_visibility_depth", "C4 vis re-align"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session-dir", required=True)
    p.add_argument("--source-pipeline", default="egoinfinity_hand_pipeline")
    p.add_argument("--output-dir", default="")
    p.add_argument("--video", default="")
    p.add_argument("--frames", default="94,212,540,562,563,579,590,591,596,616,619,626")
    p.add_argument("--worst-stage", default="phase_c2_mano_smooth_project")
    p.add_argument("--worst-count", type=int, default=12)
    p.add_argument("--hand", default="both", choices=["both", "left", "right"])
    p.add_argument("--scale", type=float, default=0.55)
    p.add_argument("--max-ref-hand-size-px", type=float, default=900.0)
    p.add_argument("--max-candidates-per-frame", type=int, default=4)
    return p.parse_args()


def parse_frame_list(text: str) -> List[int]:
    out: List[int] = []
    for item in str(text).replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            a, b = item.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(item))
    return sorted(dict.fromkeys(out))


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_clean(value.tolist())
    if isinstance(value, np.generic):
        return json_clean(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def stats(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return {}
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "p10": float(np.percentile(arr, 10)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def string_at(data: Dict[str, np.ndarray], key: str, i: int, default: str = "") -> str:
    arr = data.get(key)
    if arr is None or i >= len(arr):
        return default
    try:
        return str(arr[i])
    except Exception:
        return default


def number_at(data: Dict[str, np.ndarray], key: str, i: int, default: float = float("nan")) -> float:
    arr = data.get(key)
    if arr is None or i >= len(arr):
        return default
    try:
        return float(arr[i])
    except Exception:
        return default


def load_camera(data: Dict[str, np.ndarray], row: int) -> Tuple[float, float, float, float]:
    if "foundation_camera_json" in data and len(data["foundation_camera_json"]):
        try:
            cam = json.loads(str(np.asarray(data["foundation_camera_json"]).reshape(-1)[0]))
            return float(cam["fx"]), float(cam["fy"]), float(cam["cx"]), float(cam["cy"])
        except Exception:
            pass
    size = np.asarray(data.get("image_size", [850, 420])).reshape(-1)
    width = float(size[0]) if size.size >= 1 else 850.0
    height = float(size[1]) if size.size >= 2 else 420.0
    focal = number_at(data, "focal_length", row, float("nan"))
    if not math.isfinite(focal) or focal <= 0.0:
        focal = 334.0549853304012
    return float(focal), float(focal), width * 0.5, height * 0.5


def project_points(points: np.ndarray, data: Dict[str, np.ndarray], row: int) -> np.ndarray:
    fx, fy, cx, cy = load_camera(data, row)
    pts = np.asarray(points, dtype=np.float64)
    uv = np.full((pts.shape[0], 2), np.nan, dtype=np.float64)
    z = pts[:, 2]
    valid = np.isfinite(pts).all(axis=1) & (z > 1e-8)
    uv[valid, 0] = fx * pts[valid, 0] / z[valid] + cx
    uv[valid, 1] = fy * pts[valid, 1] / z[valid] + cy
    return uv


def hand_size_px(uv: np.ndarray) -> float:
    valid = np.isfinite(uv).all(axis=1)
    if int(np.sum(valid)) < 2:
        return float("nan")
    pts = uv[valid]
    return float(max(np.max(pts[:, 0]) - np.min(pts[:, 0]), np.max(pts[:, 1]) - np.min(pts[:, 1])))


def fixed_locked_uv(data: Dict[str, np.ndarray], row: int, max_size_px: float) -> Tuple[np.ndarray, str]:
    uv = np.asarray(data["joints_uv"][row], dtype=np.float64).copy()
    motion_infilled = bool(int(number_at(data, "motion_infilled", row, 0)) != 0)
    size = hand_size_px(uv)
    reason = "raw"
    if (motion_infilled or (math.isfinite(size) and size > float(max_size_px))) and "joints_uv_smooth_depth_camera" in data:
        repl = np.asarray(data["joints_uv_smooth_depth_camera"][row], dtype=np.float64)
        if np.isfinite(repl).any():
            uv = repl
            reason = "infilled_projected" if motion_infilled else "oversize_projected"
    return uv, reason


def stage_uv(data: Dict[str, np.ndarray], spec: StageSpec, row: int, max_ref_size_px: float) -> Tuple[np.ndarray, str]:
    if spec.mode == "uv":
        return np.asarray(data[spec.key][row], dtype=np.float64), spec.key
    if spec.mode == "project":
        return project_points(np.asarray(data[spec.key][row], dtype=np.float64), data, row), spec.key
    if spec.mode == "locked_fixed":
        return fixed_locked_uv(data, row, max_ref_size_px)
    raise RuntimeError(f"unknown stage mode: {spec.mode}")


def uv_metrics(stage: np.ndarray, ref: np.ndarray) -> Dict[str, float]:
    delta = np.asarray(stage, dtype=np.float64) - np.asarray(ref, dtype=np.float64)
    valid = np.isfinite(delta).all(axis=1)
    dist = np.linalg.norm(delta[valid], axis=1) if np.any(valid) else np.asarray([], dtype=np.float64)
    wrist = float(np.linalg.norm(delta[0])) if valid.shape[0] > 0 and bool(valid[0]) else float("nan")
    mcp_valid = valid[MCP_IDS]
    if np.any(mcp_valid):
        mcp = float(np.sqrt(np.mean(np.sum(delta[MCP_IDS][mcp_valid] ** 2, axis=1))))
    else:
        mcp = float("nan")
    return {
        "joint_count": int(np.sum(valid)),
        "rms_px": float(np.sqrt(np.mean(dist ** 2))) if dist.size else float("nan"),
        "mean_px": float(np.mean(dist)) if dist.size else float("nan"),
        "max_px": float(np.max(dist)) if dist.size else float("nan"),
        "wrist_px": wrist,
        "mcp_rms_px": mcp,
        "ref_size_px": hand_size_px(ref),
        "stage_size_px": hand_size_px(stage),
    }


def rows_by_frame(data: Dict[str, np.ndarray]) -> Dict[int, List[int]]:
    frames = np.asarray(data["frame_index"], dtype=np.int64)
    out: Dict[int, List[int]] = {}
    for i, frame in enumerate(frames.tolist()):
        out.setdefault(int(frame), []).append(i)
    return out


def read_frame(video: Path, frame: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
    ok, img = cap.read()
    cap.release()
    if not ok or img is None:
        raise RuntimeError(f"failed to read frame {frame}: {video}")
    return img


def text(img: np.ndarray, s: str, org: Tuple[int, int], color: Tuple[int, int, int] = WHITE, scale: float = 0.46) -> None:
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_skeleton(img: np.ndarray, uv: np.ndarray, color: Tuple[int, int, int], thickness: int, radius: int, alpha: float = 0.85) -> None:
    overlay = img.copy()
    h, w = img.shape[:2]
    def finite(p: np.ndarray) -> bool:
        return np.isfinite(p).all() and -50 <= float(p[0]) <= w + 50 and -50 <= float(p[1]) <= h + 50
    for a, b in HAND_EDGES:
        if finite(uv[a]) and finite(uv[b]):
            pa = (int(round(float(uv[a, 0]))), int(round(float(uv[a, 1]))))
            pb = (int(round(float(uv[b, 0]))), int(round(float(uv[b, 1]))))
            cv2.line(overlay, pa, pb, color, int(thickness), cv2.LINE_AA)
    for lid in range(min(21, uv.shape[0])):
        if finite(uv[lid]):
            p = (int(round(float(uv[lid, 0]))), int(round(float(uv[lid, 1]))))
            cv2.circle(overlay, p, int(radius) + (1 if lid == 0 else 0), color, -1, cv2.LINE_AA)
            cv2.circle(overlay, p, int(radius) + 1, WHITE, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, float(alpha), img, 1.0 - float(alpha), 0.0, dst=img)


def draw_panel(
    image: np.ndarray,
    data: Dict[str, np.ndarray],
    spec: StageSpec,
    frame: int,
    hand_filter: str,
    max_ref_size_px: float,
    max_candidates: int,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    out = image.copy()
    row_map = rows_by_frame(data)
    rows = row_map.get(int(frame), [])
    panel_rows: List[Dict[str, Any]] = []
    y = 18
    text(out, f"{spec.label} | f={frame}", (8, y), WHITE, 0.46)
    y += 18
    drawn = 0
    for row in rows:
        label = string_at(data, "hand_label", row, "")
        if hand_filter != "both" and label != hand_filter:
            continue
        if "joints_uv" not in data or spec.key not in data and spec.mode not in ("locked_fixed",):
            continue
        ref = np.asarray(data["joints_uv"][row], dtype=np.float64)
        uv, source = stage_uv(data, spec, row, max_ref_size_px)
        m = uv_metrics(uv, ref)
        ref_size = m["ref_size_px"]
        ref_ok = math.isfinite(ref_size) and ref_size <= float(max_ref_size_px)
        if ref_ok:
            draw_skeleton(out, ref, RAW_COLOR, 1, 2, 0.58)
        draw_skeleton(out, uv, STAGE_COLOR, 2, 3, 0.86)
        if "bbox_xyxy" in data:
            box = np.asarray(data["bbox_xyxy"][row], dtype=np.float64)
            if box.shape[0] == 4 and np.isfinite(box).all():
                c = RIGHT_COLOR if label == "right" else LEFT_COLOR
                cv2.rectangle(out, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), c, 1, cv2.LINE_AA)
        infilled = int(number_at(data, "motion_infilled", row, 0)) != 0
        info = f"{label} tr={int(number_at(data,'track_id',row,-1))} r={int(number_at(data,'hand_rank',row,0))} rms={m['rms_px']:.1f}px wrist={m['wrist_px']:.1f} src={source}"
        if infilled:
            info += " INFILL"
        color = RIGHT_COLOR if label == "right" else LEFT_COLOR
        text(out, info[:118], (8, y), color, 0.36)
        y += 15
        panel_rows.append({
            "stage": spec.name,
            "stage_label": spec.label,
            "frame_index": int(frame),
            "row": int(row),
            "hand_label": label,
            "track_id": int(number_at(data, "track_id", row, -1)),
            "hand_rank": int(number_at(data, "hand_rank", row, 0)),
            "motion_infilled": bool(infilled),
            "source": source,
            **m,
        })
        drawn += 1
        if drawn >= int(max_candidates):
            break
    if drawn == 0:
        text(out, "no candidate", (8, y), (120, 120, 255), 0.42)
    text(out, "magenta=raw WiLoR 2D, green=stage UV", (8, out.shape[0] - 10), WHITE, 0.36)
    return out, panel_rows


def make_sheet(
    video: Path,
    stage_data: Dict[str, Dict[str, np.ndarray]],
    specs: Sequence[StageSpec],
    frames: Sequence[int],
    output: Path,
    hand_filter: str,
    scale: float,
    max_ref_size_px: float,
    max_candidates: int,
) -> List[Dict[str, Any]]:
    panels: List[np.ndarray] = []
    metrics: List[Dict[str, Any]] = []
    for frame in frames:
        image = read_frame(video, int(frame))
        row_panels: List[np.ndarray] = []
        for spec in specs:
            data = stage_data.get(spec.name)
            if data is None:
                panel = image.copy()
                text(panel, f"{spec.label} missing", (8, 20), (80, 80, 255), 0.46)
                rows: List[Dict[str, Any]] = []
            else:
                panel, rows = draw_panel(image, data, spec, int(frame), hand_filter, max_ref_size_px, max_candidates)
            if float(scale) != 1.0:
                panel = cv2.resize(panel, (0, 0), fx=float(scale), fy=float(scale), interpolation=cv2.INTER_AREA)
            row_panels.append(panel)
            metrics.extend(rows)
        panels.append(np.concatenate(row_panels, axis=1))
    if not panels:
        raise RuntimeError("no frames selected")
    sheet = np.concatenate(panels, axis=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), sheet)
    return metrics


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fields = [
        "stage", "stage_label", "frame_index", "row", "hand_label", "track_id", "hand_rank",
        "motion_infilled", "source", "joint_count", "rms_px", "mean_px", "max_px",
        "wrist_px", "mcp_rms_px", "ref_size_px", "stage_size_px",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def summarize_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_stage: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_stage.setdefault(str(row["stage"]), []).append(row)
    out: Dict[str, Any] = {}
    for stage, items in by_stage.items():
        detected = [r for r in items if not bool(r.get("motion_infilled")) and math.isfinite(float(r.get("ref_size_px", float("nan")))) and float(r.get("ref_size_px", 1e9)) <= 900.0]
        out[stage] = {
            "rows": int(len(items)),
            "detected_rows": int(len(detected)),
            "rms_px": stats([float(r["rms_px"]) for r in detected]),
            "wrist_px": stats([float(r["wrist_px"]) for r in detected]),
            "mcp_rms_px": stats([float(r["mcp_rms_px"]) for r in detected]),
            "stage_size_px": stats([float(r["stage_size_px"]) for r in detected]),
            "ref_size_px": stats([float(r["ref_size_px"]) for r in detected]),
        }
    return out


def collect_all_stage_metrics(stage_data: Dict[str, Dict[str, np.ndarray]], specs: Sequence[StageSpec], hand_filter: str, max_ref_size_px: float) -> List[Dict[str, Any]]:
    rows_out: List[Dict[str, Any]] = []
    for spec in specs:
        data = stage_data.get(spec.name)
        if data is None or "frame_index" not in data or "joints_uv" not in data:
            continue
        n = int(len(data["frame_index"]))
        for row in range(n):
            label = string_at(data, "hand_label", row, "")
            if hand_filter != "both" and label != hand_filter:
                continue
            if spec.mode != "locked_fixed" and spec.key not in data:
                continue
            try:
                uv, source = stage_uv(data, spec, row, max_ref_size_px)
            except Exception:
                continue
            ref = np.asarray(data["joints_uv"][row], dtype=np.float64)
            m = uv_metrics(uv, ref)
            rows_out.append({
                "stage": spec.name,
                "stage_label": spec.label,
                "frame_index": int(data["frame_index"][row]),
                "row": int(row),
                "hand_label": label,
                "track_id": int(number_at(data, "track_id", row, -1)),
                "hand_rank": int(number_at(data, "hand_rank", row, 0)),
                "motion_infilled": bool(int(number_at(data, "motion_infilled", row, 0)) != 0),
                "source": source,
                **m,
            })
    return rows_out


def select_worst_frames(rows: Sequence[Dict[str, Any]], stage: str, count: int, max_ref_size_px: float) -> List[int]:
    candidates = []
    for row in rows:
        if str(row.get("stage")) != str(stage):
            continue
        if bool(row.get("motion_infilled")):
            continue
        ref_size = float(row.get("ref_size_px", float("nan")))
        rms = float(row.get("rms_px", float("nan")))
        if not math.isfinite(ref_size) or ref_size > float(max_ref_size_px) or not math.isfinite(rms):
            continue
        candidates.append((rms, int(row["frame_index"])))
    candidates.sort(reverse=True)
    frames: List[int] = []
    for _, frame in candidates:
        if frame not in frames:
            frames.append(frame)
        if len(frames) >= int(count):
            break
    return sorted(frames)


def main() -> int:
    args = parse_args()
    session_dir = Path(args.session_dir).expanduser().resolve()
    source_root = session_dir / "quality" / str(args.source_pipeline) / "stages"
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else session_dir / "quality" / "egoinfinity_hand_alignment_pipeline" / "quality_check" / "stage_overlay_alignment"
    video = Path(args.video).expanduser().resolve() if args.video else session_dir / "processed_topcam" / "left_table.mp4"
    if not video.exists():
        raise RuntimeError(f"missing video: {video}")

    stage_data: Dict[str, Dict[str, np.ndarray]] = {}
    loaded_specs: List[StageSpec] = []
    for spec in STAGES:
        path = source_root / spec.rel_npz
        if not path.exists():
            continue
        stage_data[spec.name] = load_npz(path)
        loaded_specs.append(spec)
    if not loaded_specs:
        raise RuntimeError(f"no stage NPZ files found under {source_root}")

    all_metrics = collect_all_stage_metrics(stage_data, loaded_specs, str(args.hand), float(args.max_ref_hand_size_px))
    output_dir.mkdir(parents=True, exist_ok=True)
    all_csv = output_dir / "stage_overlay_alignment_all_candidates.csv"
    write_csv(all_csv, all_metrics)

    selected_frames = parse_frame_list(str(args.frames))
    selected_sheet = output_dir / "stage_overlay_alignment_selected_frames.jpg"
    selected_metrics = make_sheet(
        video,
        stage_data,
        loaded_specs,
        selected_frames,
        selected_sheet,
        str(args.hand),
        float(args.scale),
        float(args.max_ref_hand_size_px),
        int(args.max_candidates_per_frame),
    )
    selected_csv = output_dir / "stage_overlay_alignment_selected_frames.csv"
    write_csv(selected_csv, selected_metrics)

    worst_frames = select_worst_frames(all_metrics, str(args.worst_stage), int(args.worst_count), float(args.max_ref_hand_size_px))
    worst_sheet = output_dir / f"stage_overlay_alignment_worst_{args.worst_stage}.jpg"
    worst_metrics: List[Dict[str, Any]] = []
    if worst_frames:
        worst_metrics = make_sheet(
            video,
            stage_data,
            loaded_specs,
            worst_frames,
            worst_sheet,
            str(args.hand),
            float(args.scale),
            float(args.max_ref_hand_size_px),
            int(args.max_candidates_per_frame),
        )
    worst_csv = output_dir / f"stage_overlay_alignment_worst_{args.worst_stage}.csv"
    write_csv(worst_csv, worst_metrics)

    summary = {
        "session_dir": str(session_dir),
        "source_pipeline": str(args.source_pipeline),
        "source_root": str(source_root),
        "output_dir": str(output_dir),
        "video": str(video),
        "loaded_stages": [s.name for s in loaded_specs],
        "reference": "raw WiLoR joints_uv stored in the same stage row",
        "stage_uv": "green overlay; magenta is raw joints_uv reference",
        "selected_frames": selected_frames,
        "worst_stage": str(args.worst_stage),
        "worst_frames": worst_frames,
        "all_candidates_csv": str(all_csv),
        "selected_sheet": str(selected_sheet),
        "selected_csv": str(selected_csv),
        "worst_sheet": str(worst_sheet) if worst_frames else "",
        "worst_csv": str(worst_csv),
        "summary_by_stage": summarize_metrics(all_metrics),
    }
    summary_path = output_dir / "stage_overlay_alignment_summary.json"
    summary_path.write_text(json.dumps(json_clean(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(json_clean(summary), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
