#!/usr/bin/env python3
"""Diagnose where Phase-C depth projection introduces hand overlay offset.

This script focuses on the transition:

    Phase-B raw WiLoR 2D/MANO -> Phase-C FoundationStereo depth-aligned MANO

It decomposes the error into:

1. WiLoR camera geometry reprojected with the Foundation/left-table camera.
2. Foundation depth median translation (`cam_t_depth`) projection.
3. Same depth Z, but x/y solved from raw WiLoR 2D (`xy_locked_at_depth_z`).

If (3) is good while (2) is bad, the depth Z is usable but x/y translation was
pulled by inconsistent depth samples or camera-model mismatch.  If (3) is still
bad, the depth Z/scale or MANO relative geometry is not compatible with the raw
2D hand.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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
RELIABLE_IDS = [0, 5, 9, 13, 17]
RAW_COLOR = (255, 0, 255)       # magenta
WILOR_COLOR = (0, 215, 255)     # yellow
DEPTH_COLOR = (45, 45, 255)     # red
LOCKED_COLOR = (60, 255, 60)    # green
WHITE = (245, 245, 245)
BLACK = (0, 0, 0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session-dir", required=True)
    p.add_argument("--source-pipeline", default="egoinfinity_hand_pipeline")
    p.add_argument("--output-dir", default="")
    p.add_argument("--frames", default="94,212,540,562,563,579,590,591,596,616,619,626")
    p.add_argument("--hand", default="both", choices=["both", "left", "right"])
    p.add_argument("--max-candidates-per-frame", type=int, default=4)
    p.add_argument("--worst-count", type=int, default=16)
    p.add_argument("--scale", type=float, default=0.65)
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


def parse_float_csv(text: Any) -> List[float]:
    out: List[float] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        try:
            out.append(float(item))
        except ValueError:
            pass
    return out


def parse_int_csv(text: Any) -> List[int]:
    out: List[int] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        try:
            out.append(int(item))
        except ValueError:
            pass
    return out


def load_camera(data: Dict[str, np.ndarray]) -> Dict[str, float]:
    if "foundation_camera_json" in data:
        try:
            raw = str(np.asarray(data["foundation_camera_json"]).reshape(-1)[0])
            cam = json.loads(raw)
            return {k: float(cam[k]) for k in ("fx", "fy", "cx", "cy")}
        except Exception:
            pass
    size = np.asarray(data.get("image_size", [850, 420])).reshape(-1)
    width = float(size[0]) if size.size >= 1 else 850.0
    height = float(size[1]) if size.size >= 2 else 420.0
    focal = 334.0549853304012
    return {"fx": focal, "fy": focal, "cx": width * 0.5, "cy": height * 0.5}


def project(points_cam: np.ndarray, cam: Dict[str, float]) -> np.ndarray:
    pts = np.asarray(points_cam, dtype=np.float64)
    uv = np.full((pts.shape[0], 2), np.nan, dtype=np.float64)
    z = pts[:, 2]
    valid = np.isfinite(pts).all(axis=1) & (z > 1e-8)
    uv[valid, 0] = float(cam["fx"]) * pts[valid, 0] / z[valid] + float(cam["cx"])
    uv[valid, 1] = float(cam["fy"]) * pts[valid, 1] / z[valid] + float(cam["cy"])
    return uv


def uv_metrics(stage: np.ndarray, ref: np.ndarray) -> Dict[str, float]:
    d = np.asarray(stage, dtype=np.float64) - np.asarray(ref, dtype=np.float64)
    valid = np.isfinite(d).all(axis=1)
    dist = np.linalg.norm(d[valid], axis=1) if np.any(valid) else np.asarray([], dtype=np.float64)
    reliable_valid = valid[RELIABLE_IDS]
    reliable = np.linalg.norm(d[RELIABLE_IDS][reliable_valid], axis=1) if np.any(reliable_valid) else np.asarray([], dtype=np.float64)
    return {
        "joint_count": int(np.sum(valid)),
        "rms_px": float(np.sqrt(np.mean(dist * dist))) if dist.size else float("nan"),
        "mean_px": float(np.mean(dist)) if dist.size else float("nan"),
        "max_px": float(np.max(dist)) if dist.size else float("nan"),
        "wrist_px": float(np.linalg.norm(d[0])) if valid.shape[0] > 0 and bool(valid[0]) else float("nan"),
        "reliable_rms_px": float(np.sqrt(np.mean(reliable * reliable))) if reliable.size else float("nan"),
    }


def solve_xy_from_uv(joints_rel: np.ndarray, uv: np.ndarray, tz: float, cam: Dict[str, float], joint_ids: Sequence[int] | None = None) -> Tuple[np.ndarray, Dict[str, float]]:
    rel = np.asarray(joints_rel, dtype=np.float64)
    pts_uv = np.asarray(uv, dtype=np.float64)
    ids = list(range(rel.shape[0])) if joint_ids is None else [int(v) for v in joint_ids]
    tx_vals: List[float] = []
    ty_vals: List[float] = []
    for jid in ids:
        z = rel[jid, 2] + float(tz)
        u, v = pts_uv[jid]
        if not (math.isfinite(z) and z > 1e-8 and np.isfinite([u, v]).all()):
            continue
        tx_vals.append((float(u) - float(cam["cx"])) * z / float(cam["fx"]) - rel[jid, 0])
        ty_vals.append((float(v) - float(cam["cy"])) * z / float(cam["fy"]) - rel[jid, 1])
    if not tx_vals or not ty_vals:
        return np.asarray([float("nan"), float("nan"), float(tz)], dtype=np.float64), {
            "xy_solve_count": 0,
            "tx_spread_m": float("nan"),
            "ty_spread_m": float("nan"),
        }
    tx_arr = np.asarray(tx_vals, dtype=np.float64)
    ty_arr = np.asarray(ty_vals, dtype=np.float64)
    return np.asarray([float(np.median(tx_arr)), float(np.median(ty_arr)), float(tz)], dtype=np.float64), {
        "xy_solve_count": int(min(tx_arr.size, ty_arr.size)),
        "tx_spread_m": float(np.percentile(tx_arr, 90) - np.percentile(tx_arr, 10)) if tx_arr.size >= 2 else 0.0,
        "ty_spread_m": float(np.percentile(ty_arr, 90) - np.percentile(ty_arr, 10)) if ty_arr.size >= 2 else 0.0,
    }


def fit_tz_from_2d(
    joints_rel: np.ndarray,
    uv: np.ndarray,
    cam: Dict[str, float],
    seed_tz: float,
) -> Tuple[np.ndarray, float, float]:
    """Find the camera z that best preserves raw 2D hand size/projection.

    For every tested z we solve the best common x/y translation from raw UV, then
    score the reprojection RMS.  This intentionally uses only WiLoR 2D + MANO
    relative geometry, not FoundationStereo depth.
    """
    seed = float(seed_tz) if math.isfinite(float(seed_tz)) and float(seed_tz) > 0.0 else 0.4
    lo = max(0.06, seed * 0.45)
    hi = min(1.5, seed * 1.8 + 0.05)
    best_t = np.asarray([float("nan"), float("nan"), float("nan")], dtype=np.float64)
    best_tz = float("nan")
    best_rms = float("inf")
    for grid_lo, grid_hi, steps in ((lo, hi, 90), (None, None, 80)):
        if grid_lo is None:
            if not math.isfinite(best_tz):
                continue
            grid_lo = max(0.05, best_tz - 0.04)
            grid_hi = best_tz + 0.04
        for tz in np.linspace(float(grid_lo), float(grid_hi), int(steps)):
            t, _ = solve_xy_from_uv(joints_rel, uv, float(tz), cam)
            if not np.isfinite(t).all():
                continue
            err = uv_metrics(project(joints_rel + t.reshape(1, 3), cam), uv)["rms_px"]
            if math.isfinite(err) and err < best_rms:
                best_rms = float(err)
                best_tz = float(tz)
                best_t = t
    return best_t, best_tz, best_rms


def translation_candidates(
    joints_rel: np.ndarray,
    uv: np.ndarray,
    joint_ids: Sequence[int],
    depths: Sequence[float],
    cam: Dict[str, float],
) -> np.ndarray:
    rel = np.asarray(joints_rel, dtype=np.float64)
    pts_uv = np.asarray(uv, dtype=np.float64)
    rows: List[np.ndarray] = []
    for jid, depth in zip(joint_ids, depths):
        jid = int(jid)
        d = float(depth)
        u, v = pts_uv[jid]
        if not (math.isfinite(d) and d > 0.0 and np.isfinite([u, v]).all()):
            continue
        p = np.asarray([
            (u - float(cam["cx"])) * d / float(cam["fx"]),
            (v - float(cam["cy"])) * d / float(cam["fy"]),
            d,
        ], dtype=np.float64)
        rows.append(p - rel[jid])
    if not rows:
        return np.zeros((0, 3), dtype=np.float64)
    return np.stack(rows, axis=0)


def rows_by_frame(data: Dict[str, np.ndarray]) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    for i, frame in enumerate(np.asarray(data["frame_index"], dtype=np.int64).tolist()):
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


def text(img: np.ndarray, s: str, org: Tuple[int, int], color: Tuple[int, int, int] = WHITE, scale: float = 0.42) -> None:
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_skeleton(img: np.ndarray, uv: np.ndarray, color: Tuple[int, int, int], thickness: int = 2, radius: int = 3, alpha: float = 0.80) -> None:
    overlay = img.copy()
    h, w = img.shape[:2]

    def ok(p: np.ndarray) -> bool:
        return np.isfinite(p).all() and -80 <= float(p[0]) <= w + 80 and -80 <= float(p[1]) <= h + 80

    for a, b in HAND_EDGES:
        if ok(uv[a]) and ok(uv[b]):
            pa = (int(round(float(uv[a, 0]))), int(round(float(uv[a, 1]))))
            pb = (int(round(float(uv[b, 0]))), int(round(float(uv[b, 1]))))
            cv2.line(overlay, pa, pb, color, int(thickness), cv2.LINE_AA)
    for jid in range(min(21, uv.shape[0])):
        if ok(uv[jid]):
            p = (int(round(float(uv[jid, 0]))), int(round(float(uv[jid, 1]))))
            cv2.circle(overlay, p, radius + (1 if jid in RELIABLE_IDS else 0), color, -1, cv2.LINE_AA)
            cv2.circle(overlay, p, radius + 1, WHITE, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0.0, dst=img)


def make_panel(image: np.ndarray, row: Dict[str, Any], title: str, uv: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    out = image.copy()
    draw_skeleton(out, row["raw_uv"], RAW_COLOR, 1, 2, 0.45)
    draw_skeleton(out, uv, color, 2, 3, 0.86)
    text(out, title, (8, 18), color, 0.46)
    text(out, f"raw magenta | rms={row[title + '_rms_px']:.1f}px wrist={row[title + '_wrist_px']:.1f}px", (8, 36), WHITE, 0.36)
    text(out, f"label={row['hand_label']} tr={row['track_id']} src={row['alignment_source']} used={row['used_joint_ids']}", (8, 54), WHITE, 0.34)
    return out


def tile_images(images: Sequence[np.ndarray], columns: int, scale: float) -> np.ndarray:
    if not images:
        raise RuntimeError("no images to tile")
    h, w = images[0].shape[:2]
    rows = int(math.ceil(len(images) / float(columns)))
    canvas = np.zeros((rows * h, columns * w, 3), dtype=np.uint8)
    for idx, img in enumerate(images):
        y = (idx // columns) * h
        x = (idx % columns) * w
        canvas[y:y + h, x:x + w] = img
    if scale != 1.0:
        canvas = cv2.resize(canvas, (max(1, int(canvas.shape[1] * scale)), max(1, int(canvas.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    return canvas


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fields = [
        "frame_index", "row", "hand_label", "track_id", "hand_rank", "alignment_source",
        "used_joint_ids", "used_depths_m", "used_depth_count",
        "phase_c_rms_px", "phase_c_wrist_px", "phase_c_reliable_rms_px",
        "wilor_foundation_rms_px", "wilor_foundation_wrist_px",
        "xy_locked_depth_rms_px", "xy_locked_depth_wrist_px", "xy_locked_depth_reliable_rms_px",
        "xy_locked_reliable_rms_px", "xy_locked_reliable_wrist_px",
        "depth_minus_wilor_foundation_rms_px", "phase_c_minus_xy_locked_rms_px",
        "cam_t_wilor_x", "cam_t_wilor_y", "cam_t_wilor_z",
        "cam_t_depth_x", "cam_t_depth_y", "cam_t_depth_z",
        "cam_t_depth_delta_x", "cam_t_depth_delta_y", "cam_t_depth_delta_z",
        "cam_t_depth_delta_norm_m", "cam_t_xy_locked_x", "cam_t_xy_locked_y", "cam_t_xy_locked_z",
        "cam_t_xy_locked_delta_x", "cam_t_xy_locked_delta_y", "cam_t_xy_locked_delta_norm_m",
        "translation_candidate_count", "translation_tx_spread_m", "translation_ty_spread_m",
        "translation_tz_spread_m", "translation_t_norm_spread_m", "alignment_rms_m",
        "best_2d_fit_rms_px", "best_2d_fit_z_m", "depth_over_best_2d_z",
        "cam_t_best_2d_x", "cam_t_best_2d_y", "cam_t_best_2d_z",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def main() -> int:
    args = parse_args()
    session_dir = Path(args.session_dir).expanduser().resolve()
    source_root = session_dir / "quality" / str(args.source_pipeline) / "stages"
    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else session_dir / "quality" / "egoinfinity_hand_alignment_pipeline" / "quality_check" / "phase_c_depth_project_diagnostic"
    out_dir.mkdir(parents=True, exist_ok=True)
    video = session_dir / "processed_topcam" / "left_table.mp4"

    phase_b = load_npz(source_root / "phase_b_track_postprocess" / "wilor_handresults_phase_b.npz")
    phase_c = load_npz(source_root / "phase_c_depth_align" / "wilor_handresults_phase_c_depth_aligned.npz")
    cam = load_camera(phase_c)
    frames = parse_frame_list(args.frames)
    row_map = rows_by_frame(phase_c)

    rows: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []
    for i in range(len(phase_c["frame_index"])):
        label = str(phase_c["hand_label"][i])
        if args.hand != "both" and label != args.hand:
            continue
        raw_uv = np.asarray(phase_c["joints_uv"][i], dtype=np.float64)
        joints_rel = np.asarray(phase_c["joints_3d_rel"][i], dtype=np.float64)
        cam_t_wilor = np.asarray(phase_c["cam_t_wilor"][i], dtype=np.float64)
        cam_t_depth = np.asarray(phase_c["cam_t_depth"][i], dtype=np.float64)
        uv_wilor_foundation = project(joints_rel + cam_t_wilor.reshape(1, 3), cam)
        uv_depth = project(np.asarray(phase_c["joints_cam_depth"][i], dtype=np.float64), cam)
        xy_t_all, xy_info_all = solve_xy_from_uv(joints_rel, raw_uv, float(cam_t_depth[2]), cam)
        uv_xy_locked = project(joints_rel + xy_t_all.reshape(1, 3), cam)
        used_ids = parse_int_csv(phase_c["alignment_joint_ids"][i])
        used_depths = parse_float_csv(phase_c["alignment_sampled_depths_m"][i])
        reliable_for_xy = used_ids if used_ids else RELIABLE_IDS
        xy_t_rel, xy_info_rel = solve_xy_from_uv(joints_rel, raw_uv, float(cam_t_depth[2]), cam, reliable_for_xy)
        uv_xy_rel = project(joints_rel + xy_t_rel.reshape(1, 3), cam)
        best_2d_t, best_2d_z, best_2d_rms = fit_tz_from_2d(joints_rel, raw_uv, cam, float(cam_t_depth[2]))
        trans = translation_candidates(joints_rel, raw_uv, used_ids, used_depths, cam)
        trans_spread = np.percentile(trans, 90, axis=0) - np.percentile(trans, 10, axis=0) if trans.shape[0] >= 2 else np.full(3, np.nan)
        trans_norm = np.linalg.norm(trans - np.median(trans, axis=0).reshape(1, 3), axis=1) if trans.shape[0] else np.asarray([], dtype=np.float64)

        m_depth = uv_metrics(uv_depth, raw_uv)
        m_wilor = uv_metrics(uv_wilor_foundation, raw_uv)
        m_xy = uv_metrics(uv_xy_locked, raw_uv)
        m_xy_rel = uv_metrics(uv_xy_rel, raw_uv)
        m_depth_vs_wilor = uv_metrics(uv_depth, uv_wilor_foundation)
        m_depth_vs_xy = uv_metrics(uv_depth, uv_xy_locked)

        row = {
            "frame_index": int(phase_c["frame_index"][i]),
            "row": int(i),
            "hand_label": label,
            "track_id": int(phase_c["track_id"][i]),
            "hand_rank": int(phase_c["hand_rank"][i]),
            "alignment_source": str(phase_c["alignment_source"][i]),
            "used_joint_ids": ",".join(str(v) for v in used_ids),
            "used_depths_m": ",".join(f"{v:.6f}" for v in used_depths),
            "used_depth_count": int(len(used_depths)),
            "raw_uv": raw_uv,
            "uv_depth": uv_depth,
            "uv_wilor_foundation": uv_wilor_foundation,
            "uv_xy_locked": uv_xy_locked,
            "uv_xy_locked_reliable": uv_xy_rel,
            "phase_c_rms_px": m_depth["rms_px"],
            "phase_c_wrist_px": m_depth["wrist_px"],
            "phase_c_reliable_rms_px": m_depth["reliable_rms_px"],
            "wilor_foundation_rms_px": m_wilor["rms_px"],
            "wilor_foundation_wrist_px": m_wilor["wrist_px"],
            "xy_locked_depth_rms_px": m_xy["rms_px"],
            "xy_locked_depth_wrist_px": m_xy["wrist_px"],
            "xy_locked_depth_reliable_rms_px": m_xy["reliable_rms_px"],
            "xy_locked_reliable_rms_px": m_xy_rel["rms_px"],
            "xy_locked_reliable_wrist_px": m_xy_rel["wrist_px"],
            "depth_minus_wilor_foundation_rms_px": m_depth_vs_wilor["rms_px"],
            "phase_c_minus_xy_locked_rms_px": m_depth_vs_xy["rms_px"],
            "cam_t_wilor_x": float(cam_t_wilor[0]),
            "cam_t_wilor_y": float(cam_t_wilor[1]),
            "cam_t_wilor_z": float(cam_t_wilor[2]),
            "cam_t_depth_x": float(cam_t_depth[0]),
            "cam_t_depth_y": float(cam_t_depth[1]),
            "cam_t_depth_z": float(cam_t_depth[2]),
            "cam_t_depth_delta_x": float(cam_t_depth[0] - cam_t_wilor[0]),
            "cam_t_depth_delta_y": float(cam_t_depth[1] - cam_t_wilor[1]),
            "cam_t_depth_delta_z": float(cam_t_depth[2] - cam_t_wilor[2]),
            "cam_t_depth_delta_norm_m": float(np.linalg.norm(cam_t_depth - cam_t_wilor)),
            "cam_t_xy_locked_x": float(xy_t_all[0]),
            "cam_t_xy_locked_y": float(xy_t_all[1]),
            "cam_t_xy_locked_z": float(xy_t_all[2]),
            "cam_t_xy_locked_delta_x": float(cam_t_depth[0] - xy_t_all[0]),
            "cam_t_xy_locked_delta_y": float(cam_t_depth[1] - xy_t_all[1]),
            "cam_t_xy_locked_delta_norm_m": float(np.linalg.norm(cam_t_depth[:2] - xy_t_all[:2])),
            "translation_candidate_count": int(trans.shape[0]),
            "translation_tx_spread_m": float(trans_spread[0]),
            "translation_ty_spread_m": float(trans_spread[1]),
            "translation_tz_spread_m": float(trans_spread[2]),
            "translation_t_norm_spread_m": float(np.percentile(trans_norm, 90) - np.percentile(trans_norm, 10)) if trans_norm.size >= 2 else float("nan"),
            "alignment_rms_m": float(phase_c["alignment_rms_m"][i]),
            "best_2d_fit_rms_px": float(best_2d_rms),
            "best_2d_fit_z_m": float(best_2d_z),
            "depth_over_best_2d_z": float(cam_t_depth[2] / best_2d_z) if math.isfinite(best_2d_z) and abs(best_2d_z) > 1e-9 else float("nan"),
            "cam_t_best_2d_x": float(best_2d_t[0]),
            "cam_t_best_2d_y": float(best_2d_t[1]),
            "cam_t_best_2d_z": float(best_2d_t[2]),
            "_xy_info_all": xy_info_all,
            "_xy_info_rel": xy_info_rel,
        }
        all_rows.append(row)

    csv_rows = [{k: v for k, v in row.items() if not k.startswith("_") and not isinstance(v, np.ndarray)} for row in all_rows]
    csv_path = out_dir / "phase_c_depth_project_diagnostic_all_candidates.csv"
    write_csv(csv_path, csv_rows)
    tz_fit_csv = out_dir / "phase_c_depth_project_2d_tz_fit.csv"
    write_csv(tz_fit_csv, csv_rows)

    selected_rows: List[Dict[str, Any]] = []
    for frame in frames:
        candidates = []
        for row_i in row_map.get(int(frame), []):
            label = str(phase_c["hand_label"][row_i])
            if args.hand != "both" and label != args.hand:
                continue
            match = next((r for r in all_rows if r["row"] == row_i), None)
            if match is not None:
                candidates.append(match)
        candidates.sort(key=lambda r: (str(r["hand_label"]), int(r["hand_rank"])))
        selected_rows.extend(candidates[: max(1, int(args.max_candidates_per_frame))])

    worst_rows = sorted(all_rows, key=lambda r: float(r["phase_c_rms_px"]), reverse=True)[: int(args.worst_count)]

    def make_sheet(rows_for_sheet: Sequence[Dict[str, Any]], out_path: Path) -> None:
        panels: List[np.ndarray] = []
        for row in rows_for_sheet:
            img = read_frame(video, int(row["frame_index"]))
            base = img.copy()
            text(base, f"frame={row['frame_index']} {row['hand_label']} tr={row['track_id']}", (8, 18), WHITE, 0.48)
            draw_skeleton(base, row["raw_uv"], RAW_COLOR, 2, 3, 0.86)
            panels.append(base)
            panels.append(make_panel(img, row, "wilor_foundation", row["uv_wilor_foundation"], WILOR_COLOR))
            panels.append(make_panel(img, row, "phase_c", row["uv_depth"], DEPTH_COLOR))
            panels.append(make_panel(img, row, "xy_locked_depth", row["uv_xy_locked"], LOCKED_COLOR))
        if panels:
            cv2.imwrite(str(out_path), tile_images(panels, columns=4, scale=float(args.scale)))

    selected_sheet = out_dir / "phase_c_depth_project_selected_frames.jpg"
    worst_sheet = out_dir / "phase_c_depth_project_worst_frames.jpg"
    make_sheet(selected_rows, selected_sheet)
    make_sheet(worst_rows, worst_sheet)

    summary = {
        "session_dir": str(session_dir),
        "source_pipeline": str(args.source_pipeline),
        "source_root": str(source_root),
        "output_dir": str(out_dir),
        "video": str(video),
        "camera": cam,
        "csv": str(csv_path),
        "tz_fit_csv": str(tz_fit_csv),
        "selected_sheet": str(selected_sheet),
        "worst_sheet": str(worst_sheet),
        "selected_frames": frames,
        "hand_filter": str(args.hand),
        "stage_meaning": {
            "wilor_foundation": "Phase-B MANO joints_rel + WiLoR cam_t projected with Foundation/left-table camera.",
            "phase_c": "Phase-C joints_cam_depth projected with Foundation/left-table camera.",
            "xy_locked_depth": "Use Phase-C depth z, but solve x/y translation from raw WiLoR joints_uv.",
        },
        "summary_by_metric": {
            "phase_c_rms_px": stats(r["phase_c_rms_px"] for r in all_rows),
            "wilor_foundation_rms_px": stats(r["wilor_foundation_rms_px"] for r in all_rows),
            "xy_locked_depth_rms_px": stats(r["xy_locked_depth_rms_px"] for r in all_rows),
            "xy_locked_reliable_rms_px": stats(r["xy_locked_reliable_rms_px"] for r in all_rows),
            "depth_minus_wilor_foundation_rms_px": stats(r["depth_minus_wilor_foundation_rms_px"] for r in all_rows),
            "phase_c_minus_xy_locked_rms_px": stats(r["phase_c_minus_xy_locked_rms_px"] for r in all_rows),
            "cam_t_depth_delta_norm_m": stats(r["cam_t_depth_delta_norm_m"] for r in all_rows),
            "cam_t_xy_locked_delta_norm_m": stats(r["cam_t_xy_locked_delta_norm_m"] for r in all_rows),
            "translation_tx_spread_m": stats(r["translation_tx_spread_m"] for r in all_rows),
            "translation_ty_spread_m": stats(r["translation_ty_spread_m"] for r in all_rows),
            "translation_tz_spread_m": stats(r["translation_tz_spread_m"] for r in all_rows),
            "alignment_rms_m": stats(r["alignment_rms_m"] for r in all_rows),
            "best_2d_fit_rms_px": stats(r["best_2d_fit_rms_px"] for r in all_rows),
            "best_2d_fit_z_m": stats(r["best_2d_fit_z_m"] for r in all_rows),
            "depth_over_best_2d_z": stats(r["depth_over_best_2d_z"] for r in all_rows),
        },
    }
    summary_path = out_dir / "phase_c_depth_project_diagnostic_summary.json"
    summary_path.write_text(json.dumps(json_clean(summary), indent=2), encoding="utf-8")
    print(f"[phase_c_depth_project_diagnostic] summary: {summary_path}")
    print(f"[phase_c_depth_project_diagnostic] csv: {csv_path}")
    print(f"[phase_c_depth_project_diagnostic] selected sheet: {selected_sheet}")
    print(f"[phase_c_depth_project_diagnostic] worst sheet: {worst_sheet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
