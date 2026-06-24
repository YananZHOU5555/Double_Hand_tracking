#!/usr/bin/env python3
"""EgoInfinity Phase-C depth alignment using LFV FoundationStereo depth.

This stage keeps WiLoR/MANO root-relative geometry and estimates a metric
camera-space translation (`cam_t_depth`) from FoundationStereo depth sampled at
reliable hand joints.  It is the first Phase-C stage before temporal smoothing
and MANO forward.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from egoinfinity_strict.depth_align import (  # noqa: E402
    align_hand_to_depth_multiscale_lfv,
    rescale_cam_t,
)
from egoinfinity_strict.depth_stabilize import (  # noqa: E402
    build_dynamic_mask,
    build_optical_flow_masks,
    compute_background_template,
    estimate_frame_correction,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session-dir", required=True)
    p.add_argument("--input-npz", required=True, help="Phase-B HandResult NPZ")
    p.add_argument("--depth-summary-json", required=True)
    p.add_argument("--depth-frame-csv", default="")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--frame-start", type=int, default=0)
    p.add_argument("--frame-end", type=int, default=-1)
    p.add_argument("--align-patch-size", type=int, default=7)
    p.add_argument("--min-reliable-joints", type=int, default=2)
    p.add_argument("--stabilize-depth", action="store_true")
    p.add_argument("--use-flow-mask", action="store_true")
    p.add_argument("--bbox-margin", type=float, default=0.30)
    p.add_argument("--write-stable-depth", action="store_true")
    p.add_argument("--max-warn-rms-m", type=float, default=0.035)
    p.add_argument("--max-bad-rms-m", type=float, default=0.080)
    p.add_argument("--warn-jump-m", type=float, default=0.080)
    p.add_argument("--bad-jump-m", type=float, default=0.150)
    p.add_argument("--truncated-bbox-visible-ratio", type=float, default=0.85)
    p.add_argument("--truncated-joint-visible-ratio", type=float, default=0.70)
    p.add_argument("--edge-margin-px", type=float, default=3.0)
    return p.parse_args()


def csv_float(value: float) -> str:
    return f"{float(value):.9f}" if math.isfinite(float(value)) else ""


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


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def read_depth_frame_csv(path: Path) -> Dict[int, Dict[str, str]]:
    out: Dict[int, Dict[str, str]] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            frame = int(row["frame_index"])
            out[frame] = row
    return out


def read_video_frames(video_path: Path, frames: Sequence[int]) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    out: List[np.ndarray] = []
    for frame in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
        ok, image = cap.read()
        if not ok or image is None:
            raise RuntimeError(f"failed to read video frame {frame}: {video_path}")
        out.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    cap.release()
    return out


def selected_indices(data: Dict[str, np.ndarray], frame_start: int, frame_end: int) -> np.ndarray:
    frames = data["frame_index"].astype(np.int32)
    keep = frames >= int(frame_start)
    if int(frame_end) >= 0:
        keep &= frames <= int(frame_end)
    return np.where(keep)[0].astype(np.int32)


def make_depth_sequence(
    depth_rows: Dict[int, Dict[str, str]],
    frames: Sequence[int],
) -> Tuple[List[np.ndarray], Dict[int, np.ndarray], List[int]]:
    depth_maps: List[np.ndarray] = []
    missing: List[int] = []
    by_frame: Dict[int, np.ndarray] = {}
    for frame in frames:
        row = depth_rows.get(int(frame))
        if row is None:
            missing.append(int(frame))
            continue
        path = Path(row.get("depth_npy", ""))
        if not path.exists():
            missing.append(int(frame))
            continue
        depth = np.load(path).astype(np.float32)
        depth_maps.append(depth)
        by_frame[int(frame)] = depth
    return depth_maps, by_frame, missing


def stabilize_depth_maps(
    session_dir: Path,
    raw_depth_by_frame: Dict[int, np.ndarray],
    bboxes_by_frame: Dict[int, List[np.ndarray]],
    frames: Sequence[int],
    bbox_margin: float,
    use_flow_mask: bool,
    write_dir: Optional[Path],
) -> Tuple[Dict[int, np.ndarray], Dict[str, Any]]:
    if not frames:
        return {}, {"enabled": False, "reason": "no_frames"}
    first = raw_depth_by_frame[int(frames[0])]
    h, w = first.shape[:2]
    raw_depths = [raw_depth_by_frame[int(f)] for f in frames]

    flow_masks = None
    flow_status = "disabled"
    if use_flow_mask:
        left_video = session_dir / "processed_topcam" / "left_table.mp4"
        gray = read_video_frames(left_video, frames)
        flow_masks = build_optical_flow_masks(gray, magnitude_threshold=2.0, temporal_window=3, dilate_px=7)
        flow_status = "memfof"

    dynamic_masks = []
    for i, frame in enumerate(frames):
        mask = build_dynamic_mask(h, w, bboxes_by_frame.get(int(frame), []), margin=float(bbox_margin))
        if flow_masks is not None and i < len(flow_masks):
            mask = mask | flow_masks[i]
        dynamic_masks.append(mask)

    template = compute_background_template(raw_depths, dynamic_masks)
    template_valid = np.isfinite(template) & (template > 0.05)

    stable_by_frame: Dict[int, np.ndarray] = {}
    scales: List[float] = []
    offsets: List[float] = []
    for i, frame in enumerate(frames):
        scale, offset = estimate_frame_correction(raw_depths[i], template, dynamic_masks[i])
        scales.append(float(scale))
        offsets.append(float(offset))
        stable = raw_depths[i] * float(scale) + float(offset)
        stable = np.maximum(stable, 0.0)
        stable_by_frame[int(frame)] = stable.astype(np.float32)
        if write_dir is not None:
            write_dir.mkdir(parents=True, exist_ok=True)
            np.save(write_dir / f"depth_stable_{int(frame):08d}.npy", stable_by_frame[int(frame)])

    summary = {
        "enabled": True,
        "flow_mask_status": flow_status,
        "bbox_margin": float(bbox_margin),
        "frames": len(frames),
        "template_valid_ratio": float(np.mean(template_valid)),
        "scale": stats(scales),
        "offset_m": stats(offsets),
    }
    if write_dir is not None:
        np.save(write_dir / "background_depth_template.npy", template.astype(np.float32))
        summary["stable_depth_dir"] = str(write_dir)
        summary["background_depth_template"] = str(write_dir / "background_depth_template.npy")
    return stable_by_frame, summary


def candidate_qc_flag(source: str, rms: float, jump: float, args: argparse.Namespace) -> str:
    flags: List[str] = []
    if source != "depth_reliable_joints":
        flags.append(source)
    if math.isfinite(rms):
        if rms > float(args.max_bad_rms_m):
            flags.append("bad_alignment_rms")
        elif rms > float(args.max_warn_rms_m):
            flags.append("warn_alignment_rms")
    else:
        flags.append("missing_alignment_rms")
    if math.isfinite(jump):
        if jump > float(args.bad_jump_m):
            flags.append("bad_cam_t_jump")
        elif jump > float(args.warn_jump_m):
            flags.append("warn_cam_t_jump")
    return "|".join(flags) if flags else "ok"


def clipped_bbox_visible_ratio(bbox: np.ndarray, width: int, height: int) -> float:
    if bbox.shape[0] != 4 or not np.isfinite(bbox).all():
        return float("nan")
    x1, y1, x2, y2 = [float(v) for v in bbox]
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if area <= 1e-6:
        return 0.0
    cx1, cy1 = min(max(x1, 0.0), float(width)), min(max(y1, 0.0), float(height))
    cx2, cy2 = min(max(x2, 0.0), float(width)), min(max(y2, 0.0), float(height))
    clipped = max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1)
    return float(clipped / area)


def joint_visible_ratio(uv: np.ndarray, width: int, height: int, margin_px: float) -> float:
    if uv.ndim != 2 or uv.shape[1] != 2:
        return float("nan")
    finite = np.isfinite(uv).all(axis=1)
    if not np.any(finite):
        return 0.0
    pts = uv[finite]
    m = float(margin_px)
    inside = (
        (pts[:, 0] >= -m) & (pts[:, 0] < float(width) + m) &
        (pts[:, 1] >= -m) & (pts[:, 1] < float(height) + m)
    )
    return float(np.mean(inside))


def bbox_touches_image_edge(bbox: np.ndarray, width: int, height: int, margin_px: float) -> bool:
    if bbox.shape[0] != 4 or not np.isfinite(bbox).all():
        return False
    x1, y1, x2, y2 = [float(v) for v in bbox]
    m = float(margin_px)
    return bool(x1 <= m or y1 <= m or x2 >= float(width) - m or y2 >= float(height) - m)


def candidate_diagnosis(
    source: str,
    qc_flag: str,
    rms: float,
    jump: float,
    bbox: np.ndarray,
    uv: np.ndarray,
    width: int,
    height: int,
    args: argparse.Namespace,
) -> Tuple[str, str, float, float, int]:
    """Attach human-readable diagnosis without changing the numeric QC gate."""
    bbox_ratio = clipped_bbox_visible_ratio(bbox, width, height)
    joint_ratio = joint_visible_ratio(uv, width, height, float(args.edge_margin_px))
    touches_edge = int(bbox_touches_image_edge(bbox, width, height, float(args.edge_margin_px)))

    tags: List[str] = []
    if (
        (math.isfinite(bbox_ratio) and bbox_ratio < float(args.truncated_bbox_visible_ratio)) or
        (math.isfinite(joint_ratio) and joint_ratio < float(args.truncated_joint_visible_ratio))
    ):
        tags.append("truncated_or_out_of_view")
    elif touches_edge:
        tags.append("near_image_edge")
    if source == "missing_depth_frame":
        tags.append("missing_depth_frame")
    elif source == "depth_all_joints":
        tags.append("fallback_all_joints_low_reliable_depth")
    elif source not in ("depth_reliable_joints", "not_selected"):
        tags.append(f"alignment_source_{source}")

    if "bad_cam_t_jump" in qc_flag:
        tags.append("temporal_cam_t_jump_bad")
    elif "warn_cam_t_jump" in qc_flag:
        tags.append("temporal_cam_t_jump_warn")

    if math.isfinite(rms):
        if rms > float(args.max_bad_rms_m):
            tags.append("depth_alignment_residual_high")
        elif rms > float(args.max_warn_rms_m):
            tags.append("depth_alignment_residual_warn")
    else:
        tags.append("missing_alignment_rms")

    if not tags:
        tags.append("ok")

    priority = [
        "missing_depth_frame",
        "truncated_or_out_of_view",
        "temporal_cam_t_jump_bad",
        "fallback_all_joints_low_reliable_depth",
        "depth_alignment_residual_high",
        "near_image_edge",
        "temporal_cam_t_jump_warn",
        "depth_alignment_residual_warn",
    ]
    primary = tags[0]
    for item in priority:
        if item in tags:
            primary = item
            break
    return primary, "|".join(tags), bbox_ratio, joint_ratio, touches_edge


def write_alignment_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fields = [
        "frame_index", "elapsed_sec", "candidate_index", "track_id", "hand_rank",
        "hand_label", "is_right", "det_conf", "alignment_source",
        "alignment_valid_joint_count", "alignment_joint_ids",
        "alignment_sampled_depths_m", "alignment_rms_m",
        "alignment_max_residual_m", "cam_t_wilor_x", "cam_t_wilor_y",
        "cam_t_wilor_z", "cam_t_depth_x", "cam_t_depth_y", "cam_t_depth_z",
        "cam_t_delta_m", "cam_t_jump_m", "bbox_visible_ratio",
        "joint_visible_ratio", "bbox_touches_edge", "qc_flag",
        "diagnosis_category", "qc_issue_tags",
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
    input_npz = Path(args.input_npz).expanduser().resolve()
    depth_summary_json = Path(args.depth_summary_json).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_npz.exists():
        raise RuntimeError(f"missing input npz: {input_npz}")
    if not depth_summary_json.exists():
        raise RuntimeError(f"missing depth summary: {depth_summary_json}")

    data = load_npz(input_npz)
    depth_summary = load_json(depth_summary_json)
    depth_frame_csv = Path(args.depth_frame_csv).expanduser().resolve() if args.depth_frame_csv else Path(depth_summary["outputs"]["frame_csv"])
    depth_rows = read_depth_frame_csv(depth_frame_csv)
    camera = depth_summary.get("camera") or {}
    fx = float(camera["fx"])
    fy = float(camera["fy"])
    cx = float(camera["cx"])
    cy = float(camera["cy"])

    keep = selected_indices(data, int(args.frame_start), int(args.frame_end))
    if keep.size == 0:
        raise RuntimeError(f"no Phase-B candidates in frame range {args.frame_start}..{args.frame_end}")

    selected_frames = sorted(set(int(data["frame_index"][i]) for i in keep.tolist()))
    depth_frames = [f for f in selected_frames if f in depth_rows]
    raw_depth_list, raw_depth_by_frame, missing_depth_frames = make_depth_sequence(depth_rows, depth_frames)
    if not raw_depth_by_frame:
        raise RuntimeError("no depth maps available for selected candidates")

    bboxes_by_frame: Dict[int, List[np.ndarray]] = defaultdict(list)
    for i in keep.tolist():
        bboxes_by_frame[int(data["frame_index"][i])].append(np.asarray(data["bbox_xyxy"][i], dtype=np.float32))

    stable_summary: Dict[str, Any]
    if args.stabilize_depth:
        stable_write_dir = output_dir / "depth_stable" if args.write_stable_depth else None
        depth_by_frame, stable_summary = stabilize_depth_maps(
            session_dir,
            raw_depth_by_frame,
            bboxes_by_frame,
            sorted(raw_depth_by_frame),
            float(args.bbox_margin),
            bool(args.use_flow_mask),
            stable_write_dir,
        )
    else:
        depth_by_frame = raw_depth_by_frame
        stable_summary = {"enabled": False, "flow_mask_status": "not_used"}

    n_all = int(len(data["frame_index"]))
    out: Dict[str, np.ndarray] = {}
    per_candidate_keys = {
        "frame_index", "elapsed_sec", "hand_rank", "candidate_index", "track_id",
        "hand_label", "is_right", "det_conf", "bbox_xyxy", "cam_t", "pred_cam",
        "focal_length", "global_orient", "hand_pose", "betas", "joints_3d_rel",
        "vertices_rel", "joints_cam", "vertices_cam", "joints_uv",
    }
    for key, value in data.items():
        if key in per_candidate_keys and getattr(value, "shape", (0,))[0] == n_all:
            out[key] = value[keep]
        else:
            out[key] = value

    cam_t_wilor = np.asarray(data["cam_t"][keep], dtype=np.float32)
    joints_rel = np.asarray(data["joints_3d_rel"][keep], dtype=np.float32)
    verts_rel = np.asarray(data["vertices_rel"][keep], dtype=np.float32)
    joints_uv = np.asarray(data["joints_uv"][keep], dtype=np.float32)
    focal = np.asarray(data["focal_length"][keep], dtype=np.float32)

    cam_t_depth = np.full_like(cam_t_wilor, np.nan, dtype=np.float32)
    alignment_source: List[str] = []
    alignment_valid_joint_count: List[int] = []
    alignment_joint_ids: List[str] = []
    alignment_depths: List[str] = []
    alignment_rms: List[float] = []
    alignment_max_residual: List[float] = []
    diagnosis_category: List[str] = []
    qc_issue_tags: List[str] = []
    bbox_visible_ratios: List[float] = []
    joint_visible_ratios: List[float] = []
    bbox_edge_flags: List[int] = []

    alignment_rows: List[Dict[str, Any]] = []
    previous_by_track: Dict[int, Tuple[int, np.ndarray]] = {}
    image_size = np.asarray(data.get("image_size", [int(camera.get("width", 0)), int(camera.get("height", 0))]), dtype=np.int32).reshape(-1)
    if image_size.size >= 2 and int(image_size[0]) > 0 and int(image_size[1]) > 0:
        image_width, image_height = int(image_size[0]), int(image_size[1])
    else:
        image_width, image_height = int(camera.get("width", 0)), int(camera.get("height", 0))

    for out_i, src_i in enumerate(keep.tolist()):
        frame = int(data["frame_index"][src_i])
        depth_map = depth_by_frame.get(frame)
        if depth_map is None:
            focal_ref = float((fx + fy) * 0.5)
            aligned = rescale_cam_t(cam_t_wilor[out_i], float(focal[out_i]), focal_ref)
            info = {
                "source": "missing_depth_frame",
                "valid_joint_count": 0,
                "joint_ids": [],
                "sampled_depths_m": [],
                "rms_m": float("nan"),
                "max_residual_m": float("nan"),
            }
        else:
            aligned, info = align_hand_to_depth_multiscale_lfv(
                joints_rel[out_i],
                joints_uv[out_i],
                depth_map,
                fx,
                fy,
                cx,
                cy,
                cam_t_wilor[out_i],
                float(focal[out_i]),
                patch_size=int(args.align_patch_size),
                min_reliable_joints=int(args.min_reliable_joints),
            )
        cam_t_depth[out_i] = np.asarray(aligned, dtype=np.float32)
        source = str(info.get("source", "unknown"))
        valid_count = int(info.get("valid_joint_count", 0))
        ids = [int(v) for v in info.get("joint_ids", [])]
        depths = [float(v) for v in info.get("sampled_depths_m", [])]
        rms = float(info.get("rms_m", float("nan")))
        max_res = float(info.get("max_residual_m", float("nan")))

        alignment_source.append(source)
        alignment_valid_joint_count.append(valid_count)
        alignment_joint_ids.append(",".join(str(v) for v in ids))
        alignment_depths.append(",".join(csv_float(v) for v in depths))
        alignment_rms.append(rms)
        alignment_max_residual.append(max_res)

        tid = int(data["track_id"][src_i])
        jump = float("nan")
        if tid in previous_by_track:
            _, prev = previous_by_track[tid]
            jump = float(np.linalg.norm(cam_t_depth[out_i] - prev))
        previous_by_track[tid] = (frame, cam_t_depth[out_i].copy())
        delta = float(np.linalg.norm(cam_t_depth[out_i] - cam_t_wilor[out_i]))
        flag = candidate_qc_flag(source, rms, jump, args)
        primary_diag, issue_tags, bbox_vis, joint_vis, edge_flag = candidate_diagnosis(
            source,
            flag,
            rms,
            jump,
            np.asarray(data["bbox_xyxy"][src_i], dtype=np.float32),
            np.asarray(data["joints_uv"][src_i], dtype=np.float32),
            image_width,
            image_height,
            args,
        )
        diagnosis_category.append(primary_diag)
        qc_issue_tags.append(issue_tags)
        bbox_visible_ratios.append(bbox_vis)
        joint_visible_ratios.append(joint_vis)
        bbox_edge_flags.append(edge_flag)
        alignment_rows.append(
            {
                "frame_index": frame,
                "elapsed_sec": csv_float(float(data["elapsed_sec"][src_i])),
                "candidate_index": int(data["candidate_index"][src_i]) if "candidate_index" in data else src_i,
                "track_id": tid,
                "hand_rank": int(data["hand_rank"][src_i]),
                "hand_label": str(data["hand_label"][src_i]),
                "is_right": int(round(float(data["is_right"][src_i]))),
                "det_conf": csv_float(float(data["det_conf"][src_i])),
                "alignment_source": source,
                "alignment_valid_joint_count": valid_count,
                "alignment_joint_ids": ",".join(str(v) for v in ids),
                "alignment_sampled_depths_m": ",".join(csv_float(v) for v in depths),
                "alignment_rms_m": csv_float(rms),
                "alignment_max_residual_m": csv_float(max_res),
                "cam_t_wilor_x": csv_float(float(cam_t_wilor[out_i, 0])),
                "cam_t_wilor_y": csv_float(float(cam_t_wilor[out_i, 1])),
                "cam_t_wilor_z": csv_float(float(cam_t_wilor[out_i, 2])),
                "cam_t_depth_x": csv_float(float(cam_t_depth[out_i, 0])),
                "cam_t_depth_y": csv_float(float(cam_t_depth[out_i, 1])),
                "cam_t_depth_z": csv_float(float(cam_t_depth[out_i, 2])),
                "cam_t_delta_m": csv_float(delta),
                "cam_t_jump_m": csv_float(jump),
                "bbox_visible_ratio": csv_float(bbox_vis),
                "joint_visible_ratio": csv_float(joint_vis),
                "bbox_touches_edge": int(edge_flag),
                "qc_flag": flag,
                "diagnosis_category": primary_diag,
                "qc_issue_tags": issue_tags,
            }
        )

    joints_cam_depth = joints_rel + cam_t_depth[:, None, :]
    vertices_cam_depth = verts_rel + cam_t_depth[:, None, :]

    out["cam_t_wilor"] = cam_t_wilor
    out["cam_t_depth"] = cam_t_depth
    out["joints_cam_depth"] = joints_cam_depth.astype(np.float32)
    out["vertices_cam_depth"] = vertices_cam_depth.astype(np.float32)
    out["alignment_source"] = np.asarray(alignment_source)
    out["alignment_valid_joint_count"] = np.asarray(alignment_valid_joint_count, dtype=np.int32)
    out["alignment_joint_ids"] = np.asarray(alignment_joint_ids)
    out["alignment_sampled_depths_m"] = np.asarray(alignment_depths)
    out["alignment_rms_m"] = np.asarray(alignment_rms, dtype=np.float32)
    out["alignment_max_residual_m"] = np.asarray(alignment_max_residual, dtype=np.float32)
    out["diagnosis_category"] = np.asarray(diagnosis_category)
    out["qc_issue_tags"] = np.asarray(qc_issue_tags)
    out["bbox_visible_ratio"] = np.asarray(bbox_visible_ratios, dtype=np.float32)
    out["joint_visible_ratio"] = np.asarray(joint_visible_ratios, dtype=np.float32)
    out["bbox_touches_edge"] = np.asarray(bbox_edge_flags, dtype=np.int32)
    out["foundation_camera_json"] = np.asarray([json.dumps(camera, ensure_ascii=False)])

    output_npz = output_dir / "wilor_handresults_phase_c_depth_aligned.npz"
    output_csv = output_dir / "phase_c_alignment_quality.csv"
    summary_json = output_dir / "phase_c_alignment_summary.json"
    np.savez_compressed(output_npz, **out)
    write_alignment_csv(output_csv, alignment_rows)

    source_counts = Counter(alignment_source)
    qc_flags = Counter(row["qc_flag"] for row in alignment_rows)
    diagnosis_counts = Counter(diagnosis_category)
    issue_tag_counts = Counter()
    for tags in qc_issue_tags:
        issue_tag_counts.update(tag for tag in str(tags).split("|") if tag)
    fallback_count = sum(v for k, v in source_counts.items() if k != "depth_reliable_joints")
    hard_errors: List[str] = []
    warnings: List[str] = []
    if missing_depth_frames:
        warnings.append(f"missing_depth_frames:{len(missing_depth_frames)}")
    if not np.isfinite(cam_t_depth).all():
        hard_errors.append("non_finite_cam_t_depth")
    fallback_ratio = float(fallback_count / max(1, len(alignment_source)))
    if fallback_ratio > 0.50:
        warnings.append(f"fallback_ratio_high:{fallback_ratio:.3f}")
    bad_flag_count = sum(1 for row in alignment_rows if "bad_" in row["qc_flag"])
    if bad_flag_count:
        warnings.append(f"bad_alignment_or_jump_flags:{bad_flag_count}")

    summary = {
        "semantic": "LFV EgoInfinity Phase-C depth alignment with FoundationStereo metric depth",
        "session_dir": str(session_dir),
        "input_npz": str(input_npz),
        "depth_summary_json": str(depth_summary_json),
        "depth_frame_csv": str(depth_frame_csv),
        "output_npz": str(output_npz),
        "alignment_quality_csv": str(output_csv),
        "candidates_in_phase_b": int(n_all),
        "candidates_selected": int(len(keep)),
        "frame_start": int(args.frame_start),
        "frame_end": int(args.frame_end),
        "frames_selected": len(selected_frames),
        "depth_frames_available": len(raw_depth_by_frame),
        "missing_depth_frames": [int(v) for v in missing_depth_frames[:50]],
        "missing_depth_frame_count": int(len(missing_depth_frames)),
        "camera": camera,
        "alignment_source_counts": dict(source_counts),
        "qc_flag_counts": dict(qc_flags),
        "diagnosis_category_counts": dict(diagnosis_counts),
        "qc_issue_tag_counts": dict(issue_tag_counts),
        "fallback_ratio": fallback_ratio,
        "valid_joint_count": stats(alignment_valid_joint_count),
        "alignment_rms_m": stats(alignment_rms),
        "alignment_max_residual_m": stats(alignment_max_residual),
        "bbox_visible_ratio": stats(bbox_visible_ratios),
        "joint_visible_ratio": stats(joint_visible_ratios),
        "cam_t_delta_m": stats(float(row["cam_t_delta_m"]) for row in alignment_rows if row["cam_t_delta_m"]),
        "cam_t_jump_m": stats(float(row["cam_t_jump_m"]) for row in alignment_rows if row["cam_t_jump_m"]),
        "depth_stabilization": stable_summary,
        "hard_errors": hard_errors,
        "warnings": warnings,
        "ok": len(hard_errors) == 0,
    }
    summary_json.write_text(json.dumps(json_clean(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(json_clean(summary), indent=2, ensure_ascii=False))
    return 0 if len(hard_errors) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
