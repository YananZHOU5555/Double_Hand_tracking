#!/usr/bin/env python3
"""Phase-C1c EgoInfinity MotionInfiller adapter for LFV HandResult NPZ.

This node converts the LFV candidate-table NPZ into EgoInfinity HandResult
lists, runs the original HaWoR/EgoInfinity MotionInfiller, then converts newly
filled missing hand frames back into the NPZ format.  Detected rows are kept
unchanged.  Filled rows are appended and explicitly marked with
``motion_infilled=1``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from egoinfinity_strict.hand_result import HandResult  # noqa: E402
from egoinfinity_strict.motion_infiller import MotionInfiller  # noqa: E402
from phase_c2_mano_temporal_smooth import load_wilor_mano  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session-dir", required=True)
    p.add_argument("--input-npz", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--wilor-root", default="/home/yannan/workspace/learning-from-video/WiLor")
    p.add_argument("--checkpoint", default="/home/yannan/workspace/EgoInfinity/pretrained_models/infiller.pt")
    p.add_argument("--wilor-checkpoint", default="")
    p.add_argument("--model-config", default="")
    p.add_argument("--device", default="cuda")
    p.add_argument("--total-frames", type=int, default=0)
    p.add_argument("--video", default="")
    p.add_argument("--max-warn-wrist-jump-m", type=float, default=0.120)
    p.add_argument("--max-bad-wrist-jump-m", type=float, default=0.220)
    return p.parse_args()


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


def finite_ratio(arr: np.ndarray) -> float:
    arr = np.asarray(arr)
    if arr.size == 0:
        return 0.0
    return float(np.mean(np.isfinite(arr)))


def csv_float(value: float) -> str:
    return f"{float(value):.9f}" if math.isfinite(float(value)) else ""


def string_array(values: Sequence[str]) -> np.ndarray:
    max_len = max([1] + [len(str(v)) for v in values])
    return np.asarray([str(v) for v in values], dtype=f"<U{max_len}")


def video_frame_count(video: Path) -> int:
    if not video.exists():
        return 0
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def estimate_elapsed(frame: int, data: Dict[str, np.ndarray]) -> float:
    if "elapsed_sec" in data and "frame_index" in data and len(data["frame_index"]) >= 2:
        frames = np.asarray(data["frame_index"], dtype=np.float64)
        elapsed = np.asarray(data["elapsed_sec"], dtype=np.float64)
        valid = np.isfinite(frames) & np.isfinite(elapsed)
        if int(np.sum(valid)) >= 2:
            order = np.argsort(frames[valid])
            f = frames[valid][order]
            e = elapsed[valid][order]
            df = np.diff(f)
            de = np.diff(e)
            good = (df > 0) & np.isfinite(de)
            if int(np.sum(good)) > 0:
                sec_per_frame = float(np.median(de[good] / df[good]))
                base = float(e[0] - f[0] * sec_per_frame)
                return base + frame * sec_per_frame
    return float(frame) / 30.0


def get_cam_t(data: Dict[str, np.ndarray], i: int) -> np.ndarray:
    for key in ("cam_t_depth", "cam_t_smooth", "cam_t"):
        if key in data:
            return np.asarray(data[key][i], dtype=np.float32)
    raise RuntimeError("input NPZ has no cam_t-like field")


def get_joints_cam(data: Dict[str, np.ndarray], i: int, cam_t: np.ndarray) -> np.ndarray:
    for key in ("joints_cam_depth", "joints_cam_smooth", "joints_cam"):
        if key in data:
            return np.asarray(data[key][i], dtype=np.float32)
    return np.asarray(data["joints_3d_rel"][i], dtype=np.float32) + cam_t[None, :]


def get_vertices_cam(data: Dict[str, np.ndarray], i: int, cam_t: np.ndarray) -> np.ndarray:
    for key in ("vertices_cam_depth", "vertices_cam_smooth", "vertices_cam"):
        if key in data:
            return np.asarray(data[key][i], dtype=np.float32)
    return np.asarray(data["vertices_rel"][i], dtype=np.float32) + cam_t[None, :]


def make_hand_result(data: Dict[str, np.ndarray], i: int) -> HandResult:
    cam_t = get_cam_t(data, i)
    joints_cam = get_joints_cam(data, i, cam_t)
    vertices_cam = get_vertices_cam(data, i, cam_t)
    joints_rel = np.asarray(data.get("joints_3d_rel", joints_cam - cam_t[None, :])[i], dtype=np.float32) \
        if "joints_3d_rel" in data else (joints_cam - cam_t[None, :]).astype(np.float32)
    vertices_rel = np.asarray(data.get("vertices_rel", vertices_cam - cam_t[None, :])[i], dtype=np.float32) \
        if "vertices_rel" in data else (vertices_cam - cam_t[None, :]).astype(np.float32)
    focal = float(data["focal_length"][i]) if "focal_length" in data else 1.0
    confidence = float(data["det_conf"][i]) if "det_conf" in data else 1.0
    track_id = int(data["track_id"][i]) if "track_id" in data else -1
    return HandResult(
        global_orient=np.asarray(data["global_orient"][i], dtype=np.float32),
        hand_pose=np.asarray(data["hand_pose"][i], dtype=np.float32),
        betas=np.asarray(data["betas"][i], dtype=np.float32),
        joints_3d=joints_cam.astype(np.float32),
        joints_3d_rel=joints_rel.astype(np.float32),
        vertices=vertices_cam.astype(np.float32),
        cam_t=cam_t.astype(np.float32),
        joints_2d=np.asarray(data["joints_uv"][i], dtype=np.float32),
        scaled_focal=focal,
        is_right=bool(float(data["is_right"][i]) >= 0.5),
        bbox=np.asarray(data["bbox_xyxy"][i], dtype=np.float32),
        confidence=confidence,
        track_id=track_id,
    )


def choose_total_frames(args: argparse.Namespace, data: Dict[str, np.ndarray]) -> int:
    if int(args.total_frames) > 0:
        return int(args.total_frames)
    video = Path(args.video).expanduser().resolve() if args.video else Path(args.session_dir).expanduser().resolve() / "processed_topcam" / "left_table.mp4"
    n = video_frame_count(video)
    if n > 0:
        return n
    frames = np.asarray(data["frame_index"], dtype=np.int32)
    return int(np.max(frames)) + 1 if len(frames) else 0


def nearest_original_index(
    frame: int,
    is_right: bool,
    indices_by_label: Dict[bool, List[int]],
    frame_index: np.ndarray,
) -> int:
    candidates = indices_by_label.get(bool(is_right), [])
    if not candidates:
        candidates = list(range(len(frame_index)))
    return min(candidates, key=lambda i: abs(int(frame_index[i]) - int(frame)))


def gap_length_for_label(frame: int, is_right: bool, frames_by_label: Dict[bool, List[int]]) -> Tuple[int, int]:
    frames = frames_by_label.get(bool(is_right), [])
    if not frames:
        return 999, -1
    prev_frames = [f for f in frames if f < frame]
    next_frames = [f for f in frames if f > frame]
    if not prev_frames or not next_frames:
        nearest = min(frames, key=lambda f: abs(f - frame))
        return 999, int(nearest)
    prev_f = max(prev_frames)
    next_f = min(next_frames)
    nearest = prev_f if abs(frame - prev_f) <= abs(next_f - frame) else next_f
    return int(next_f - prev_f - 1), int(nearest)


def core_value(
    key: str,
    hr: HandResult,
    frame: int,
    new_candidate_index: int,
    data: Dict[str, np.ndarray],
    ref_idx: int,
) -> Any:
    joints_rel = np.asarray(hr.joints_3d_rel, dtype=np.float32)
    vertices_rel = np.asarray(hr.vertices - hr.cam_t[None, :], dtype=np.float32)
    hand_label = "right" if hr.is_right else "left"
    overrides: Dict[str, Any] = {
        "frame_index": int(frame),
        "elapsed_sec": estimate_elapsed(frame, data),
        "hand_rank": 0,
        "candidate_index": int(new_candidate_index),
        "track_id": int(hr.track_id),
        "hand_label": hand_label,
        "is_right": float(1.0 if hr.is_right else 0.0),
        "det_conf": float(0.0),
        "bbox_xyxy": np.asarray(hr.bbox, dtype=np.float32),
        "cam_t": np.asarray(hr.cam_t, dtype=np.float32),
        "pred_cam": np.zeros((3,), dtype=np.float32),
        "focal_length": float(hr.scaled_focal),
        "global_orient": np.asarray(hr.global_orient, dtype=np.float32),
        "hand_pose": np.asarray(hr.hand_pose, dtype=np.float32),
        "betas": np.asarray(hr.betas, dtype=np.float32),
        "joints_3d_rel": joints_rel,
        "vertices_rel": vertices_rel,
        "joints_cam": np.asarray(hr.joints_3d, dtype=np.float32),
        "vertices_cam": np.asarray(hr.vertices, dtype=np.float32),
        "joints_uv": np.asarray(hr.joints_2d, dtype=np.float32),
        "cam_t_wilor": np.asarray(hr.cam_t, dtype=np.float32),
        "cam_t_depth": np.asarray(hr.cam_t, dtype=np.float32),
        "joints_cam_depth": np.asarray(hr.joints_3d, dtype=np.float32),
        "vertices_cam_depth": np.asarray(hr.vertices, dtype=np.float32),
        "cam_t_depth_smooth": np.asarray(hr.cam_t, dtype=np.float32),
        "joints_cam_depth_smooth": np.asarray(hr.joints_3d, dtype=np.float32),
        "vertices_cam_depth_smooth": np.asarray(hr.vertices, dtype=np.float32),
        "cam_t_depth_before_depth_smooth": np.asarray(hr.cam_t, dtype=np.float32),
        "alignment_source": "motion_infilled",
        "alignment_valid_joint_count": 0,
        "alignment_joint_ids": "",
        "alignment_sampled_depths_m": "",
        "alignment_rms_m": float("nan"),
        "alignment_max_residual_m": float("nan"),
        "diagnosis_category": "motion_infilled",
        "qc_issue_tags": "motion_infilled",
        "depth_smooth_status": "motion_infilled",
        "depth_smooth_qc_flag": "motion_infilled",
        "depth_smooth_anchor_z_m": float("nan"),
        "depth_smooth_anchor_z_smoothed_m": float(hr.cam_t[2]),
        "depth_smooth_delta_z_m": 0.0,
        "depth_smooth_vertex_mean_delta_z_m": 0.0,
        "depth_smooth_trust": 0.0,
        "depth_smooth_valid_sample_count": 0,
    }
    if key in overrides:
        return overrides[key]
    if key in data and np.asarray(data[key]).shape[0] > ref_idx:
        return data[key][ref_idx]
    raise KeyError(key)


def assemble_npz(
    data: Dict[str, np.ndarray],
    entries: Sequence[Dict[str, Any]],
    n_input: int,
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for key, arr in data.items():
        arr_np = np.asarray(arr)
        if arr_np.shape[:1] != (n_input,):
            out[key] = arr
            continue
        values: List[Any] = []
        string_like = arr_np.dtype.kind in ("U", "S", "O")
        for entry in entries:
            if entry["source_index"] is not None:
                values.append(arr_np[int(entry["source_index"])])
            else:
                values.append(core_value(
                    key,
                    entry["hand_result"],
                    int(entry["frame_index"]),
                    int(entry["candidate_index"]),
                    data,
                    int(entry["reference_index"]),
                ))
        if string_like or any(isinstance(v, str) for v in values):
            out[key] = string_array([str(v) for v in values])
        else:
            try:
                stacked = np.stack(values, axis=0)
            except Exception:
                stacked = np.asarray(values)
            try:
                out[key] = stacked.astype(arr_np.dtype, copy=False)
            except Exception:
                out[key] = stacked
    return out


def write_quality_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fields = [
        "frame_index", "candidate_index", "track_id", "hand_label", "is_right",
        "motion_infilled", "motion_infiller_method", "gap_len",
        "nearest_detected_frame", "cam_t_jump_m", "wrist_jump_m",
        "geometry_finite_ratio", "qc_flag",
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
    output_dir = Path(args.output_dir).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    wilor_root = Path(args.wilor_root).expanduser().resolve()
    wilor_checkpoint = Path(args.wilor_checkpoint).expanduser().resolve() if args.wilor_checkpoint else wilor_root / "pretrained_models" / "wilor_final.ckpt"
    model_config = Path(args.model_config).expanduser().resolve() if args.model_config else wilor_root / "pretrained_models" / "model_config.yaml"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_npz.exists():
        raise RuntimeError(f"missing input NPZ: {input_npz}")
    if not checkpoint.exists():
        raise RuntimeError(f"missing MotionInfiller checkpoint: {checkpoint}")

    data = load_npz(input_npz)
    required = [
        "frame_index", "candidate_index", "track_id", "hand_label", "is_right",
        "bbox_xyxy", "det_conf", "global_orient", "hand_pose", "betas",
        "joints_3d_rel", "vertices_rel", "joints_uv",
    ]
    missing = [key for key in required if key not in data]
    if missing:
        raise RuntimeError(f"input NPZ missing required fields: {missing}")

    n_input = int(len(data["frame_index"]))
    total_frames = choose_total_frames(args, data)
    if total_frames <= 0:
        raise RuntimeError("could not determine total frame count")

    frame_index = np.asarray(data["frame_index"], dtype=np.int32)
    is_right_arr = np.asarray(data["is_right"], dtype=np.float32) >= 0.5
    original_by_frame_label = {(int(f), bool(r)) for f, r in zip(frame_index.tolist(), is_right_arr.tolist())}
    frames_by_label: Dict[bool, List[int]] = {
        False: sorted(set(int(frame_index[i]) for i in range(n_input) if not bool(is_right_arr[i]))),
        True: sorted(set(int(frame_index[i]) for i in range(n_input) if bool(is_right_arr[i]))),
    }
    indices_by_label: Dict[bool, List[int]] = {
        False: [i for i in range(n_input) if not bool(is_right_arr[i])],
        True: [i for i in range(n_input) if bool(is_right_arr[i])],
    }

    hand_results_per_frame: List[List[HandResult]] = [[] for _ in range(total_frames)]
    for i in range(n_input):
        frame = int(frame_index[i])
        if frame < 0 or frame >= total_frames:
            continue
        hand_results_per_frame[frame].append(make_hand_result(data, i))

    left_frames = set(frames_by_label[False])
    right_frames = set(frames_by_label[True])
    both_valid = len(left_frames & right_frames)
    total_valid = len(left_frames | right_frames)
    dual_ratio = float(both_valid / max(total_valid, 1))
    long_gap_method = "transformer" if both_valid >= 2 and dual_ratio >= 0.15 else "slerp_fallback"

    mano = load_wilor_mano(wilor_root, wilor_checkpoint, model_config)
    try:
        mano = mano.cpu().eval()
    except Exception:
        mano = mano.eval()
    infiller = MotionInfiller(str(checkpoint), device=str(args.device), mano_model=mano)
    filled_per_frame = infiller.fill_missing_frames(hand_results_per_frame, total_frames)

    max_candidate = int(np.max(np.asarray(data["candidate_index"], dtype=np.int64))) if n_input else -1
    entries: List[Dict[str, Any]] = []
    for i in range(n_input):
        entries.append({"source_index": i})

    filled_entries: List[Dict[str, Any]] = []
    for frame, hands in enumerate(filled_per_frame):
        for hr in hands:
            key = (int(frame), bool(hr.is_right))
            if key in original_by_frame_label:
                continue
            max_candidate += 1
            ref_idx = nearest_original_index(frame, bool(hr.is_right), indices_by_label, frame_index)
            gap_len, nearest_frame = gap_length_for_label(frame, bool(hr.is_right), frames_by_label)
            method = "neighbor_interp" if gap_len <= 2 else long_gap_method
            filled_entries.append(
                {
                    "source_index": None,
                    "hand_result": hr,
                    "frame_index": int(frame),
                    "candidate_index": int(max_candidate),
                    "reference_index": int(ref_idx),
                    "gap_len": int(gap_len),
                    "nearest_detected_frame": int(nearest_frame),
                    "motion_infiller_method": method,
                }
            )

    entries.extend(filled_entries)
    entries.sort(key=lambda e: (
        int(frame_index[int(e["source_index"])]) if e["source_index"] is not None else int(e["frame_index"]),
        int(data["candidate_index"][int(e["source_index"])]) if e["source_index"] is not None else int(e["candidate_index"]),
    ))

    out = assemble_npz(data, entries, n_input)
    n_out = len(entries)
    motion_infilled = np.zeros((n_out,), dtype=np.int32)
    method_values: List[str] = ["detected"] * n_out
    gap_len_values = np.zeros((n_out,), dtype=np.int32)
    nearest_values = np.full((n_out,), -1, dtype=np.int32)
    reference_values = np.full((n_out,), -1, dtype=np.int32)
    for idx, entry in enumerate(entries):
        if entry["source_index"] is not None:
            reference_values[idx] = int(entry["source_index"])
            continue
        motion_infilled[idx] = 1
        method_values[idx] = str(entry["motion_infiller_method"])
        gap_len_values[idx] = int(entry["gap_len"])
        nearest_values[idx] = int(entry["nearest_detected_frame"])
        reference_values[idx] = int(entry["reference_index"])

    out["motion_infilled"] = motion_infilled
    out["motion_infiller_method"] = string_array(method_values)
    out["motion_infiller_gap_len"] = gap_len_values
    out["motion_infiller_nearest_detected_frame"] = nearest_values
    out["motion_infiller_reference_index"] = reference_values

    frame_out = np.asarray(out["frame_index"], dtype=np.int32)
    track_out = np.asarray(out["track_id"], dtype=np.int32)
    label_out = np.asarray(out["hand_label"]).astype(str)
    is_right_out = np.asarray(out["is_right"], dtype=np.float32) >= 0.5
    cam_t_out = get_cam_t(out, 0)  # validates key presence; ignored below
    del cam_t_out
    cam_t_key = "cam_t_depth" if "cam_t_depth" in out else "cam_t"
    joints_key = "joints_cam_depth" if "joints_cam_depth" in out else "joints_cam"
    cam_t_arr = np.asarray(out[cam_t_key], dtype=np.float32)
    joints_arr = np.asarray(out[joints_key], dtype=np.float32)

    cam_jump = np.full((n_out,), np.nan, dtype=np.float32)
    wrist_jump = np.full((n_out,), np.nan, dtype=np.float32)
    finite_geom = np.ones((n_out,), dtype=np.float32)
    for tid in sorted(set(track_out.tolist())):
        idxs = [i for i in range(n_out) if int(track_out[i]) == int(tid)]
        idxs.sort(key=lambda i: (int(frame_out[i]), int(out["candidate_index"][i])))
        if len(idxs) > 1:
            ct = cam_t_arr[idxs]
            wr = joints_arr[idxs, 0]
            cam_jump[np.asarray(idxs[1:], dtype=np.int32)] = np.linalg.norm(ct[1:] - ct[:-1], axis=1)
            wrist_jump[np.asarray(idxs[1:], dtype=np.int32)] = np.linalg.norm(wr[1:] - wr[:-1], axis=1)
    for i in range(n_out):
        finite_geom[i] = min(finite_ratio(cam_t_arr[i]), finite_ratio(joints_arr[i]))

    qc_flags: List[str] = []
    quality_rows: List[Dict[str, Any]] = []
    for i in range(n_out):
        parts: List[str] = []
        if finite_geom[i] < 1.0:
            parts.append("bad_non_finite_geometry")
        if math.isfinite(float(wrist_jump[i])):
            if float(wrist_jump[i]) > float(args.max_bad_wrist_jump_m):
                parts.append("bad_wrist_jump")
            elif float(wrist_jump[i]) > float(args.max_warn_wrist_jump_m):
                parts.append("warn_wrist_jump")
        if motion_infilled[i] and gap_len_values[i] >= 999:
            parts.append("warn_edge_gap_fill")
        flag = "|".join(parts) if parts else "ok"
        qc_flags.append(flag)
        quality_rows.append(
            {
                "frame_index": int(frame_out[i]),
                "candidate_index": int(out["candidate_index"][i]),
                "track_id": int(track_out[i]),
                "hand_label": str(label_out[i]),
                "is_right": int(is_right_out[i]),
                "motion_infilled": int(motion_infilled[i]),
                "motion_infiller_method": str(method_values[i]),
                "gap_len": int(gap_len_values[i]),
                "nearest_detected_frame": int(nearest_values[i]),
                "cam_t_jump_m": csv_float(float(cam_jump[i])),
                "wrist_jump_m": csv_float(float(wrist_jump[i])),
                "geometry_finite_ratio": csv_float(float(finite_geom[i])),
                "qc_flag": flag,
            }
        )

    out["motion_infiller_cam_t_jump_m"] = cam_jump.astype(np.float32)
    out["motion_infiller_wrist_jump_m"] = wrist_jump.astype(np.float32)
    out["motion_infiller_geometry_finite_ratio"] = finite_geom.astype(np.float32)
    out["motion_infiller_qc_flag"] = string_array(qc_flags)

    output_npz = output_dir / "wilor_handresults_phase_c1c_motion_infilled.npz"
    quality_csv = output_dir / "motion_infiller_quality.csv"
    summary_json = output_dir / "motion_infiller_summary.json"
    np.savez_compressed(output_npz, **out)
    write_quality_csv(quality_csv, quality_rows)

    qc_counts = Counter(qc_flags)
    method_counts = Counter(method_values)
    hard_errors: List[str] = []
    warnings: List[str] = []
    if finite_ratio(cam_t_arr) < 1.0 or finite_ratio(joints_arr) < 1.0:
        hard_errors.append("non_finite_motion_infiller_geometry")
    bad_count = sum(v for k, v in qc_counts.items() if "bad_" in str(k))
    warn_count = sum(v for k, v in qc_counts.items() if "warn_" in str(k))
    if bad_count:
        warnings.append(f"motion_infiller_bad_flags:{bad_count}")
    if warn_count:
        warnings.append(f"motion_infiller_warn_flags:{warn_count}")

    summary = {
        "semantic": "LFV Phase-C1c EgoInfinity MotionInfiller adapter",
        "session_dir": str(session_dir),
        "input_npz": str(input_npz),
        "output_npz": str(output_npz),
        "quality_csv": str(quality_csv),
        "checkpoint": str(checkpoint),
        "wilor_root": str(wilor_root),
        "total_frames": int(total_frames),
        "candidates_in": int(n_input),
        "candidates_out": int(n_out),
        "motion_infilled_candidates": int(np.sum(motion_infilled)),
        "left_detected_frames": int(len(left_frames)),
        "right_detected_frames": int(len(right_frames)),
        "both_hands_detected_frames": int(both_valid),
        "dual_valid_ratio": float(dual_ratio),
        "long_gap_method": long_gap_method,
        "method_counts": dict(method_counts),
        "qc_flag_counts": dict(qc_counts),
        "cam_t_jump_m": stats(cam_jump),
        "wrist_jump_m": stats(wrist_jump),
        "geometry_finite_ratio": stats(finite_geom),
        "hard_errors": hard_errors,
        "warnings": warnings,
        "ok": len(hard_errors) == 0,
    }
    summary_json.write_text(json.dumps(json_clean(summary), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(json_clean(summary), indent=2, ensure_ascii=False))
    return 0 if not hard_errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
