#!/usr/bin/env python3
"""Phase-C2 MANO temporal smoothing and forward pass.

This is the LFV wrapper around the EgoInfinity strict MANO smoothing logic:
rotation parameters are smoothed per physical track, biomechanical limits are
re-applied, and MANO forward is run again.  It deliberately does not translate
or deform an old mesh as a shortcut.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from egoinfinity_strict.biomech_constraints import clamp_hand_pose  # noqa: E402
from egoinfinity_strict.infiller_utils.rotation import angle_axis_to_rotation_matrix  # noqa: E402
from egoinfinity_strict.mano_smoothing import (  # noqa: E402
    _batch_mano_forward,
    _ensure_quat_continuity,
    _quat_to_aa,
    _rotmat_to_quat,
    _savgol_quat,
)


LANDMARK_COUNT = 21


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session-dir", required=True)
    p.add_argument("--input-npz", required=True, help="Phase-C depth-aligned NPZ")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--wilor-root", default="/home/yannan/workspace/learning-from-video/WiLor")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--model-config", default="")
    p.add_argument("--smooth-window", type=int, default=7)
    p.add_argument("--smooth-polyorder", type=int, default=2)
    p.add_argument("--min-track-frames", type=int, default=3)
    p.add_argument("--warn-input-joint-rms-m", type=float, default=0.060)
    p.add_argument("--bad-input-joint-rms-m", type=float, default=0.120)
    p.add_argument("--warn-smooth-wrist-jump-m", type=float, default=0.090)
    p.add_argument("--bad-smooth-wrist-jump-m", type=float, default=0.160)
    return p.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise RuntimeError(f"missing {label}: {path}")


def patch_torch_load_for_legacy_ultralytics(torch_mod: Any) -> None:
    original_load = torch_mod.load
    if getattr(original_load, "_lfv_legacy_ultralytics_patch", False):
        return

    def load_compat(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    load_compat._lfv_legacy_ultralytics_patch = True  # type: ignore[attr-defined]
    torch_mod.load = load_compat


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
    vals = []
    for value in values:
        try:
            f = float(value)
        except Exception:
            continue
        if math.isfinite(f):
            vals.append(f)
    arr = np.asarray(vals, dtype=np.float64)
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


def load_wilor_mano(wilor_root: Path, checkpoint: Path, model_config: Path) -> Any:
    require_file(checkpoint, "WiLoR checkpoint")
    require_file(model_config, "WiLoR model config")
    require_file(wilor_root / "mano_data" / "MANO_RIGHT.pkl", "MANO_RIGHT.pkl")
    require_file(wilor_root / "mano_data" / "mano_mean_params.npz", "mano_mean_params.npz")

    sys.path.insert(0, str(wilor_root))
    old_cwd = os.getcwd()
    os.chdir(str(wilor_root))
    try:
        import torch
        from wilor.models import load_wilor

        patch_torch_load_for_legacy_ultralytics(torch)
        model, _ = load_wilor(checkpoint_path=str(checkpoint), cfg_path=str(model_config))
        mano = model.mano.eval()
        return mano
    except Exception as exc:
        raise RuntimeError(f"failed to load WiLoR MANO model: {exc!r}") from exc
    finally:
        os.chdir(old_cwd)


def aa_to_rotmat_np(axis_angle: np.ndarray) -> np.ndarray:
    import torch

    flat = np.asarray(axis_angle, dtype=np.float32).reshape(-1, 3)
    with torch.no_grad():
        mats = angle_axis_to_rotation_matrix(torch.from_numpy(flat).float()).cpu().numpy()
    return mats.reshape(*np.asarray(axis_angle).shape[:-1], 3, 3).astype(np.float32)


def rotation_delta_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    rel = np.matmul(np.swapaxes(a, -1, -2), b)
    trace = np.trace(rel, axis1=-2, axis2=-1)
    cos_angle = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return np.rad2deg(np.arccos(cos_angle))


def frame_to_frame_jump(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((0,), dtype=np.float32)
    out = np.full((len(points),), np.nan, dtype=np.float32)
    if len(points) > 1:
        out[1:] = np.linalg.norm(points[1:] - points[:-1], axis=-1)
    return out


def project_points(points: np.ndarray, camera_json: str) -> np.ndarray:
    if not camera_json:
        return np.full((*points.shape[:-1], 2), np.nan, dtype=np.float32)
    try:
        camera = json.loads(str(camera_json))
        fx, fy = float(camera["fx"]), float(camera["fy"])
        cx, cy = float(camera["cx"]), float(camera["cy"])
    except Exception:
        return np.full((*points.shape[:-1], 2), np.nan, dtype=np.float32)

    pts = np.asarray(points, dtype=np.float32)
    z = pts[..., 2]
    uv = np.full((*pts.shape[:-1], 2), np.nan, dtype=np.float32)
    valid = np.isfinite(z) & (z > 1e-6)
    uv[..., 0][valid] = fx * pts[..., 0][valid] / z[valid] + cx
    uv[..., 1][valid] = fy * pts[..., 1][valid] / z[valid] + cy
    return uv


def write_candidate_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fields = [
        "frame_index", "candidate_index", "track_id", "hand_label", "is_right",
        "smoothing_status", "smoothing_event_tags", "biomech_clamped",
        "input_to_smooth_joint_rms_m", "input_to_smooth_wrist_delta_m",
        "raw_wrist_jump_m", "smooth_wrist_jump_m",
        "raw_vertex_centroid_jump_m", "smooth_vertex_centroid_jump_m",
        "global_rot_delta_deg", "hand_rot_delta_deg_mean", "hand_rot_delta_deg_max",
        "smooth_finite_ratio", "qc_flag",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def write_track_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fields = [
        "track_id", "hand_label", "is_right", "frame_count", "frame_min", "frame_max",
        "status", "biomech_clamped_frames", "betas_std_mean", "betas_std_max",
        "global_rot_delta_deg_p95", "hand_rot_delta_deg_mean_p95",
        "input_to_smooth_joint_rms_m_p95", "raw_wrist_jump_m_p95",
        "smooth_wrist_jump_m_p95", "raw_vertex_centroid_jump_m_p95",
        "smooth_vertex_centroid_jump_m_p95",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def quality_flag(
    joint_rms: float,
    smooth_jump: float,
    finite: float,
    status: str,
    args: argparse.Namespace,
) -> str:
    flags: List[str] = []
    if status != "smoothed_mano_forward":
        flags.append(status)
    if finite < 1.0:
        flags.append("bad_non_finite_smooth_geometry")
    if math.isfinite(joint_rms):
        if joint_rms > float(args.bad_input_joint_rms_m):
            flags.append("bad_input_to_smooth_joint_rms")
        elif joint_rms > float(args.warn_input_joint_rms_m):
            flags.append("warn_input_to_smooth_joint_rms")
    if math.isfinite(smooth_jump):
        if smooth_jump > float(args.bad_smooth_wrist_jump_m):
            flags.append("bad_smooth_wrist_jump")
        elif smooth_jump > float(args.warn_smooth_wrist_jump_m):
            flags.append("warn_smooth_wrist_jump")
    return "|".join(flags) if flags else "ok"


def main() -> int:
    args = parse_args()
    session_dir = Path(args.session_dir).expanduser().resolve()
    input_npz = Path(args.input_npz).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    wilor_root = Path(args.wilor_root).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else wilor_root / "pretrained_models" / "wilor_final.ckpt"
    model_config = Path(args.model_config).expanduser().resolve() if args.model_config else wilor_root / "pretrained_models" / "model_config.yaml"

    require_file(input_npz, "Phase-C depth-aligned NPZ")
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_npz(input_npz)
    n = int(len(data.get("frame_index", [])))
    if n == 0:
        raise RuntimeError(f"empty input npz: {input_npz}")

    required = [
        "frame_index", "track_id", "hand_label", "is_right", "candidate_index",
        "global_orient", "hand_pose", "betas", "cam_t_depth",
        "joints_cam_depth", "vertices_cam_depth", "faces",
    ]
    missing = [key for key in required if key not in data]
    if missing:
        raise RuntimeError(f"input npz missing required fields: {missing}")

    mano = load_wilor_mano(wilor_root, checkpoint, model_config)

    frame_index = data["frame_index"].astype(np.int32)
    track_id = data["track_id"].astype(np.int32)
    hand_label = np.asarray(data["hand_label"]).astype(str)
    is_right_arr = np.asarray(data["is_right"], dtype=np.float32)
    go_raw = np.asarray(data["global_orient"], dtype=np.float32)
    hp_raw = np.asarray(data["hand_pose"], dtype=np.float32)
    betas_raw = np.asarray(data["betas"], dtype=np.float32)
    cam_t_input = np.asarray(data["cam_t_depth"], dtype=np.float32)
    joints_input = np.asarray(data["joints_cam_depth"], dtype=np.float32)
    verts_input = np.asarray(data["vertices_cam_depth"], dtype=np.float32)

    cam_t_smooth = cam_t_input.copy()
    joints_smooth = joints_input.copy()
    verts_smooth = verts_input.copy()
    go_smooth_rot = go_raw.copy()
    hp_smooth_rot = hp_raw.copy()
    betas_smooth = betas_raw.copy()
    joints_rel_smooth = np.asarray(data.get("joints_3d_rel", joints_input - cam_t_input[:, None, :]), dtype=np.float32).copy()
    verts_rel_smooth = np.asarray(data.get("vertices_rel", verts_input - cam_t_input[:, None, :]), dtype=np.float32).copy()

    status = ["not_processed"] * n
    event_tags = [""] * n
    clamped_flags = np.zeros((n,), dtype=np.int32)
    input_to_smooth_rms = np.full((n,), np.nan, dtype=np.float32)
    input_to_smooth_wrist_delta = np.full((n,), np.nan, dtype=np.float32)
    raw_wrist_jump = np.full((n,), np.nan, dtype=np.float32)
    smooth_wrist_jump = np.full((n,), np.nan, dtype=np.float32)
    raw_vertex_centroid_jump = np.full((n,), np.nan, dtype=np.float32)
    smooth_vertex_centroid_jump = np.full((n,), np.nan, dtype=np.float32)
    global_delta = np.full((n,), np.nan, dtype=np.float32)
    hand_delta_mean = np.full((n,), np.nan, dtype=np.float32)
    hand_delta_max = np.full((n,), np.nan, dtype=np.float32)
    smooth_finite = np.ones((n,), dtype=np.float32)

    indices_by_track: Dict[int, List[int]] = defaultdict(list)
    for i, tid in enumerate(track_id.tolist()):
        indices_by_track[int(tid)].append(i)

    candidate_rows: List[Dict[str, Any]] = []
    track_rows: List[Dict[str, Any]] = []
    summary_events = Counter()

    for tid, indices_unsorted in sorted(indices_by_track.items()):
        indices = sorted(indices_unsorted, key=lambda i: (int(frame_index[i]), int(data["candidate_index"][i])))
        frames = frame_index[indices]
        labels = hand_label[indices]
        label_counts = Counter(labels.tolist())
        label = label_counts.most_common(1)[0][0] if label_counts else "unknown"
        is_right = bool(np.mean(is_right_arr[indices] >= 0.5) >= 0.5)
        T = len(indices)

        if T < int(args.min_track_frames):
            for i in indices:
                status[i] = "skipped_short_track"
                event_tags[i] = f"track_frames<{int(args.min_track_frames)}"
            summary_events["skipped_short_track"] += 1
            track_rows.append(
                {
                    "track_id": tid,
                    "hand_label": label,
                    "is_right": int(is_right),
                    "frame_count": T,
                    "frame_min": int(frames[0]),
                    "frame_max": int(frames[-1]),
                    "status": "skipped_short_track",
                }
            )
            continue

        go_quat = np.zeros((T, 4), dtype=np.float32)
        hp_quat = np.zeros((T, 15, 4), dtype=np.float32)
        for t, i in enumerate(indices):
            go_quat[t] = _rotmat_to_quat(go_raw[i, 0])
            for j in range(15):
                hp_quat[t, j] = _rotmat_to_quat(hp_raw[i, j])

        go_quat = _ensure_quat_continuity(go_quat)
        go_quat_smooth = _savgol_quat(go_quat, int(args.smooth_window), int(args.smooth_polyorder))
        hp_quat_smooth = np.zeros_like(hp_quat)
        for j in range(15):
            hp_quat[:, j] = _ensure_quat_continuity(hp_quat[:, j])
            hp_quat_smooth[:, j] = _savgol_quat(hp_quat[:, j], int(args.smooth_window), int(args.smooth_polyorder))

        go_aa = np.zeros((T, 3), dtype=np.float32)
        hp_aa = np.zeros((T, 45), dtype=np.float32)
        clamped_count = 0
        for t in range(T):
            go_aa[t] = _quat_to_aa(go_quat_smooth[t])
            hp_frame = np.zeros((15, 3), dtype=np.float32)
            for j in range(15):
                hp_frame[j] = _quat_to_aa(hp_quat_smooth[t, j])
            hp_frame, changed = clamp_hand_pose(hp_frame)
            if changed:
                clamped_count += 1
                clamped_flags[indices[t]] = 1
            hp_aa[t] = hp_frame.reshape(-1)

        betas_track = betas_raw[indices]
        betas_std = np.nanstd(betas_track, axis=0)
        betas_median = np.nanmedian(betas_track, axis=0, keepdims=True).astype(np.float32).repeat(T, axis=0)
        cam_ts = cam_t_input[indices].astype(np.float32)

        joints_track, verts_track = _batch_mano_forward(
            mano,
            go_aa.astype(np.float32),
            hp_aa.astype(np.float32),
            betas_median,
            is_right,
            cam_ts,
        )
        joints_track = joints_track.astype(np.float32)
        verts_track = verts_track.astype(np.float32)

        go_track_rot = aa_to_rotmat_np(go_aa).reshape(T, 1, 3, 3)
        hp_track_rot = aa_to_rotmat_np(hp_aa.reshape(T, 15, 3)).reshape(T, 15, 3, 3)

        joints_smooth[indices] = joints_track
        verts_smooth[indices] = verts_track
        joints_rel_smooth[indices] = joints_track - cam_ts[:, None, :]
        verts_rel_smooth[indices] = verts_track - cam_ts[:, None, :]
        go_smooth_rot[indices] = go_track_rot
        hp_smooth_rot[indices] = hp_track_rot
        betas_smooth[indices] = betas_median

        raw_wrist_j = frame_to_frame_jump(joints_input[indices, 0])
        smooth_wrist_j = frame_to_frame_jump(joints_track[:, 0])
        raw_centroid = np.nanmean(verts_input[indices], axis=1)
        smooth_centroid = np.nanmean(verts_track, axis=1)
        raw_centroid_j = frame_to_frame_jump(raw_centroid)
        smooth_centroid_j = frame_to_frame_jump(smooth_centroid)

        go_delta_track = rotation_delta_deg(go_raw[indices, 0], go_track_rot[:, 0]).astype(np.float32)
        hp_delta_track = rotation_delta_deg(hp_raw[indices], hp_track_rot).astype(np.float32)
        joint_delta = joints_track - joints_input[indices]
        joint_rms_track = np.sqrt(np.nanmean(np.sum(joint_delta * joint_delta, axis=-1), axis=1)).astype(np.float32)
        wrist_delta_track = np.linalg.norm(joint_delta[:, 0], axis=-1).astype(np.float32)

        for t, i in enumerate(indices):
            status[i] = "smoothed_mano_forward"
            tags: List[str] = []
            if clamped_flags[i]:
                tags.append("biomech_clamped")
            if T < int(args.smooth_window):
                tags.append("window_larger_than_track")
            event_tags[i] = "|".join(tags)
            input_to_smooth_rms[i] = joint_rms_track[t]
            input_to_smooth_wrist_delta[i] = wrist_delta_track[t]
            raw_wrist_jump[i] = raw_wrist_j[t]
            smooth_wrist_jump[i] = smooth_wrist_j[t]
            raw_vertex_centroid_jump[i] = raw_centroid_j[t]
            smooth_vertex_centroid_jump[i] = smooth_centroid_j[t]
            global_delta[i] = go_delta_track[t]
            hand_delta_mean[i] = float(np.nanmean(hp_delta_track[t]))
            hand_delta_max[i] = float(np.nanmax(hp_delta_track[t]))
            smooth_finite[i] = min(finite_ratio(joints_track[t]), finite_ratio(verts_track[t]))

        summary_events["smoothed_track"] += 1
        if clamped_count:
            summary_events["biomech_clamped_track"] += 1

        track_rows.append(
            {
                "track_id": tid,
                "hand_label": label,
                "is_right": int(is_right),
                "frame_count": T,
                "frame_min": int(frames[0]),
                "frame_max": int(frames[-1]),
                "status": "smoothed_mano_forward",
                "biomech_clamped_frames": clamped_count,
                "betas_std_mean": csv_float(float(np.nanmean(betas_std))),
                "betas_std_max": csv_float(float(np.nanmax(betas_std))),
                "global_rot_delta_deg_p95": csv_float(float(np.nanpercentile(go_delta_track, 95))),
                "hand_rot_delta_deg_mean_p95": csv_float(float(np.nanpercentile(np.nanmean(hp_delta_track, axis=1), 95))),
                "input_to_smooth_joint_rms_m_p95": csv_float(float(np.nanpercentile(joint_rms_track, 95))),
                "raw_wrist_jump_m_p95": csv_float(float(np.nanpercentile(raw_wrist_j, 95))),
                "smooth_wrist_jump_m_p95": csv_float(float(np.nanpercentile(smooth_wrist_j, 95))),
                "raw_vertex_centroid_jump_m_p95": csv_float(float(np.nanpercentile(raw_centroid_j, 95))),
                "smooth_vertex_centroid_jump_m_p95": csv_float(float(np.nanpercentile(smooth_centroid_j, 95))),
            }
        )

    qc_flags: List[str] = []
    for i in range(n):
        flag = quality_flag(
            float(input_to_smooth_rms[i]),
            float(smooth_wrist_jump[i]),
            float(smooth_finite[i]),
            status[i],
            args,
        )
        qc_flags.append(flag)
        candidate_rows.append(
            {
                "frame_index": int(frame_index[i]),
                "candidate_index": int(data["candidate_index"][i]) if "candidate_index" in data else i,
                "track_id": int(track_id[i]),
                "hand_label": str(hand_label[i]),
                "is_right": int(round(float(is_right_arr[i]))),
                "smoothing_status": status[i],
                "smoothing_event_tags": event_tags[i],
                "biomech_clamped": int(clamped_flags[i]),
                "input_to_smooth_joint_rms_m": csv_float(float(input_to_smooth_rms[i])),
                "input_to_smooth_wrist_delta_m": csv_float(float(input_to_smooth_wrist_delta[i])),
                "raw_wrist_jump_m": csv_float(float(raw_wrist_jump[i])),
                "smooth_wrist_jump_m": csv_float(float(smooth_wrist_jump[i])),
                "raw_vertex_centroid_jump_m": csv_float(float(raw_vertex_centroid_jump[i])),
                "smooth_vertex_centroid_jump_m": csv_float(float(smooth_vertex_centroid_jump[i])),
                "global_rot_delta_deg": csv_float(float(global_delta[i])),
                "hand_rot_delta_deg_mean": csv_float(float(hand_delta_mean[i])),
                "hand_rot_delta_deg_max": csv_float(float(hand_delta_max[i])),
                "smooth_finite_ratio": csv_float(float(smooth_finite[i])),
                "qc_flag": flag,
            }
        )

    camera_json = ""
    if "foundation_camera_json" in data and len(data["foundation_camera_json"]):
        camera_json = str(data["foundation_camera_json"][0])
    joints_uv_smooth = project_points(joints_smooth, camera_json)

    output_npz = output_dir / "wilor_handresults_phase_c2_mano_smooth.npz"
    quality_csv = output_dir / "mano_smoothing_quality.csv"
    track_csv = output_dir / "mano_smoothing_track_summary.csv"
    summary_json = output_dir / "mano_smoothing_summary.json"

    out = dict(data)
    out["cam_t_smooth"] = cam_t_smooth.astype(np.float32)
    out["joints_cam_smooth"] = joints_smooth.astype(np.float32)
    out["vertices_cam_smooth"] = verts_smooth.astype(np.float32)
    out["joints_3d_rel_smooth"] = joints_rel_smooth.astype(np.float32)
    out["vertices_rel_smooth"] = verts_rel_smooth.astype(np.float32)
    out["global_orient_smooth"] = go_smooth_rot.astype(np.float32)
    out["hand_pose_smooth"] = hp_smooth_rot.astype(np.float32)
    out["betas_smooth"] = betas_smooth.astype(np.float32)
    out["joints_uv_smooth_depth_camera"] = joints_uv_smooth.astype(np.float32)
    out["mano_smoothing_status"] = string_array(status)
    out["mano_smoothing_event_tags"] = string_array(event_tags)
    out["mano_biomech_clamped"] = clamped_flags.astype(np.int32)
    out["mano_input_to_smooth_joint_rms_m"] = input_to_smooth_rms.astype(np.float32)
    out["mano_input_to_smooth_wrist_delta_m"] = input_to_smooth_wrist_delta.astype(np.float32)
    out["mano_raw_wrist_jump_m"] = raw_wrist_jump.astype(np.float32)
    out["mano_smooth_wrist_jump_m"] = smooth_wrist_jump.astype(np.float32)
    out["mano_global_rot_delta_deg"] = global_delta.astype(np.float32)
    out["mano_hand_rot_delta_deg_mean"] = hand_delta_mean.astype(np.float32)
    out["mano_hand_rot_delta_deg_max"] = hand_delta_max.astype(np.float32)
    out["mano_smoothing_qc_flag"] = string_array(qc_flags)

    np.savez_compressed(output_npz, **out)
    write_candidate_csv(quality_csv, candidate_rows)
    write_track_csv(track_csv, track_rows)

    qc_counter = Counter(qc_flags)
    status_counter = Counter(status)
    hard_errors: List[str] = []
    warnings: List[str] = []
    if finite_ratio(joints_smooth) < 1.0 or finite_ratio(verts_smooth) < 1.0:
        hard_errors.append("non_finite_smooth_geometry")
    bad_count = sum(v for k, v in qc_counter.items() if "bad_" in str(k))
    warn_count = sum(v for k, v in qc_counter.items() if "warn_" in str(k))
    if bad_count:
        warnings.append(f"phase_c2_bad_smoothing_flags:{bad_count}")
    if warn_count:
        warnings.append(f"phase_c2_warn_smoothing_flags:{warn_count}")
    if status_counter.get("skipped_short_track", 0):
        warnings.append(f"phase_c2_skipped_short_track_candidates:{status_counter['skipped_short_track']}")

    summary = {
        "semantic": "LFV Phase-C2 EgoInfinity-style MANO temporal smoothing, biomech clamp, and MANO forward",
        "session_dir": str(session_dir),
        "input_npz": str(input_npz),
        "output_npz": str(output_npz),
        "quality_csv": str(quality_csv),
        "track_summary_csv": str(track_csv),
        "candidates": n,
        "tracks": int(len(indices_by_track)),
        "smooth_window": int(args.smooth_window),
        "smooth_polyorder": int(args.smooth_polyorder),
        "min_track_frames": int(args.min_track_frames),
        "status_counts": dict(status_counter),
        "qc_flag_counts": dict(qc_counter),
        "event_counts": dict(summary_events),
        "input_to_smooth_joint_rms_m": stats(input_to_smooth_rms),
        "input_to_smooth_wrist_delta_m": stats(input_to_smooth_wrist_delta),
        "raw_wrist_jump_m": stats(raw_wrist_jump),
        "smooth_wrist_jump_m": stats(smooth_wrist_jump),
        "raw_vertex_centroid_jump_m": stats(raw_vertex_centroid_jump),
        "smooth_vertex_centroid_jump_m": stats(smooth_vertex_centroid_jump),
        "global_rot_delta_deg": stats(global_delta),
        "hand_rot_delta_deg_mean": stats(hand_delta_mean),
        "hand_rot_delta_deg_max": stats(hand_delta_max),
        "smooth_joints_finite_ratio": finite_ratio(joints_smooth),
        "smooth_vertices_finite_ratio": finite_ratio(verts_smooth),
        "biomech_clamped_candidates": int(np.sum(clamped_flags)),
        "hard_errors": hard_errors,
        "warnings": warnings,
        "ok": len(hard_errors) == 0,
    }
    summary_json.write_text(json.dumps(json_clean(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(json_clean(summary), indent=2, ensure_ascii=False))
    return 0 if len(hard_errors) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
