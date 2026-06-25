#!/usr/bin/env python3
"""Repair detected-but-bad MANO pose frames after Phase-C2.

This stage is intentionally conservative.  It does not rerun detection and it
does not touch ordinary detected frames.  It marks suspicious detected frames
using Phase-C2 orientation/jump diagnostics, bridges tiny good gaps inside a
bad span, interpolates MANO pose/cam_t from same-track trusted neighbors, then
runs MANO forward again for the repaired frames.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from egoinfinity_strict.mano_smoothing import _batch_mano_forward  # noqa: E402
from phase_c2_mano_temporal_smooth import (  # noqa: E402
    aa_to_rotmat_np,
    csv_float,
    finite_ratio,
    json_clean,
    load_wilor_mano,
    project_points,
    rotation_delta_deg,
    stats,
    string_array,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session-dir", required=True)
    p.add_argument("--input-npz", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--wilor-root", default="/home/yannan/workspace/learning-from-video/WiLor")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--model-config", default="")
    p.add_argument("--bad-global-rot-delta-deg", type=float, default=30.0)
    p.add_argument("--bad-raw-global-rot-jump-deg", type=float, default=90.0)
    p.add_argument("--bad-smooth-global-rot-jump-deg", type=float, default=90.0)
    p.add_argument("--bad-infilled-wrist-jump-m", type=float, default=0.080)
    p.add_argument("--neighbor-window-frames", type=int, default=12)
    p.add_argument("--bridge-good-gap-frames", type=int, default=2)
    p.add_argument("--repair-motion-infilled-jumps", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--repair-orientation-neighbors", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def require_keys(data: Dict[str, np.ndarray], keys: Sequence[str]) -> None:
    missing = [k for k in keys if k not in data]
    if missing:
        raise RuntimeError(f"input NPZ missing required fields: {missing}")


def append_flag(existing: Any, flag: str) -> str:
    parts = [p for p in str(existing or "").split("|") if p and p != "ok"]
    if flag not in parts:
        parts.append(flag)
    return "|".join(parts) if parts else "ok"


def finite_metric(data: Dict[str, np.ndarray], key: str, n: int, default: float = float("nan")) -> np.ndarray:
    if key in data and np.asarray(data[key]).shape[:1] == (n,):
        return np.asarray(data[key], dtype=np.float32)
    return np.full((n,), default, dtype=np.float32)


def bool_metric(data: Dict[str, np.ndarray], key: str, n: int) -> np.ndarray:
    if key in data and np.asarray(data[key]).shape[:1] == (n,):
        return np.asarray(data[key], dtype=np.int32).astype(bool)
    return np.zeros((n,), dtype=bool)


def rotmat_slerp(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    r = Rotation.from_matrix(np.stack([a, b], axis=0))
    return Slerp([0.0, 1.0], r)([float(alpha)]).as_matrix()[0].astype(np.float32)


def interp_hand_pose(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    out = np.zeros_like(a, dtype=np.float32)
    for j in range(a.shape[0]):
        out[j] = rotmat_slerp(a[j], b[j], alpha)
    return out


def rotmat_to_aa(mat: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(mat).as_rotvec().astype(np.float32)


def bridge_short_gaps(mask: np.ndarray, frames: np.ndarray, max_gap: int) -> np.ndarray:
    out = mask.copy()
    bad_positions = np.where(mask)[0]
    if bad_positions.size < 2 or int(max_gap) <= 0:
        return out
    for left_pos, right_pos in zip(bad_positions[:-1], bad_positions[1:]):
        if right_pos <= left_pos + 1:
            continue
        frame_gap = int(frames[right_pos] - frames[left_pos] - 1)
        row_gap = int(right_pos - left_pos - 1)
        if frame_gap <= int(max_gap) and row_gap <= int(max_gap):
            out[left_pos + 1:right_pos] = True
    return out


def nearest_good_bracket(local_index: int, good: np.ndarray) -> Tuple[int, int]:
    prevs = np.where(good[:local_index])[0]
    nexts = np.where(good[local_index + 1:])[0]
    if prevs.size == 0 or nexts.size == 0:
        return -1, -1
    return int(prevs[-1]), int(local_index + 1 + nexts[0])


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fields = [
        "frame_index", "candidate_index", "track_id", "hand_label", "is_right",
        "motion_infilled", "repair_bad_pose", "repair_repaired", "repair_reason",
        "repair_method", "prev_frame", "next_frame", "alpha",
        "global_rot_delta_deg_before", "global_rot_delta_deg_after",
        "wrist_delta_m", "qc_flag",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def main() -> int:
    t0 = time.time()
    args = parse_args()
    session_dir = Path(args.session_dir).expanduser().resolve()
    input_npz = Path(args.input_npz).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_npz.exists():
        raise RuntimeError(f"missing input NPZ: {input_npz}")

    data = load_npz(input_npz)
    require_keys(
        data,
        [
            "frame_index", "candidate_index", "track_id", "hand_label", "is_right",
            "cam_t_smooth", "global_orient_smooth", "hand_pose_smooth", "betas_smooth",
            "joints_cam_smooth", "vertices_cam_smooth", "joints_3d_rel_smooth",
            "vertices_rel_smooth",
        ],
    )

    n = int(len(data["frame_index"]))
    frame_index = np.asarray(data["frame_index"], dtype=np.int32)
    candidate_index = np.asarray(data["candidate_index"], dtype=np.int32)
    track_id = np.asarray(data["track_id"], dtype=np.int32)
    hand_label = np.asarray(data["hand_label"]).astype(str)
    is_right_arr = np.asarray(data["is_right"], dtype=np.float32) >= 0.5
    motion_infilled = np.asarray(data.get("motion_infilled", np.zeros((n,), dtype=np.int32)), dtype=np.int32).astype(bool)

    cam_t = np.asarray(data["cam_t_smooth"], dtype=np.float32).copy()
    go = np.asarray(data["global_orient_smooth"], dtype=np.float32).copy()
    hp = np.asarray(data["hand_pose_smooth"], dtype=np.float32).copy()
    betas = np.asarray(data["betas_smooth"], dtype=np.float32).copy()
    joints = np.asarray(data["joints_cam_smooth"], dtype=np.float32).copy()
    verts = np.asarray(data["vertices_cam_smooth"], dtype=np.float32).copy()
    joints_rel = np.asarray(data["joints_3d_rel_smooth"], dtype=np.float32).copy()
    verts_rel = np.asarray(data["vertices_rel_smooth"], dtype=np.float32).copy()

    global_delta = finite_metric(data, "mano_global_rot_delta_deg", n)
    raw_global_jump = finite_metric(data, "mano_raw_global_rot_jump_deg", n)
    smooth_global_jump = finite_metric(data, "mano_smooth_global_rot_jump_deg", n)
    smooth_wrist_jump = finite_metric(data, "mano_smooth_wrist_jump_m", n)
    orientation_core = bool_metric(data, "mano_orientation_flip_core", n)
    orientation_neighbor = bool_metric(data, "mano_orientation_flip_neighbor", n)

    bad_core = orientation_core.copy()
    bad_core |= np.isfinite(global_delta) & (global_delta > float(args.bad_global_rot_delta_deg))
    bad_core |= np.isfinite(raw_global_jump) & (raw_global_jump > float(args.bad_raw_global_rot_jump_deg))
    bad_core |= np.isfinite(smooth_global_jump) & (smooth_global_jump > float(args.bad_smooth_global_rot_jump_deg))
    if bool(args.repair_motion_infilled_jumps):
        bad_core |= motion_infilled & np.isfinite(smooth_wrist_jump) & (smooth_wrist_jump > float(args.bad_infilled_wrist_jump_m))

    if not bool(args.repair_orientation_neighbors):
        bad_core &= ~orientation_neighbor
    bad_core &= np.isfinite(cam_t).all(axis=1)

    repair_mask = bad_core.copy()
    groups: Dict[Tuple[str, int], List[int]] = defaultdict(list)
    for i in range(n):
        groups[(str(hand_label[i]), int(track_id[i]))].append(i)
    for _, idxs_unsorted in groups.items():
        idxs = sorted(idxs_unsorted, key=lambda i: (int(frame_index[i]), int(candidate_index[i])))
        local = np.asarray([bool(repair_mask[i]) for i in idxs], dtype=bool)
        frames = np.asarray([int(frame_index[i]) for i in idxs], dtype=np.int32)
        bridged = bridge_short_gaps(local, frames, int(args.bridge_good_gap_frames))
        for flag, i in zip(bridged.tolist(), idxs):
            repair_mask[i] = bool(flag)

    output_go = go.copy()
    output_hp = hp.copy()
    output_betas = betas.copy()
    output_cam_t = cam_t.copy()
    repaired = np.zeros((n,), dtype=np.int32)
    reason: List[str] = ["ok"] * n
    method: List[str] = ["keep"] * n
    prev_frame = np.full((n,), -1, dtype=np.int32)
    next_frame = np.full((n,), -1, dtype=np.int32)
    alpha_arr = np.full((n,), np.nan, dtype=np.float32)
    after_global_delta = np.full((n,), np.nan, dtype=np.float32)
    wrist_delta = np.full((n,), np.nan, dtype=np.float32)
    qc_flags: List[str] = ["ok"] * n
    rows: List[Dict[str, Any]] = []

    for i in range(n):
        parts: List[str] = []
        if bool(orientation_core[i]):
            parts.append("orientation_flip_core")
        if bool(orientation_neighbor[i]):
            parts.append("orientation_flip_neighbor")
        if math.isfinite(float(global_delta[i])) and float(global_delta[i]) > float(args.bad_global_rot_delta_deg):
            parts.append("large_global_rot_delta")
        if math.isfinite(float(raw_global_jump[i])) and float(raw_global_jump[i]) > float(args.bad_raw_global_rot_jump_deg):
            parts.append("large_raw_global_rot_jump")
        if math.isfinite(float(smooth_global_jump[i])) and float(smooth_global_jump[i]) > float(args.bad_smooth_global_rot_jump_deg):
            parts.append("large_smooth_global_rot_jump")
        if bool(motion_infilled[i]) and math.isfinite(float(smooth_wrist_jump[i])) and float(smooth_wrist_jump[i]) > float(args.bad_infilled_wrist_jump_m):
            parts.append("large_infilled_wrist_jump")
        if bool(repair_mask[i]) and not parts:
            parts.append("bridged_short_bad_pose_gap")
        reason[i] = "|".join(parts) if parts else "ok"

    for key, idxs_unsorted in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        idxs = sorted(idxs_unsorted, key=lambda i: (int(frame_index[i]), int(candidate_index[i])))
        local_bad = np.asarray([bool(repair_mask[i]) for i in idxs], dtype=bool)
        local_good = ~local_bad
        if int(np.sum(local_bad)) == 0:
            continue
        is_right = bool(np.mean(is_right_arr[idxs]) >= 0.5)
        repair_local_positions = np.where(local_bad)[0]
        frame_vals = np.asarray([int(frame_index[i]) for i in idxs], dtype=np.float32)
        repaired_indices: List[int] = []
        go_aa: List[np.ndarray] = []
        hp_aa: List[np.ndarray] = []
        betas_new: List[np.ndarray] = []
        cam_t_new: List[np.ndarray] = []

        for pos in repair_local_positions.tolist():
            i = idxs[pos]
            left_pos, right_pos = nearest_good_bracket(pos, local_good)
            if left_pos < 0 or right_pos < 0:
                method[i] = "no_good_bracket_keep"
                qc_flags[i] = "warn_bad_pose_no_good_bracket"
                continue
            left_i, right_i = idxs[left_pos], idxs[right_pos]
            if int(args.neighbor_window_frames) > 0:
                left_gap = int(frame_index[i] - frame_index[left_i])
                right_gap = int(frame_index[right_i] - frame_index[i])
                if left_gap > int(args.neighbor_window_frames) or right_gap > int(args.neighbor_window_frames):
                    method[i] = "no_near_good_bracket_keep"
                    qc_flags[i] = "warn_bad_pose_no_near_good_bracket"
                    prev_frame[i] = int(frame_index[left_i])
                    next_frame[i] = int(frame_index[right_i])
                    continue
            denom = max(float(frame_vals[right_pos] - frame_vals[left_pos]), 1.0)
            alpha = float(np.clip((float(frame_index[i]) - float(frame_index[left_i])) / denom, 0.0, 1.0))
            prev_frame[i] = int(frame_index[left_i])
            next_frame[i] = int(frame_index[right_i])
            alpha_arr[i] = alpha
            output_cam_t[i] = ((1.0 - alpha) * cam_t[left_i] + alpha * cam_t[right_i]).astype(np.float32)
            output_betas[i] = ((1.0 - alpha) * betas[left_i] + alpha * betas[right_i]).astype(np.float32)
            output_go[i, 0] = rotmat_slerp(go[left_i, 0], go[right_i, 0], alpha)
            output_hp[i] = interp_hand_pose(hp[left_i], hp[right_i], alpha)
            repaired_indices.append(i)
            go_aa.append(rotmat_to_aa(output_go[i, 0]))
            hp_aa.append(Rotation.from_matrix(output_hp[i]).as_rotvec().reshape(45).astype(np.float32))
            betas_new.append(output_betas[i])
            cam_t_new.append(output_cam_t[i])
            repaired[i] = 1
            method[i] = "same_track_slerp_mano_forward"

        if repaired_indices:
            j_new, v_new = _batch_mano_forward(
                mano_model,
                np.asarray(go_aa, dtype=np.float32),
                np.asarray(hp_aa, dtype=np.float32),
                np.asarray(betas_new, dtype=np.float32),
                is_right,
                np.asarray(cam_t_new, dtype=np.float32),
            )
            j_new = j_new.astype(np.float32)
            v_new = v_new.astype(np.float32)
            for local_out, i in enumerate(repaired_indices):
                joints[i] = j_new[local_out]
                verts[i] = v_new[local_out]
                joints_rel[i] = joints[i] - output_cam_t[i][None, :]
                verts_rel[i] = verts[i] - output_cam_t[i][None, :]
                after_global_delta[i] = float(rotation_delta_deg(go[i, 0], output_go[i, 0]))
                wrist_delta[i] = float(np.linalg.norm(joints[i, 0] - np.asarray(data["joints_cam_smooth"], dtype=np.float32)[i, 0]))
                qc_flags[i] = "pose_repaired"

    camera_json = ""
    if "foundation_camera_json" in data and len(data["foundation_camera_json"]):
        camera_json = str(data["foundation_camera_json"][0])
    joints_uv = project_points(joints, camera_json)

    base_qc = np.asarray(data.get("mano_smoothing_qc_flag", np.asarray(["ok"] * n))).astype(str)
    repair_qc = []
    for i in range(n):
        if repaired[i]:
            repair_qc.append(append_flag(base_qc[i], "pose_repaired_bad_detected"))
        elif repair_mask[i]:
            repair_qc.append(append_flag(base_qc[i], "bad_pose_repair_failed"))
        else:
            repair_qc.append(str(base_qc[i]))
        rows.append(
            {
                "frame_index": int(frame_index[i]),
                "candidate_index": int(candidate_index[i]),
                "track_id": int(track_id[i]),
                "hand_label": str(hand_label[i]),
                "is_right": int(bool(is_right_arr[i])),
                "motion_infilled": int(bool(motion_infilled[i])),
                "repair_bad_pose": int(bool(repair_mask[i])),
                "repair_repaired": int(bool(repaired[i])),
                "repair_reason": reason[i],
                "repair_method": method[i],
                "prev_frame": int(prev_frame[i]),
                "next_frame": int(next_frame[i]),
                "alpha": csv_float(float(alpha_arr[i])),
                "global_rot_delta_deg_before": csv_float(float(global_delta[i])),
                "global_rot_delta_deg_after": csv_float(float(after_global_delta[i])),
                "wrist_delta_m": csv_float(float(wrist_delta[i])),
                "qc_flag": qc_flags[i],
            }
        )

    out = dict(data)
    out["cam_t_smooth_before_pose_repair"] = cam_t.astype(np.float32)
    out["joints_cam_smooth_before_pose_repair"] = np.asarray(data["joints_cam_smooth"], dtype=np.float32)
    out["vertices_cam_smooth_before_pose_repair"] = np.asarray(data["vertices_cam_smooth"], dtype=np.float32)
    out["cam_t_smooth"] = output_cam_t.astype(np.float32)
    out["global_orient_smooth"] = output_go.astype(np.float32)
    out["hand_pose_smooth"] = output_hp.astype(np.float32)
    out["betas_smooth"] = output_betas.astype(np.float32)
    out["joints_cam_smooth"] = joints.astype(np.float32)
    out["vertices_cam_smooth"] = verts.astype(np.float32)
    out["joints_3d_rel_smooth"] = joints_rel.astype(np.float32)
    out["vertices_rel_smooth"] = verts_rel.astype(np.float32)
    out["joints_uv_smooth_depth_camera"] = joints_uv.astype(np.float32)
    out["mano_smoothing_qc_flag"] = string_array(repair_qc)
    out["pose_repair_bad_pose"] = repair_mask.astype(np.int32)
    out["pose_repair_repaired"] = repaired.astype(np.int32)
    out["pose_repair_reason"] = string_array(reason)
    out["pose_repair_method"] = string_array(method)
    out["pose_repair_prev_frame"] = prev_frame.astype(np.int32)
    out["pose_repair_next_frame"] = next_frame.astype(np.int32)
    out["pose_repair_alpha"] = alpha_arr.astype(np.float32)
    out["pose_repair_global_rot_delta_after_deg"] = after_global_delta.astype(np.float32)
    out["pose_repair_wrist_delta_m"] = wrist_delta.astype(np.float32)
    out["pose_repair_qc_flag"] = string_array(qc_flags)

    output_npz = output_dir / "wilor_handresults_phase_c2b_bad_pose_repaired.npz"
    quality_csv = output_dir / "bad_pose_repair_quality.csv"
    summary_json = output_dir / "bad_pose_repair_summary.json"
    np.savez_compressed(output_npz, **out)
    write_csv(quality_csv, rows)

    qc_counter = Counter(qc_flags)
    reason_counter = Counter(r for r in reason if r != "ok")
    hard_errors: List[str] = []
    warnings: List[str] = []
    failed = int(np.sum(repair_mask & (repaired == 0)))
    if finite_ratio(joints) < 1.0 or finite_ratio(verts) < 1.0:
        hard_errors.append("non_finite_repaired_geometry")
    if failed:
        warnings.append(f"pose_repair_failed:{failed}")
    elapsed = float(time.time() - t0)
    summary = {
        "semantic": "LFV Phase-C2b detected-bad-pose repair by same-track MANO interpolation",
        "session_dir": str(session_dir),
        "input_npz": str(input_npz),
        "output_npz": str(output_npz),
        "quality_csv": str(quality_csv),
        "candidates": int(n),
        "bad_pose_candidates": int(np.sum(repair_mask)),
        "repaired_candidates": int(np.sum(repaired)),
        "failed_candidates": failed,
        "bad_global_rot_delta_deg": float(args.bad_global_rot_delta_deg),
        "bad_infilled_wrist_jump_m": float(args.bad_infilled_wrist_jump_m),
        "neighbor_window_frames": int(args.neighbor_window_frames),
        "bridge_good_gap_frames": int(args.bridge_good_gap_frames),
        "repair_motion_infilled_jumps": bool(args.repair_motion_infilled_jumps),
        "reason_counts": dict(reason_counter),
        "qc_flag_counts": dict(qc_counter),
        "wrist_delta_m": stats(wrist_delta),
        "global_rot_delta_after_deg": stats(after_global_delta),
        "elapsed_sec": elapsed,
        "hard_errors": hard_errors,
        "warnings": warnings,
        "ok": len(hard_errors) == 0,
    }
    summary_json.write_text(json.dumps(json_clean(summary), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(json_clean(summary), indent=2, ensure_ascii=False))
    return 0 if not hard_errors else 2


if __name__ == "__main__":
    args_for_paths = parse_args()
    wilor_root = Path(args_for_paths.wilor_root).expanduser().resolve()
    checkpoint = Path(args_for_paths.checkpoint).expanduser().resolve() if args_for_paths.checkpoint else wilor_root / "pretrained_models" / "wilor_final.ckpt"
    model_config = Path(args_for_paths.model_config).expanduser().resolve() if args_for_paths.model_config else wilor_root / "pretrained_models" / "model_config.yaml"
    mano_model = load_wilor_mano(wilor_root, checkpoint, model_config)
    raise SystemExit(main())
