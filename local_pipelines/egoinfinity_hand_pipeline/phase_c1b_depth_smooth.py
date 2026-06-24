#!/usr/bin/env python3
"""Phase-C1b EgoInfinity-style hand depth smoothing.

This stage adapts the hand part of EgoInfinity's ``depth_smooth.py`` to the
local LFV NPZ pipeline.  It samples FoundationStereo depth at wrist + MCP
locations, rejects local depth outliers, smooths the per-track hand Z anchor
with a weighted Gaussian, and translates each whole hand along camera Z.

The output NPZ intentionally overwrites ``cam_t_depth``, ``joints_cam_depth``
and ``vertices_cam_depth`` so downstream Phase-C2 can consume it without a
special path.  The original ``cam_t_depth`` is preserved as
``cam_t_depth_before_depth_smooth``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


RELIABLE_JOINT_IDS = [0, 5, 9, 13, 17]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session-dir", required=True)
    p.add_argument("--input-npz", required=True, help="Phase-C depth-aligned NPZ")
    p.add_argument("--depth-summary-json", required=True)
    p.add_argument("--depth-frame-csv", default="")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--joint-ids", default="0,5,9,13,17")
    p.add_argument("--uv-field", default="joints_uv")
    p.add_argument("--sigma-z", type=float, default=5.0)
    p.add_argument("--mad-factor", type=float, default=2.0)
    p.add_argument("--min-inliers", type=int, default=3)
    p.add_argument("--patch-size", type=int, default=7)
    p.add_argument("--temporal-mad-window", type=int, default=5)
    p.add_argument("--temporal-mad-factor", type=float, default=5.0)
    p.add_argument("--max-delta-z-m", type=float, default=0.30)
    p.add_argument("--min-track-anchors", type=int, default=4)
    p.add_argument("--smooth-vertex-mean-z", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--warn-delta-z-m", type=float, default=0.080)
    p.add_argument("--bad-delta-z-m", type=float, default=0.160)
    p.add_argument("--warn-after-jump-m", type=float, default=0.090)
    p.add_argument("--bad-after-jump-m", type=float, default=0.160)
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


def finite_ratio(arr: np.ndarray) -> float:
    arr = np.asarray(arr)
    if arr.size == 0:
        return 0.0
    return float(np.mean(np.isfinite(arr)))


def string_array(values: Sequence[str]) -> np.ndarray:
    max_len = max([1] + [len(str(v)) for v in values])
    return np.asarray([str(v) for v in values], dtype=f"<U{max_len}")


def parse_joint_ids(text: str) -> List[int]:
    out: List[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise RuntimeError("empty --joint-ids")
    return out


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def read_depth_frame_csv(path: Path) -> Dict[int, Dict[str, str]]:
    rows: Dict[int, Dict[str, str]] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[int(row["frame_index"])] = row
    return rows


class DepthCache:
    def __init__(self, rows: Dict[int, Dict[str, str]], max_items: int = 16):
        self.rows = rows
        self.max_items = int(max_items)
        self.cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.missing_frames: set[int] = set()

    def get(self, frame: int) -> np.ndarray | None:
        frame = int(frame)
        if frame in self.cache:
            value = self.cache.pop(frame)
            self.cache[frame] = value
            return value
        row = self.rows.get(frame)
        if row is None:
            self.missing_frames.add(frame)
            return None
        path = Path(row.get("depth_npy", ""))
        if not path.exists():
            self.missing_frames.add(frame)
            return None
        depth = np.load(path).astype(np.float32)
        self.cache[frame] = depth
        while len(self.cache) > self.max_items:
            self.cache.popitem(last=False)
        return depth


def sample_depth_patch(depth: np.ndarray, u: float, v: float, half: int) -> float:
    h, w = depth.shape[:2]
    if not np.isfinite(u) or not np.isfinite(v):
        return float("nan")
    x = int(round(float(u)))
    y = int(round(float(v)))
    if x < 0 or x >= w or y < 0 or y >= h:
        return float("nan")
    x0, x1 = max(0, x - int(half)), min(w, x + int(half) + 1)
    y0, y1 = max(0, y - int(half)), min(h, y + int(half) + 1)
    patch = depth[y0:y1, x0:x1]
    valid = patch[np.isfinite(patch) & (patch > 0.01)]
    return float(np.median(valid)) if valid.size else float("nan")


def gaussian_smooth_weighted(values: np.ndarray, weights: np.ndarray, sigma: float) -> np.ndarray:
    n = len(values)
    if n == 0:
        return values
    values = np.where(np.isfinite(values), values, 0.0).astype(np.float64)
    weights = np.where(np.isfinite(weights), weights, 0.0).astype(np.float64)
    half = max(int(np.ceil(3.0 * float(sigma))), 1)
    kernel = np.exp(-0.5 * (np.arange(-half, half + 1, dtype=np.float64) / float(sigma)) ** 2)
    out = np.zeros(n, dtype=np.float64)
    norm = np.zeros(n, dtype=np.float64)
    for k, kw in enumerate(kernel):
        offset = k - half
        i0 = max(0, -offset)
        i1 = min(n, n - offset)
        if i1 <= i0:
            continue
        src = slice(i0 + offset, i1 + offset)
        dst = slice(i0, i1)
        out[dst] += float(kw) * weights[src] * values[src]
        norm[dst] += float(kw) * weights[src]
    return np.where(norm > 1e-9, out / np.maximum(norm, 1e-9), values)


def temporal_outlier_reject(
    series: np.ndarray,
    trust: np.ndarray,
    window: int,
    mad_factor: float,
) -> np.ndarray:
    n = len(series)
    out = trust.astype(np.float64).copy()
    for t in range(n):
        if not np.isfinite(series[t]) or trust[t] <= 0:
            continue
        lo = max(0, t - int(window))
        hi = min(n, t + int(window) + 1)
        valid = np.isfinite(series[lo:hi]) & (trust[lo:hi] > 0)
        if int(np.sum(valid)) < 3:
            continue
        vals = series[lo:hi][valid]
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med))) + 1e-6
        if abs(float(series[t]) - med) > float(mad_factor) * mad:
            out[t] = 0.0
    return out


def frame_to_frame_jump(values: np.ndarray) -> np.ndarray:
    out = np.full((len(values),), np.nan, dtype=np.float32)
    if len(values) > 1:
        out[1:] = np.abs(np.diff(values)).astype(np.float32)
    return out


def qc_flag_for_candidate(
    status: str,
    delta_z: float,
    clipped: int,
    after_jump: float,
    args: argparse.Namespace,
) -> str:
    flags: List[str] = []
    if status != "depth_smooth_ok":
        flags.append(status)
    abs_delta = abs(float(delta_z)) if math.isfinite(float(delta_z)) else float("nan")
    if math.isfinite(abs_delta):
        if abs_delta > float(args.bad_delta_z_m):
            flags.append("bad_large_depth_smooth_delta")
        elif abs_delta > float(args.warn_delta_z_m):
            flags.append("warn_large_depth_smooth_delta")
    if clipped:
        flags.append("warn_delta_z_clipped")
    if math.isfinite(float(after_jump)):
        if float(after_jump) > float(args.bad_after_jump_m):
            flags.append("bad_after_depth_smooth_jump")
        elif float(after_jump) > float(args.warn_after_jump_m):
            flags.append("warn_after_depth_smooth_jump")
    return "|".join(flags) if flags else "ok"


def write_quality_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fields = [
        "frame_index", "elapsed_sec", "candidate_index", "track_id", "hand_label",
        "is_right", "current_wrist_z_m", "anchor_z_m", "anchor_z_smooth_m",
        "trust", "valid_depth_count", "joint_ids", "sampled_depths_m",
        "delta_z_m", "vertex_mean_delta_z_m", "delta_z_clipped",
        "before_wrist_z_jump_m", "after_wrist_z_jump_m", "status", "qc_flag",
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
        "valid_anchor_count", "smoothed", "before_wrist_z_jump_p95_m",
        "after_wrist_z_jump_p95_m", "delta_z_p95_m", "delta_z_max_m",
        "vertex_mean_delta_z_p95_m", "vertex_mean_delta_z_max_m",
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
    depth_cache = DepthCache(depth_rows)
    joint_ids = parse_joint_ids(args.joint_ids)
    half = max(0, int(args.patch_size) // 2)

    required = [
        "frame_index", "track_id", "hand_label", "is_right", "candidate_index",
        "cam_t_depth", "joints_cam_depth", "vertices_cam_depth", args.uv_field,
    ]
    missing = [key for key in required if key not in data]
    if missing:
        raise RuntimeError(f"input npz missing fields: {missing}")

    n = int(len(data["frame_index"]))
    frame_index = np.asarray(data["frame_index"], dtype=np.int32)
    track_id = np.asarray(data["track_id"], dtype=np.int32)
    hand_label = np.asarray(data["hand_label"]).astype(str)
    is_right = np.asarray(data["is_right"], dtype=np.float32)
    cam_t_before = np.asarray(data["cam_t_depth"], dtype=np.float32)
    joints_before = np.asarray(data["joints_cam_depth"], dtype=np.float32)
    verts_before = np.asarray(data["vertices_cam_depth"], dtype=np.float32)
    uv = np.asarray(data[args.uv_field], dtype=np.float32)

    current_z = joints_before[:, 0, 2].astype(np.float64)
    anchor_z = np.full((n,), np.nan, dtype=np.float64)
    trust_raw = np.zeros((n,), dtype=np.float64)
    trust_final = np.zeros((n,), dtype=np.float64)
    valid_counts = np.zeros((n,), dtype=np.int32)
    sampled_depths_text: List[str] = [""] * n
    joint_ids_text: List[str] = [""] * n
    status = ["no_depth_anchor"] * n

    for i in range(n):
        depth = depth_cache.get(int(frame_index[i]))
        if depth is None:
            status[i] = "missing_depth_frame"
            continue
        depths: List[float] = []
        used_ids: List[int] = []
        for jid in joint_ids:
            d = sample_depth_patch(depth, float(uv[i, jid, 0]), float(uv[i, jid, 1]), half)
            if math.isfinite(d) and d > 0.0:
                depths.append(d)
                used_ids.append(jid)
        valid_counts[i] = len(depths)
        joint_ids_text[i] = ",".join(str(v) for v in used_ids)
        sampled_depths_text[i] = ",".join(csv_float(v) for v in depths)
        if len(depths) < int(args.min_inliers):
            status[i] = "low_valid_depth_count"
            continue
        arr = np.asarray(depths, dtype=np.float64)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med))) + 1e-6
        inliers = arr[np.abs(arr - med) <= float(args.mad_factor) * mad]
        if len(inliers) < int(args.min_inliers):
            status[i] = "low_inlier_depth_count"
            continue
        anchor_z[i] = float(np.mean(inliers))
        trust_raw[i] = 1.0
        status[i] = "anchor_ok"

    delta_z = np.zeros((n,), dtype=np.float64)
    anchor_smooth = np.where(np.isfinite(anchor_z), anchor_z, current_z).astype(np.float64)
    delta_clipped = np.zeros((n,), dtype=np.int32)
    before_jump = np.full((n,), np.nan, dtype=np.float32)
    after_jump = np.full((n,), np.nan, dtype=np.float32)
    vertex_delta_z = np.zeros((n,), dtype=np.float64)

    indices_by_track: Dict[int, List[int]] = defaultdict(list)
    for i, tid in enumerate(track_id.tolist()):
        indices_by_track[int(tid)].append(i)

    track_rows: List[Dict[str, Any]] = []
    for tid, indices_unsorted in sorted(indices_by_track.items()):
        indices = sorted(indices_unsorted, key=lambda i: (int(frame_index[i]), int(data["candidate_index"][i])))
        valid_count = int(np.sum(trust_raw[indices] > 0))
        smoothed = valid_count >= int(args.min_track_anchors)
        current_track = current_z[indices]
        before_track_jump = frame_to_frame_jump(current_track)
        before_jump[indices] = before_track_jump
        if smoothed:
            trust_track = temporal_outlier_reject(
                anchor_z[indices],
                trust_raw[indices],
                int(args.temporal_mad_window),
                float(args.temporal_mad_factor),
            )
            trust_final[indices] = trust_track
            filled = np.where(np.isfinite(anchor_z[indices]), anchor_z[indices], current_track)
            smooth = gaussian_smooth_weighted(filled, trust_track, float(args.sigma_z))
            raw_delta = smooth - current_track
            clipped_delta = np.clip(raw_delta, -float(args.max_delta_z_m), float(args.max_delta_z_m))
            delta_z[indices] = np.where(np.isfinite(clipped_delta), clipped_delta, 0.0)
            delta_clipped[indices] = (np.abs(raw_delta - clipped_delta) > 1e-9).astype(np.int32)
            anchor_smooth[indices] = smooth
            for i, t in zip(indices, trust_track):
                if status[i] == "anchor_ok":
                    status[i] = "depth_smooth_ok" if t > 0 else "depth_anchor_temporal_outlier"
        else:
            trust_final[indices] = trust_raw[indices]
            for i in indices:
                if status[i] == "anchor_ok":
                    status[i] = "track_skipped_low_anchor_count"
        after_track_jump = frame_to_frame_jump(current_track + delta_z[indices])
        after_jump[indices] = after_track_jump

        abs_delta = np.abs(delta_z[indices])
        v_abs_delta = np.abs(vertex_delta_z[indices])
        track_rows.append(
            {
                "track_id": tid,
                "hand_label": Counter(hand_label[indices].tolist()).most_common(1)[0][0],
                "is_right": int(np.mean(is_right[indices] >= 0.5) >= 0.5),
                "frame_count": len(indices),
                "frame_min": int(frame_index[indices[0]]),
                "frame_max": int(frame_index[indices[-1]]),
                "valid_anchor_count": valid_count,
                "smoothed": int(smoothed),
                "before_wrist_z_jump_p95_m": csv_float(float(np.nanpercentile(before_track_jump, 95))),
                "after_wrist_z_jump_p95_m": csv_float(float(np.nanpercentile(after_track_jump, 95))),
                "delta_z_p95_m": csv_float(float(np.nanpercentile(abs_delta, 95))),
                "delta_z_max_m": csv_float(float(np.nanmax(abs_delta))),
                "vertex_mean_delta_z_p95_m": csv_float(float(np.nanpercentile(v_abs_delta, 95))),
                "vertex_mean_delta_z_max_m": csv_float(float(np.nanmax(v_abs_delta))),
            }
        )

    cam_t_after = cam_t_before.copy()
    joints_after = joints_before.copy()
    verts_after = verts_before.copy()
    cam_t_after[:, 2] += delta_z.astype(np.float32)
    joints_after[:, :, 2] += delta_z[:, None].astype(np.float32)
    verts_after[:, :, 2] += delta_z[:, None].astype(np.float32)

    if bool(args.smooth_vertex_mean_z):
        for tid, indices_unsorted in sorted(indices_by_track.items()):
            indices = sorted(indices_unsorted, key=lambda i: (int(frame_index[i]), int(data["candidate_index"][i])))
            vmz = np.nanmean(verts_after[indices, :, 2], axis=1).astype(np.float64)
            valid = np.isfinite(vmz).astype(np.float64)
            if int(np.sum(valid > 0)) < int(args.min_track_anchors):
                continue
            trust = temporal_outlier_reject(vmz, valid, int(args.temporal_mad_window), float(args.temporal_mad_factor))
            smooth = gaussian_smooth_weighted(vmz, trust, float(args.sigma_z))
            raw_delta = smooth - vmz
            vd = np.clip(raw_delta, -float(args.max_delta_z_m), float(args.max_delta_z_m))
            vertex_delta_z[indices] = np.where(np.isfinite(vd), vd, 0.0)
        verts_after[:, :, 2] += vertex_delta_z[:, None].astype(np.float32)

    track_rows = []
    for tid, indices_unsorted in sorted(indices_by_track.items()):
        indices = sorted(indices_unsorted, key=lambda i: (int(frame_index[i]), int(data["candidate_index"][i])))
        valid_count = int(np.sum(trust_raw[indices] > 0))
        abs_delta = np.abs(delta_z[indices])
        v_abs_delta = np.abs(vertex_delta_z[indices])
        track_rows.append(
            {
                "track_id": tid,
                "hand_label": Counter(hand_label[indices].tolist()).most_common(1)[0][0],
                "is_right": int(np.mean(is_right[indices] >= 0.5) >= 0.5),
                "frame_count": len(indices),
                "frame_min": int(frame_index[indices[0]]),
                "frame_max": int(frame_index[indices[-1]]),
                "valid_anchor_count": valid_count,
                "smoothed": int(valid_count >= int(args.min_track_anchors)),
                "before_wrist_z_jump_p95_m": csv_float(float(np.nanpercentile(before_jump[indices], 95))),
                "after_wrist_z_jump_p95_m": csv_float(float(np.nanpercentile(after_jump[indices], 95))),
                "delta_z_p95_m": csv_float(float(np.nanpercentile(abs_delta, 95))),
                "delta_z_max_m": csv_float(float(np.nanmax(abs_delta))),
                "vertex_mean_delta_z_p95_m": csv_float(float(np.nanpercentile(v_abs_delta, 95))),
                "vertex_mean_delta_z_max_m": csv_float(float(np.nanmax(v_abs_delta))),
            }
        )

    qc_flags: List[str] = []
    quality_rows: List[Dict[str, Any]] = []
    for i in range(n):
        flag = qc_flag_for_candidate(status[i], float(delta_z[i]), int(delta_clipped[i]), float(after_jump[i]), args)
        qc_flags.append(flag)
        quality_rows.append(
            {
                "frame_index": int(frame_index[i]),
                "elapsed_sec": csv_float(float(data["elapsed_sec"][i])) if "elapsed_sec" in data else "",
                "candidate_index": int(data["candidate_index"][i]) if "candidate_index" in data else i,
                "track_id": int(track_id[i]),
                "hand_label": str(hand_label[i]),
                "is_right": int(round(float(is_right[i]))),
                "current_wrist_z_m": csv_float(float(current_z[i])),
                "anchor_z_m": csv_float(float(anchor_z[i])),
                "anchor_z_smooth_m": csv_float(float(anchor_smooth[i])),
                "trust": csv_float(float(trust_final[i])),
                "valid_depth_count": int(valid_counts[i]),
                "joint_ids": joint_ids_text[i],
                "sampled_depths_m": sampled_depths_text[i],
                "delta_z_m": csv_float(float(delta_z[i])),
                "vertex_mean_delta_z_m": csv_float(float(vertex_delta_z[i])),
                "delta_z_clipped": int(delta_clipped[i]),
                "before_wrist_z_jump_m": csv_float(float(before_jump[i])),
                "after_wrist_z_jump_m": csv_float(float(after_jump[i])),
                "status": status[i],
                "qc_flag": flag,
            }
        )

    output_npz = output_dir / "wilor_handresults_phase_c1b_depth_smooth.npz"
    quality_csv = output_dir / "depth_smooth_quality.csv"
    track_csv = output_dir / "depth_smooth_track_summary.csv"
    summary_json = output_dir / "depth_smooth_summary.json"

    out = dict(data)
    out["cam_t_depth_before_depth_smooth"] = cam_t_before.astype(np.float32)
    out["cam_t_depth_smooth"] = cam_t_after.astype(np.float32)
    out["joints_cam_depth_smooth"] = joints_after.astype(np.float32)
    out["vertices_cam_depth_smooth"] = verts_after.astype(np.float32)
    out["cam_t_depth"] = cam_t_after.astype(np.float32)
    out["joints_cam_depth"] = joints_after.astype(np.float32)
    out["vertices_cam_depth"] = verts_after.astype(np.float32)
    out["depth_smooth_anchor_z_m"] = anchor_z.astype(np.float32)
    out["depth_smooth_anchor_z_smoothed_m"] = anchor_smooth.astype(np.float32)
    out["depth_smooth_delta_z_m"] = delta_z.astype(np.float32)
    out["depth_smooth_vertex_mean_delta_z_m"] = vertex_delta_z.astype(np.float32)
    out["depth_smooth_trust"] = trust_final.astype(np.float32)
    out["depth_smooth_valid_sample_count"] = valid_counts.astype(np.int32)
    out["depth_smooth_status"] = string_array(status)
    out["depth_smooth_qc_flag"] = string_array(qc_flags)
    np.savez_compressed(output_npz, **out)
    write_quality_csv(quality_csv, quality_rows)
    write_track_csv(track_csv, track_rows)

    status_counts = Counter(status)
    qc_counts = Counter(qc_flags)
    hard_errors: List[str] = []
    warnings: List[str] = []
    if finite_ratio(cam_t_after) < 1.0 or finite_ratio(joints_after) < 1.0 or finite_ratio(verts_after) < 1.0:
        hard_errors.append("non_finite_depth_smooth_geometry")
    skipped = status_counts.get("track_skipped_low_anchor_count", 0)
    if skipped:
        warnings.append(f"depth_smooth_skipped_low_anchor_candidates:{skipped}")
    outlier = status_counts.get("depth_anchor_temporal_outlier", 0)
    if outlier:
        warnings.append(f"depth_smooth_temporal_outlier_anchors:{outlier}")
    bad_count = sum(v for k, v in qc_counts.items() if "bad_" in str(k))
    warn_count = sum(v for k, v in qc_counts.items() if "warn_" in str(k))
    if bad_count:
        warnings.append(f"depth_smooth_bad_flags:{bad_count}")
    if warn_count:
        warnings.append(f"depth_smooth_warn_flags:{warn_count}")
    if depth_cache.missing_frames:
        warnings.append(f"depth_smooth_missing_depth_frames:{len(depth_cache.missing_frames)}")

    before_valid = before_jump[np.isfinite(before_jump)]
    after_valid = after_jump[np.isfinite(after_jump)]
    summary = {
        "semantic": "LFV Phase-C1b EgoInfinity-style hand Z depth smoothing",
        "session_dir": str(session_dir),
        "input_npz": str(input_npz),
        "depth_summary_json": str(depth_summary_json),
        "depth_frame_csv": str(depth_frame_csv),
        "output_npz": str(output_npz),
        "quality_csv": str(quality_csv),
        "track_summary_csv": str(track_csv),
        "candidates": n,
        "tracks": int(len(indices_by_track)),
        "joint_ids": joint_ids,
        "uv_field": str(args.uv_field),
        "sigma_z": float(args.sigma_z),
        "mad_factor": float(args.mad_factor),
        "min_inliers": int(args.min_inliers),
        "patch_size": int(args.patch_size),
        "temporal_mad_window": int(args.temporal_mad_window),
        "temporal_mad_factor": float(args.temporal_mad_factor),
        "max_delta_z_m": float(args.max_delta_z_m),
        "smooth_vertex_mean_z": bool(args.smooth_vertex_mean_z),
        "status_counts": dict(status_counts),
        "qc_flag_counts": dict(qc_counts),
        "anchor_z_m": stats(anchor_z),
        "delta_z_m": stats(delta_z),
        "abs_delta_z_m": stats(np.abs(delta_z)),
        "vertex_mean_delta_z_m": stats(vertex_delta_z),
        "valid_sample_count": stats(valid_counts),
        "trust": stats(trust_final),
        "before_wrist_z_jump_m": stats(before_valid),
        "after_wrist_z_jump_m": stats(after_valid),
        "cam_t_depth_finite_ratio": finite_ratio(cam_t_after),
        "joints_cam_depth_finite_ratio": finite_ratio(joints_after),
        "vertices_cam_depth_finite_ratio": finite_ratio(verts_after),
        "missing_depth_frame_count": int(len(depth_cache.missing_frames)),
        "missing_depth_frames": sorted(int(v) for v in depth_cache.missing_frames)[:50],
        "hard_errors": hard_errors,
        "warnings": warnings,
        "ok": len(hard_errors) == 0,
    }
    summary_json.write_text(json.dumps(json_clean(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(json_clean(summary), indent=2, ensure_ascii=False))
    return 0 if len(hard_errors) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
