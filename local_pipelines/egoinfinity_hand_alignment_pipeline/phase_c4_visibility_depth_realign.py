#!/usr/bin/env python3
"""Experimental visibility-aware depth re-alignment.

This is not an EgoInfinity original stage.  It uses Phase-C3 MANO mesh
visibility to choose which hand joints are allowed to touch FoundationStereo
depth.  It is deliberately isolated from the main chain: it writes its own
``cam_t_visibility_depth`` output and does not overwrite ``cam_t_smooth``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


RELIABLE_JOINT_IDS = [0, 5, 9, 13, 17]
STABLE_VISIBLE_JOINT_IDS = [0, 1, 2, 5, 6, 9, 10, 13, 14, 17, 18]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session-dir", required=True)
    p.add_argument("--input-npz", required=True, help="Phase-C3 mesh-visibility NPZ")
    p.add_argument("--depth-summary-json", required=True)
    p.add_argument("--depth-frame-csv", default="")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--patch-size", type=int, default=7)
    p.add_argument("--min-visible-reliable-joints", type=int, default=2)
    p.add_argument("--min-visible-stable-joints", type=int, default=3)
    p.add_argument("--enable-all-visible-fallback", action="store_true")
    p.add_argument("--min-all-visible-joints", type=int, default=4)
    p.add_argument("--max-patch-spread-m", type=float, default=0.120)
    p.add_argument("--max-patch-mad-m", type=float, default=0.040)
    p.add_argument("--min-patch-valid-pixels", type=int, default=4)
    p.add_argument("--warn-rms-m", type=float, default=0.035)
    p.add_argument("--bad-rms-m", type=float, default=0.080)
    p.add_argument("--warn-delta-m", type=float, default=0.080)
    p.add_argument("--bad-delta-m", type=float, default=0.160)
    p.add_argument("--keep-previous-on-bad-rms", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cam-t-field", default="cam_t_smooth")
    p.add_argument("--joints-rel-field", default="joints_3d_rel_smooth")
    p.add_argument("--vertices-rel-field", default="vertices_rel_smooth")
    p.add_argument("--joints-uv-field", default="joints_uv_smooth_depth_camera")
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


def sample_depth_patch(
    depth: np.ndarray,
    u: float,
    v: float,
    half: int,
    min_pixels: int,
    max_spread_m: float,
    max_mad_m: float,
) -> Tuple[float, str, float, float, int]:
    h, w = depth.shape[:2]
    if not np.isfinite(u) or not np.isfinite(v):
        return float("nan"), "invalid_uv", float("nan"), float("nan"), 0
    x = int(round(float(u)))
    y = int(round(float(v)))
    if x < 0 or x >= w or y < 0 or y >= h:
        return float("nan"), "out_of_view", float("nan"), float("nan"), 0
    x0, x1 = max(0, x - int(half)), min(w, x + int(half) + 1)
    y0, y1 = max(0, y - int(half)), min(h, y + int(half) + 1)
    patch = depth[y0:y1, x0:x1]
    valid = patch[np.isfinite(patch) & (patch > 0.01)]
    if valid.size < int(min_pixels):
        return float("nan"), "low_patch_valid_pixels", float("nan"), float("nan"), int(valid.size)
    med = float(np.median(valid))
    spread = float(np.percentile(valid, 90) - np.percentile(valid, 10))
    mad = float(np.median(np.abs(valid - med)))
    if spread > float(max_spread_m):
        return float("nan"), "patch_spread_high", spread, mad, int(valid.size)
    if mad > float(max_mad_m):
        return float("nan"), "patch_mad_high", spread, mad, int(valid.size)
    return med, "ok", spread, mad, int(valid.size)


def backproject(u: np.ndarray, v: np.ndarray, z: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    pts = np.zeros((len(z), 3), dtype=np.float32)
    pts[:, 2] = z.astype(np.float32)
    pts[:, 0] = ((u - float(cx)) * z / float(fx)).astype(np.float32)
    pts[:, 1] = ((v - float(cy)) * z / float(fy)).astype(np.float32)
    return pts


def choose_visible_joint_ids(
    visible: np.ndarray,
    sampled_valid: Dict[int, bool],
    args: argparse.Namespace,
) -> Tuple[List[int], str]:
    reliable = [jid for jid in RELIABLE_JOINT_IDS if int(visible[jid]) > 0 and sampled_valid.get(jid, False)]
    if len(reliable) >= int(args.min_visible_reliable_joints):
        return reliable, "visible_reliable_joints"

    stable = [jid for jid in STABLE_VISIBLE_JOINT_IDS if int(visible[jid]) > 0 and sampled_valid.get(jid, False)]
    if len(stable) >= int(args.min_visible_stable_joints):
        return stable, "visible_stable_joints"

    if bool(args.enable_all_visible_fallback):
        all_visible = [jid for jid in range(len(visible)) if int(visible[jid]) > 0 and sampled_valid.get(jid, False)]
        if len(all_visible) >= int(args.min_all_visible_joints):
            return all_visible, "visible_all_joints_fallback"

    return [], "keep_previous_low_visible_depth"


def qc_flag_for_candidate(
    source: str,
    rms: float,
    delta: float,
    rejected_bad_rms: int,
    args: argparse.Namespace,
) -> str:
    flags: List[str] = []
    if source not in ("visible_reliable_joints", "visible_stable_joints"):
        flags.append(source)
    if rejected_bad_rms:
        flags.append("bad_rms_rejected_keep_previous")
    if math.isfinite(float(rms)):
        if float(rms) > float(args.bad_rms_m):
            flags.append("bad_visibility_realign_rms")
        elif float(rms) > float(args.warn_rms_m):
            flags.append("warn_visibility_realign_rms")
    else:
        flags.append("missing_visibility_realign_rms")
    if math.isfinite(float(delta)):
        if float(delta) > float(args.bad_delta_m):
            flags.append("bad_visibility_realign_delta")
        elif float(delta) > float(args.warn_delta_m):
            flags.append("warn_visibility_realign_delta")
    return "|".join(flags) if flags else "ok"


def write_quality_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fields = [
        "frame_index", "elapsed_sec", "candidate_index", "track_id", "hand_label",
        "is_right", "source", "selected_joint_count", "selected_joint_ids",
        "sampled_depths_m", "patch_reject_counts", "rms_m", "max_residual_m",
        "delta_m", "candidate_delta_m", "rejected_bad_rms_keep_previous",
        "visible_reliable_joint_count", "visible_joint_count", "qc_flag",
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
    depth_cache = DepthCache(read_depth_frame_csv(depth_frame_csv))
    camera = depth_summary.get("camera") or {}
    fx, fy = float(camera["fx"]), float(camera["fy"])
    cx, cy = float(camera["cx"]), float(camera["cy"])
    half = max(0, int(args.patch_size) // 2)

    cam_t_field = str(args.cam_t_field)
    joints_rel_field = str(args.joints_rel_field)
    vertices_rel_field = str(args.vertices_rel_field)
    joints_uv_field = str(args.joints_uv_field)
    required = [
        "frame_index", "track_id", "hand_label", "is_right", "candidate_index",
        cam_t_field, joints_rel_field, vertices_rel_field,
        joints_uv_field, "mano_joint_visible",
        "mano_visible_reliable_joint_count", "mano_visible_joint_count",
    ]
    missing = [key for key in required if key not in data]
    if missing:
        raise RuntimeError(f"input npz missing fields: {missing}")

    frame_index = np.asarray(data["frame_index"], dtype=np.int32)
    n = int(len(frame_index))
    cam_t_prev = np.asarray(data[cam_t_field], dtype=np.float32)
    joints_rel = np.asarray(data[joints_rel_field], dtype=np.float32)
    verts_rel = np.asarray(data[vertices_rel_field], dtype=np.float32)
    joints_uv = np.asarray(data[joints_uv_field], dtype=np.float32)
    visible = np.asarray(data["mano_joint_visible"], dtype=np.uint8)

    cam_t_candidate = cam_t_prev.copy()
    cam_t_final = cam_t_prev.copy()
    joints_final = joints_rel + cam_t_final[:, None, :]
    verts_final = verts_rel + cam_t_final[:, None, :]
    source: List[str] = []
    selected_counts = np.zeros((n,), dtype=np.int32)
    selected_text: List[str] = []
    sampled_depth_text: List[str] = []
    patch_reject_text: List[str] = []
    rms_arr = np.full((n,), np.nan, dtype=np.float32)
    max_res_arr = np.full((n,), np.nan, dtype=np.float32)
    delta_final = np.zeros((n,), dtype=np.float32)
    delta_candidate = np.zeros((n,), dtype=np.float32)
    rejected_bad_rms = np.zeros((n,), dtype=np.int32)
    qc_flags: List[str] = []
    rows: List[Dict[str, Any]] = []

    for i in range(n):
        depth = depth_cache.get(int(frame_index[i]))
        sampled: Dict[int, float] = {}
        sampled_valid: Dict[int, bool] = {}
        reject_counts: Counter[str] = Counter()
        if depth is not None:
            for jid in range(joints_uv.shape[1]):
                d, reason, _, _, _ = sample_depth_patch(
                    depth,
                    float(joints_uv[i, jid, 0]),
                    float(joints_uv[i, jid, 1]),
                    half,
                    int(args.min_patch_valid_pixels),
                    float(args.max_patch_spread_m),
                    float(args.max_patch_mad_m),
                )
                if reason == "ok" and math.isfinite(d):
                    sampled[jid] = d
                    sampled_valid[jid] = True
                else:
                    sampled_valid[jid] = False
                    reject_counts[reason] += 1
        else:
            reject_counts["missing_depth_frame"] += 1

        ids, src = choose_visible_joint_ids(visible[i], sampled_valid, args)
        source.append(src)
        selected_counts[i] = len(ids)
        selected_text.append(",".join(str(v) for v in ids))
        sampled_depth_text.append(",".join(csv_float(sampled[jid]) for jid in ids))
        patch_reject_text.append(",".join(f"{k}:{v}" for k, v in sorted(reject_counts.items())))

        if ids:
            uv_i = joints_uv[i, ids]
            depths = np.asarray([sampled[jid] for jid in ids], dtype=np.float32)
            obs = backproject(uv_i[:, 0], uv_i[:, 1], depths, fx, fy, cx, cy)
            rel = joints_rel[i, ids].astype(np.float32)
            translations = obs - rel
            proposed = np.nanmedian(translations, axis=0).astype(np.float32)
            cam_t_candidate[i] = proposed
            pred = rel + proposed.reshape(1, 3)
            residual = np.linalg.norm(pred - obs, axis=1)
            rms = float(np.sqrt(np.mean(residual * residual))) if residual.size else float("nan")
            max_res = float(np.max(residual)) if residual.size else float("nan")
            rms_arr[i] = rms
            max_res_arr[i] = max_res
            delta_candidate[i] = float(np.linalg.norm(proposed - cam_t_prev[i]))
            if bool(args.keep_previous_on_bad_rms) and math.isfinite(rms) and rms > float(args.bad_rms_m):
                rejected_bad_rms[i] = 1
                source[i] = f"{src}_bad_rms_keep_previous"
                cam_t_final[i] = cam_t_prev[i]
            else:
                cam_t_final[i] = proposed
        else:
            cam_t_candidate[i] = cam_t_prev[i]
            cam_t_final[i] = cam_t_prev[i]

        delta_final[i] = float(np.linalg.norm(cam_t_final[i] - cam_t_prev[i]))
        flag = qc_flag_for_candidate(source[i], float(rms_arr[i]), float(delta_final[i]), int(rejected_bad_rms[i]), args)
        qc_flags.append(flag)
        rows.append(
            {
                "frame_index": int(frame_index[i]),
                "elapsed_sec": csv_float(float(data["elapsed_sec"][i])) if "elapsed_sec" in data else "",
                "candidate_index": int(data["candidate_index"][i]) if "candidate_index" in data else i,
                "track_id": int(data["track_id"][i]),
                "hand_label": str(data["hand_label"][i]),
                "is_right": int(round(float(data["is_right"][i]))),
                "source": source[i],
                "selected_joint_count": int(selected_counts[i]),
                "selected_joint_ids": selected_text[i],
                "sampled_depths_m": sampled_depth_text[i],
                "patch_reject_counts": patch_reject_text[i],
                "rms_m": csv_float(float(rms_arr[i])),
                "max_residual_m": csv_float(float(max_res_arr[i])),
                "delta_m": csv_float(float(delta_final[i])),
                "candidate_delta_m": csv_float(float(delta_candidate[i])),
                "rejected_bad_rms_keep_previous": int(rejected_bad_rms[i]),
                "visible_reliable_joint_count": int(data["mano_visible_reliable_joint_count"][i]),
                "visible_joint_count": int(data["mano_visible_joint_count"][i]),
                "qc_flag": flag,
            }
        )

    joints_final = joints_rel + cam_t_final[:, None, :]
    verts_final = verts_rel + cam_t_final[:, None, :]

    output_npz = output_dir / "wilor_handresults_phase_c4_visibility_depth_realign.npz"
    quality_csv = output_dir / "visibility_depth_realign_quality.csv"
    summary_json = output_dir / "visibility_depth_realign_summary.json"

    out = dict(data)
    out["cam_t_visibility_depth_candidate"] = cam_t_candidate.astype(np.float32)
    out["cam_t_visibility_depth"] = cam_t_final.astype(np.float32)
    out["joints_cam_visibility_depth"] = joints_final.astype(np.float32)
    out["vertices_cam_visibility_depth"] = verts_final.astype(np.float32)
    out["visibility_realign_source"] = string_array(source)
    out["visibility_realign_selected_joint_count"] = selected_counts.astype(np.int32)
    out["visibility_realign_selected_joint_ids"] = string_array(selected_text)
    out["visibility_realign_sampled_depths_m"] = string_array(sampled_depth_text)
    out["visibility_realign_rms_m"] = rms_arr.astype(np.float32)
    out["visibility_realign_max_residual_m"] = max_res_arr.astype(np.float32)
    out["visibility_realign_delta_m"] = delta_final.astype(np.float32)
    out["visibility_realign_candidate_delta_m"] = delta_candidate.astype(np.float32)
    out["visibility_realign_rejected_bad_rms"] = rejected_bad_rms.astype(np.int32)
    out["visibility_realign_qc_flag"] = string_array(qc_flags)
    np.savez_compressed(output_npz, **out)
    write_quality_csv(quality_csv, rows)

    source_counts = Counter(source)
    qc_counts = Counter(qc_flags)
    hard_errors: List[str] = []
    warnings: List[str] = []
    if finite_ratio(cam_t_final) < 1.0 or finite_ratio(joints_final) < 1.0 or finite_ratio(verts_final) < 1.0:
        hard_errors.append("non_finite_visibility_realign_geometry")
    bad_count = sum(v for k, v in qc_counts.items() if "bad_" in str(k))
    warn_count = sum(v for k, v in qc_counts.items() if "warn_" in str(k))
    keep_prev_count = sum(v for k, v in source_counts.items() if "keep_previous" in str(k))
    if bad_count:
        warnings.append(f"visibility_realign_bad_flags:{bad_count}")
    if warn_count:
        warnings.append(f"visibility_realign_warn_flags:{warn_count}")
    if keep_prev_count:
        warnings.append(f"visibility_realign_kept_previous_candidates:{keep_prev_count}")
    if depth_cache.missing_frames:
        warnings.append(f"visibility_realign_missing_depth_frames:{len(depth_cache.missing_frames)}")

    summary = {
        "semantic": "LFV Phase-C4 experimental visibility-aware depth re-alignment",
        "session_dir": str(session_dir),
        "input_npz": str(input_npz),
        "depth_summary_json": str(depth_summary_json),
        "depth_frame_csv": str(depth_frame_csv),
        "output_npz": str(output_npz),
        "quality_csv": str(quality_csv),
        "candidates": n,
        "cam_t_field": cam_t_field,
        "joints_rel_field": joints_rel_field,
        "vertices_rel_field": vertices_rel_field,
        "joints_uv_field": joints_uv_field,
        "enabled_by_default": False,
        "keep_previous_on_bad_rms": bool(args.keep_previous_on_bad_rms),
        "enable_all_visible_fallback": bool(args.enable_all_visible_fallback),
        "patch_size": int(args.patch_size),
        "min_visible_reliable_joints": int(args.min_visible_reliable_joints),
        "min_visible_stable_joints": int(args.min_visible_stable_joints),
        "source_counts": dict(source_counts),
        "qc_flag_counts": dict(qc_counts),
        "selected_joint_count": stats(selected_counts),
        "rms_m": stats(rms_arr),
        "max_residual_m": stats(max_res_arr),
        "delta_m": stats(delta_final),
        "candidate_delta_m": stats(delta_candidate),
        "rejected_bad_rms_count": int(np.sum(rejected_bad_rms)),
        "missing_depth_frame_count": int(len(depth_cache.missing_frames)),
        "missing_depth_frames": sorted(int(v) for v in depth_cache.missing_frames)[:50],
        "cam_t_visibility_depth_finite_ratio": finite_ratio(cam_t_final),
        "joints_cam_visibility_depth_finite_ratio": finite_ratio(joints_final),
        "vertices_cam_visibility_depth_finite_ratio": finite_ratio(verts_final),
        "hard_errors": hard_errors,
        "warnings": warnings,
        "ok": len(hard_errors) == 0,
    }
    summary_json.write_text(json.dumps(json_clean(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(json_clean(summary), indent=2, ensure_ascii=False))
    return 0 if len(hard_errors) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
