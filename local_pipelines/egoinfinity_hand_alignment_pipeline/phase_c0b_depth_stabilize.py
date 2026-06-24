#!/usr/bin/env python3
"""Phase-C0b EgoInfinity-style static-camera depth stabilization.

This node wraps EgoInfinity's background-template depth stabilization as an
independent LFV stage.  It reads the FoundationStereo frame CSV, builds dynamic
masks from Phase-B hand bboxes, estimates a static background depth template,
then writes corrected depth maps plus a new depth summary JSON.  Downstream
Phase-C nodes can consume that summary exactly like the original
FoundationStereo summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from egoinfinity_strict.depth_stabilize import (  # noqa: E402
    build_dynamic_mask,
    build_optical_flow_masks,
    compute_background_template,
    estimate_frame_correction,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session-dir", required=True)
    p.add_argument("--phase-b-npz", required=True)
    p.add_argument("--depth-summary-json", required=True)
    p.add_argument("--depth-frame-csv", default="")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--bbox-margin", type=float, default=0.30)
    p.add_argument("--template-min-valid-ratio", type=float, default=0.30)
    p.add_argument("--near-m", type=float, default=0.10)
    p.add_argument("--far-m", type=float, default=10.0)
    p.add_argument("--template-stride", type=int, default=1)
    p.add_argument("--use-flow-mask", action="store_true")
    p.add_argument("--flow-magnitude-threshold", type=float, default=2.0)
    p.add_argument("--flow-temporal-window", type=int, default=3)
    p.add_argument("--flow-dilate-px", type=int, default=7)
    p.add_argument("--write-dynamic-masks", action="store_true")
    p.add_argument("--write-template", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def csv_float(value: float) -> str:
    return f"{float(value):.9f}" if math.isfinite(float(value)) else ""


def finite_ratio(arr: np.ndarray) -> float:
    arr = np.asarray(arr)
    if arr.size == 0:
        return 0.0
    return float(np.mean(np.isfinite(arr)))


def read_depth_rows(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def write_depth_rows(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fields = [
        "frame_index", "disparity_npy", "depth_npy", "lr_mask_npy",
        "valid_depth_ratio", "depth_median_m", "depth_p95_m",
        "original_depth_npy", "depth_stabilize_scale", "depth_stabilize_offset",
        "dynamic_mask_ratio", "template_valid_ratio",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def write_correction_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fields = [
        "frame_index", "scale", "offset_m", "dynamic_mask_ratio",
        "valid_depth_ratio_before", "valid_depth_ratio_after",
        "depth_median_before_m", "depth_median_after_m",
        "depth_p95_before_m", "depth_p95_after_m",
        "template_background_rmse_before_m", "template_background_rmse_after_m",
        "qc_flag",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def depth_stats(depth: np.ndarray) -> Dict[str, float]:
    valid = np.asarray(depth)[np.isfinite(depth) & (depth > 0.01)]
    if valid.size == 0:
        return {
            "valid_depth_ratio": 0.0,
            "depth_median_m": float("nan"),
            "depth_p95_m": float("nan"),
        }
    return {
        "valid_depth_ratio": float(valid.size / depth.size),
        "depth_median_m": float(np.median(valid)),
        "depth_p95_m": float(np.percentile(valid, 95)),
    }


def background_rmse(depth: np.ndarray, template: np.ndarray, mask: np.ndarray, near: float, far: float) -> float:
    valid = (
        (~mask)
        & np.isfinite(depth)
        & np.isfinite(template)
        & (depth > near)
        & (depth < far)
        & (template > near)
        & (template < far)
    )
    if int(np.sum(valid)) < 100:
        return float("nan")
    residual = depth[valid] - template[valid]
    return float(np.sqrt(np.mean(residual * residual)))


def bboxes_by_frame(data: Dict[str, np.ndarray]) -> Dict[int, List[np.ndarray]]:
    out: Dict[int, List[np.ndarray]] = defaultdict(list)
    if "frame_index" not in data or "bbox_xyxy" not in data:
        return out
    frames = np.asarray(data["frame_index"], dtype=np.int32)
    bboxes = np.asarray(data["bbox_xyxy"], dtype=np.float32)
    for frame, bbox in zip(frames.tolist(), bboxes):
        if np.all(np.isfinite(bbox)):
            out[int(frame)].append(np.asarray(bbox, dtype=np.float32))
    return out


def read_video_grays(video_path: Path, frames: Sequence[int]) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video for flow masks: {video_path}")
    grays: List[np.ndarray] = []
    for frame in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
        ok, image = cap.read()
        if not ok or image is None:
            raise RuntimeError(f"failed to read video frame {frame}: {video_path}")
        grays.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    cap.release()
    return grays


def main() -> int:
    args = parse_args()
    session_dir = Path(args.session_dir).expanduser().resolve()
    phase_b_npz = Path(args.phase_b_npz).expanduser().resolve()
    depth_summary_json = Path(args.depth_summary_json).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not phase_b_npz.exists():
        raise RuntimeError(f"missing Phase-B NPZ: {phase_b_npz}")
    if not depth_summary_json.exists():
        raise RuntimeError(f"missing depth summary: {depth_summary_json}")

    depth_summary = load_json(depth_summary_json)
    depth_frame_csv = Path(args.depth_frame_csv).expanduser().resolve() if args.depth_frame_csv else Path(depth_summary["outputs"]["frame_csv"])
    depth_rows_in = read_depth_rows(depth_frame_csv)
    if not depth_rows_in:
        raise RuntimeError(f"empty depth frame CSV: {depth_frame_csv}")

    frames = [int(row["frame_index"]) for row in depth_rows_in]
    phase_b = load_npz(phase_b_npz)
    bboxes = bboxes_by_frame(phase_b)

    depth_maps: List[np.ndarray] = []
    original_depth_paths: List[Path] = []
    for row in depth_rows_in:
        path = Path(row["depth_npy"])
        if not path.exists():
            raise RuntimeError(f"missing depth npy: {path}")
        depth = np.load(path).astype(np.float32)
        depth_maps.append(depth)
        original_depth_paths.append(path)

    h, w = depth_maps[0].shape[:2]
    if any(d.shape[:2] != (h, w) for d in depth_maps):
        raise RuntimeError("depth maps have inconsistent shapes")

    dynamic_masks: List[np.ndarray] = []
    for frame in frames:
        dynamic_masks.append(build_dynamic_mask(h, w, bboxes.get(int(frame), []), float(args.bbox_margin)))

    flow_used = False
    if bool(args.use_flow_mask):
        left_video = Path(depth_summary.get("left_video") or (session_dir / "processed_topcam" / "left_table.mp4"))
        grays = read_video_grays(left_video, frames)
        flow_masks = build_optical_flow_masks(
            grays,
            magnitude_threshold=float(args.flow_magnitude_threshold),
            temporal_window=int(args.flow_temporal_window),
            dilate_px=int(args.flow_dilate_px),
        )
        dynamic_masks = [a | b for a, b in zip(dynamic_masks, flow_masks)]
        flow_used = True

    template_indices = list(range(0, len(depth_maps), max(1, int(args.template_stride))))
    template = compute_background_template(
        [depth_maps[i] for i in template_indices],
        [dynamic_masks[i] for i in template_indices],
        min_valid_ratio=float(args.template_min_valid_ratio),
    )
    template_valid_ratio = float(np.mean(np.isfinite(template) & (template > float(args.near_m))))

    depth_out_dir = output_dir / "depth_stabilized"
    mask_out_dir = output_dir / "dynamic_masks"
    depth_out_dir.mkdir(parents=True, exist_ok=True)
    if bool(args.write_dynamic_masks):
        mask_out_dir.mkdir(parents=True, exist_ok=True)

    out_rows: List[Dict[str, Any]] = []
    correction_rows: List[Dict[str, Any]] = []
    scales: List[float] = []
    offsets: List[float] = []
    rmse_before: List[float] = []
    rmse_after: List[float] = []
    dynamic_ratios: List[float] = []
    qc_counts: Counter[str] = Counter()

    for row, frame, depth, mask, orig_path in zip(depth_rows_in, frames, depth_maps, dynamic_masks, original_depth_paths):
        scale, offset = estimate_frame_correction(
            depth,
            template,
            mask,
            near=float(args.near_m),
            far=float(args.far_m),
        )
        corrected = np.maximum(depth * np.float32(scale) + np.float32(offset), 0.0).astype(np.float32)
        before = depth_stats(depth)
        after = depth_stats(corrected)
        dyn_ratio = float(np.mean(mask))
        bg_before = background_rmse(depth, template, mask, float(args.near_m), float(args.far_m))
        bg_after = background_rmse(corrected, template, mask, float(args.near_m), float(args.far_m))

        flag_parts: List[str] = []
        if abs(float(scale) - 1.0) > 0.10:
            flag_parts.append("warn_large_scale_correction")
        if abs(float(offset)) > 0.10:
            flag_parts.append("warn_large_offset_correction")
        if not math.isfinite(bg_after):
            flag_parts.append("warn_no_background_rmse")
        elif math.isfinite(bg_before) and bg_after > bg_before * 1.10:
            flag_parts.append("warn_background_rmse_worse")
        qc_flag = "|".join(flag_parts) if flag_parts else "ok"
        qc_counts[qc_flag] += 1

        out_path = depth_out_dir / f"depth_stabilized_{int(frame):08d}.npy"
        np.save(out_path, corrected)
        if bool(args.write_dynamic_masks):
            np.save(mask_out_dir / f"dynamic_mask_{int(frame):08d}.npy", mask.astype(np.bool_))

        scales.append(float(scale))
        offsets.append(float(offset))
        rmse_before.append(bg_before)
        rmse_after.append(bg_after)
        dynamic_ratios.append(dyn_ratio)

        out_rows.append(
            {
                "frame_index": int(frame),
                "disparity_npy": row.get("disparity_npy", ""),
                "depth_npy": str(out_path),
                "lr_mask_npy": row.get("lr_mask_npy", ""),
                "valid_depth_ratio": csv_float(after["valid_depth_ratio"]),
                "depth_median_m": csv_float(after["depth_median_m"]),
                "depth_p95_m": csv_float(after["depth_p95_m"]),
                "original_depth_npy": str(orig_path),
                "depth_stabilize_scale": csv_float(float(scale)),
                "depth_stabilize_offset": csv_float(float(offset)),
                "dynamic_mask_ratio": csv_float(dyn_ratio),
                "template_valid_ratio": csv_float(template_valid_ratio),
            }
        )
        correction_rows.append(
            {
                "frame_index": int(frame),
                "scale": csv_float(float(scale)),
                "offset_m": csv_float(float(offset)),
                "dynamic_mask_ratio": csv_float(dyn_ratio),
                "valid_depth_ratio_before": csv_float(before["valid_depth_ratio"]),
                "valid_depth_ratio_after": csv_float(after["valid_depth_ratio"]),
                "depth_median_before_m": csv_float(before["depth_median_m"]),
                "depth_median_after_m": csv_float(after["depth_median_m"]),
                "depth_p95_before_m": csv_float(before["depth_p95_m"]),
                "depth_p95_after_m": csv_float(after["depth_p95_m"]),
                "template_background_rmse_before_m": csv_float(bg_before),
                "template_background_rmse_after_m": csv_float(bg_after),
                "qc_flag": qc_flag,
            }
        )

    frame_csv_out = output_dir / "foundationstereo_depth_stabilized_frames.csv"
    correction_csv = output_dir / "depth_stabilize_corrections.csv"
    summary_json = output_dir / "foundationstereo_depth_stabilized_summary.json"
    template_path = output_dir / "background_depth_template.npy"
    write_depth_rows(frame_csv_out, out_rows)
    write_correction_csv(correction_csv, correction_rows)
    if bool(args.write_template):
        np.save(template_path, template.astype(np.float32))

    stabilized_summary = dict(depth_summary)
    stabilized_summary.update(
        {
            "semantic": "LFV FoundationStereo depth after EgoInfinity-style static-camera background stabilization",
            "input_depth_summary_json": str(depth_summary_json),
            "input_depth_frame_csv": str(depth_frame_csv),
            "phase_b_npz": str(phase_b_npz),
            "backend": f"{depth_summary.get('backend', 'foundationstereo')}+depth_stabilize",
            "outputs": dict(depth_summary.get("outputs") or {}),
            "depth_stabilize": {
                "bbox_margin": float(args.bbox_margin),
                "template_min_valid_ratio": float(args.template_min_valid_ratio),
                "template_stride": int(args.template_stride),
                "template_valid_ratio": template_valid_ratio,
                "use_flow_mask": bool(args.use_flow_mask),
                "flow_used": bool(flow_used),
                "correction_csv": str(correction_csv),
                "template_npy": str(template_path) if bool(args.write_template) else "",
                "dynamic_mask_dir": str(mask_out_dir) if bool(args.write_dynamic_masks) else "",
                "scale": stats(scales),
                "offset_m": stats(offsets),
                "dynamic_mask_ratio": stats(dynamic_ratios),
                "background_rmse_before_m": stats(rmse_before),
                "background_rmse_after_m": stats(rmse_after),
                "qc_flag_counts": dict(qc_counts),
            },
        }
    )
    stabilized_summary["outputs"]["frame_csv"] = str(frame_csv_out)
    stabilized_summary["outputs"]["depth_dir"] = str(depth_out_dir)
    stabilized_summary["outputs"]["depth_stabilize_correction_csv"] = str(correction_csv)
    stabilized_summary["valid_depth_ratio"] = stats(
        float(row["valid_depth_ratio"]) for row in out_rows if str(row["valid_depth_ratio"])
    )
    stabilized_summary["frames_exported"] = int(len(out_rows))

    hard_errors: List[str] = []
    warnings: List[str] = []
    if template_valid_ratio < 0.10:
        hard_errors.append(f"depth_stabilize_template_valid_ratio_too_low:{template_valid_ratio:.6f}")
    warn_count = sum(v for k, v in qc_counts.items() if "warn_" in str(k))
    if warn_count:
        warnings.append(f"depth_stabilize_warn_flags:{warn_count}")
    stabilized_summary["hard_errors"] = hard_errors
    stabilized_summary["warnings"] = warnings
    stabilized_summary["ok"] = len(hard_errors) == 0

    summary_json.write_text(json.dumps(json_clean(stabilized_summary), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(json_clean(stabilized_summary), indent=2, ensure_ascii=False))
    return 0 if not hard_errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
