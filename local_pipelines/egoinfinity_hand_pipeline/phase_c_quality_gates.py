#!/usr/bin/env python3
"""Node-level quality gates for the LFV EgoInfinity hand pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import cv2
import numpy as np


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def finite_ratio(arr: np.ndarray) -> float:
    arr = np.asarray(arr)
    if arr.size == 0:
        return 0.0
    return float(np.mean(np.isfinite(arr)))


def stats(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return {}
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


@dataclass
class NodeQC:
    node: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    hard_errors: List[str] = field(default_factory=list)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def error(self, msg: str) -> None:
        self.hard_errors.append(msg)

    @property
    def ok(self) -> bool:
        return not self.hard_errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node": self.node,
            "ok": self.ok,
            "metrics": self.metrics,
            "warnings": self.warnings,
            "hard_errors": self.hard_errors,
        }


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def video_info(path: Path) -> Dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"exists": path.exists(), "opened": False}
    out = {
        "exists": True,
        "opened": True,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
    }
    cap.release()
    return out


def check_topcam(session: Path) -> NodeQC:
    qc = NodeQC("A_processed_topcam_preflight")
    left = session / "processed_topcam" / "left_table.mp4"
    right = session / "processed_topcam" / "right_table.mp4"
    meta_path = session / "processed_topcam" / "processing_metadata.json"
    calib_path = session / "config" / "stereo_calibration_fisheye.json"
    table_path = Path("/home/yannan/workspace/ros1_docker-main/data/lfv_calibration/table_frame_latest.json")

    left_info = video_info(left)
    right_info = video_info(right)
    qc.metrics.update({
        "left_video": str(left),
        "right_video": str(right),
        "left": left_info,
        "right": right_info,
        "processing_metadata": str(meta_path),
        "calibration_json": str(calib_path),
        "table_frame_latest": str(table_path),
        "table_frame_latest_exists": table_path.exists(),
    })

    if not left_info.get("opened"):
        qc.error(f"left_table_missing_or_unreadable:{left}")
    if not right_info.get("opened"):
        qc.error(f"right_table_missing_or_unreadable:{right}")
    if qc.hard_errors:
        return qc

    if int(left_info["frame_count"]) != int(right_info["frame_count"]):
        qc.error(f"left_right_frame_count_mismatch:{left_info['frame_count']}!={right_info['frame_count']}")
    if int(left_info["width"]) != int(right_info["width"]) or int(left_info["height"]) != int(right_info["height"]):
        qc.error("left_right_image_size_mismatch")
    if not meta_path.exists():
        qc.error(f"processing_metadata_missing:{meta_path}")
    else:
        meta = read_json(meta_path)
        crop = meta.get("crop") or {}
        qc.metrics["crop"] = crop
        if int(crop.get("width", -1)) != int(left_info["width"]) or int(crop.get("height", -1)) != int(left_info["height"]):
            qc.error("metadata_crop_does_not_match_video_size")
    if not calib_path.exists():
        qc.error(f"calibration_missing:{calib_path}")
    if not table_path.exists():
        qc.warn(f"table_frame_latest_missing:{table_path}")
    return qc


def check_phase_b_npz(npz_path: Path) -> NodeQC:
    qc = NodeQC("C_phase_b_handresults")
    qc.metrics["npz"] = str(npz_path)
    if not npz_path.exists():
        qc.error(f"phase_b_npz_missing:{npz_path}")
        return qc
    data = np.load(npz_path, allow_pickle=True)
    n = int(len(data["frame_index"])) if "frame_index" in data else 0
    qc.metrics["candidates"] = n
    if n == 0:
        qc.error("phase_b_has_no_candidates")
        return qc

    required = [
        "track_id", "hand_label", "is_right", "bbox_xyxy", "det_conf",
        "global_orient", "hand_pose", "betas", "cam_t",
        "joints_3d_rel", "vertices_rel", "joints_uv",
    ]
    for key in required:
        if key not in data:
            qc.error(f"missing_npz_field:{key}")
    if qc.hard_errors:
        return qc

    for key in ["global_orient", "hand_pose", "betas", "cam_t", "joints_3d_rel", "vertices_rel", "joints_uv"]:
        ratio = finite_ratio(data[key])
        qc.metrics[f"{key}_finite_ratio"] = ratio
        if ratio < 1.0:
            qc.error(f"non_finite_{key}:ratio={ratio:.6f}")

    frame_index = data["frame_index"].astype(np.int32)
    track_id = data["track_id"].astype(np.int32)
    labels = np.asarray(data["hand_label"]).astype(str)
    qc.metrics["frame_min"] = int(np.min(frame_index))
    qc.metrics["frame_max"] = int(np.max(frame_index))
    qc.metrics["track_count"] = int(len(set(track_id.tolist())))
    qc.metrics["track_count_by_label"] = {
        label: int(len(set(track_id[labels == label].tolist())))
        for label in sorted(set(labels.tolist()))
    }
    qc.metrics["candidate_count_by_label"] = {
        label: int(np.sum(labels == label))
        for label in sorted(set(labels.tolist()))
    }

    # Same-label same-frame overlap is a hard Phase-B failure if still high.
    hard_overlap_count = 0
    by_frame: Dict[int, List[int]] = {}
    for i, f in enumerate(frame_index.tolist()):
        by_frame.setdefault(int(f), []).append(i)
    bbox = data["bbox_xyxy"].astype(np.float32)
    for indices in by_frame.values():
        for a_pos, i in enumerate(indices):
            for j in indices[a_pos + 1:]:
                if labels[i] != labels[j]:
                    continue
                iou = bbox_iou(bbox[i], bbox[j])
                if iou > 0.30:
                    hard_overlap_count += 1
    qc.metrics["remaining_same_label_overlap_iou_gt_0p30"] = int(hard_overlap_count)
    if hard_overlap_count:
        qc.error(f"remaining_same_label_duplicate_overlap:{hard_overlap_count}")
    return qc


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    ix2, iy2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return float(inter / union) if union > 1e-9 else 0.0


def check_foundation_depth(
    depth_summary_path: Path,
    depth_frames_csv: Optional[Path] = None,
    total_frames: Optional[int] = None,
    require_full_coverage: bool = True,
) -> NodeQC:
    qc = NodeQC("D_foundationstereo_depth")
    qc.metrics["summary_json"] = str(depth_summary_path)
    if not depth_summary_path.exists():
        qc.error(f"foundation_depth_summary_missing:{depth_summary_path}")
        return qc
    summary = read_json(depth_summary_path)
    outputs = summary.get("outputs") or {}
    if depth_frames_csv is None:
        depth_frames_csv = Path(outputs.get("frame_csv", ""))
    qc.metrics["frame_csv"] = str(depth_frames_csv)
    qc.metrics["camera"] = summary.get("camera") or {}
    qc.metrics["frame_start"] = int(summary.get("frame_start", -1))
    qc.metrics["frame_end"] = int(summary.get("frame_end", -1))
    qc.metrics["stride"] = int(summary.get("stride", 0))
    qc.metrics["frames_exported"] = int(summary.get("frames_exported", 0))
    qc.metrics["valid_depth_ratio_summary"] = summary.get("valid_depth_ratio") or {}
    if qc.metrics["frames_exported"] <= 0:
        qc.error("foundation_depth_has_no_frames")
    if not depth_frames_csv.exists():
        qc.error(f"foundation_depth_frame_csv_missing:{depth_frames_csv}")
        return qc

    ratios: List[float] = []
    med_depths: List[float] = []
    frame_indices: List[int] = []
    missing_depth_files = 0
    with depth_frames_csv.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            frame_indices.append(int(row["frame_index"]))
            ratios.append(safe_float(row.get("valid_depth_ratio")))
            med_depths.append(safe_float(row.get("depth_median_m")))
            depth_file = Path(row.get("depth_npy", ""))
            if not depth_file.exists():
                missing_depth_files += 1
    unique_frames = sorted(set(frame_indices))
    qc.metrics["coverage"] = {
        "required_full": bool(require_full_coverage),
        "video_total_frames": int(total_frames or 0),
        "frame_count": int(len(unique_frames)),
        "frame_min": int(unique_frames[0]) if unique_frames else None,
        "frame_max": int(unique_frames[-1]) if unique_frames else None,
        "coverage_ratio": float(len(unique_frames) / total_frames) if total_frames else None,
    }
    qc.metrics["valid_depth_ratio"] = stats(ratios)
    qc.metrics["median_depth_m"] = stats(med_depths)
    qc.metrics["missing_depth_npy_count"] = int(missing_depth_files)
    if missing_depth_files:
        qc.error(f"missing_depth_npy_count:{missing_depth_files}")
    median_valid = qc.metrics["valid_depth_ratio"].get("median", 0.0)
    if median_valid < 0.05:
        qc.error(f"valid_depth_ratio_too_low:{median_valid:.6f}")
    elif median_valid < 0.15:
        qc.warn(f"valid_depth_ratio_low:{median_valid:.6f}")
    if require_full_coverage and total_frames and total_frames > 0:
        if qc.metrics["stride"] != 1:
            qc.error(f"foundation_depth_stride_not_full_rate:{qc.metrics['stride']}")
        if len(unique_frames) != total_frames or not unique_frames or unique_frames[0] != 0 or unique_frames[-1] != total_frames - 1:
            qc.error(
                "foundation_depth_partial_coverage:"
                f"{qc.metrics['coverage']['frame_min']}..{qc.metrics['coverage']['frame_max']}"
                f"/{total_frames} count={len(unique_frames)}"
            )
    return qc


def check_phase_c0b_depth_stabilize(
    summary_path: Path,
    frame_csv: Path,
    correction_csv: Path,
    total_frames: Optional[int] = None,
    require_full_coverage: bool = True,
) -> NodeQC:
    qc = NodeQC("D1_phase_c0b_depth_stabilize")
    qc.metrics.update({
        "summary_json": str(summary_path),
        "frame_csv": str(frame_csv),
        "correction_csv": str(correction_csv),
    })
    if not summary_path.exists():
        qc.warn(f"phase_c0b_depth_stabilize_not_run_yet:{summary_path}")
        return qc

    summary = read_json(summary_path)
    outputs = summary.get("outputs") or {}
    if not str(frame_csv):
        frame_csv = Path(outputs.get("frame_csv", ""))
    if not str(correction_csv):
        correction_csv = Path(outputs.get("depth_stabilize_correction_csv", ""))
    details = summary.get("depth_stabilize") or {}
    qc.metrics["summary_ok"] = bool(summary.get("ok", False))
    qc.metrics["frames_exported"] = int(summary.get("frames_exported", 0))
    qc.metrics["template_valid_ratio"] = safe_float(details.get("template_valid_ratio"), 0.0)
    qc.metrics["scale"] = details.get("scale") or {}
    qc.metrics["offset_m"] = details.get("offset_m") or {}
    qc.metrics["background_rmse_before_m"] = details.get("background_rmse_before_m") or {}
    qc.metrics["background_rmse_after_m"] = details.get("background_rmse_after_m") or {}
    qc.metrics["qc_flag_counts"] = details.get("qc_flag_counts") or {}

    for err in summary.get("hard_errors") or []:
        qc.error(f"phase_c0b_summary_error:{err}")
    for warn in summary.get("warnings") or []:
        qc.warn(f"phase_c0b_summary_warning:{warn}")
    if not summary.get("ok", False):
        qc.error("phase_c0b_summary_not_ok")
    if qc.metrics["template_valid_ratio"] < 0.10:
        qc.error(f"phase_c0b_template_valid_ratio_too_low:{qc.metrics['template_valid_ratio']:.6f}")

    if not frame_csv.exists():
        qc.error(f"phase_c0b_frame_csv_missing:{frame_csv}")
        return qc
    if not correction_csv.exists():
        qc.error(f"phase_c0b_correction_csv_missing:{correction_csv}")
        return qc

    frame_indices: List[int] = []
    ratios: List[float] = []
    missing_depth = 0
    with frame_csv.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            frame_indices.append(int(row["frame_index"]))
            ratios.append(safe_float(row.get("valid_depth_ratio")))
            if not Path(row.get("depth_npy", "")).exists():
                missing_depth += 1
    correction_rows = 0
    with correction_csv.open("r", newline="", encoding="utf-8") as f:
        for _ in csv.DictReader(f):
            correction_rows += 1
    unique_frames = sorted(set(frame_indices))
    qc.metrics["frame_count"] = int(len(unique_frames))
    qc.metrics["frame_min"] = int(unique_frames[0]) if unique_frames else None
    qc.metrics["frame_max"] = int(unique_frames[-1]) if unique_frames else None
    qc.metrics["valid_depth_ratio"] = stats(ratios)
    qc.metrics["missing_stabilized_depth_npy_count"] = int(missing_depth)
    qc.metrics["correction_csv_rows"] = int(correction_rows)
    if missing_depth:
        qc.error(f"phase_c0b_missing_stabilized_depth_npy:{missing_depth}")
    if correction_rows != len(frame_indices):
        qc.error(f"phase_c0b_correction_csv_row_mismatch:{correction_rows}!={len(frame_indices)}")
    if require_full_coverage and total_frames and total_frames > 0:
        if len(unique_frames) != total_frames or not unique_frames or unique_frames[0] != 0 or unique_frames[-1] != total_frames - 1:
            qc.error(
                "phase_c0b_partial_depth_coverage:"
                f"{qc.metrics['frame_min']}..{qc.metrics['frame_max']}/{total_frames} count={len(unique_frames)}"
            )
    warn_count = sum(int(v) for k, v in qc.metrics["qc_flag_counts"].items() if "warn_" in str(k))
    qc.metrics["warn_depth_stabilize_flag_count"] = int(warn_count)
    if warn_count:
        qc.warn(f"phase_c0b_warn_depth_stabilize_flags:{warn_count}")
    return qc


def check_phase_c_alignment(
    npz_path: Path,
    summary_path: Path,
    quality_csv: Path,
    expected_candidates: Optional[int] = None,
    require_full_coverage: bool = True,
) -> NodeQC:
    qc = NodeQC("E_phase_c_depth_alignment")
    qc.metrics.update({
        "npz": str(npz_path),
        "summary_json": str(summary_path),
        "quality_csv": str(quality_csv),
    })
    if not summary_path.exists():
        qc.warn(f"phase_c_alignment_not_run_yet:{summary_path}")
        return qc

    summary = read_json(summary_path)
    qc.metrics["summary_ok"] = bool(summary.get("ok", False))
    qc.metrics["candidates_selected"] = int(summary.get("candidates_selected", 0))
    qc.metrics["frames_selected"] = int(summary.get("frames_selected", 0))
    qc.metrics["alignment_source_counts"] = summary.get("alignment_source_counts") or {}
    qc.metrics["qc_flag_counts"] = summary.get("qc_flag_counts") or {}
    qc.metrics["diagnosis_category_counts"] = summary.get("diagnosis_category_counts") or {}
    qc.metrics["qc_issue_tag_counts"] = summary.get("qc_issue_tag_counts") or {}
    qc.metrics["fallback_ratio"] = safe_float(summary.get("fallback_ratio"), 0.0)
    qc.metrics["frame_start"] = int(summary.get("frame_start", -1))
    qc.metrics["frame_end"] = int(summary.get("frame_end", -1))
    qc.metrics["valid_joint_count"] = summary.get("valid_joint_count") or {}
    qc.metrics["alignment_rms_m"] = summary.get("alignment_rms_m") or {}
    qc.metrics["cam_t_jump_m"] = summary.get("cam_t_jump_m") or {}

    for err in summary.get("hard_errors") or []:
        qc.error(f"phase_c_summary_error:{err}")
    for warn in summary.get("warnings") or []:
        qc.warn(f"phase_c_summary_warning:{warn}")
    if not summary.get("ok", False):
        qc.error("phase_c_summary_not_ok")
    if qc.metrics["candidates_selected"] <= 0:
        qc.error("phase_c_has_no_selected_candidates")

    if not npz_path.exists():
        qc.error(f"phase_c_npz_missing:{npz_path}")
        return qc
    if not quality_csv.exists():
        qc.error(f"phase_c_quality_csv_missing:{quality_csv}")
        return qc

    data = np.load(npz_path, allow_pickle=True)
    required = [
        "frame_index", "track_id", "hand_label", "bbox_xyxy", "cam_t",
        "cam_t_wilor", "cam_t_depth", "joints_cam_depth", "vertices_cam_depth",
        "alignment_source", "alignment_valid_joint_count",
        "alignment_joint_ids", "alignment_sampled_depths_m",
        "alignment_rms_m", "alignment_max_residual_m",
        "diagnosis_category", "qc_issue_tags",
        "bbox_visible_ratio", "joint_visible_ratio", "bbox_touches_edge",
    ]
    for key in required:
        if key not in data:
            qc.error(f"missing_phase_c_npz_field:{key}")
    if qc.hard_errors:
        return qc

    n = int(len(data["frame_index"]))
    qc.metrics["npz_candidates"] = n
    frame_index = data["frame_index"].astype(np.int32)
    frame_min = int(np.min(frame_index)) if n else None
    frame_max = int(np.max(frame_index)) if n else None
    qc.metrics["coverage"] = {
        "required_full": bool(require_full_coverage),
        "expected_candidates": int(expected_candidates or 0),
        "candidate_count": int(n),
        "frame_min": frame_min,
        "frame_max": frame_max,
        "candidate_coverage_ratio": float(n / expected_candidates) if expected_candidates else None,
    }
    for key in ["cam_t_depth", "joints_cam_depth", "vertices_cam_depth"]:
        ratio = finite_ratio(data[key])
        qc.metrics[f"{key}_finite_ratio"] = ratio
        if ratio < 1.0:
            qc.error(f"non_finite_{key}:ratio={ratio:.6f}")

    alignment_source = np.asarray(data["alignment_source"]).astype(str)
    qc.metrics["npz_alignment_source_counts"] = {
        source: int(np.sum(alignment_source == source))
        for source in sorted(set(alignment_source.tolist()))
    }
    valid_counts = np.asarray(data["alignment_valid_joint_count"], dtype=np.float64)
    rms = np.asarray(data["alignment_rms_m"], dtype=np.float64)
    selected_mask = alignment_source != "not_selected"
    qc.metrics["npz_selected_candidates"] = int(np.sum(selected_mask))
    qc.metrics["npz_valid_joint_count"] = stats(valid_counts[selected_mask])
    qc.metrics["npz_alignment_rms_m"] = stats(rms[selected_mask])
    if qc.metrics["npz_selected_candidates"] != qc.metrics["candidates_selected"]:
        qc.error(
            "phase_c_selected_count_mismatch:"
            f"{qc.metrics['npz_selected_candidates']}!={qc.metrics['candidates_selected']}"
        )
    if require_full_coverage and expected_candidates and expected_candidates > 0:
        if n != expected_candidates or qc.metrics["candidates_selected"] != expected_candidates:
            qc.error(
                "phase_c_partial_candidate_coverage:"
                f"npz={n} selected={qc.metrics['candidates_selected']} expected={expected_candidates}"
            )

    csv_rows = 0
    with quality_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for _ in reader:
            csv_rows += 1
    qc.metrics["quality_csv_rows"] = int(csv_rows)
    if csv_rows != qc.metrics["candidates_selected"]:
        qc.error(f"phase_c_quality_csv_row_mismatch:{csv_rows}!={qc.metrics['candidates_selected']}")

    flag_counts = qc.metrics["qc_flag_counts"]
    bad_count = sum(int(v) for k, v in flag_counts.items() if "bad_" in str(k))
    warn_count = sum(int(v) for k, v in flag_counts.items() if "warn_" in str(k))
    qc.metrics["bad_alignment_flag_count"] = int(bad_count)
    qc.metrics["warn_alignment_flag_count"] = int(warn_count)
    if bad_count:
        qc.warn(f"phase_c_bad_alignment_flags:{bad_count}")
    if qc.metrics["fallback_ratio"] > 0.50:
        qc.warn(f"phase_c_high_fallback_ratio:{qc.metrics['fallback_ratio']:.3f}")
    return qc


def check_phase_c2_mano_smoothing(
    npz_path: Path,
    summary_path: Path,
    quality_csv: Path,
    expected_candidates: Optional[int] = None,
) -> NodeQC:
    qc = NodeQC("F_phase_c2_mano_temporal_smoothing")
    qc.metrics.update({
        "npz": str(npz_path),
        "summary_json": str(summary_path),
        "quality_csv": str(quality_csv),
    })
    if not summary_path.exists():
        qc.warn(f"phase_c2_mano_smoothing_not_run_yet:{summary_path}")
        return qc

    summary = read_json(summary_path)
    qc.metrics["summary_ok"] = bool(summary.get("ok", False))
    qc.metrics["candidates"] = int(summary.get("candidates", 0))
    qc.metrics["tracks"] = int(summary.get("tracks", 0))
    qc.metrics["status_counts"] = summary.get("status_counts") or {}
    qc.metrics["qc_flag_counts"] = summary.get("qc_flag_counts") or {}
    qc.metrics["event_counts"] = summary.get("event_counts") or {}
    qc.metrics["input_to_smooth_joint_rms_m"] = summary.get("input_to_smooth_joint_rms_m") or {}
    qc.metrics["smooth_wrist_jump_m"] = summary.get("smooth_wrist_jump_m") or {}
    qc.metrics["smooth_joints_finite_ratio"] = safe_float(summary.get("smooth_joints_finite_ratio"), 0.0)
    qc.metrics["smooth_vertices_finite_ratio"] = safe_float(summary.get("smooth_vertices_finite_ratio"), 0.0)
    qc.metrics["biomech_clamped_candidates"] = int(summary.get("biomech_clamped_candidates", 0))

    for err in summary.get("hard_errors") or []:
        qc.error(f"phase_c2_summary_error:{err}")
    for warn in summary.get("warnings") or []:
        qc.warn(f"phase_c2_summary_warning:{warn}")
    if not summary.get("ok", False):
        qc.error("phase_c2_summary_not_ok")
    if expected_candidates and expected_candidates > 0 and qc.metrics["candidates"] != expected_candidates:
        qc.error(f"phase_c2_candidate_count_mismatch:{qc.metrics['candidates']}!={expected_candidates}")

    if not npz_path.exists():
        qc.error(f"phase_c2_npz_missing:{npz_path}")
        return qc
    if not quality_csv.exists():
        qc.error(f"phase_c2_quality_csv_missing:{quality_csv}")
        return qc

    data = np.load(npz_path, allow_pickle=True)
    required = [
        "frame_index", "track_id", "hand_label", "cam_t_smooth",
        "joints_cam_smooth", "vertices_cam_smooth",
        "joints_3d_rel_smooth", "vertices_rel_smooth",
        "global_orient_smooth", "hand_pose_smooth", "betas_smooth",
        "joints_uv_smooth_depth_camera",
        "mano_smoothing_status", "mano_smoothing_qc_flag",
    ]
    for key in required:
        if key not in data:
            qc.error(f"missing_phase_c2_npz_field:{key}")
    if qc.hard_errors:
        return qc

    n = int(len(data["frame_index"]))
    qc.metrics["npz_candidates"] = n
    if expected_candidates and expected_candidates > 0 and n != expected_candidates:
        qc.error(f"phase_c2_npz_candidate_count_mismatch:{n}!={expected_candidates}")
    for key in ["cam_t_smooth", "joints_cam_smooth", "vertices_cam_smooth", "global_orient_smooth", "hand_pose_smooth"]:
        ratio = finite_ratio(data[key])
        qc.metrics[f"{key}_finite_ratio"] = ratio
        if ratio < 1.0:
            qc.error(f"non_finite_{key}:ratio={ratio:.6f}")

    status = np.asarray(data["mano_smoothing_status"]).astype(str)
    flags = np.asarray(data["mano_smoothing_qc_flag"]).astype(str)
    qc.metrics["npz_status_counts"] = {
        value: int(np.sum(status == value))
        for value in sorted(set(status.tolist()))
    }
    qc.metrics["npz_qc_flag_counts"] = {
        value: int(np.sum(flags == value))
        for value in sorted(set(flags.tolist()))
    }
    csv_rows = 0
    with quality_csv.open("r", newline="", encoding="utf-8") as f:
        for _ in csv.DictReader(f):
            csv_rows += 1
    qc.metrics["quality_csv_rows"] = int(csv_rows)
    if csv_rows != n:
        qc.error(f"phase_c2_quality_csv_row_mismatch:{csv_rows}!={n}")

    bad_count = sum(int(v) for k, v in qc.metrics["qc_flag_counts"].items() if "bad_" in str(k))
    warn_count = sum(int(v) for k, v in qc.metrics["qc_flag_counts"].items() if "warn_" in str(k))
    qc.metrics["bad_smoothing_flag_count"] = int(bad_count)
    qc.metrics["warn_smoothing_flag_count"] = int(warn_count)
    if bad_count:
        qc.warn(f"phase_c2_bad_smoothing_flags:{bad_count}")
    if warn_count:
        qc.warn(f"phase_c2_warn_smoothing_flags:{warn_count}")
    return qc


def check_phase_c1b_depth_smooth(
    npz_path: Path,
    summary_path: Path,
    quality_csv: Path,
    track_csv: Path,
    expected_candidates: Optional[int] = None,
) -> NodeQC:
    qc = NodeQC("F0_phase_c1b_depth_smooth")
    qc.metrics.update({
        "npz": str(npz_path),
        "summary_json": str(summary_path),
        "quality_csv": str(quality_csv),
        "track_csv": str(track_csv),
    })
    if not summary_path.exists():
        qc.warn(f"phase_c1b_depth_smooth_not_run_yet:{summary_path}")
        return qc

    summary = read_json(summary_path)
    qc.metrics["summary_ok"] = bool(summary.get("ok", False))
    qc.metrics["candidates"] = int(summary.get("candidates", 0))
    qc.metrics["tracks"] = int(summary.get("tracks", 0))
    qc.metrics["status_counts"] = summary.get("status_counts") or {}
    qc.metrics["qc_flag_counts"] = summary.get("qc_flag_counts") or {}
    qc.metrics["delta_z_m"] = summary.get("delta_z_m") or {}
    qc.metrics["before_wrist_z_jump_m"] = summary.get("before_wrist_z_jump_m") or {}
    qc.metrics["after_wrist_z_jump_m"] = summary.get("after_wrist_z_jump_m") or {}
    qc.metrics["cam_t_depth_finite_ratio"] = safe_float(summary.get("cam_t_depth_finite_ratio"), 0.0)
    qc.metrics["joints_cam_depth_finite_ratio"] = safe_float(summary.get("joints_cam_depth_finite_ratio"), 0.0)
    qc.metrics["vertices_cam_depth_finite_ratio"] = safe_float(summary.get("vertices_cam_depth_finite_ratio"), 0.0)

    for err in summary.get("hard_errors") or []:
        qc.error(f"phase_c1b_summary_error:{err}")
    for warn in summary.get("warnings") or []:
        qc.warn(f"phase_c1b_summary_warning:{warn}")
    if not summary.get("ok", False):
        qc.error("phase_c1b_summary_not_ok")
    if expected_candidates and expected_candidates > 0 and qc.metrics["candidates"] != expected_candidates:
        qc.error(f"phase_c1b_candidate_count_mismatch:{qc.metrics['candidates']}!={expected_candidates}")

    if not npz_path.exists():
        qc.error(f"phase_c1b_npz_missing:{npz_path}")
        return qc
    if not quality_csv.exists():
        qc.error(f"phase_c1b_quality_csv_missing:{quality_csv}")
        return qc
    if not track_csv.exists():
        qc.error(f"phase_c1b_track_csv_missing:{track_csv}")
        return qc

    data = np.load(npz_path, allow_pickle=True)
    required = [
        "frame_index", "track_id", "cam_t_depth_before_depth_smooth",
        "cam_t_depth", "joints_cam_depth", "vertices_cam_depth",
        "cam_t_depth_smooth", "joints_cam_depth_smooth", "vertices_cam_depth_smooth",
        "depth_smooth_delta_z_m", "depth_smooth_trust",
        "depth_smooth_valid_sample_count", "depth_smooth_status",
        "depth_smooth_qc_flag",
    ]
    for key in required:
        if key not in data:
            qc.error(f"missing_phase_c1b_npz_field:{key}")
    if qc.hard_errors:
        return qc

    n = int(len(data["frame_index"]))
    qc.metrics["npz_candidates"] = n
    if expected_candidates and expected_candidates > 0 and n != expected_candidates:
        qc.error(f"phase_c1b_npz_candidate_count_mismatch:{n}!={expected_candidates}")
    for key in ["cam_t_depth", "joints_cam_depth", "vertices_cam_depth"]:
        ratio = finite_ratio(data[key])
        qc.metrics[f"{key}_finite_ratio"] = ratio
        if ratio < 1.0:
            qc.error(f"non_finite_{key}:ratio={ratio:.6f}")
    flags = np.asarray(data["depth_smooth_qc_flag"]).astype(str)
    qc.metrics["npz_qc_flag_counts"] = {
        value: int(np.sum(flags == value))
        for value in sorted(set(flags.tolist()))
    }
    csv_rows = 0
    with quality_csv.open("r", newline="", encoding="utf-8") as f:
        for _ in csv.DictReader(f):
            csv_rows += 1
    qc.metrics["quality_csv_rows"] = int(csv_rows)
    if csv_rows != n:
        qc.error(f"phase_c1b_quality_csv_row_mismatch:{csv_rows}!={n}")
    bad_count = sum(int(v) for k, v in qc.metrics["qc_flag_counts"].items() if "bad_" in str(k))
    warn_count = sum(int(v) for k, v in qc.metrics["qc_flag_counts"].items() if "warn_" in str(k))
    qc.metrics["bad_depth_smooth_flag_count"] = int(bad_count)
    qc.metrics["warn_depth_smooth_flag_count"] = int(warn_count)
    if bad_count:
        qc.warn(f"phase_c1b_bad_depth_smooth_flags:{bad_count}")
    if warn_count:
        qc.warn(f"phase_c1b_warn_depth_smooth_flags:{warn_count}")
    return qc


def check_phase_c1c_motion_infiller(
    npz_path: Path,
    summary_path: Path,
    quality_csv: Path,
    expected_input_candidates: Optional[int] = None,
) -> NodeQC:
    qc = NodeQC("F1_phase_c1c_motion_infiller")
    qc.metrics.update({
        "npz": str(npz_path),
        "summary_json": str(summary_path),
        "quality_csv": str(quality_csv),
    })
    if not summary_path.exists():
        qc.warn(f"phase_c1c_motion_infiller_not_run_yet:{summary_path}")
        return qc

    summary = read_json(summary_path)
    qc.metrics["summary_ok"] = bool(summary.get("ok", False))
    qc.metrics["candidates_in"] = int(summary.get("candidates_in", 0))
    qc.metrics["candidates_out"] = int(summary.get("candidates_out", 0))
    qc.metrics["motion_infilled_candidates"] = int(summary.get("motion_infilled_candidates", 0))
    qc.metrics["dual_valid_ratio"] = safe_float(summary.get("dual_valid_ratio"), 0.0)
    qc.metrics["long_gap_method"] = summary.get("long_gap_method", "")
    qc.metrics["method_counts"] = summary.get("method_counts") or {}
    qc.metrics["qc_flag_counts"] = summary.get("qc_flag_counts") or {}
    qc.metrics["cam_t_jump_m"] = summary.get("cam_t_jump_m") or {}
    qc.metrics["wrist_jump_m"] = summary.get("wrist_jump_m") or {}

    for err in summary.get("hard_errors") or []:
        qc.error(f"phase_c1c_summary_error:{err}")
    for warn in summary.get("warnings") or []:
        qc.warn(f"phase_c1c_summary_warning:{warn}")
    if not summary.get("ok", False):
        qc.error("phase_c1c_summary_not_ok")
    if expected_input_candidates and expected_input_candidates > 0 and qc.metrics["candidates_in"] != expected_input_candidates:
        qc.error(f"phase_c1c_input_candidate_count_mismatch:{qc.metrics['candidates_in']}!={expected_input_candidates}")
    if qc.metrics["candidates_out"] < qc.metrics["candidates_in"]:
        qc.error(f"phase_c1c_candidate_count_decreased:{qc.metrics['candidates_out']}<{qc.metrics['candidates_in']}")

    if not npz_path.exists():
        qc.error(f"phase_c1c_npz_missing:{npz_path}")
        return qc
    if not quality_csv.exists():
        qc.error(f"phase_c1c_quality_csv_missing:{quality_csv}")
        return qc

    data = np.load(npz_path, allow_pickle=True)
    required = [
        "frame_index", "candidate_index", "track_id", "hand_label",
        "motion_infilled", "motion_infiller_method",
        "motion_infiller_gap_len", "motion_infiller_qc_flag",
        "cam_t_depth", "joints_cam_depth", "vertices_cam_depth",
    ]
    for key in required:
        if key not in data:
            qc.error(f"missing_phase_c1c_npz_field:{key}")
    if qc.hard_errors:
        return qc

    n = int(len(data["frame_index"]))
    qc.metrics["npz_candidates"] = n
    if n != qc.metrics["candidates_out"]:
        qc.error(f"phase_c1c_npz_candidate_count_mismatch:{n}!={qc.metrics['candidates_out']}")
    for key in ["cam_t_depth", "joints_cam_depth", "vertices_cam_depth"]:
        ratio = finite_ratio(data[key])
        qc.metrics[f"{key}_finite_ratio"] = ratio
        if ratio < 1.0:
            qc.error(f"non_finite_{key}:ratio={ratio:.6f}")

    motion_infilled = np.asarray(data["motion_infilled"], dtype=np.int32)
    flags = np.asarray(data["motion_infiller_qc_flag"]).astype(str)
    methods = np.asarray(data["motion_infiller_method"]).astype(str)
    qc.metrics["npz_motion_infilled_count"] = int(np.sum(motion_infilled > 0))
    qc.metrics["npz_method_counts"] = {
        value: int(np.sum(methods == value))
        for value in sorted(set(methods.tolist()))
    }
    qc.metrics["npz_qc_flag_counts"] = {
        value: int(np.sum(flags == value))
        for value in sorted(set(flags.tolist()))
    }
    if qc.metrics["npz_motion_infilled_count"] != qc.metrics["motion_infilled_candidates"]:
        qc.error(
            "phase_c1c_infilled_count_mismatch:"
            f"{qc.metrics['npz_motion_infilled_count']}!={qc.metrics['motion_infilled_candidates']}"
        )

    csv_rows = 0
    with quality_csv.open("r", newline="", encoding="utf-8") as f:
        for _ in csv.DictReader(f):
            csv_rows += 1
    qc.metrics["quality_csv_rows"] = int(csv_rows)
    if csv_rows != n:
        qc.error(f"phase_c1c_quality_csv_row_mismatch:{csv_rows}!={n}")

    bad_count = sum(int(v) for k, v in qc.metrics["qc_flag_counts"].items() if "bad_" in str(k))
    warn_count = sum(int(v) for k, v in qc.metrics["qc_flag_counts"].items() if "warn_" in str(k))
    qc.metrics["bad_motion_infiller_flag_count"] = int(bad_count)
    qc.metrics["warn_motion_infiller_flag_count"] = int(warn_count)
    if bad_count:
        qc.warn(f"phase_c1c_bad_motion_infiller_flags:{bad_count}")
    if warn_count:
        qc.warn(f"phase_c1c_warn_motion_infiller_flags:{warn_count}")
    return qc


def check_phase_c3_mesh_visibility(
    npz_path: Path,
    summary_path: Path,
    joint_csv: Path,
    candidate_csv: Path,
    expected_candidates: Optional[int] = None,
) -> NodeQC:
    qc = NodeQC("G_phase_c3_mano_mesh_visibility")
    qc.metrics.update({
        "npz": str(npz_path),
        "summary_json": str(summary_path),
        "joint_csv": str(joint_csv),
        "candidate_csv": str(candidate_csv),
    })
    if not summary_path.exists():
        qc.warn(f"phase_c3_mesh_visibility_not_run_yet:{summary_path}")
        return qc

    summary = read_json(summary_path)
    qc.metrics["summary_ok"] = bool(summary.get("ok", False))
    qc.metrics["candidates"] = int(summary.get("candidates", 0))
    qc.metrics["qc_flag_counts"] = summary.get("qc_flag_counts") or {}
    qc.metrics["visible_vertex_ratio"] = summary.get("visible_vertex_ratio") or {}
    qc.metrics["visible_joint_count"] = summary.get("visible_joint_count") or {}
    qc.metrics["visible_reliable_joint_count"] = summary.get("visible_reliable_joint_count") or {}
    qc.metrics["visible_mcp_count"] = summary.get("visible_mcp_count") or {}
    qc.metrics["zbuffer_pixel_count"] = summary.get("zbuffer_pixel_count") or {}
    qc.metrics["joint_mesh_visible_ratio"] = summary.get("joint_mesh_visible_ratio") or {}

    for err in summary.get("hard_errors") or []:
        qc.error(f"phase_c3_summary_error:{err}")
    for warn in summary.get("warnings") or []:
        qc.warn(f"phase_c3_summary_warning:{warn}")
    if not summary.get("ok", False):
        qc.error("phase_c3_summary_not_ok")
    if expected_candidates and expected_candidates > 0 and qc.metrics["candidates"] != expected_candidates:
        qc.error(f"phase_c3_candidate_count_mismatch:{qc.metrics['candidates']}!={expected_candidates}")

    if not npz_path.exists():
        qc.error(f"phase_c3_npz_missing:{npz_path}")
        return qc
    if not joint_csv.exists():
        qc.error(f"phase_c3_joint_csv_missing:{joint_csv}")
        return qc
    if not candidate_csv.exists():
        qc.error(f"phase_c3_candidate_csv_missing:{candidate_csv}")
        return qc

    data = np.load(npz_path, allow_pickle=True)
    required = [
        "frame_index", "track_id", "hand_label",
        "mano_vertex_visible", "mano_joint_visible",
        "mano_joint_mesh_visible_ratio", "mano_joint_mesh_surface_margin_m",
        "mano_visible_vertex_ratio", "mano_visible_joint_count",
        "mano_visible_reliable_joint_count", "mano_visibility_qc_flag",
    ]
    for key in required:
        if key not in data:
            qc.error(f"missing_phase_c3_npz_field:{key}")
    if qc.hard_errors:
        return qc

    n = int(len(data["frame_index"]))
    qc.metrics["npz_candidates"] = n
    if expected_candidates and expected_candidates > 0 and n != expected_candidates:
        qc.error(f"phase_c3_npz_candidate_count_mismatch:{n}!={expected_candidates}")
    for key in ["mano_visible_vertex_ratio", "mano_joint_mesh_visible_ratio", "mano_joint_mesh_surface_margin_m"]:
        ratio = finite_ratio(data[key])
        qc.metrics[f"{key}_finite_ratio"] = ratio
        if ratio < 0.50:
            qc.warn(f"low_finite_{key}:ratio={ratio:.6f}")

    flags = np.asarray(data["mano_visibility_qc_flag"]).astype(str)
    qc.metrics["npz_qc_flag_counts"] = {
        value: int(np.sum(flags == value))
        for value in sorted(set(flags.tolist()))
    }
    joint_rows = 0
    with joint_csv.open("r", newline="", encoding="utf-8") as f:
        for _ in csv.DictReader(f):
            joint_rows += 1
    candidate_rows = 0
    with candidate_csv.open("r", newline="", encoding="utf-8") as f:
        for _ in csv.DictReader(f):
            candidate_rows += 1
    qc.metrics["joint_csv_rows"] = int(joint_rows)
    qc.metrics["candidate_csv_rows"] = int(candidate_rows)
    if candidate_rows != n:
        qc.error(f"phase_c3_candidate_csv_row_mismatch:{candidate_rows}!={n}")
    if joint_rows != n * 21:
        qc.error(f"phase_c3_joint_csv_row_mismatch:{joint_rows}!={n * 21}")

    bad_count = sum(int(v) for k, v in qc.metrics["qc_flag_counts"].items() if "bad_" in str(k))
    warn_count = sum(int(v) for k, v in qc.metrics["qc_flag_counts"].items() if "warn_" in str(k))
    qc.metrics["bad_visibility_flag_count"] = int(bad_count)
    qc.metrics["warn_visibility_flag_count"] = int(warn_count)
    if bad_count:
        qc.warn(f"phase_c3_bad_visibility_flags:{bad_count}")
    if warn_count:
        qc.warn(f"phase_c3_warn_visibility_flags:{warn_count}")
    return qc


def check_phase_c4_visibility_realign(
    npz_path: Path,
    summary_path: Path,
    quality_csv: Path,
    expected_candidates: Optional[int] = None,
) -> NodeQC:
    qc = NodeQC("H_phase_c4_visibility_depth_realign_experimental")
    qc.metrics.update({
        "npz": str(npz_path),
        "summary_json": str(summary_path),
        "quality_csv": str(quality_csv),
    })
    if not summary_path.exists():
        qc.warn(f"phase_c4_visibility_realign_not_run_yet:{summary_path}")
        return qc

    summary = read_json(summary_path)
    qc.metrics["summary_ok"] = bool(summary.get("ok", False))
    qc.metrics["candidates"] = int(summary.get("candidates", 0))
    qc.metrics["enabled_by_default"] = bool(summary.get("enabled_by_default", False))
    qc.metrics["keep_previous_on_bad_rms"] = bool(summary.get("keep_previous_on_bad_rms", False))
    qc.metrics["enable_all_visible_fallback"] = bool(summary.get("enable_all_visible_fallback", False))
    qc.metrics["source_counts"] = summary.get("source_counts") or {}
    qc.metrics["qc_flag_counts"] = summary.get("qc_flag_counts") or {}
    qc.metrics["selected_joint_count"] = summary.get("selected_joint_count") or {}
    qc.metrics["rms_m"] = summary.get("rms_m") or {}
    qc.metrics["delta_m"] = summary.get("delta_m") or {}
    qc.metrics["rejected_bad_rms_count"] = int(summary.get("rejected_bad_rms_count", 0))
    qc.metrics["cam_t_visibility_depth_finite_ratio"] = safe_float(summary.get("cam_t_visibility_depth_finite_ratio"), 0.0)
    qc.metrics["joints_cam_visibility_depth_finite_ratio"] = safe_float(summary.get("joints_cam_visibility_depth_finite_ratio"), 0.0)
    qc.metrics["vertices_cam_visibility_depth_finite_ratio"] = safe_float(summary.get("vertices_cam_visibility_depth_finite_ratio"), 0.0)

    for err in summary.get("hard_errors") or []:
        qc.error(f"phase_c4_summary_error:{err}")
    for warn in summary.get("warnings") or []:
        qc.warn(f"phase_c4_summary_warning:{warn}")
    if not summary.get("ok", False):
        qc.error("phase_c4_summary_not_ok")
    if summary.get("enabled_by_default", False):
        qc.warn("phase_c4_visibility_realign_should_remain_default_off")
    if expected_candidates and expected_candidates > 0 and qc.metrics["candidates"] != expected_candidates:
        qc.error(f"phase_c4_candidate_count_mismatch:{qc.metrics['candidates']}!={expected_candidates}")

    if not npz_path.exists():
        qc.error(f"phase_c4_npz_missing:{npz_path}")
        return qc
    if not quality_csv.exists():
        qc.error(f"phase_c4_quality_csv_missing:{quality_csv}")
        return qc

    data = np.load(npz_path, allow_pickle=True)
    required = [
        "frame_index", "track_id", "cam_t_visibility_depth_candidate",
        "cam_t_visibility_depth", "joints_cam_visibility_depth",
        "vertices_cam_visibility_depth", "visibility_realign_source",
        "visibility_realign_selected_joint_count", "visibility_realign_rms_m",
        "visibility_realign_delta_m", "visibility_realign_qc_flag",
    ]
    for key in required:
        if key not in data:
            qc.error(f"missing_phase_c4_npz_field:{key}")
    if qc.hard_errors:
        return qc

    n = int(len(data["frame_index"]))
    qc.metrics["npz_candidates"] = n
    if expected_candidates and expected_candidates > 0 and n != expected_candidates:
        qc.error(f"phase_c4_npz_candidate_count_mismatch:{n}!={expected_candidates}")
    for key in ["cam_t_visibility_depth", "joints_cam_visibility_depth", "vertices_cam_visibility_depth"]:
        ratio = finite_ratio(data[key])
        qc.metrics[f"{key}_finite_ratio"] = ratio
        if ratio < 1.0:
            qc.error(f"non_finite_{key}:ratio={ratio:.6f}")
    flags = np.asarray(data["visibility_realign_qc_flag"]).astype(str)
    qc.metrics["npz_qc_flag_counts"] = {
        value: int(np.sum(flags == value))
        for value in sorted(set(flags.tolist()))
    }
    csv_rows = 0
    with quality_csv.open("r", newline="", encoding="utf-8") as f:
        for _ in csv.DictReader(f):
            csv_rows += 1
    qc.metrics["quality_csv_rows"] = int(csv_rows)
    if csv_rows != n:
        qc.error(f"phase_c4_quality_csv_row_mismatch:{csv_rows}!={n}")
    bad_count = sum(int(v) for k, v in qc.metrics["qc_flag_counts"].items() if "bad_" in str(k))
    warn_count = sum(int(v) for k, v in qc.metrics["qc_flag_counts"].items() if "warn_" in str(k))
    qc.metrics["bad_visibility_realign_flag_count"] = int(bad_count)
    qc.metrics["warn_visibility_realign_flag_count"] = int(warn_count)
    if bad_count:
        qc.warn(f"phase_c4_bad_visibility_realign_flags:{bad_count}")
    if warn_count:
        qc.warn(f"phase_c4_warn_visibility_realign_flags:{warn_count}")
    return qc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--phase-b-npz", default="")
    parser.add_argument("--foundation-depth-summary", default="")
    parser.add_argument("--phase-c0b-summary", default="")
    parser.add_argument("--phase-c0b-frame-csv", default="")
    parser.add_argument("--phase-c0b-correction-csv", default="")
    parser.add_argument("--phase-c-npz", default="")
    parser.add_argument("--phase-c-summary", default="")
    parser.add_argument("--phase-c-quality-csv", default="")
    parser.add_argument("--phase-c1b-npz", default="")
    parser.add_argument("--phase-c1b-summary", default="")
    parser.add_argument("--phase-c1b-quality-csv", default="")
    parser.add_argument("--phase-c1b-track-csv", default="")
    parser.add_argument("--phase-c1c-npz", default="")
    parser.add_argument("--phase-c1c-summary", default="")
    parser.add_argument("--phase-c1c-quality-csv", default="")
    parser.add_argument("--phase-c2-npz", default="")
    parser.add_argument("--phase-c2-summary", default="")
    parser.add_argument("--phase-c2-quality-csv", default="")
    parser.add_argument("--phase-c3-npz", default="")
    parser.add_argument("--phase-c3-summary", default="")
    parser.add_argument("--phase-c3-joint-csv", default="")
    parser.add_argument("--phase-c3-candidate-csv", default="")
    parser.add_argument("--phase-c4-npz", default="")
    parser.add_argument("--phase-c4-summary", default="")
    parser.add_argument("--phase-c4-quality-csv", default="")
    parser.add_argument("--allow-partial-coverage", action="store_true")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    session = Path(args.session_dir).expanduser().resolve()
    phase_b = Path(args.phase_b_npz).expanduser().resolve() if args.phase_b_npz else (
        session / "quality" / "egoinfinity_hand_pipeline" / "stages" / "phase_b_track_postprocess" / "wilor_handresults_phase_b.npz"
    )
    depth_summary = Path(args.foundation_depth_summary).expanduser().resolve() if args.foundation_depth_summary else (
        session / "quality" / "egoinfinity_hand_pipeline" / "stages" / "foundationstereo_depth" / "foundationstereo_depth_summary.json"
    )
    phase_c0b_summary = Path(args.phase_c0b_summary).expanduser().resolve() if args.phase_c0b_summary else (
        session / "quality" / "egoinfinity_hand_pipeline" / "stages" / "foundationstereo_depth_stabilized" / "foundationstereo_depth_stabilized_summary.json"
    )
    phase_c0b_frame_csv = Path(args.phase_c0b_frame_csv).expanduser().resolve() if args.phase_c0b_frame_csv else (
        session / "quality" / "egoinfinity_hand_pipeline" / "stages" / "foundationstereo_depth_stabilized" / "foundationstereo_depth_stabilized_frames.csv"
    )
    phase_c0b_correction_csv = Path(args.phase_c0b_correction_csv).expanduser().resolve() if args.phase_c0b_correction_csv else (
        session / "quality" / "egoinfinity_hand_pipeline" / "stages" / "foundationstereo_depth_stabilized" / "depth_stabilize_corrections.csv"
    )
    phase_c_npz = Path(args.phase_c_npz).expanduser().resolve() if args.phase_c_npz else (
        session / "quality" / "egoinfinity_hand_pipeline" / "stages" / "phase_c_depth_align" / "wilor_handresults_phase_c_depth_aligned.npz"
    )
    phase_c_summary = Path(args.phase_c_summary).expanduser().resolve() if args.phase_c_summary else (
        session / "quality" / "egoinfinity_hand_pipeline" / "stages" / "phase_c_depth_align" / "phase_c_alignment_summary.json"
    )
    phase_c_quality_csv = Path(args.phase_c_quality_csv).expanduser().resolve() if args.phase_c_quality_csv else (
        session / "quality" / "egoinfinity_hand_pipeline" / "stages" / "phase_c_depth_align" / "phase_c_alignment_quality.csv"
    )
    phase_c1b_npz = Path(args.phase_c1b_npz).expanduser().resolve() if args.phase_c1b_npz else (
        session / "quality" / "egoinfinity_hand_pipeline" / "stages" / "phase_c_depth_smooth" / "wilor_handresults_phase_c1b_depth_smooth.npz"
    )
    phase_c1b_summary = Path(args.phase_c1b_summary).expanduser().resolve() if args.phase_c1b_summary else (
        session / "quality" / "egoinfinity_hand_pipeline" / "stages" / "phase_c_depth_smooth" / "depth_smooth_summary.json"
    )
    phase_c1b_quality_csv = Path(args.phase_c1b_quality_csv).expanduser().resolve() if args.phase_c1b_quality_csv else (
        session / "quality" / "egoinfinity_hand_pipeline" / "stages" / "phase_c_depth_smooth" / "depth_smooth_quality.csv"
    )
    phase_c1b_track_csv = Path(args.phase_c1b_track_csv).expanduser().resolve() if args.phase_c1b_track_csv else (
        session / "quality" / "egoinfinity_hand_pipeline" / "stages" / "phase_c_depth_smooth" / "depth_smooth_track_summary.csv"
    )
    phase_c1c_npz = Path(args.phase_c1c_npz).expanduser().resolve() if args.phase_c1c_npz else (
        session / "quality" / "egoinfinity_hand_pipeline" / "stages" / "phase_c_motion_infiller" / "wilor_handresults_phase_c1c_motion_infilled.npz"
    )
    phase_c1c_summary = Path(args.phase_c1c_summary).expanduser().resolve() if args.phase_c1c_summary else (
        session / "quality" / "egoinfinity_hand_pipeline" / "stages" / "phase_c_motion_infiller" / "motion_infiller_summary.json"
    )
    phase_c1c_quality_csv = Path(args.phase_c1c_quality_csv).expanduser().resolve() if args.phase_c1c_quality_csv else (
        session / "quality" / "egoinfinity_hand_pipeline" / "stages" / "phase_c_motion_infiller" / "motion_infiller_quality.csv"
    )
    phase_c2_npz = Path(args.phase_c2_npz).expanduser().resolve() if args.phase_c2_npz else (
        session / "quality" / "egoinfinity_hand_pipeline" / "stages" / "phase_c_mano_smooth" / "wilor_handresults_phase_c2_mano_smooth.npz"
    )
    phase_c2_summary = Path(args.phase_c2_summary).expanduser().resolve() if args.phase_c2_summary else (
        session / "quality" / "egoinfinity_hand_pipeline" / "stages" / "phase_c_mano_smooth" / "mano_smoothing_summary.json"
    )
    phase_c2_quality_csv = Path(args.phase_c2_quality_csv).expanduser().resolve() if args.phase_c2_quality_csv else (
        session / "quality" / "egoinfinity_hand_pipeline" / "stages" / "phase_c_mano_smooth" / "mano_smoothing_quality.csv"
    )
    phase_c3_npz = Path(args.phase_c3_npz).expanduser().resolve() if args.phase_c3_npz else (
        session / "quality" / "egoinfinity_hand_pipeline" / "stages" / "phase_c_mesh_visibility" / "wilor_handresults_phase_c3_mesh_visibility.npz"
    )
    phase_c3_summary = Path(args.phase_c3_summary).expanduser().resolve() if args.phase_c3_summary else (
        session / "quality" / "egoinfinity_hand_pipeline" / "stages" / "phase_c_mesh_visibility" / "mano_mesh_visibility_summary.json"
    )
    phase_c3_joint_csv = Path(args.phase_c3_joint_csv).expanduser().resolve() if args.phase_c3_joint_csv else (
        session / "quality" / "egoinfinity_hand_pipeline" / "stages" / "phase_c_mesh_visibility" / "mano_mesh_visibility_joints.csv"
    )
    phase_c3_candidate_csv = Path(args.phase_c3_candidate_csv).expanduser().resolve() if args.phase_c3_candidate_csv else (
        session / "quality" / "egoinfinity_hand_pipeline" / "stages" / "phase_c_mesh_visibility" / "mano_mesh_visibility_candidates.csv"
    )
    phase_c4_npz = Path(args.phase_c4_npz).expanduser().resolve() if args.phase_c4_npz else (
        session / "quality" / "egoinfinity_hand_pipeline" / "stages" / "phase_c_visibility_depth_realign" / "wilor_handresults_phase_c4_visibility_depth_realign.npz"
    )
    phase_c4_summary = Path(args.phase_c4_summary).expanduser().resolve() if args.phase_c4_summary else (
        session / "quality" / "egoinfinity_hand_pipeline" / "stages" / "phase_c_visibility_depth_realign" / "visibility_depth_realign_summary.json"
    )
    phase_c4_quality_csv = Path(args.phase_c4_quality_csv).expanduser().resolve() if args.phase_c4_quality_csv else (
        session / "quality" / "egoinfinity_hand_pipeline" / "stages" / "phase_c_visibility_depth_realign" / "visibility_depth_realign_quality.csv"
    )

    topcam_qc = check_topcam(session)
    total_frames = int(((topcam_qc.metrics.get("left") or {}).get("frame_count") or 0))
    phase_b_expected_candidates = 0
    if phase_b.exists():
        phase_b_data = np.load(phase_b, allow_pickle=True)
        if "frame_index" in phase_b_data:
            phase_b_expected_candidates = int(len(phase_b_data["frame_index"]))
    downstream_expected_candidates = phase_b_expected_candidates
    if phase_c1c_summary.exists():
        try:
            motion_summary = read_json(phase_c1c_summary)
            downstream_expected_candidates = int(motion_summary.get("candidates_out", downstream_expected_candidates))
        except Exception:
            pass
    require_full_coverage = not bool(args.allow_partial_coverage)

    nodes = [
        topcam_qc,
        check_phase_b_npz(phase_b),
    ]
    if depth_summary.exists():
        nodes.append(check_foundation_depth(
            depth_summary,
            total_frames=total_frames,
            require_full_coverage=require_full_coverage,
        ))
    else:
        qc = NodeQC("D_foundationstereo_depth")
        qc.warn(f"foundation_depth_not_run_yet:{depth_summary}")
        nodes.append(qc)
    if phase_c0b_summary.exists() or args.phase_c0b_summary:
        nodes.append(check_phase_c0b_depth_stabilize(
            phase_c0b_summary,
            phase_c0b_frame_csv,
            phase_c0b_correction_csv,
            total_frames=total_frames,
            require_full_coverage=require_full_coverage,
        ))
    nodes.append(check_phase_c_alignment(
        phase_c_npz,
        phase_c_summary,
        phase_c_quality_csv,
        expected_candidates=phase_b_expected_candidates,
        require_full_coverage=require_full_coverage,
    ))
    if phase_c1b_summary.exists() or args.phase_c1b_summary:
        nodes.append(check_phase_c1b_depth_smooth(
            phase_c1b_npz,
            phase_c1b_summary,
            phase_c1b_quality_csv,
            phase_c1b_track_csv,
            expected_candidates=phase_b_expected_candidates,
        ))
    if phase_c1c_summary.exists() or args.phase_c1c_summary:
        nodes.append(check_phase_c1c_motion_infiller(
            phase_c1c_npz,
            phase_c1c_summary,
            phase_c1c_quality_csv,
            expected_input_candidates=phase_b_expected_candidates,
        ))
    if phase_c2_summary.exists() or args.phase_c2_summary:
        nodes.append(check_phase_c2_mano_smoothing(
            phase_c2_npz,
            phase_c2_summary,
            phase_c2_quality_csv,
            expected_candidates=downstream_expected_candidates,
        ))
    if phase_c3_summary.exists() or args.phase_c3_summary:
        nodes.append(check_phase_c3_mesh_visibility(
            phase_c3_npz,
            phase_c3_summary,
            phase_c3_joint_csv,
            phase_c3_candidate_csv,
            expected_candidates=downstream_expected_candidates,
        ))
    if phase_c4_summary.exists() or args.phase_c4_summary:
        nodes.append(check_phase_c4_visibility_realign(
            phase_c4_npz,
            phase_c4_summary,
            phase_c4_quality_csv,
            expected_candidates=downstream_expected_candidates,
        ))

    report = {
        "semantic": "Node-level quality gate report for LFV EgoInfinity hand pipeline",
        "session_dir": str(session),
        "coverage_policy": "full_video_required" if require_full_coverage else "partial_allowed",
        "nodes": [n.to_dict() for n in nodes],
        "hard_errors": [err for n in nodes for err in n.hard_errors],
        "warnings": [warn for n in nodes for warn in n.warnings],
    }
    report["ok"] = len(report["hard_errors"]) == 0
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.json_out:
        out = Path(args.json_out).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
