#!/usr/bin/env python3
"""Stabilize WiLoR dual-hand candidates before gripper mapping.

This is a CSV-level port of the first EgoInfinity WiLoR post-process layer:

1. Track physical hand candidates by bbox continuity.
2. Correct handedness with per-track majority vote.
3. Remove duplicate overlapping candidates in each frame.
4. Filter short / position-outlier / size-outlier tracks.
5. Reject isolated 3D wrist spikes before selecting the target hand lane.

The output schema remains compatible with the downstream LFV gripper mapping:
one target-hand candidate per frame, still represented as 21 landmark rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


LANDMARK_COUNT = 21


def fnum(value: Any, default: float = float("nan")) -> float:
    if value is None or value == "":
        return default
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def inum(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except Exception:
        return default


def csv_float(value: Any) -> str:
    v = fnum(value)
    return f"{v:.9f}" if math.isfinite(v) else ""


def read_csv(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [dict(row) for row in reader]
        fields = list(reader.fieldnames or [])
    if not fields:
        raise RuntimeError(f"empty/headerless CSV: {path}")
    return rows, fields


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def add_fields(fields: Sequence[str], extras: Sequence[str]) -> List[str]:
    out = list(fields)
    for field in extras:
        if field not in out:
            out.append(field)
    return out


def bbox_area(bbox: Sequence[float]) -> float:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_center(bbox: Sequence[float]) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return np.asarray([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float64)


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = bbox_area(a) + bbox_area(b) - inter
    return float(inter / union) if union > 1e-9 else 0.0


def normal_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return float("nan")
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(math.degrees(math.acos(dot)))


def group_predictions(rows: Sequence[Dict[str, str]]) -> Dict[Tuple[int, int], Dict[str, Any]]:
    groups: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for row in rows:
        frame = inum(row.get("frame_index"), -1)
        rank = inum(row.get("hand_rank"), -1)
        lid = inum(row.get("landmark_id"), -1)
        if frame < 0 or rank < 0 or not (0 <= lid < LANDMARK_COUNT):
            continue
        key = (frame, rank)
        group = groups.setdefault(
            key,
            {
                "frame": frame,
                "rank": rank,
                "rows": [None] * LANDMARK_COUNT,
                "label": str(row.get("hand_label", "")).lower(),
                "det_conf": fnum(row.get("det_conf"), 0.0),
                "elapsed_sec": fnum(row.get("elapsed_sec"), float("nan")),
                "bbox": np.asarray(
                    [
                        fnum(row.get("bbox_x1")),
                        fnum(row.get("bbox_y1")),
                        fnum(row.get("bbox_x2")),
                        fnum(row.get("bbox_y2")),
                    ],
                    dtype=np.float64,
                ),
                "cam_t": np.asarray(
                    [
                        fnum(row.get("cam_t_x")),
                        fnum(row.get("cam_t_y")),
                        fnum(row.get("cam_t_z")),
                    ],
                    dtype=np.float64,
                ),
            },
        )
        group["rows"][lid] = row
    return {
        key: group
        for key, group in groups.items()
        if all(np.isfinite(group["bbox"])) and sum(row is not None for row in group["rows"]) >= 10
    }


def group_points(group: Dict[str, Any]) -> np.ndarray:
    pts = np.full((LANDMARK_COUNT, 3), np.nan, dtype=np.float64)
    for lid, row in enumerate(group["rows"]):
        if row is None:
            continue
        pts[lid] = [
            fnum(row.get("cam_x_m")),
            fnum(row.get("cam_y_m")),
            fnum(row.get("cam_z_m")),
        ]
    return pts


def wrist_position(group: Dict[str, Any]) -> np.ndarray:
    pts = group_points(group)
    if np.isfinite(pts[0]).all():
        return pts[0]
    cam_t = np.asarray(group.get("cam_t", np.full((3,), np.nan)), dtype=np.float64)
    return cam_t if np.isfinite(cam_t).all() else np.full((3,), np.nan, dtype=np.float64)


def palm_normal(group: Dict[str, Any]) -> np.ndarray:
    pts = group_points(group)
    for lid in (0, 5, 17):
        if not np.isfinite(pts[lid]).all():
            return np.full((3,), np.nan, dtype=np.float64)
    n = np.cross(pts[5] - pts[0], pts[17] - pts[0])
    norm = float(np.linalg.norm(n))
    if norm < 1e-9:
        return np.full((3,), np.nan, dtype=np.float64)
    return n / norm


def is_right_value(label: str) -> str:
    return "1.000000000" if str(label).lower() == "right" else "0.000000000"


def make_event(
    frame: int,
    event_type: str,
    target_label: str,
    cand: Optional[Dict[str, Any]] = None,
    **extra: Any,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "frame_index": frame,
        "event_type": event_type,
        "track_id": "",
        "target_label": target_label,
        "stable_label": "",
        "observed_label": "",
        "observed_rank": "",
        "source": "",
        "reason": "",
        "hold_age_frames": "",
        "det_conf": "",
        "track_iou_prev": "",
        "track_center_px": "",
        "palm_angle_deg": "",
        "palm_speed_deg_s": "",
        "duplicate_with": "",
        "track_frame_count": "",
        "track_duration_frames": "",
        "track_filter_status": "",
        "track_filter_reason": "",
        "spike_distance_m": "",
    }
    if cand is not None:
        row.update(
            {
                "track_id": cand.get("track_id", ""),
                "stable_label": cand.get("majority_label", cand.get("stable_label", "")),
                "observed_label": cand.get("label", ""),
                "observed_rank": cand.get("rank", ""),
                "det_conf": csv_float(cand.get("det_conf")),
                "track_iou_prev": csv_float(cand.get("track_iou_prev")),
                "track_center_px": csv_float(cand.get("track_center_px")),
                "palm_angle_deg": csv_float(cand.get("palm_angle_deg")),
                "palm_speed_deg_s": csv_float(cand.get("palm_speed_deg_s")),
                "duplicate_with": cand.get("duplicate_with", ""),
                "track_frame_count": cand.get("track_frame_count", ""),
                "track_duration_frames": cand.get("track_duration_frames", ""),
                "track_filter_status": cand.get("track_filter_status", ""),
                "track_filter_reason": cand.get("track_filter_reason", ""),
                "spike_distance_m": csv_float(cand.get("spike_distance_m")),
            }
        )
    row.update(extra)
    return row


def retime_rows(
    source_rows: Sequence[Optional[Dict[str, str]]],
    frame: int,
    elapsed: float,
    target_label: str,
    source: str,
    track_id: int,
    hold_age: int,
    observed_group: Optional[Dict[str, Any]],
    reason: str,
    cand: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for lid, src in enumerate(source_rows):
        if src is None:
            continue
        row: Dict[str, Any] = dict(src)
        row["frame_index"] = int(frame)
        row["elapsed_sec"] = csv_float(elapsed)
        row["hand_rank"] = 0
        row["hand_label"] = target_label
        row["is_right"] = is_right_value(target_label)
        row["landmark_id"] = lid
        row["stabilized"] = 1
        row["stable_track_id"] = int(track_id)
        row["stabilization_source"] = source
        row["stabilization_reason"] = reason
        row["hold_age_frames"] = int(hold_age)
        if observed_group is not None:
            row["observed_hand_rank"] = int(observed_group["rank"])
            row["observed_hand_label"] = observed_group["label"]
            row["observed_det_conf"] = csv_float(observed_group["det_conf"])
        else:
            row["observed_hand_rank"] = ""
            row["observed_hand_label"] = ""
            row["observed_det_conf"] = ""
        if cand is not None:
            row["track_majority_label"] = cand.get("majority_label", "")
            row["track_majority_weight"] = csv_float(cand.get("majority_weight"))
            row["track_frame_count"] = cand.get("track_frame_count", "")
            row["track_duration_frames"] = cand.get("track_duration_frames", "")
            row["track_filter_status"] = cand.get("track_filter_status", "")
            row["track_filter_reason"] = cand.get("track_filter_reason", "")
            row["spike_rejected"] = int(cand.get("spike_bad", 0))
            row["spike_distance_m"] = csv_float(cand.get("spike_distance_m"))
        else:
            row["track_majority_label"] = ""
            row["track_majority_weight"] = ""
            row["track_frame_count"] = ""
            row["track_duration_frames"] = ""
            row["track_filter_status"] = ""
            row["track_filter_reason"] = ""
            row["spike_rejected"] = 0
            row["spike_distance_m"] = ""
        out.append(row)
    return out


def candidate_from_group(group: Dict[str, Any], fps: float) -> Dict[str, Any]:
    bbox = np.asarray(group["bbox"], dtype=np.float64)
    elapsed = float(group["elapsed_sec"]) if math.isfinite(float(group["elapsed_sec"])) else int(group["frame"]) / fps
    return {
        "key": (int(group["frame"]), int(group["rank"])),
        "group": group,
        "frame": int(group["frame"]),
        "rank": int(group["rank"]),
        "label": str(group["label"]).lower(),
        "det_conf": float(group["det_conf"]),
        "elapsed_sec": elapsed,
        "bbox": bbox,
        "center": bbox_center(bbox),
        "area": bbox_area(bbox),
        "wrist": wrist_position(group),
        "cam_t": np.asarray(group.get("cam_t", np.full((3,), np.nan)), dtype=np.float64),
        "palm_normal": palm_normal(group),
        "duplicate": False,
        "duplicate_drop": False,
        "duplicate_with": "",
        "track_id": 0,
        "track_iou_prev": float("nan"),
        "track_center_px": float("nan"),
        "track_gap": 0,
        "palm_angle_deg": float("nan"),
        "palm_speed_deg_s": float("nan"),
        "palm_bad": 0,
        "majority_label": "",
        "majority_weight": float("nan"),
        "track_frame_count": 0,
        "track_duration_frames": 0,
        "track_filter_status": "ok",
        "track_filter_reason": "",
        "spike_bad": 0,
        "spike_distance_m": float("nan"),
    }


def mark_duplicate_candidates(
    frame_candidates: List[Dict[str, Any]],
    duplicate_iou: float,
    duplicate_center_px: float,
) -> None:
    duplicate_clusters: List[List[int]] = []
    for i in range(len(frame_candidates)):
        for j in range(i + 1, len(frame_candidates)):
            ci = frame_candidates[i]
            cj = frame_candidates[j]
            iou = bbox_iou(ci["bbox"], cj["bbox"])
            center_dist = float(np.linalg.norm(ci["center"] - cj["center"]))
            area_ratio = min(float(ci["area"]), float(cj["area"])) / max(float(ci["area"]), float(cj["area"]), 1.0)
            duplicate = bool(iou >= duplicate_iou or (center_dist <= duplicate_center_px and area_ratio >= 0.35))
            if not duplicate:
                continue
            ci["duplicate"] = True
            cj["duplicate"] = True
            ci["duplicate_with"] = str(cj["rank"])
            cj["duplicate_with"] = str(ci["rank"])
            found = False
            for cluster in duplicate_clusters:
                if i in cluster or j in cluster:
                    if i not in cluster:
                        cluster.append(i)
                    if j not in cluster:
                        cluster.append(j)
                    found = True
                    break
            if not found:
                duplicate_clusters.append([i, j])
    for cluster in duplicate_clusters:
        best_i = max(cluster, key=lambda idx: float(frame_candidates[idx]["det_conf"]))
        for idx in cluster:
            frame_candidates[idx]["duplicate_drop"] = idx != best_i


def build_tracks(
    groups: Dict[Tuple[int, int], Dict[str, Any]],
    keys_by_frame: Dict[int, List[Tuple[int, int]]],
    args: argparse.Namespace,
    fps: float,
    target_label: str,
) -> Tuple[Dict[int, List[Dict[str, Any]]], Dict[int, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    active_tracks: Dict[int, Dict[str, Any]] = {}
    track_history: Dict[int, List[Dict[str, Any]]] = {}
    records_by_frame: Dict[int, List[Dict[str, Any]]] = {}
    event_rows: List[Dict[str, Any]] = []
    next_track_id = 1

    missing_bridge_frames = max(1, int(round(float(args.missing_bridge_ms) * fps / 1000.0)))
    palm_window_frames = max(1, int(round(float(args.palm_flip_window_ms) * fps / 1000.0)))

    for frame in sorted(keys_by_frame):
        frame_candidates = [candidate_from_group(groups[key], fps) for key in keys_by_frame[frame]]
        mark_duplicate_candidates(frame_candidates, float(args.duplicate_iou), float(args.duplicate_center_px))

        matches: List[Tuple[float, int, int, float, float, int]] = []
        for ci, cand in enumerate(frame_candidates):
            for track_id, track in active_tracks.items():
                gap = int(frame - int(track["last_frame"]))
                if gap < 1 or gap > missing_bridge_frames:
                    continue
                iou = bbox_iou(cand["bbox"], track["bbox"])
                center_dist = float(np.linalg.norm(cand["center"] - track["center"]))
                if iou < float(args.track_min_iou) and center_dist > float(args.track_max_center_px):
                    continue
                score = iou * 3.0 - center_dist / max(1.0, float(args.track_max_center_px)) - 0.25 * float(gap - 1)
                matches.append((score, ci, int(track_id), iou, center_dist, gap))
        matches.sort(key=lambda item: item[0], reverse=True)

        assigned_candidates = set()
        assigned_tracks = set()
        frame_track_updates: Dict[int, Dict[str, Any]] = {}
        for _, ci, track_id, iou, center_dist, gap in matches:
            if ci in assigned_candidates or track_id in assigned_tracks:
                continue
            assigned_candidates.add(ci)
            assigned_tracks.add(track_id)
            cand = frame_candidates[ci]
            track = active_tracks[track_id]
            palm_angle = normal_angle_deg(np.asarray(track.get("last_palm_normal", np.full((3,), np.nan))), cand["palm_normal"])
            dt = max(1.0 / fps, float(gap) / fps)
            palm_speed = palm_angle / dt if math.isfinite(palm_angle) else float("nan")
            palm_bad = bool(
                gap <= palm_window_frames
                and math.isfinite(palm_angle)
                and (
                    palm_angle >= float(args.palm_flip_angle_deg)
                    or (math.isfinite(palm_speed) and palm_speed >= float(args.hard_palm_speed_deg_s))
                )
            )
            cand["track_id"] = track_id
            cand["track_iou_prev"] = iou
            cand["track_center_px"] = center_dist
            cand["track_gap"] = gap
            cand["palm_angle_deg"] = palm_angle
            cand["palm_speed_deg_s"] = palm_speed
            cand["palm_bad"] = int(palm_bad)
            frame_track_updates[track_id] = cand
            if palm_bad:
                event_rows.append(
                    make_event(
                        frame,
                        "palm_motion_bad",
                        target_label,
                        cand,
                        source="palm_normal_gate",
                        reason="angle_or_speed",
                    )
                )

        for ci, cand in enumerate(frame_candidates):
            if ci in assigned_candidates:
                continue
            track_id = next_track_id
            next_track_id += 1
            cand["track_id"] = track_id
            frame_track_updates[track_id] = cand
            event_rows.append(make_event(frame, "track_start", target_label, cand, source="new_track"))

        for cand in frame_candidates:
            if cand["duplicate_drop"]:
                event_rows.append(
                    make_event(
                        frame,
                        "duplicate_drop",
                        target_label,
                        cand,
                        source="duplicate_overlap",
                        reason="lower_conf_overlap",
                    )
                )

        for track_id, cand in frame_track_updates.items():
            active_tracks[track_id] = {
                "last_frame": int(frame),
                "bbox": np.asarray(cand["bbox"], dtype=np.float64),
                "center": np.asarray(cand["center"], dtype=np.float64),
                "last_palm_normal": np.asarray(cand["palm_normal"], dtype=np.float64),
            }
            track_history.setdefault(track_id, []).append(cand)
            records_by_frame.setdefault(frame, []).append(cand)

        active_tracks = {
            int(tid): tr
            for tid, tr in active_tracks.items()
            if int(frame - int(tr["last_frame"])) <= missing_bridge_frames
        }

    return records_by_frame, track_history, event_rows


def apply_track_majority_vote(
    track_history: Dict[int, List[Dict[str, Any]]],
    target_label: str,
    event_rows: List[Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    track_stats: Dict[int, Dict[str, Any]] = {}
    for track_id, recs in track_history.items():
        weights: Dict[str, float] = {}
        for rec in recs:
            label = str(rec.get("label", "")).lower()
            weights[label] = weights.get(label, 0.0) + max(0.01, float(rec.get("det_conf", 0.0)))
        majority_label = max(weights, key=lambda key: float(weights[key])) if weights else ""
        majority_weight = float(weights.get(majority_label, 0.0))
        frames = [int(rec["frame"]) for rec in recs]
        duration = max(frames) - min(frames) + 1 if frames else 0
        track_stats[track_id] = {
            "majority_label": majority_label,
            "majority_weight": majority_weight,
            "label_weights": weights,
            "frame_count": len(recs),
            "duration_frames": duration,
            "frames": frames,
        }
        for rec in recs:
            rec["majority_label"] = majority_label
            rec["majority_weight"] = majority_weight
            rec["track_frame_count"] = len(recs)
            rec["track_duration_frames"] = duration
            if str(rec.get("label", "")).lower() != majority_label:
                event_rows.append(
                    make_event(
                        int(rec["frame"]),
                        "track_majority_label_fix",
                        target_label,
                        rec,
                        source="track_majority_vote",
                        reason=f"{rec.get('label','')}->{majority_label}",
                    )
                )
    return track_stats


def finite_median(values: Sequence[np.ndarray]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return np.full((3,), np.nan, dtype=np.float64)
    valid = np.isfinite(arr).all(axis=1)
    if not valid.any():
        return np.full((3,), np.nan, dtype=np.float64)
    return np.median(arr[valid], axis=0)


def median_track_size(recs: Sequence[Dict[str, Any]]) -> float:
    vals: List[float] = []
    for rec in recs:
        depth = fnum(np.asarray(rec.get("cam_t", np.full((3,), np.nan)))[2])
        area = fnum(rec.get("area"))
        if math.isfinite(depth) and depth > 0.1 and math.isfinite(area) and area > 0.0:
            vals.append(area / (depth * depth))
    return float(np.median(vals)) if vals else float("nan")


def apply_track_filtering(
    track_history: Dict[int, List[Dict[str, Any]]],
    track_stats: Dict[int, Dict[str, Any]],
    args: argparse.Namespace,
    fps: float,
    target_label: str,
    event_rows: List[Dict[str, Any]],
) -> None:
    min_track_frames = max(1, int(args.min_track_frames))
    if float(args.min_track_ms) > 0:
        min_track_frames = max(min_track_frames, int(round(float(args.min_track_ms) * fps / 1000.0)))

    dominant: Dict[str, int] = {}
    for side in ("left", "right"):
        side_tracks = [
            (tid, stats)
            for tid, stats in track_stats.items()
            if str(stats.get("majority_label", "")).lower() == side
        ]
        if side_tracks:
            dominant[side] = max(side_tracks, key=lambda item: int(item[1]["frame_count"]))[0]

    remove_reason: Dict[int, str] = {}
    for track_id, recs in track_history.items():
        stats = track_stats[track_id]
        side = str(stats.get("majority_label", "")).lower()
        n_frames = int(stats["frame_count"])
        if n_frames < min_track_frames:
            remove_reason[track_id] = f"short_track<{min_track_frames}"
            continue

        dom_id = dominant.get(side)
        if dom_id is None or dom_id == track_id:
            continue
        dom_recs = track_history[dom_id]
        dom_frames = int(track_stats[dom_id]["frame_count"])
        if n_frames >= max(1, dom_frames) * float(args.outlier_max_relative_len):
            continue

        dom_wrist = finite_median([np.asarray(rec["wrist"], dtype=np.float64) for rec in dom_recs])
        my_wrist = finite_median([np.asarray(rec["wrist"], dtype=np.float64) for rec in recs])
        if np.isfinite(dom_wrist).all() and np.isfinite(my_wrist).all():
            dist = float(np.linalg.norm(my_wrist - dom_wrist))
            if dist > float(args.track_outlier_distance_m):
                remove_reason[track_id] = f"position_outlier>{args.track_outlier_distance_m:g}m"
                continue

        dom_size = median_track_size(dom_recs)
        my_size = median_track_size(recs)
        if math.isfinite(dom_size) and dom_size > 1e-9 and math.isfinite(my_size):
            ratio = my_size / dom_size
            if ratio > float(args.size_outlier_ratio) or ratio < 1.0 / float(args.size_outlier_ratio):
                remove_reason[track_id] = f"size_outlier_ratio={ratio:.2f}"

    for track_id, recs in track_history.items():
        reason = remove_reason.get(track_id, "")
        for rec in recs:
            if reason:
                rec["track_filter_status"] = "removed"
                rec["track_filter_reason"] = reason
            else:
                rec["track_filter_status"] = "ok"
                rec["track_filter_reason"] = ""
        if reason:
            event_rows.append(
                make_event(
                    int(recs[0]["frame"]),
                    "track_filter_removed",
                    target_label,
                    recs[0],
                    source="track_level_filter",
                    reason=reason,
                )
            )


def apply_spike_rejection(
    track_history: Dict[int, List[Dict[str, Any]]],
    args: argparse.Namespace,
    target_label: str,
    event_rows: List[Dict[str, Any]],
) -> None:
    if not bool(args.enable_spike_reject):
        return
    max_gap = max(1, int(args.spike_neighbor_max_gap_frames))
    for track_id, recs_unsorted in track_history.items():
        recs = [
            rec
            for rec in sorted(recs_unsorted, key=lambda item: int(item["frame"]))
            if rec.get("track_filter_status") == "ok" and not bool(rec.get("duplicate_drop", False))
        ]
        if len(recs) < 3:
            continue
        positions = [np.asarray(rec["wrist"], dtype=np.float64) for rec in recs]
        frames = [int(rec["frame"]) for rec in recs]
        steps = []
        for i in range(1, len(recs)):
            if frames[i] - frames[i - 1] <= max_gap and np.isfinite(positions[i]).all() and np.isfinite(positions[i - 1]).all():
                steps.append(float(np.linalg.norm(positions[i] - positions[i - 1])))
        if not steps:
            continue
        median_step = float(np.median(steps))
        threshold = max(float(args.spike_min_distance_m), float(args.spike_threshold_factor) * median_step)
        for i in range(1, len(recs) - 1):
            if frames[i] - frames[i - 1] > max_gap or frames[i + 1] - frames[i] > max_gap:
                continue
            p0, p1, p2 = positions[i - 1], positions[i], positions[i + 1]
            if not (np.isfinite(p0).all() and np.isfinite(p1).all() and np.isfinite(p2).all()):
                continue
            d_prev = float(np.linalg.norm(p1 - p0))
            d_next = float(np.linalg.norm(p2 - p1))
            d_skip = float(np.linalg.norm(p2 - p0))
            if d_prev > threshold and d_next > threshold and d_skip < threshold * 0.75:
                recs[i]["spike_bad"] = 1
                recs[i]["spike_distance_m"] = max(d_prev, d_next)
                event_rows.append(
                    make_event(
                        int(recs[i]["frame"]),
                        "spike_rejected",
                        target_label,
                        recs[i],
                        source="egoinfinity_style_spike_reject",
                        reason=f"single_frame_wrist_spike>{threshold:.4f}m",
                    )
                )


def select_target_rows(
    records_by_frame: Dict[int, List[Dict[str, Any]]],
    target_label: str,
    fps: float,
    args: argparse.Namespace,
    event_rows: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    output_rows: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    target_last_rows: Optional[List[Optional[Dict[str, str]]]] = None
    target_last_frame: Optional[int] = None
    target_last_track_id = 0
    target_last_cand: Optional[Dict[str, Any]] = None
    missing_bridge_frames = max(1, int(round(float(args.missing_bridge_ms) * fps / 1000.0)))
    identity_hold_frames = max(1, int(round(float(args.identity_hold_ms) * fps / 1000.0)))

    def bump(name: str) -> None:
        counts[name] = counts.get(name, 0) + 1

    for frame in sorted(records_by_frame):
        frame_candidates = records_by_frame[frame]
        good_candidates = [
            cand
            for cand in frame_candidates
            if str(cand.get("majority_label", "")).lower() == target_label
            and not bool(cand.get("duplicate_drop", False))
            and cand.get("track_filter_status") == "ok"
            and not int(cand.get("spike_bad", 0))
            and not int(cand.get("palm_bad", 0))
        ]
        good_candidates.sort(
            key=lambda cand: (
                -int(cand.get("track_frame_count", 0)),
                -float(cand.get("det_conf", 0.0)),
            )
        )
        bad_target_candidates = [
            cand
            for cand in frame_candidates
            if str(cand.get("majority_label", "")).lower() == target_label
            and not bool(cand.get("duplicate_drop", False))
        ]
        bad_target_candidates.sort(key=lambda cand: -float(cand.get("det_conf", 0.0)))

        selected: Optional[Dict[str, Any]] = good_candidates[0] if good_candidates else None
        bad_selected: Optional[Dict[str, Any]] = None if selected is not None else (bad_target_candidates[0] if bad_target_candidates else None)
        frame_elapsed = (
            float(selected["elapsed_sec"])
            if selected is not None
            else (
                float(bad_selected["elapsed_sec"])
                if bad_selected is not None
                else (float(frame_candidates[0]["elapsed_sec"]) if frame_candidates else frame / fps)
            )
        )
        source = ""
        reason = ""
        hold_age = 0
        source_rows: Optional[List[Optional[Dict[str, str]]]] = None
        observed_group: Optional[Dict[str, Any]] = selected["group"] if selected is not None else (bad_selected["group"] if bad_selected is not None else None)
        track_id = int(selected["track_id"]) if selected is not None else (int(bad_selected["track_id"]) if bad_selected is not None else target_last_track_id)
        selected_for_metadata = selected

        if selected is not None:
            source_rows = selected["group"]["rows"]
            source = "raw_candidate"
            reason = "ok"
            hold_age = 0
        elif bad_selected is not None and target_last_rows is not None and target_last_frame is not None:
            hold_age = int(frame - target_last_frame)
            bad_hold_limit = max(missing_bridge_frames, identity_hold_frames)
            if hold_age <= bad_hold_limit:
                source_rows = target_last_rows
                source = "hold_bad_candidate"
                if bad_selected.get("track_filter_status") != "ok":
                    reason = str(bad_selected.get("track_filter_reason", "track_filtered"))
                elif int(bad_selected.get("spike_bad", 0)):
                    reason = "spike_rejected"
                elif int(bad_selected.get("palm_bad", 0)):
                    reason = "palm_motion_bad"
                else:
                    reason = "candidate_bad"
                selected_for_metadata = bad_selected
        elif selected is None and target_last_rows is not None and target_last_frame is not None:
            hold_age = int(frame - target_last_frame)
            if hold_age <= missing_bridge_frames:
                source_rows = target_last_rows
                source = "hold_missing"
                reason = "missing_bridge"
                selected_for_metadata = target_last_cand

        if source_rows is None:
            bump("dropped")
            continue

        output_rows.extend(
            retime_rows(
                source_rows,
                frame,
                frame_elapsed,
                target_label,
                source,
                track_id,
                hold_age,
                observed_group,
                reason,
                selected_for_metadata,
            )
        )
        bump(source)
        if source == "raw_candidate" and selected is not None:
            target_last_rows = selected["group"]["rows"]
            target_last_frame = frame
            target_last_track_id = track_id
            target_last_cand = selected

    return output_rows, counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--target-label", required=True, choices=["left", "right"])
    parser.add_argument("--events-csv", default="")
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--identity-hold-ms", type=float, default=400.0)
    parser.add_argument("--missing-bridge-ms", type=float, default=200.0)
    parser.add_argument("--palm-flip-window-ms", type=float, default=300.0)
    parser.add_argument("--palm-flip-angle-deg", type=float, default=100.0)
    parser.add_argument("--hard-palm-speed-deg-s", type=float, default=900.0)
    parser.add_argument("--track-min-iou", type=float, default=0.12)
    parser.add_argument("--track-max-center-px", type=float, default=130.0)
    parser.add_argument("--duplicate-iou", type=float, default=0.45)
    parser.add_argument("--duplicate-center-px", type=float, default=35.0)
    parser.add_argument("--min-track-frames", type=int, default=5)
    parser.add_argument("--min-track-ms", type=float, default=0.0)
    parser.add_argument("--track-outlier-distance-m", type=float, default=0.5)
    parser.add_argument("--outlier-max-relative-len", type=float, default=0.30)
    parser.add_argument("--size-outlier-ratio", type=float, default=3.0)
    parser.add_argument("--enable-spike-reject", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--spike-threshold-factor", type=float, default=3.0)
    parser.add_argument("--spike-min-distance-m", type=float, default=0.08)
    parser.add_argument("--spike-neighbor-max-gap-frames", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_csv = Path(args.input_csv).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    events_csv = Path(args.events_csv).expanduser().resolve() if args.events_csv else output_csv.with_name(output_csv.stem + "_events.csv")
    summary_json = Path(args.summary_json).expanduser().resolve() if args.summary_json else output_csv.with_suffix(".json")
    target_label = str(args.target_label).lower()
    fps = float(args.fps)

    raw_rows, raw_fields = read_csv(input_csv)
    groups = group_predictions(raw_rows)
    keys_by_frame: Dict[int, List[Tuple[int, int]]] = {}
    for key in sorted(groups):
        keys_by_frame.setdefault(key[0], []).append(key)

    out_fields = add_fields(
        raw_fields,
        [
            "stabilized",
            "stable_track_id",
            "stabilization_source",
            "stabilization_reason",
            "hold_age_frames",
            "observed_hand_rank",
            "observed_hand_label",
            "observed_det_conf",
            "track_majority_label",
            "track_majority_weight",
            "track_frame_count",
            "track_duration_frames",
            "track_filter_status",
            "track_filter_reason",
            "spike_rejected",
            "spike_distance_m",
        ],
    )
    event_fields = [
        "frame_index",
        "event_type",
        "track_id",
        "target_label",
        "stable_label",
        "observed_label",
        "observed_rank",
        "source",
        "reason",
        "hold_age_frames",
        "det_conf",
        "track_iou_prev",
        "track_center_px",
        "palm_angle_deg",
        "palm_speed_deg_s",
        "duplicate_with",
        "track_frame_count",
        "track_duration_frames",
        "track_filter_status",
        "track_filter_reason",
        "spike_distance_m",
    ]

    records_by_frame, track_history, event_rows = build_tracks(groups, keys_by_frame, args, fps, target_label)
    track_stats = apply_track_majority_vote(track_history, target_label, event_rows)
    apply_track_filtering(track_history, track_stats, args, fps, target_label, event_rows)
    apply_spike_rejection(track_history, args, target_label, event_rows)
    output_rows, source_counts = select_target_rows(records_by_frame, target_label, fps, args, event_rows)

    write_csv(output_csv, output_rows, out_fields)
    write_csv(events_csv, event_rows, event_fields)

    frame_count = len({inum(row.get("frame_index"), -1) for row in output_rows if inum(row.get("frame_index"), -1) >= 0})
    tracks_removed = sorted(
        int(tid)
        for tid, recs in track_history.items()
        if recs and recs[0].get("track_filter_status") == "removed"
    )
    summary = {
        "semantic": "EgoInfinity-style CSV-level WiLoR pre-gripper temporal stabilizer.",
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "events_csv": str(events_csv),
        "target_label": target_label,
        "fps": fps,
        "params": {
            "identity_hold_ms": float(args.identity_hold_ms),
            "identity_hold_frames": max(1, int(round(float(args.identity_hold_ms) * fps / 1000.0))),
            "missing_bridge_ms": float(args.missing_bridge_ms),
            "missing_bridge_frames": max(1, int(round(float(args.missing_bridge_ms) * fps / 1000.0))),
            "palm_flip_window_ms": float(args.palm_flip_window_ms),
            "palm_flip_window_frames": max(1, int(round(float(args.palm_flip_window_ms) * fps / 1000.0))),
            "palm_flip_angle_deg": float(args.palm_flip_angle_deg),
            "hard_palm_speed_deg_s": float(args.hard_palm_speed_deg_s),
            "track_min_iou": float(args.track_min_iou),
            "track_max_center_px": float(args.track_max_center_px),
            "duplicate_iou": float(args.duplicate_iou),
            "duplicate_center_px": float(args.duplicate_center_px),
            "min_track_frames": int(args.min_track_frames),
            "min_track_ms": float(args.min_track_ms),
            "track_outlier_distance_m": float(args.track_outlier_distance_m),
            "outlier_max_relative_len": float(args.outlier_max_relative_len),
            "size_outlier_ratio": float(args.size_outlier_ratio),
            "enable_spike_reject": bool(args.enable_spike_reject),
            "spike_threshold_factor": float(args.spike_threshold_factor),
            "spike_min_distance_m": float(args.spike_min_distance_m),
            "spike_neighbor_max_gap_frames": int(args.spike_neighbor_max_gap_frames),
        },
        "rows_in": len(raw_rows),
        "groups_in": len(groups),
        "tracks_total": len(track_history),
        "tracks_removed": len(tracks_removed),
        "tracks_removed_ids": tracks_removed,
        "rows_out": len(output_rows),
        "frames_out": int(frame_count),
        "source_counts": source_counts,
        "event_counts": {
            name: sum(1 for row in event_rows if row.get("event_type") == name)
            for name in sorted({str(row.get("event_type", "")) for row in event_rows})
        },
        "track_stats": {
            str(tid): {
                "majority_label": stats.get("majority_label", ""),
                "frame_count": int(stats.get("frame_count", 0)),
                "duration_frames": int(stats.get("duration_frames", 0)),
                "label_weights": stats.get("label_weights", {}),
                "filter_status": track_history[tid][0].get("track_filter_status", "") if track_history.get(tid) else "",
                "filter_reason": track_history[tid][0].get("track_filter_reason", "") if track_history.get(tid) else "",
            }
            for tid, stats in sorted(track_stats.items())
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
