#!/usr/bin/env python3
"""EgoInfinity Phase-B style post-processing for LFV WiLoR HandResult NPZ."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


LANDMARK_COUNT = 21


def csv_float(value: float) -> str:
    return f"{float(value):.9f}" if np.isfinite(float(value)) else ""


def bbox_area(b: np.ndarray) -> float:
    return float(max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]))


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    ix2, iy2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = bbox_area(a) + bbox_area(b) - inter
    return float(inter / union) if union > 1e-9 else 0.0


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.keys()}


def make_events_row(event_type: str, frame: int, idx: int, track_id: int, reason: str = "", **extra: Any) -> Dict[str, Any]:
    row = {
        "event_type": event_type,
        "frame_index": int(frame),
        "candidate_index": int(idx),
        "track_id": int(track_id),
        "reason": reason,
    }
    row.update(extra)
    return row


def assign_tracks_like_egoinfinity(frame_index: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Fallback tracker matching EgoInfinity HandDetector's previous-frame IoU rule."""
    n = len(frame_index)
    track_ids = np.full(n, -1, dtype=np.int32)
    prev_indices: List[int] = []
    next_tid = 0

    by_frame: Dict[int, List[int]] = defaultdict(list)
    for i, f in enumerate(frame_index):
        by_frame[int(f)].append(i)

    for f in sorted(by_frame):
        indices = list(by_frame[f])
        if not prev_indices:
            for i in indices:
                track_ids[i] = next_tid
                next_tid += 1
            prev_indices = indices
            continue

        pairs = []
        for pi in prev_indices:
            for ci in indices:
                pairs.append((bbox_iou(bbox[pi], bbox[ci]), pi, ci))
        pairs.sort(reverse=True)

        used_prev = set()
        used_curr = set()
        for iou_val, pi, ci in pairs:
            if iou_val < 0.2:
                break
            if pi in used_prev or ci in used_curr:
                continue
            track_ids[ci] = int(track_ids[pi])
            used_prev.add(pi)
            used_curr.add(ci)

        for ci in indices:
            if ci not in used_curr:
                track_ids[ci] = next_tid
                next_tid += 1
        prev_indices = indices
    return track_ids


def dedup_frame(indices: Sequence[int], bbox: np.ndarray, conf: np.ndarray, iou_thresh: float) -> Tuple[set, List[Tuple[int, int, float]]]:
    removed: set = set()
    pairs: List[Tuple[int, int, float]] = []
    order = sorted(indices, key=lambda i: float(conf[i]), reverse=True)
    for ai, i in enumerate(order):
        if i in removed:
            continue
        for j in order[ai + 1 :]:
            if j in removed:
                continue
            iou = bbox_iou(bbox[i], bbox[j])
            if iou > iou_thresh:
                removed.add(j)
                pairs.append((j, i, iou))
    return removed, pairs


def export_predictions_csv(path: Path, data: Dict[str, np.ndarray], indices: np.ndarray, track_ids: np.ndarray) -> None:
    fields = [
        "frame_index", "elapsed_sec", "hand_rank", "candidate_index", "track_id",
        "hand_label", "is_right", "det_conf", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        "cam_t_x", "cam_t_y", "cam_t_z", "scaled_focal_length", "landmark_id",
        "local_x_m", "local_y_m", "local_z_m", "cam_x_m", "cam_y_m", "cam_z_m", "u_px", "v_px",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        ranks_by_frame: Dict[int, int] = defaultdict(int)
        for out_i, src_i in enumerate(indices.tolist()):
            frame = int(data["frame_index"][src_i])
            rank = ranks_by_frame[frame]
            ranks_by_frame[frame] += 1
            label = str(data["hand_label"][src_i])
            bbox = data["bbox_xyxy"][src_i]
            cam_t = data["cam_t"][src_i]
            focal = float(data["focal_length"][src_i])
            for lid in range(LANDMARK_COUNT):
                local = data["joints_3d_rel"][src_i, lid]
                cam = data["joints_cam"][src_i, lid]
                uv = data["joints_uv"][src_i, lid]
                writer.writerow(
                    {
                        "frame_index": frame,
                        "elapsed_sec": csv_float(float(data["elapsed_sec"][src_i])),
                        "hand_rank": rank,
                        "candidate_index": int(data["candidate_index"][src_i]) if "candidate_index" in data else int(src_i),
                        "track_id": int(track_ids[src_i]),
                        "hand_label": label,
                        "is_right": csv_float(float(data["is_right"][src_i])),
                        "det_conf": csv_float(float(data["det_conf"][src_i])),
                        "bbox_x1": csv_float(float(bbox[0])),
                        "bbox_y1": csv_float(float(bbox[1])),
                        "bbox_x2": csv_float(float(bbox[2])),
                        "bbox_y2": csv_float(float(bbox[3])),
                        "cam_t_x": csv_float(float(cam_t[0])),
                        "cam_t_y": csv_float(float(cam_t[1])),
                        "cam_t_z": csv_float(float(cam_t[2])),
                        "scaled_focal_length": csv_float(focal),
                        "landmark_id": lid,
                        "local_x_m": csv_float(float(local[0])),
                        "local_y_m": csv_float(float(local[1])),
                        "local_z_m": csv_float(float(local[2])),
                        "cam_x_m": csv_float(float(cam[0])),
                        "cam_y_m": csv_float(float(cam[1])),
                        "cam_z_m": csv_float(float(cam[2])),
                        "u_px": csv_float(float(uv[0])),
                        "v_px": csv_float(float(uv[1])),
                    }
                )


def save_subset_npz(path: Path, data: Dict[str, np.ndarray], keep_indices: np.ndarray, track_ids: np.ndarray) -> None:
    out: Dict[str, np.ndarray] = {}
    n = len(data["frame_index"])
    per_candidate = {
        "frame_index", "elapsed_sec", "hand_rank", "candidate_index", "hand_label", "is_right",
        "det_conf", "bbox_xyxy", "cam_t", "pred_cam", "focal_length", "global_orient",
        "hand_pose", "betas", "joints_3d_rel", "vertices_rel", "joints_cam", "vertices_cam", "joints_uv",
    }
    for key, value in data.items():
        if key in per_candidate and value.shape[0] == n:
            out[key] = value[keep_indices]
        else:
            out[key] = value
    out["track_id"] = track_ids[keep_indices].astype(np.int32)
    ranks_by_frame: Dict[int, int] = defaultdict(int)
    new_ranks = np.zeros(len(keep_indices), dtype=np.int32)
    for out_i, src_i in enumerate(keep_indices.tolist()):
        frame = int(data["frame_index"][src_i])
        new_ranks[out_i] = ranks_by_frame[frame]
        ranks_by_frame[frame] += 1
    out["hand_rank"] = new_ranks
    np.savez_compressed(path, **out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dedup-iou", type=float, default=0.30)
    parser.add_argument("--min-track-frames", type=int, default=5)
    parser.add_argument("--position-outlier-m", type=float, default=0.50)
    args = parser.parse_args()

    input_npz = Path(args.input_npz).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_npz = output_dir / "wilor_handresults_phase_b.npz"
    output_csv = output_dir / "wilor_predictions_phase_b.csv"
    events_csv = output_dir / "wilor_phase_b_events.csv"
    summary_json = output_dir / "wilor_phase_b_summary.json"

    data = load_npz(input_npz)
    n = int(len(data.get("frame_index", [])))
    events: List[Dict[str, Any]] = []
    if n == 0:
        save_subset_npz(output_npz, data, np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32))
        export_predictions_csv(output_csv, data, np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32))
        summary = {"semantic": "EgoInfinity Phase-B style hand postprocess", "input_npz": str(input_npz), "candidates_in": 0, "candidates_out": 0}
        summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return 0

    frame_index = data["frame_index"].astype(np.int32)
    bbox = data["bbox_xyxy"].astype(np.float32)
    conf = data["det_conf"].astype(np.float32)
    is_right = data["is_right"].astype(np.float32).copy()
    labels = np.asarray(data["hand_label"]).astype(str).copy()
    raw_track_ids = data.get("track_id")
    if raw_track_ids is not None and len(raw_track_ids) == n and np.any(np.asarray(raw_track_ids, dtype=np.int32) >= 0):
        track_ids = np.asarray(raw_track_ids, dtype=np.int32).copy()
    else:
        track_ids = assign_tracks_like_egoinfinity(frame_index, bbox)

    frame_to_indices: Dict[int, List[int]] = defaultdict(list)
    for i, f in enumerate(frame_index.tolist()):
        frame_to_indices[int(f)].append(i)

    drop = np.zeros(n, dtype=bool)
    for frame, indices in frame_to_indices.items():
        removed, pairs = dedup_frame(indices, bbox, conf, float(args.dedup_iou))
        for idx, kept, iou in pairs:
            drop[idx] = True
            events.append(make_events_row("duplicate_drop", frame, idx, int(track_ids[idx]), f"overlap_iou>{args.dedup_iou}", kept_candidate=int(kept), iou=csv_float(iou)))

    # Majority handedness per physical track.
    track_votes: Dict[int, Counter] = defaultdict(Counter)
    for i in range(n):
        if drop[i]:
            continue
        track_votes[int(track_ids[i])][bool(is_right[i] >= 0.5)] += float(conf[i])
    track_majority: Dict[int, bool] = {}
    for tid, votes in track_votes.items():
        majority = votes.most_common(1)[0][0]
        track_majority[tid] = bool(majority)
    for i in range(n):
        tid = int(track_ids[i])
        if tid not in track_majority or drop[i]:
            continue
        majority = track_majority[tid]
        if bool(is_right[i] >= 0.5) != majority:
            old_label = labels[i]
            is_right[i] = 1.0 if majority else 0.0
            labels[i] = "right" if majority else "left"
            data["is_right"][i] = is_right[i]
            data["hand_label"][i] = labels[i]
            # EgoInfinity Phase B mirrors geometry when correcting handedness.
            data["joints_3d_rel"][i, :, 0] *= -1.0
            data["vertices_rel"][i, :, 0] *= -1.0
            data["cam_t"][i, 0] *= -1.0
            data["joints_cam"][i] = data["joints_3d_rel"][i] + data["cam_t"][i].reshape(1, 3)
            data["vertices_cam"][i] = data["vertices_rel"][i] + data["cam_t"][i].reshape(1, 3)
            events.append(make_events_row("track_majority_label_fix", int(frame_index[i]), i, tid, f"{old_label}->{labels[i]}"))

    # Track-level filtering.
    track_stats: Dict[int, Dict[str, Any]] = defaultdict(lambda: {"frames": [], "wrists": [], "bbox_areas": [], "depths": [], "is_right": None})
    for i in range(n):
        if drop[i]:
            continue
        tid = int(track_ids[i])
        ts = track_stats[tid]
        ts["frames"].append(int(frame_index[i]))
        ts["wrists"].append(np.asarray(data["cam_t"][i], dtype=np.float32).copy())
        ts["depths"].append(float(data["cam_t"][i, 2]))
        ts["bbox_areas"].append(bbox_area(bbox[i]))
        ts["is_right"] = bool(is_right[i] >= 0.5)

    dominant: Dict[bool, int] = {}
    for side in [True, False]:
        side_tracks = [(tid, ts) for tid, ts in track_stats.items() if ts["is_right"] == side]
        if side_tracks:
            dominant[side] = max(side_tracks, key=lambda item: len(item[1]["frames"]))[0]

    tracks_to_remove: Dict[int, str] = {}
    for tid, ts in track_stats.items():
        side = bool(ts["is_right"])
        count = len(ts["frames"])
        if count < int(args.min_track_frames):
            tracks_to_remove[tid] = f"short_track<{args.min_track_frames}"
            continue
        if side in dominant and dominant[side] != tid:
            dom_ts = track_stats[dominant[side]]
            dom_wrist = np.median(np.asarray(dom_ts["wrists"], dtype=np.float32), axis=0)
            my_wrist = np.median(np.asarray(ts["wrists"], dtype=np.float32), axis=0)
            dist = float(np.linalg.norm(my_wrist - dom_wrist))
            if dist > float(args.position_outlier_m) and count < len(dom_ts["frames"]) * 0.3:
                tracks_to_remove[tid] = f"position_outlier>{args.position_outlier_m}m"
                continue
            dom_depth = np.asarray(dom_ts["depths"], dtype=np.float32)
            dom_area = np.asarray(dom_ts["bbox_areas"], dtype=np.float32)
            my_depth = np.asarray(ts["depths"], dtype=np.float32)
            my_area = np.asarray(ts["bbox_areas"], dtype=np.float32)
            dom_valid = dom_depth > 0.1
            my_valid = my_depth > 0.1
            if dom_valid.sum() > 0 and my_valid.sum() > 0:
                dom_size = float(np.median(dom_area[dom_valid] / np.maximum(dom_depth[dom_valid] ** 2, 1e-6)))
                my_size = float(np.median(my_area[my_valid] / np.maximum(my_depth[my_valid] ** 2, 1e-6)))
                ratio = my_size / max(dom_size, 1e-6)
                if (ratio > 3.0 or ratio < 0.33) and count < len(dom_ts["frames"]) * 0.3:
                    tracks_to_remove[tid] = "size_outlier_ratio"

    for i in range(n):
        tid = int(track_ids[i])
        if tid in tracks_to_remove:
            drop[i] = True
    for tid, reason in tracks_to_remove.items():
        frame = min(track_stats[tid]["frames"]) if track_stats[tid]["frames"] else -1
        events.append(make_events_row("track_filter_removed", frame, -1, tid, reason, frame_count=len(track_stats[tid]["frames"])))

    keep_indices = np.asarray([i for i in range(n) if not drop[i]], dtype=np.int32)
    save_subset_npz(output_npz, data, keep_indices, track_ids)
    export_predictions_csv(output_csv, data, keep_indices, track_ids)

    event_fields = ["event_type", "frame_index", "candidate_index", "track_id", "reason", "kept_candidate", "iou", "frame_count"]
    with events_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=event_fields, extrasaction="ignore")
        writer.writeheader()
        for row in events:
            writer.writerow(row)

    summary = {
        "semantic": "EgoInfinity Phase-B style hand postprocess on LFV HandResult NPZ.",
        "input_npz": str(input_npz),
        "output_npz": str(output_npz),
        "output_predictions_csv": str(output_csv),
        "events_csv": str(events_csv),
        "candidates_in": int(n),
        "candidates_out": int(len(keep_indices)),
        "tracks_total": int(len(set(track_ids.tolist()))),
        "tracks_removed": int(len(tracks_to_remove)),
        "removed_tracks": {str(k): v for k, v in sorted(tracks_to_remove.items())},
        "event_counts": dict(Counter(row["event_type"] for row in events)),
        "dominant_tracks": {("right" if k else "left"): int(v) for k, v in dominant.items()},
        "track_summary": {
            str(tid): {
                "frame_count": len(ts["frames"]),
                "frame_start": min(ts["frames"]) if ts["frames"] else None,
                "frame_end": max(ts["frames"]) if ts["frames"] else None,
                "label": "right" if ts["is_right"] else "left",
                "removed_reason": tracks_to_remove.get(tid, ""),
            }
            for tid, ts in sorted(track_stats.items())
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
