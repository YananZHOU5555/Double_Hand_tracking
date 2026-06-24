#!/usr/bin/env python3
"""Quality checks for EgoInfinity-style LFV HandResult NPZ outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


REQUIRED_FIELDS = {
    "global_orient": (1, 3, 3),
    "hand_pose": (15, 3, 3),
    "betas": (10,),
    "joints_3d_rel": (21, 3),
    "vertices_rel": (778, 3),
    "joints_cam": (21, 3),
    "vertices_cam": (778, 3),
    "cam_t": (3,),
    "joints_uv": (21, 2),
    "bbox_xyxy": (4,),
}


def csv_float(value: float) -> str:
    return f"{float(value):.9f}" if np.isfinite(float(value)) else ""


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.keys()}


def finite_ratio(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 1.0
    return float(np.isfinite(arr).sum() / arr.size)


def safe_label(value: Any) -> str:
    return str(value).strip().lower()


def image_size(data: Dict[str, np.ndarray]) -> Tuple[int, int]:
    size = data.get("image_size")
    if size is not None and len(size) >= 2:
        return int(size[0]), int(size[1])
    return 0, 0


def per_candidate_rows(name: str, data: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
    n = int(len(data.get("frame_index", [])))
    w, h = image_size(data)
    rows: List[Dict[str, Any]] = []
    track_id = data.get("track_id", np.full((n,), -1, dtype=np.int32))
    candidate_index = data.get("candidate_index", np.arange(n, dtype=np.int32))
    for i in range(n):
        bbox = np.asarray(data["bbox_xyxy"][i], dtype=np.float32)
        uv = np.asarray(data["joints_uv"][i], dtype=np.float32)
        cam_t = np.asarray(data["cam_t"][i], dtype=np.float32)
        bbox_valid = bool(np.isfinite(bbox).all() and bbox[2] > bbox[0] and bbox[3] > bbox[1])
        bbox_in_range = False
        if bbox_valid and w > 0 and h > 0:
            margin_x = 0.25 * w
            margin_y = 0.25 * h
            bbox_in_range = bool(
                bbox[2] >= -margin_x
                and bbox[0] <= w + margin_x
                and bbox[3] >= -margin_y
                and bbox[1] <= h + margin_y
            )
        uv_finite = np.isfinite(uv).all(axis=1)
        if w > 0 and h > 0 and uv.size:
            uv_in = uv_finite & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
            uv_in_ratio = float(uv_in.sum() / max(1, len(uv)))
        else:
            uv_in_ratio = float("nan")
        hard_reasons = []
        warning_reasons = []
        if not (np.isfinite(cam_t[2]) and cam_t[2] > 0.0):
            hard_reasons.append("non_positive_cam_t_z")
        if not bbox_valid:
            hard_reasons.append("invalid_bbox")
        elif not bbox_in_range:
            hard_reasons.append("bbox_out_of_image_range")
        joints_finite = finite_ratio(np.asarray(data["joints_cam"][i]))
        verts_finite = finite_ratio(np.asarray(data["vertices_cam"][i]))
        if joints_finite < 1.0:
            hard_reasons.append("non_finite_joints_cam")
        if verts_finite < 1.0:
            hard_reasons.append("non_finite_vertices_cam")
        if np.isfinite(uv_in_ratio) and uv_in_ratio < 0.35:
            warning_reasons.append("low_uv_in_image_ratio")
        rows.append(
            {
                "stage": name,
                "row_index": i,
                "frame_index": int(data["frame_index"][i]),
                "candidate_index": int(candidate_index[i]),
                "track_id": int(track_id[i]) if len(track_id) > i else -1,
                "hand_label": safe_label(data["hand_label"][i]) if "hand_label" in data else "",
                "is_right": csv_float(float(data["is_right"][i])) if "is_right" in data else "",
                "det_conf": csv_float(float(data["det_conf"][i])) if "det_conf" in data else "",
                "cam_t_x": csv_float(float(cam_t[0])),
                "cam_t_y": csv_float(float(cam_t[1])),
                "cam_t_z": csv_float(float(cam_t[2])),
                "cam_t_z_positive": int(bool(np.isfinite(cam_t[2]) and cam_t[2] > 0.0)),
                "bbox_valid": int(bbox_valid),
                "bbox_in_range": int(bbox_in_range),
                "uv_in_image_ratio": csv_float(uv_in_ratio),
                "joints_cam_finite_ratio": csv_float(joints_finite),
                "vertices_cam_finite_ratio": csv_float(verts_finite),
                "hard_error": int(bool(hard_reasons)),
                "warning": int(bool(warning_reasons)),
                "hard_reasons": ";".join(hard_reasons),
                "warning_reasons": ";".join(warning_reasons),
            }
        )
    return rows


def summarize_stage(name: str, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
    n = int(len(data.get("frame_index", [])))
    w, h = image_size(data)
    shape_checks: Dict[str, Dict[str, Any]] = {}
    for key, tail_shape in REQUIRED_FIELDS.items():
        if key not in data:
            shape_checks[key] = {"present": False, "ok": False, "shape": None}
            continue
        arr = data[key]
        ok = arr.ndim >= 1 and tuple(arr.shape[1:]) == tuple(tail_shape)
        shape_checks[key] = {"present": True, "ok": bool(ok), "shape": list(arr.shape)}

    finite_checks = {
        key: finite_ratio(np.asarray(data[key]))
        for key in REQUIRED_FIELDS
        if key in data and np.issubdtype(np.asarray(data[key]).dtype, np.number)
    }
    labels = [safe_label(v) for v in data.get("hand_label", np.asarray([], dtype=str))]
    track_ids = data.get("track_id")
    rows = per_candidate_rows(name, data) if n else []
    cam_z = np.asarray(data.get("cam_t", np.zeros((0, 3), dtype=np.float32)))[:, 2] if n else np.zeros((0,), dtype=np.float32)
    hard_rows = [r for r in rows if r.get("hard_error") == 1]
    warning_rows = [r for r in rows if r.get("warning") == 1]
    track_summary: Dict[str, Any] = {}
    if track_ids is not None and len(track_ids) == n:
        for tid in sorted(set(int(v) for v in track_ids.tolist())):
            mask = track_ids == tid
            frames = np.asarray(data["frame_index"])[mask]
            tid_labels = [labels[i] for i in np.flatnonzero(mask).tolist()]
            track_summary[str(tid)] = {
                "frame_count": int(mask.sum()),
                "frame_start": int(frames.min()) if len(frames) else None,
                "frame_end": int(frames.max()) if len(frames) else None,
                "label_counts": dict(Counter(tid_labels)),
            }
    return {
        "stage": name,
        "candidates": n,
        "frames": int(len(set(int(v) for v in data.get("frame_index", [])))),
        "image_size": [w, h],
        "shape_checks": shape_checks,
        "all_required_shapes_ok": bool(all(v["ok"] for v in shape_checks.values())),
        "finite_ratios": finite_checks,
        "label_counts": dict(Counter(labels)),
        "cam_t_z": {
            "min": float(np.nanmin(cam_z)) if len(cam_z) else None,
            "median": float(np.nanmedian(cam_z)) if len(cam_z) else None,
            "max": float(np.nanmax(cam_z)) if len(cam_z) else None,
            "positive_count": int(np.count_nonzero(np.isfinite(cam_z) & (cam_z > 0))),
        },
        "hard_error_candidate_count": int(len(hard_rows)),
        "warning_candidate_count": int(len(warning_rows)),
        "hard_error_examples": hard_rows[:20],
        "warning_examples": warning_rows[:20],
        "track_summary": track_summary,
    }


def write_candidate_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fields = [
        "stage",
        "row_index",
        "frame_index",
        "candidate_index",
        "track_id",
        "hand_label",
        "is_right",
        "det_conf",
        "cam_t_x",
        "cam_t_y",
        "cam_t_z",
        "cam_t_z_positive",
        "bbox_valid",
        "bbox_in_range",
        "uv_in_image_ratio",
        "joints_cam_finite_ratio",
        "vertices_cam_finite_ratio",
        "hard_error",
        "warning",
        "hard_reasons",
        "warning_reasons",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def color_for_label(label: str) -> Tuple[int, int, int]:
    label = safe_label(label)
    if label == "left":
        return (255, 170, 0)
    if label == "right":
        return (0, 210, 255)
    return (210, 210, 210)


def draw_timeline(path: Path, raw: Dict[str, np.ndarray], phase_b: Dict[str, np.ndarray]) -> None:
    import cv2

    frames = []
    for data in (raw, phase_b):
        if "frame_index" in data:
            frames.extend(int(v) for v in data["frame_index"].tolist())
    if not frames:
        return
    f0, f1 = min(frames), max(frames)
    span = max(1, f1 - f0 + 1)
    scale = max(1, int(math.ceil(span / 1200)))
    width = max(240, int(math.ceil(span / scale)) + 120)
    tracks = sorted(set(int(v) for v in phase_b.get("track_id", np.asarray([], dtype=np.int32)).tolist()))
    height = 100 + max(1, len(tracks)) * 28
    img = np.full((height, width, 3), 28, dtype=np.uint8)

    cv2.putText(img, f"HandResult QA timeline frames {f0}-{f1}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(img, "raw candidates", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1, cv2.LINE_AA)
    raw_counts = defaultdict(int)
    for f in raw.get("frame_index", []):
        raw_counts[int(f)] += 1
    for f, count in raw_counts.items():
        x = 110 + int((f - f0) / scale)
        cv2.line(img, (x, 42), (x, 42 + min(18, count * 8)), (180, 180, 180), 1)

    cv2.putText(img, "phase-B", (12, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1, cv2.LINE_AA)
    track_y = {tid: 94 + k * 28 for k, tid in enumerate(tracks)}
    labels = phase_b.get("hand_label", np.asarray([], dtype=str))
    for i, f in enumerate(phase_b.get("frame_index", [])):
        tid = int(phase_b["track_id"][i]) if "track_id" in phase_b else -1
        y = track_y.get(tid, 94)
        x = 110 + int((int(f) - f0) / scale)
        color = color_for_label(str(labels[i]) if len(labels) > i else "")
        cv2.rectangle(img, (x, y - 9), (x + max(1, int(1 / scale)), y + 9), color, -1)
    for tid, y in track_y.items():
        cv2.putText(img, f"track {tid}", (12, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), img)


def load_video_frame(video: Path, frame_index: int) -> Optional[np.ndarray]:
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def draw_contact_sheet(path: Path, video: Path, data: Dict[str, np.ndarray], max_frames: int) -> None:
    import cv2

    if not video.exists() or len(data.get("frame_index", [])) == 0:
        return
    unique_frames = sorted(set(int(v) for v in data["frame_index"].tolist()))
    if len(unique_frames) > max_frames:
        idxs = np.linspace(0, len(unique_frames) - 1, max_frames).round().astype(int)
        unique_frames = [unique_frames[i] for i in idxs]
    frame_to_indices: Dict[int, List[int]] = defaultdict(list)
    for i, f in enumerate(data["frame_index"].tolist()):
        frame_to_indices[int(f)].append(i)

    tiles = []
    labels = data.get("hand_label", np.asarray([], dtype=str))
    track_ids = data.get("track_id", np.full((len(data["frame_index"]),), -1, dtype=np.int32))
    for f in unique_frames:
        img = load_video_frame(video, f)
        if img is None:
            continue
        for i in frame_to_indices[f]:
            bbox = np.asarray(data["bbox_xyxy"][i], dtype=np.float32)
            label = str(labels[i]) if len(labels) > i else ""
            color = color_for_label(label)
            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            txt = f"f={f} t={int(track_ids[i])} {label} {float(data['det_conf'][i]):.2f}"
            cv2.putText(img, txt, (x1, max(18, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)
            uv = np.asarray(data["joints_uv"][i], dtype=np.float32)
            for p in uv:
                if np.isfinite(p).all():
                    cv2.circle(img, (int(round(float(p[0]))), int(round(float(p[1])))), 2, color, -1)
        cv2.putText(img, f"frame={f}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
        tiles.append(img)
    if not tiles:
        return
    h, w = tiles[0].shape[:2]
    cols = min(4, len(tiles))
    rows = int(math.ceil(len(tiles) / cols))
    sheet = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * h : (r + 1) * h, c * w : (c + 1) * w] = tile
    cv2.imwrite(str(path), sheet)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-npz", required=True)
    parser.add_argument("--phase-b-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--video", default="")
    parser.add_argument("--max-contact-frames", type=int, default=24)
    args = parser.parse_args()

    raw_npz = Path(args.raw_npz).expanduser().resolve()
    phase_b_npz = Path(args.phase_b_npz).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    video = Path(args.video).expanduser().resolve() if args.video else None

    raw = load_npz(raw_npz)
    phase_b = load_npz(phase_b_npz)
    raw_summary = summarize_stage("raw", raw)
    phase_b_summary = summarize_stage("phase_b", phase_b)
    rows = per_candidate_rows("raw", raw) + per_candidate_rows("phase_b", phase_b)

    per_candidate_csv = output_dir / "handresults_quality_per_candidate.csv"
    summary_json = output_dir / "handresults_quality_summary.json"
    timeline_png = output_dir / "handresults_track_timeline.png"
    contact_sheet = output_dir / "handresults_phase_b_contact_sheet.jpg"

    write_candidate_csv(per_candidate_csv, rows)
    draw_timeline(timeline_png, raw, phase_b)
    if video is not None:
        draw_contact_sheet(contact_sheet, video, phase_b, int(args.max_contact_frames))

    ok = bool(
        raw_summary["all_required_shapes_ok"]
        and phase_b_summary["all_required_shapes_ok"]
        and raw_summary["hard_error_candidate_count"] == 0
        and phase_b_summary["hard_error_candidate_count"] == 0
    )
    summary = {
        "semantic": "Quality check for EgoInfinity-style LFV HandResult outputs.",
        "ok": ok,
        "raw_npz": str(raw_npz),
        "phase_b_npz": str(phase_b_npz),
        "per_candidate_csv": str(per_candidate_csv),
        "timeline_png": str(timeline_png),
        "contact_sheet": str(contact_sheet if video is not None else ""),
        "raw": raw_summary,
        "phase_b": phase_b_summary,
        "candidate_delta": int(phase_b_summary["candidates"] - raw_summary["candidates"]),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
