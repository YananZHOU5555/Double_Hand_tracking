#!/usr/bin/env python3
"""Visualize Phase-C bad alignment frames.

Outputs:
  - phase_c_bad_alignment_rows.csv: rows whose qc_flag contains "bad_"
  - phase_c_bad_alignment_overlay.mp4: bad frames plus temporal context
  - phase_c_bad_alignment_contact_sheet.jpg: bad frames only
  - bad_frame_images/*.jpg: individual annotated bad frames
  - phase_c_bad_alignment_viz_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def csv_float(value: float) -> str:
    return f"{float(value):.9f}" if math.isfinite(float(value)) else ""


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            row["row_index"] = str(i)
            rows.append(row)
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_int_list(text: str) -> List[int]:
    out: List[int] = []
    for part in str(text or "").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            pass
    return out


def color_for_label(label: str, bad: bool = False, warn: bool = False) -> Tuple[int, int, int]:
    if bad:
        return (40, 40, 255)
    if warn:
        return (0, 220, 255)
    label = str(label).strip().lower()
    if label == "left":
        return (255, 160, 40)
    if label == "right":
        return (60, 220, 60)
    return (220, 220, 220)


def put_text(
    img: np.ndarray,
    text: str,
    org: Tuple[int, int],
    color: Tuple[int, int, int] = (240, 240, 240),
    scale: float = 0.45,
    thickness: int = 1,
) -> None:
    x, y = org
    cv2.putText(img, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def read_frame(cap: cv2.VideoCapture, frame_index: int) -> Optional[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    return frame if ok and frame is not None else None


def row_bool(row: Dict[str, str], needle: str) -> bool:
    return needle in str(row.get("qc_flag", ""))


def draw_candidate(
    img: np.ndarray,
    data: Dict[str, np.ndarray],
    row: Dict[str, str],
    row_index: int,
    highlight: bool,
) -> None:
    h, w = img.shape[:2]
    bad = row_bool(row, "bad_")
    warn = row_bool(row, "warn_")
    color = color_for_label(row.get("hand_label", ""), bad=bad, warn=warn)
    if not highlight and not bad and not warn:
        color = tuple(int(v * 0.55) for v in color)

    bbox = np.asarray(data["bbox_xyxy"][row_index], dtype=np.float32)
    if np.isfinite(bbox).all():
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        x1, x2 = max(-200, x1), min(w + 200, x2)
        y1, y2 = max(-200, y1), min(h + 200, y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3 if bad else 2)
        label = (
            f"r{row.get('hand_rank')} t{row.get('track_id')} {row.get('hand_label')} "
            f"{row.get('diagnosis_category') or row.get('alignment_source')} "
            f"rms={safe_float(row.get('alignment_rms_m')):.3f}"
        )
        put_text(img, label, (max(4, x1), max(18, y1 - 7)), color, 0.42, 1)

    uv = np.asarray(data["joints_uv"][row_index], dtype=np.float32)
    sampled_ids = set(parse_int_list(row.get("alignment_joint_ids", "")))
    source = str(row.get("alignment_source", ""))
    sample_color = (255, 255, 0) if source == "depth_reliable_joints" else (255, 0, 255)

    for a, b in HAND_EDGES:
        if a >= len(uv) or b >= len(uv):
            continue
        pa, pb = uv[a], uv[b]
        if np.isfinite(pa).all() and np.isfinite(pb).all():
            cv2.line(
                img,
                (int(round(float(pa[0]))), int(round(float(pa[1])))),
                (int(round(float(pb[0]))), int(round(float(pb[1])))),
                color,
                2 if bad else 1,
                cv2.LINE_AA,
            )
    for jid, p in enumerate(uv):
        if not np.isfinite(p).all():
            continue
        x, y = int(round(float(p[0]))), int(round(float(p[1])))
        if jid in sampled_ids:
            cv2.circle(img, (x, y), 6 if bad else 5, sample_color, 2, cv2.LINE_AA)
            put_text(img, str(jid), (x + 5, y - 5), sample_color, 0.36, 1)
        else:
            cv2.circle(img, (x, y), 2, color, -1, cv2.LINE_AA)


def make_canvas(frame: np.ndarray, frame_index: int, rows: Sequence[Dict[str, str]], bad_frame: bool) -> np.ndarray:
    h, w = frame.shape[:2]
    header_h = 132
    canvas = np.zeros((h + header_h, w, 3), dtype=np.uint8)
    canvas[:header_h] = (18, 18, 18)
    canvas[header_h:] = frame

    bad_rows = [r for r in rows if row_bool(r, "bad_")]
    warn_rows = [r for r in rows if row_bool(r, "warn_")]
    title_color = (0, 255, 255) if bad_frame else (200, 200, 200)
    put_text(
        canvas,
        f"Phase-C alignment audit | frame={frame_index} | bad={len(bad_rows)} warn={len(warn_rows)} candidates={len(rows)}",
        (12, 28),
        title_color,
        0.65,
        2,
    )
    put_text(
        canvas,
        "cyan circle = reliable sampled joint, magenta circle = all-joint fallback sample, red bbox = bad alignment",
        (12, 58),
        (230, 230, 230),
        0.47,
        1,
    )
    y = 84
    for r in bad_rows[:2]:
        msg = (
            f"BAD row={r.get('row_index')} {r.get('hand_label')} track={r.get('track_id')} "
            f"diag={r.get('diagnosis_category', '')} src={r.get('alignment_source')} valid={r.get('alignment_valid_joint_count')} "
            f"rms={safe_float(r.get('alignment_rms_m')):.3f} max={safe_float(r.get('alignment_max_residual_m')):.3f} "
            f"jump={safe_float(r.get('cam_t_jump_m')):.3f}"
        )
        put_text(canvas, msg, (12, y), (50, 80, 255), 0.44, 1)
        y += 22
    if len(bad_rows) > 2:
        put_text(canvas, f"... {len(bad_rows) - 2} more bad candidates on this frame", (12, y), (50, 80, 255), 0.44, 1)
    return canvas


def draw_frame_overlay(
    frame: np.ndarray,
    frame_index: int,
    rows: Sequence[Dict[str, str]],
    data: Dict[str, np.ndarray],
    bad_frame: bool,
) -> np.ndarray:
    canvas = make_canvas(frame.copy(), frame_index, rows, bad_frame)
    header_h = canvas.shape[0] - frame.shape[0]
    image_area = canvas[header_h:]
    for row in rows:
        row_index = int(row["row_index"])
        draw_candidate(image_area, data, row, row_index, highlight=bad_frame)
    return canvas


def resize_tile(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w == width:
        return img
    scale = float(width) / max(1, w)
    return cv2.resize(img, (width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)


def write_contact_sheet(path: Path, images: Sequence[np.ndarray], cols: int = 3, tile_width: int = 520) -> None:
    if not images:
        return
    tiles = [resize_tile(img, tile_width) for img in images]
    th = max(t.shape[0] for t in tiles)
    tw = tile_width
    cols = max(1, min(cols, len(tiles)))
    rows = int(math.ceil(len(tiles) / cols))
    sheet = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        y0 = r * th
        x0 = c * tw
        sheet[y0:y0 + tile.shape[0], x0:x0 + tile.shape[1]] = tile
    cv2.imwrite(str(path), sheet)


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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session-dir", required=True)
    p.add_argument("--phase-c-npz", required=True)
    p.add_argument("--alignment-csv", required=True)
    p.add_argument("--video", default="")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--context-frames", type=int, default=2)
    p.add_argument("--max-sheet-frames", type=int, default=80)
    p.add_argument("--fps", type=float, default=6.0)
    args = p.parse_args()

    session_dir = Path(args.session_dir).expanduser().resolve()
    phase_c_npz = Path(args.phase_c_npz).expanduser().resolve()
    alignment_csv = Path(args.alignment_csv).expanduser().resolve()
    video = Path(args.video).expanduser().resolve() if args.video else session_dir / "processed_topcam" / "left_table.mp4"
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "bad_frame_images"
    images_dir.mkdir(parents=True, exist_ok=True)

    data = load_npz(phase_c_npz)
    rows = read_csv_rows(alignment_csv)
    if len(rows) != len(data.get("frame_index", [])):
        raise RuntimeError(f"alignment csv rows != phase-c npz candidates: {len(rows)} != {len(data.get('frame_index', []))}")

    bad_rows = [r for r in rows if row_bool(r, "bad_")]
    warn_rows = [r for r in rows if row_bool(r, "warn_")]
    bad_frame_set = {int(r["frame_index"]) for r in bad_rows}
    frames_for_video = set(bad_frame_set)
    for frame in list(bad_frame_set):
        for delta in range(-int(args.context_frames), int(args.context_frames) + 1):
            if frame + delta >= 0:
                frames_for_video.add(frame + delta)
    frames_for_video_sorted = sorted(frames_for_video)
    bad_frames_sorted = sorted(bad_frame_set)

    rows_by_frame: Dict[int, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_frame[int(row["frame_index"])].append(row)

    bad_csv = output_dir / "phase_c_bad_alignment_rows.csv"
    overlay_mp4 = output_dir / "phase_c_bad_alignment_overlay.mp4"
    contact_sheet = output_dir / "phase_c_bad_alignment_contact_sheet.jpg"
    summary_json = output_dir / "phase_c_bad_alignment_viz_summary.json"

    write_csv(bad_csv, bad_rows)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video}")
    video_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer: Optional[cv2.VideoWriter] = None
    contact_images: List[np.ndarray] = []
    written_overlay_frames = 0
    written_bad_images = 0

    for frame_index in frames_for_video_sorted:
        if frame_index >= frame_count:
            continue
        frame = read_frame(cap, frame_index)
        if frame is None:
            continue
        canvas = draw_frame_overlay(
            frame,
            frame_index,
            rows_by_frame.get(frame_index, []),
            data,
            frame_index in bad_frame_set,
        )
        if writer is None:
            h, w = canvas.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(overlay_mp4), fourcc, float(args.fps), (w, h))
        writer.write(canvas)
        written_overlay_frames += 1

        if frame_index in bad_frame_set:
            out_img = images_dir / f"phase_c_bad_frame_{frame_index:08d}.jpg"
            cv2.imwrite(str(out_img), canvas)
            written_bad_images += 1
            if len(contact_images) < int(args.max_sheet_frames):
                contact_images.append(canvas)

    cap.release()
    if writer is not None:
        writer.release()
    write_contact_sheet(contact_sheet, contact_images)

    source_counts = Counter(r.get("alignment_source", "") for r in bad_rows)
    label_counts = Counter(r.get("hand_label", "") for r in bad_rows)
    flag_counts = Counter(r.get("qc_flag", "") for r in bad_rows)
    diagnosis_counts = Counter(r.get("diagnosis_category", "") for r in bad_rows)
    issue_tag_counts = Counter()
    for row in bad_rows:
        issue_tag_counts.update(tag for tag in str(row.get("qc_issue_tags", "")).split("|") if tag)
    summary = {
        "semantic": "Phase-C bad alignment frame visualization",
        "session_dir": str(session_dir),
        "phase_c_npz": str(phase_c_npz),
        "alignment_csv": str(alignment_csv),
        "video": str(video),
        "video_frame_count": int(frame_count),
        "video_fps": float(video_fps),
        "total_candidates": int(len(rows)),
        "bad_candidate_count": int(len(bad_rows)),
        "warn_candidate_count": int(len(warn_rows)),
        "bad_unique_frame_count": int(len(bad_frame_set)),
        "bad_frame_min": int(min(bad_frame_set)) if bad_frame_set else None,
        "bad_frame_max": int(max(bad_frame_set)) if bad_frame_set else None,
        "bad_by_label": dict(label_counts),
        "bad_by_alignment_source": dict(source_counts),
        "bad_by_flag": dict(flag_counts),
        "bad_by_diagnosis_category": dict(diagnosis_counts),
        "bad_by_issue_tag": dict(issue_tag_counts),
        "bad_alignment_rms_m": stats(safe_float(r.get("alignment_rms_m")) for r in bad_rows),
        "bad_cam_t_jump_m": stats(safe_float(r.get("cam_t_jump_m")) for r in bad_rows),
        "context_frames": int(args.context_frames),
        "overlay_frame_count": int(written_overlay_frames),
        "bad_image_count": int(written_bad_images),
        "outputs": {
            "bad_rows_csv": str(bad_csv),
            "overlay_mp4": str(overlay_mp4),
            "contact_sheet_jpg": str(contact_sheet),
            "bad_frame_images_dir": str(images_dir),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
