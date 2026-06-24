#!/usr/bin/env python3
"""Glue utilities for the QZY local WiLoR fallback pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


LM_THUMB_TIP = 4
LM_INDEX_TIP = 8


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(round(float(value)))
    except Exception:
        return default


def fmt_float(value: Any) -> str:
    v = safe_float(value)
    return f"{v:.9f}" if math.isfinite(v) else ""


def read_csv(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if not fields:
        raise RuntimeError(f"empty or headerless CSV: {path}")
    return rows, fields


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def frame_of(row: Dict[str, Any]) -> int:
    return safe_int(row.get("frame_index"), -1)


def filter_frame_rows(path: Path, start: int, end: int, output: Path) -> Dict[str, Any]:
    rows, fields = read_csv(path)
    kept = [row for row in rows if start <= frame_of(row) <= end]
    write_csv(output, kept, fields)
    return {
        "input": str(path),
        "output": str(output),
        "rows_in": len(rows),
        "rows_out": len(kept),
        "frame_start": start,
        "frame_end": end,
    }


def cmd_trim(args: argparse.Namespace) -> None:
    report = json.loads(Path(args.hand21_report).read_text(encoding="utf-8"))
    active = report.get("detected_active_segment") or {}
    if "start_frame" not in active or "end_frame" not in active:
        raise RuntimeError(f"detected_active_segment missing in {args.hand21_report}")
    base_start = int(active["start_frame"])
    base_end = int(active["end_frame"])
    trim = int(args.trim_frames)
    start = base_start + trim
    end = base_end - trim
    if start > end:
        raise RuntimeError(f"empty active trim: active={base_start}..{base_end}, trim={trim}")

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    hand_out = out_dir / f"{Path(args.hand21_csv).stem}_active_trim{trim}.csv"
    wilor_out = out_dir / f"{Path(args.wilor_predictions_csv).stem}_active_trim{trim}.csv"
    meta_out = out_dir / "active_trim_metadata.json"

    summary = {
        "semantic": "Detected active segment with fixed boundary trim for QZY WiLoR fallback pipeline.",
        "detected_active_segment": active,
        "trim_frames_each_side": trim,
        "trimmed_frame_start": start,
        "trimmed_frame_end": end,
        "hand21": filter_frame_rows(Path(args.hand21_csv).expanduser().resolve(), start, end, hand_out),
        "wilor_predictions": filter_frame_rows(
            Path(args.wilor_predictions_csv).expanduser().resolve(),
            start,
            end,
            wilor_out,
        ),
    }
    meta_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def load_hand21_points(path: Path) -> Dict[int, Dict[int, np.ndarray]]:
    rows, _ = read_csv(path)
    out: Dict[int, Dict[int, np.ndarray]] = {}
    for row in rows:
        if str(row.get("valid", "1")).strip().lower() in {"0", "false"}:
            continue
        frame = frame_of(row)
        lm = safe_int(row.get("landmark_id"), -1)
        if frame < 0 or not (0 <= lm < 21):
            continue
        x_key = "table_x_m" if "table_x_m" in row else "x_m"
        y_key = "table_y_m" if "table_y_m" in row else "y_m"
        z_key = "table_z_m" if "table_z_m" in row else "z_m"
        xyz = np.asarray(
            [safe_float(row.get(x_key)), safe_float(row.get(y_key)), safe_float(row.get(z_key))],
            dtype=np.float64,
        )
        if np.isfinite(xyz).all():
            out.setdefault(frame, {})[lm] = xyz
    return out


def by_frame(rows: Iterable[Dict[str, str]]) -> Dict[int, Dict[str, str]]:
    out: Dict[int, Dict[str, str]] = {}
    for row in rows:
        frame = frame_of(row)
        if frame >= 0:
            out[frame] = row
    return out


def cmd_audit(args: argparse.Namespace) -> None:
    constrained_csv = Path(args.constrained_hand21_csv).expanduser().resolve()
    per_frame_csv = Path(args.per_frame_csv).expanduser().resolve()
    output = Path(args.output_csv).expanduser().resolve()
    points = load_hand21_points(constrained_csv)
    per_rows, _ = read_csv(per_frame_csv)
    per = by_frame(per_rows)

    fields = [
        "frame_index",
        "elapsed_sec",
        "valid_pose",
        "candidate_valid",
        "fit_source",
        "align_points",
        "align_rms_m",
        "scale_raw",
        "scale_smooth",
        "translation_step_m",
        "reason",
        "jaw_width_m",
    ]
    rows: List[Dict[str, Any]] = []
    for frame in sorted(points):
        meta = per.get(frame, {})
        thumb = points[frame].get(LM_THUMB_TIP)
        index = points[frame].get(LM_INDEX_TIP)
        jaw = float(np.linalg.norm(index - thumb)) if thumb is not None and index is not None else float("nan")
        rows.append(
            {
                "frame_index": frame,
                "elapsed_sec": meta.get("elapsed_sec", fmt_float(frame / float(args.fps))),
                "valid_pose": "1" if math.isfinite(jaw) else "0",
                "candidate_valid": meta.get("candidate_valid", ""),
                "fit_source": meta.get("source", ""),
                "align_points": meta.get("align_points", ""),
                "align_rms_m": meta.get("align_rms_m", ""),
                "scale_raw": meta.get("scale_raw", ""),
                "scale_smooth": meta.get("scale_smooth", ""),
                "translation_step_m": meta.get("translation_step_m", ""),
                "reason": meta.get("reason", ""),
                "jaw_width_m": fmt_float(jaw),
            }
        )
    write_csv(output, rows, fields)
    summary = {
        "semantic": "Minimal per-frame audit metadata for WiLoR MCP-X core export.",
        "constrained_hand21_csv": str(constrained_csv),
        "per_frame_csv": str(per_frame_csv),
        "output_csv": str(output),
        "rows": len(rows),
        "frame_min": rows[0]["frame_index"] if rows else None,
        "frame_max": rows[-1]["frame_index"] if rows else None,
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def add_fields(fields: List[str], extras: Sequence[str]) -> List[str]:
    out = list(fields)
    for field in extras:
        if field not in out:
            out.append(field)
    return out


def cmd_best_inputs(args: argparse.Namespace) -> None:
    tcp_rows, tcp_fields = read_csv(Path(args.tcp_core_csv).expanduser().resolve())
    state_rows, state_fields = read_csv(Path(args.v0_state_core_csv).expanduser().resolve())
    ori_rows, _ = read_csv(Path(args.orientation_core_csv).expanduser().resolve())
    tcp_by = by_frame(tcp_rows)
    state_by = by_frame(state_rows)
    ori_by = by_frame(ori_rows)
    frames = sorted(set(tcp_by) & set(state_by) & set(ori_by))
    if not frames:
        raise RuntimeError("no common frames between TCP, V0 state, and orientation core CSVs")

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(args.prefix)
    tcp_out = out_dir / f"gripper_pose_table_core_{prefix}_tcp.csv"
    state_ori_out = out_dir / f"gripper_pose_table_core_{prefix}_state_orientation.csv"
    summary_out = out_dir / f"{prefix}_best_inputs_summary.json"

    tcp_kept = [dict(tcp_by[f]) for f in frames]
    state_ori: List[Dict[str, Any]] = []
    for idx, frame in enumerate(frames):
        state = dict(state_by[frame])
        ori = ori_by[frame]
        state["core_frame_index"] = idx
        state["frame_index"] = frame
        for key in ("yaw_rad", "jaw_yaw_rad", "roll_deg", "pitch_deg", "yaw_deg"):
            if key in ori:
                state[key] = ori[key]
        for j in range(6):
            state[f"rot6d_{j}"] = ori.get(f"rot6d_{j}", "")
        state["pose_source"] = "v0_state+qzy_wilor_index_middle_thumb_index_yz_tilt_gm1p0_orientation"
        state["orientation_mode"] = "wilor_v2_index_middle_thumb_index_yz_tilt_so3"
        state["orientation_source_csv"] = str(Path(args.orientation_core_csv).expanduser().resolve())
        state["model_orientation_mode"] = ori.get("orientation_mode", "")
        state["model_pose_source"] = ori.get("pose_source", "")
        state["model_align_rms_m"] = ori.get("align_rms_m", "")
        state["model_scale_smooth"] = ori.get("scale_smooth", "")
        state["model_fit_source"] = ori.get("fit_source", "")
        state_ori.append(state)

    state_fields_out = add_fields(
        state_fields,
        [
            "orientation_source_csv",
            "model_orientation_mode",
            "model_pose_source",
            "model_align_rms_m",
            "model_scale_smooth",
            "model_fit_source",
            "roll_deg",
            "pitch_deg",
            "yaw_deg",
        ],
    )
    write_csv(tcp_out, tcp_kept, tcp_fields)
    write_csv(state_ori_out, state_ori, state_fields_out)

    state_counts: Dict[str, int] = {}
    for row in state_ori:
        state = str(row.get("state", "")).strip().lower() or "unknown"
        state_counts[state] = state_counts.get(state, 0) + 1
    summary = {
        "semantic": "QZY best local pipeline inputs: WiLoR TCP plus V0 state and index-middle/thumb-YZ orientation.",
        "tcp_core_csv": str(Path(args.tcp_core_csv).expanduser().resolve()),
        "v0_state_core_csv": str(Path(args.v0_state_core_csv).expanduser().resolve()),
        "orientation_core_csv": str(Path(args.orientation_core_csv).expanduser().resolve()),
        "tcp_output_csv": str(tcp_out),
        "state_orientation_output_csv": str(state_ori_out),
        "rows": len(frames),
        "frame_min": int(frames[0]),
        "frame_max": int(frames[-1]),
        "state_counts": state_counts,
    }
    summary_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    trim = sub.add_parser("trim-active")
    trim.add_argument("--hand21-report", required=True)
    trim.add_argument("--hand21-csv", required=True)
    trim.add_argument("--wilor-predictions-csv", required=True)
    trim.add_argument("--output-dir", required=True)
    trim.add_argument("--trim-frames", type=int, default=20)
    trim.set_defaults(func=cmd_trim)

    audit = sub.add_parser("make-audit")
    audit.add_argument("--constrained-hand21-csv", required=True)
    audit.add_argument("--per-frame-csv", required=True)
    audit.add_argument("--output-csv", required=True)
    audit.add_argument("--fps", type=float, default=60.0)
    audit.set_defaults(func=cmd_audit)

    best = sub.add_parser("build-best-inputs")
    best.add_argument("--tcp-core-csv", required=True)
    best.add_argument("--v0-state-core-csv", required=True)
    best.add_argument("--orientation-core-csv", required=True)
    best.add_argument("--output-dir", required=True)
    best.add_argument("--prefix", default="qzy_wilor_best_gm1p0_button_up")
    best.set_defaults(func=cmd_best_inputs)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

