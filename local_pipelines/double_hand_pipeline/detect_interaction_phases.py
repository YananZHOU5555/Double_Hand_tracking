#!/usr/bin/env python3
"""Detect coarse hand-object interaction phases from table-frame TCP motion.

This is a deliberately lightweight first pass.  It does not require object
masks; it uses the regularized gripper/TCP CSV already produced by the
double-hand pipeline.  When visual object masks become stable, their contact
evidence can be added as another input without changing the output schema.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PHASES: Dict[str, Dict[str, Any]] = {
    "pre_contact_approach": {
        "id": 0,
        "color": "#2f80ed",
        "description": "free hand motion before first object contact",
    },
    "touch_contact": {
        "id": 1,
        "color": "#f2994a",
        "description": "first touch / grasp-settle contact window",
    },
    "grasp_object_motion": {
        "id": 2,
        "color": "#27ae60",
        "description": "object-coupled motion after grasp/contact",
    },
    "place_release_contact": {
        "id": 3,
        "color": "#9b51e0",
        "description": "placing / release contact window",
    },
    "post_release_retreat": {
        "id": 4,
        "color": "#828282",
        "description": "free hand motion after release/place",
    },
    "unknown": {
        "id": 9,
        "color": "#bdbdbd",
        "description": "not enough evidence",
    },
}


@dataclass
class PhaseRange:
    label: str
    start_idx: int
    end_idx: int
    confidence: float
    reason: str


def parse_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value in ("", None):
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def csv_float(value: float) -> str:
    return f"{float(value):.9f}" if np.isfinite(float(value)) else ""


def finite_stats(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return {}
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "valid", "ok"}


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return values.copy()
    window = max(1, min(int(window), len(values)))
    if window <= 1:
        return values.copy()
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    kernel = np.ones((window,), dtype=np.float64) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def read_tcp_csv(path: Path) -> Tuple[List[Dict[str, str]], np.ndarray, np.ndarray, List[str], np.ndarray]:
    rows = list(csv.DictReader(path.open("r", newline="", encoding="utf-8")))
    if not rows:
        raise RuntimeError(f"empty TCP CSV: {path}")
    fields = set(rows[0].keys())
    if {"center_x_m", "center_y_m", "center_z_m"}.issubset(fields):
        xyz_keys = ("center_x_m", "center_y_m", "center_z_m")
    elif {"table_x_m", "table_y_m", "table_z_m"}.issubset(fields):
        xyz_keys = ("table_x_m", "table_y_m", "table_z_m")
    elif {"x_table_m", "y_table_m", "z_table_m"}.issubset(fields):
        xyz_keys = ("x_table_m", "y_table_m", "z_table_m")
    else:
        raise RuntimeError(f"cannot find table-frame XYZ columns in {path}")

    keep_rows: List[Dict[str, str]] = []
    t_values: List[float] = []
    xyz_values: List[np.ndarray] = []
    states: List[str] = []
    grasp_values: List[float] = []
    for i, row in enumerate(rows):
        if "pose_valid" in row and not truthy(row.get("pose_valid")):
            continue
        xyz = np.asarray([parse_float(row.get(k)) for k in xyz_keys], dtype=np.float64)
        if not np.isfinite(xyz).all():
            continue
        elapsed = parse_float(row.get("elapsed_sec"), float("nan"))
        if not np.isfinite(elapsed):
            elapsed = parse_float(row.get("frame_index"), i) / 30.0
        keep_rows.append(row)
        t_values.append(float(elapsed))
        xyz_values.append(xyz)
        states.append(str(row.get("state", "")).strip().lower())
        grasp_values.append(parse_float(row.get("grasp_binary"), float("nan")))
    if len(keep_rows) < 3:
        raise RuntimeError(f"not enough valid TCP samples in {path}")
    order = np.argsort(np.asarray(t_values, dtype=np.float64))
    keep_rows = [keep_rows[int(i)] for i in order]
    t = np.asarray([t_values[int(i)] for i in order], dtype=np.float64)
    xyz = np.asarray([xyz_values[int(i)] for i in order], dtype=np.float64)
    states = [states[int(i)] for i in order]
    grasp = np.asarray([grasp_values[int(i)] for i in order], dtype=np.float64)
    t = t - float(t[0])
    return keep_rows, t, xyz, states, grasp


def estimate_speed(t: np.ndarray, xyz: np.ndarray, csv_rows: Sequence[Dict[str, str]]) -> np.ndarray:
    direct = np.asarray([parse_float(r.get("speed_mps")) for r in csv_rows], dtype=np.float64)
    if np.isfinite(direct).sum() >= max(3, int(0.5 * len(direct))):
        med = np.nanmedian(direct[np.isfinite(direct)])
        direct[~np.isfinite(direct)] = med
        return direct
    speed = np.zeros((len(t),), dtype=np.float64)
    dt = np.diff(t)
    dist = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    valid = np.isfinite(dt) & (dt > 1.0e-6) & np.isfinite(dist)
    speed[1:][valid] = dist[valid] / dt[valid]
    if len(speed) > 1:
        speed[0] = speed[1]
    return speed


def contiguous_ranges(mask: np.ndarray, max_gap: int = 1, min_len: int = 1) -> List[Tuple[int, int]]:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    ranges: List[Tuple[int, int]] = []
    s = int(idx[0])
    e = int(idx[0])
    for raw in idx[1:]:
        i = int(raw)
        if i <= e + max(1, int(max_gap)) + 1:
            e = i
        else:
            if e - s + 1 >= int(min_len):
                ranges.append((s, e))
            s = e = i
    if e - s + 1 >= int(min_len):
        ranges.append((s, e))
    return ranges


def state_transition_indices(states: Sequence[str], grasp: np.ndarray) -> Tuple[Optional[int], Optional[int]]:
    closed = np.zeros((len(states),), dtype=bool)
    for i, state in enumerate(states):
        if state in {"closed", "close", "grasp", "grasped"}:
            closed[i] = True
        elif state in {"open", "opened"}:
            closed[i] = False
        elif np.isfinite(grasp[i]):
            closed[i] = grasp[i] >= 0.5
    close_idx: Optional[int] = None
    release_idx: Optional[int] = None
    for i in range(1, len(closed)):
        if not closed[i - 1] and closed[i]:
            close_idx = i
            break
    if close_idx is not None:
        for i in range(max(close_idx + 1, len(closed) // 3), len(closed)):
            if closed[i - 1] and not closed[i]:
                release_idx = i
                break
    return close_idx, release_idx


def choose_contact_ranges(
    t: np.ndarray,
    xyz: np.ndarray,
    speed: np.ndarray,
    states: Sequence[str],
    grasp: np.ndarray,
    fps: float,
) -> Tuple[Tuple[int, int], Optional[Tuple[int, int]], Dict[str, Any]]:
    n = len(t)
    smooth_window = max(3, int(round(0.35 * float(fps))))
    z_s = moving_average(xyz[:, 2], smooth_window)
    speed_s = moving_average(speed, smooth_window)
    z_low = float(np.percentile(z_s, 30))
    speed_low = float(np.percentile(speed_s, 38))
    low_slow = (z_s <= z_low) & (speed_s <= speed_low)
    min_len = max(2, int(round(0.12 * float(fps))))
    ranges = contiguous_ranges(low_slow, max_gap=max(1, int(round(0.08 * float(fps)))), min_len=min_len)

    close_idx, release_idx = state_transition_indices(states, grasp)
    contact_half = max(2, int(round(0.20 * float(fps))))

    source = "low_z_slow"
    confidence = 0.62
    pick_range: Optional[Tuple[int, int]] = None
    release_range: Optional[Tuple[int, int]] = None

    if close_idx is not None:
        pick_range = (max(0, close_idx - contact_half), min(n - 1, close_idx + contact_half))
        source = "gripper_close_transition"
        confidence = 0.86
    elif ranges:
        first_limit = int(round(0.45 * (n - 1)))
        early = [r for r in ranges if r[0] <= first_limit]
        pick_range = early[0] if early else ranges[0]

    if release_idx is not None:
        release_range = (max(0, release_idx - contact_half), min(n - 1, release_idx + contact_half))
    elif ranges:
        late_start = int(round(0.45 * (n - 1)))
        late = [r for r in ranges if r[0] >= late_start]
        if late:
            release_range = late[-1]
        elif len(ranges) >= 2:
            release_range = ranges[-1]

    if pick_range is None:
        # Fallback: first low height point in the first half.
        first_half = max(2, n // 2)
        center = int(np.argmin(z_s[:first_half]))
        pick_range = (max(0, center - contact_half), min(n - 1, center + contact_half))
        source = "fallback_first_low_height"
        confidence = 0.38

    if release_range is not None and release_range[0] <= pick_range[1] + min_len:
        release_range = None

    meta = {
        "contact_source": source,
        "base_confidence": confidence,
        "fps": float(fps),
        "smooth_window_frames": int(smooth_window),
        "low_z_threshold_m": z_low,
        "low_speed_threshold_mps": speed_low,
        "low_slow_ranges": [
            {
                "start_index": int(s),
                "end_index": int(e),
                "start_frame": str(csv_frame_placeholder(s)),
                "end_frame": str(csv_frame_placeholder(e)),
                "duration_sec": float(t[e] - t[s]) if e > s else 0.0,
            }
            for s, e in ranges
        ],
        "gripper_close_index": int(close_idx) if close_idx is not None else None,
        "gripper_release_index": int(release_idx) if release_idx is not None else None,
    }
    return pick_range, release_range, meta


def csv_frame_placeholder(index: int) -> int:
    return int(index)


def build_phase_ranges(
    t: np.ndarray,
    xyz: np.ndarray,
    speed: np.ndarray,
    states: Sequence[str],
    grasp: np.ndarray,
    fps: float,
) -> Tuple[List[PhaseRange], Dict[str, Any]]:
    n = len(t)
    pick, release, meta = choose_contact_ranges(t, xyz, speed, states, grasp, fps)
    base_conf = float(meta.get("base_confidence", 0.5))
    ranges: List[PhaseRange] = []

    p0, p1 = pick
    if p0 > 0:
        ranges.append(PhaseRange("pre_contact_approach", 0, p0 - 1, max(0.25, base_conf - 0.08), "before first contact window"))
    ranges.append(PhaseRange("touch_contact", p0, p1, base_conf, str(meta.get("contact_source", "contact"))))

    if release is not None:
        r0, r1 = release
        if r0 > p1 + 1:
            ranges.append(PhaseRange("grasp_object_motion", p1 + 1, r0 - 1, max(0.30, base_conf - 0.05), "between first contact and release/place contact"))
        ranges.append(PhaseRange("place_release_contact", r0, r1, max(0.25, base_conf - 0.08), "late low/slow contact window"))
        if r1 < n - 1:
            ranges.append(PhaseRange("post_release_retreat", r1 + 1, n - 1, max(0.20, base_conf - 0.15), "after release/place contact"))
    else:
        if p1 < n - 1:
            ranges.append(PhaseRange("grasp_object_motion", p1 + 1, n - 1, max(0.25, base_conf - 0.12), "after first contact; no separate release detected"))

    # Fill any gaps defensively.
    covered = np.zeros((n,), dtype=bool)
    for r in ranges:
        covered[r.start_idx : r.end_idx + 1] = True
    for s, e in contiguous_ranges(~covered, max_gap=0, min_len=1):
        ranges.append(PhaseRange("unknown", s, e, 0.10, "gap between inferred phases"))
    ranges.sort(key=lambda r: r.start_idx)
    return ranges, meta


def per_frame_phase(
    n: int,
    ranges: Sequence[PhaseRange],
) -> Tuple[List[str], np.ndarray, List[str]]:
    labels = ["unknown"] * n
    conf = np.zeros((n,), dtype=np.float64)
    reasons = [""] * n
    for r in ranges:
        for i in range(max(0, r.start_idx), min(n - 1, r.end_idx) + 1):
            labels[i] = r.label
            conf[i] = float(r.confidence)
            reasons[i] = r.reason
    return labels, conf, reasons


def write_phase_csv(
    path: Path,
    rows: Sequence[Dict[str, str]],
    t: np.ndarray,
    xyz: np.ndarray,
    speed: np.ndarray,
    speed_s: np.ndarray,
    z_s: np.ndarray,
    labels: Sequence[str],
    conf: np.ndarray,
    reasons: Sequence[str],
) -> None:
    fields = [
        "sample_index",
        "frame_index",
        "elapsed_sec",
        "phase_id",
        "phase_label",
        "phase_confidence",
        "phase_reason",
        "phase_color",
        "x_table_m",
        "y_table_m",
        "z_table_m",
        "speed_mps",
        "speed_smooth_mps",
        "z_smooth_m",
        "state",
        "grasp_binary",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(rows):
            phase = PHASES.get(labels[i], PHASES["unknown"])
            writer.writerow(
                {
                    "sample_index": i,
                    "frame_index": row.get("frame_index", row.get("core_frame_index", i)),
                    "elapsed_sec": csv_float(float(t[i])),
                    "phase_id": int(phase["id"]),
                    "phase_label": labels[i],
                    "phase_confidence": csv_float(float(conf[i])),
                    "phase_reason": reasons[i],
                    "phase_color": str(phase["color"]),
                    "x_table_m": csv_float(float(xyz[i, 0])),
                    "y_table_m": csv_float(float(xyz[i, 1])),
                    "z_table_m": csv_float(float(xyz[i, 2])),
                    "speed_mps": csv_float(float(speed[i])),
                    "speed_smooth_mps": csv_float(float(speed_s[i])),
                    "z_smooth_m": csv_float(float(z_s[i])),
                    "state": row.get("state", ""),
                    "grasp_binary": row.get("grasp_binary", ""),
                }
            )


def range_summary(rows: Sequence[Dict[str, str]], t: np.ndarray, ranges: Sequence[PhaseRange]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in ranges:
        phase = PHASES.get(r.label, PHASES["unknown"])
        out.append(
            {
                "phase_id": int(phase["id"]),
                "phase_label": r.label,
                "description": phase["description"],
                "start_sample_index": int(r.start_idx),
                "end_sample_index": int(r.end_idx),
                "start_frame": rows[r.start_idx].get("frame_index", rows[r.start_idx].get("core_frame_index", "")),
                "end_frame": rows[r.end_idx].get("frame_index", rows[r.end_idx].get("core_frame_index", "")),
                "start_elapsed_sec": float(t[r.start_idx]),
                "end_elapsed_sec": float(t[r.end_idx]),
                "duration_sec": float(max(0.0, t[r.end_idx] - t[r.start_idx])),
                "confidence": float(r.confidence),
                "reason": r.reason,
                "color": str(phase["color"]),
            }
        )
    return out


def plot_phase_timeline(path: Path, t: np.ndarray, xyz: np.ndarray, speed_s: np.ndarray, labels: Sequence[str], ranges: Sequence[PhaseRange]) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    names = ["x", "y", "z"]
    for dim, ax in enumerate(axes[:3]):
        for r in ranges:
            color = PHASES.get(r.label, PHASES["unknown"])["color"]
            ax.axvspan(t[r.start_idx], t[r.end_idx], color=color, alpha=0.13, linewidth=0)
        ax.plot(t, xyz[:, dim], color="#111111", linewidth=1.4)
        ax.set_ylabel(f"table {names[dim]} (m)")
        ax.grid(True, alpha=0.25)
    ax = axes[3]
    for r in ranges:
        color = PHASES.get(r.label, PHASES["unknown"])["color"]
        ax.axvspan(t[r.start_idx], t[r.end_idx], color=color, alpha=0.18, linewidth=0)
    ax.plot(t, speed_s, color="#111111", linewidth=1.3)
    ax.set_ylabel("speed smooth (m/s)")
    ax.set_xlabel("elapsed sec")
    ax.grid(True, alpha=0.25)
    handles = []
    used = []
    for label in labels:
        if label not in used:
            used.append(label)
            handles.append(plt.Line2D([0], [0], color=PHASES.get(label, PHASES["unknown"])["color"], lw=5, label=label))
    axes[0].legend(handles=handles, loc="upper right")
    axes[0].set_title("Interaction phases inferred from table-frame TCP motion")
    fig.tight_layout()
    fig.savefig(str(path), dpi=160)
    plt.close(fig)


def plot_phase_3d(path: Path, xyz: np.ndarray, labels: Sequence[str], ranges: Sequence[PhaseRange]) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    for r in ranges:
        seg = xyz[r.start_idx : r.end_idx + 1]
        if len(seg) == 0:
            continue
        color = PHASES.get(r.label, PHASES["unknown"])["color"]
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=color, linewidth=2.4, label=r.label)
        ax.scatter(seg[0, 0], seg[0, 1], seg[0, 2], color=color, s=20)
    ax.scatter(xyz[0, 0], xyz[0, 1], xyz[0, 2], color="#000000", marker="o", s=45, label="start")
    ax.scatter(xyz[-1, 0], xyz[-1, 1], xyz[-1, 2], color="#000000", marker="x", s=55, label="end")
    finite = np.isfinite(xyz).all(axis=1)
    if np.any(finite):
        mins = np.min(xyz[finite], axis=0)
        maxs = np.max(xyz[finite], axis=0)
        xs = np.linspace(mins[0] - 0.04, maxs[0] + 0.04, 2)
        ys = np.linspace(mins[1] - 0.04, maxs[1] + 0.04, 2)
        xx, yy = np.meshgrid(xs, ys)
        ax.plot_surface(xx, yy, np.zeros_like(xx), color="gray", alpha=0.14, linewidth=0, shade=False)
    ax.set_xlabel("table x (m)")
    ax.set_ylabel("table y (m)")
    ax.set_zlabel("table z (m)")
    ax.set_title("Table-frame TCP by inferred interaction phase")
    # Deduplicate legend entries.
    handles, leg_labels = ax.get_legend_handles_labels()
    keep: Dict[str, Any] = {}
    for h, l in zip(handles, leg_labels):
        keep.setdefault(l, h)
    ax.legend(keep.values(), keep.keys(), loc="upper right")
    ax.view_init(elev=28, azim=-58)
    fig.tight_layout()
    fig.savefig(str(path), dpi=160)
    plt.close(fig)


def image_data_uri(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def write_html(path: Path, summary: Dict[str, Any], images: Sequence[Tuple[str, Path]]) -> None:
    css = """
body{font-family:system-ui,Arial,sans-serif;margin:24px;line-height:1.45;color:#1f2933}
code,pre{background:#f6f8fa;border-radius:6px}
pre{padding:12px;overflow:auto}
img{max-width:100%;border:1px solid #ddd;margin:8px 0 24px}
.legend{display:flex;flex-wrap:wrap;gap:10px;margin:12px 0 20px}
.pill{display:inline-flex;align-items:center;gap:6px;border:1px solid #ddd;border-radius:999px;padding:5px 10px}
.sw{width:14px;height:14px;border-radius:3px;display:inline-block}
table{border-collapse:collapse;margin:12px 0 24px}
th,td{border:1px solid #ddd;padding:6px 8px;text-align:left}
th{background:#f6f8fa}
"""
    parts = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>Interaction Phase Detection</title>",
        f"<style>{css}</style>",
        "<h1>Interaction Phase Detection</h1>",
        "<p>第一版阶段检测基于 table-frame TCP 轨迹、速度、高度和 gripper state。没有 object mask 时，接触帧是弱监督估计。</p>",
        "<div class='legend'>",
    ]
    for label, spec in PHASES.items():
        if label == "unknown":
            continue
        parts.append(f"<span class='pill'><span class='sw' style='background:{spec['color']}'></span>{label}</span>")
    parts.append("</div>")
    parts.append("<h2>Phase Ranges</h2>")
    parts.append("<table><tr><th>phase</th><th>frames</th><th>elapsed sec</th><th>duration</th><th>confidence</th><th>reason</th></tr>")
    for r in summary.get("phase_ranges", []):
        parts.append(
            "<tr>"
            f"<td>{r['phase_label']}</td>"
            f"<td>{r['start_frame']} - {r['end_frame']}</td>"
            f"<td>{r['start_elapsed_sec']:.3f} - {r['end_elapsed_sec']:.3f}</td>"
            f"<td>{r['duration_sec']:.3f}</td>"
            f"<td>{r['confidence']:.2f}</td>"
            f"<td>{r['reason']}</td>"
            "</tr>"
        )
    parts.append("</table>")
    parts.append("<h2>Summary JSON</h2>")
    parts.append("<pre>" + json.dumps(summary, indent=2, ensure_ascii=False) + "</pre>")
    for title, img in images:
        parts.append(f"<h2>{title}</h2>")
        parts.append(f"<img src='{image_data_uri(img)}'>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tcp-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="interaction_phases")
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    tcp_csv = Path(args.tcp_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, t, xyz, states, grasp = read_tcp_csv(tcp_csv)
    speed = estimate_speed(t, xyz, rows)
    smooth_window = max(3, int(round(0.35 * float(args.fps))))
    speed_s = moving_average(speed, smooth_window)
    z_s = moving_average(xyz[:, 2], smooth_window)
    ranges, phase_meta = build_phase_ranges(t, xyz, speed, states, grasp, float(args.fps))
    labels, conf, reasons = per_frame_phase(len(rows), ranges)

    csv_path = output_dir / f"{args.prefix}.csv"
    json_path = output_dir / f"{args.prefix}.json"
    timeline_png = output_dir / f"{args.prefix}_timeline.png"
    table3d_png = output_dir / f"{args.prefix}_3d.png"
    html_path = output_dir / f"{args.prefix}.html"

    write_phase_csv(csv_path, rows, t, xyz, speed, speed_s, z_s, labels, conf, reasons)
    plot_phase_timeline(timeline_png, t, xyz, speed_s, labels, ranges)
    plot_phase_3d(table3d_png, xyz, labels, ranges)

    phase_ranges = range_summary(rows, t, ranges)
    summary = {
        "schema": "lfv_interaction_phases_v1",
        "semantic": "Coarse hand-object interaction phase detection from regularized table-frame TCP motion.",
        "tcp_csv": str(tcp_csv),
        "frame": "table_frame",
        "method": {
            "version": "kinematic_low_z_low_speed_v1",
            "inputs": ["center_xyz", "speed_mps", "state", "grasp_binary"],
            "limits": [
                "Without object masks, touch/place contact is inferred from low-height low-speed windows and gripper transitions.",
                "If active trim removed pre-contact frames, pre_contact_approach can only cover the remaining visible samples.",
            ],
            **phase_meta,
        },
        "samples": int(len(rows)),
        "duration_sec": float(t[-1] - t[0]),
        "phase_ranges": phase_ranges,
        "phase_counts": {label: int(sum(1 for x in labels if x == label)) for label in sorted(set(labels))},
        "xyz_table_m": {
            "x": finite_stats(xyz[:, 0]),
            "y": finite_stats(xyz[:, 1]),
            "z": finite_stats(xyz[:, 2]),
        },
        "speed_mps": finite_stats(speed),
        "outputs": {
            "phase_csv": str(csv_path),
            "phase_json": str(json_path),
            "phase_html": str(html_path),
            "timeline_png": str(timeline_png),
            "table3d_png": str(table3d_png),
        },
    }
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_html(html_path, summary, [("Phase Timeline", timeline_png), ("3D TCP Phase View", table3d_png)])

    print("[interaction_phases] csv:", csv_path)
    print("[interaction_phases] json:", json_path)
    print("[interaction_phases] html:", html_path)
    for r in phase_ranges:
        print(
            f"[interaction_phases] {r['phase_label']}: "
            f"frames {r['start_frame']}..{r['end_frame']} "
            f"t={r['start_elapsed_sec']:.3f}..{r['end_elapsed_sec']:.3f}s "
            f"conf={r['confidence']:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
