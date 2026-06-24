#!/usr/bin/env python3
"""Export Phase-C WiLoR/MANO hand skeletons in LFV table frame as HTML."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

LANDMARK_NAMES = [
    "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-npz", required=True)
    p.add_argument("--table-frame-json", default="/home/yannan/workspace/ros1_docker-main/data/lfv_calibration/table_frame_latest.json")
    p.add_argument("--output-html", required=True)
    p.add_argument("--output-json", default="")
    p.add_argument("--video", default="")
    p.add_argument("--stage-name", default="")
    p.add_argument("--joints-key", default="auto")
    p.add_argument("--uv-key", default="auto")
    p.add_argument(
        "--infilled-uv-key",
        default="auto",
        help="UV source for motion-infilled rows when --reproject-from-uv-depth is used. Use none to keep --uv-key.",
    )
    p.add_argument("--max-uv-hand-size-px", type=float, default=900.0)
    p.add_argument(
        "--reproject-from-uv-depth",
        action="store_true",
        help="Build camera-frame 3D from selected 2D UV and selected joints camera Z before table transform.",
    )
    p.add_argument("--frame-start", type=int, default=-1)
    p.add_argument("--frame-end", type=int, default=-1)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-candidates", type=int, default=0)
    p.add_argument("--prefer-scaled-table-transform", action="store_true")
    return p.parse_args()


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_clean(value.tolist())
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def rel_or_abs(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except Exception:
        return str(path.resolve())


def choose_joints_key(npz: np.lib.npyio.NpzFile, requested: str) -> str:
    if requested and requested != "auto":
        if requested not in npz.files:
            raise RuntimeError(f"missing requested joints key: {requested}")
        return requested
    for key in ("joints_cam_visibility_depth", "joints_cam_smooth", "joints_cam_depth_smooth", "joints_cam_depth", "joints_cam"):
        if key in npz.files:
            return key
    raise RuntimeError("no usable joints_cam key found")


def choose_uv_key(npz: np.lib.npyio.NpzFile, requested: str, joints_key: str) -> str:
    if requested and requested != "auto":
        if requested not in npz.files:
            raise RuntimeError(f"missing requested uv key: {requested}")
        return requested
    if joints_key == "joints_cam":
        candidates = ("joints_uv", "joints_uv_smooth_depth_camera")
    else:
        candidates = ("joints_uv_smooth_depth_camera", "joints_uv")
    for key in candidates:
        if key in npz.files:
            return key
    raise RuntimeError("no usable joints_uv key found")


def choose_infilled_uv_key(npz: np.lib.npyio.NpzFile, requested: str, primary_uv_key: str) -> str:
    requested = str(requested or "auto")
    if requested.lower() in ("none", "off", "false", "0"):
        return ""
    if requested != "auto":
        if requested not in npz.files:
            raise RuntimeError(f"missing requested infilled uv key: {requested}")
        return requested
    for key in ("joints_uv_smooth_depth_camera", "joints_uv"):
        if key in npz.files and key != primary_uv_key:
            return key
    return ""


def uv_hand_size(uv: np.ndarray) -> float:
    valid = np.isfinite(uv).all(axis=1)
    if int(np.sum(valid)) < 2:
        return float("nan")
    pts = uv[valid]
    return float(max(np.max(pts[:, 0]) - np.min(pts[:, 0]), np.max(pts[:, 1]) - np.min(pts[:, 1])))


def replace_bad_or_infilled_uv(
    primary_uv: np.ndarray,
    replacement_uv: np.ndarray,
    motion_infilled: np.ndarray,
    max_size_px: float,
) -> Tuple[np.ndarray, int, int]:
    uv = np.asarray(primary_uv, dtype=np.float64).copy()
    repl = np.asarray(replacement_uv, dtype=np.float64)
    replaced_infilled = 0
    replaced_oversize = 0
    for i in range(uv.shape[0]):
        size = uv_hand_size(uv[i])
        use_replacement = False
        if bool(motion_infilled[i]):
            use_replacement = True
            replaced_infilled += 1
        elif math.isfinite(size) and size > float(max_size_px):
            use_replacement = True
            replaced_oversize += 1
        if use_replacement and repl.shape[:2] == uv.shape[:2] and np.isfinite(repl[i]).any():
            uv[i] = repl[i]
    return uv, replaced_infilled, replaced_oversize


def load_camera(npz: np.lib.npyio.NpzFile) -> Dict[str, float]:
    if "foundation_camera_json" in npz.files and len(npz["foundation_camera_json"]):
        try:
            raw = str(np.asarray(npz["foundation_camera_json"]).reshape(-1)[0])
            data = json.loads(raw)
            return {k: float(data[k]) for k in ("fx", "fy", "cx", "cy")}
        except Exception:
            pass
    if "image_size" in npz.files:
        size = np.asarray(npz["image_size"]).reshape(-1)
        width, height = float(size[0]), float(size[1])
    else:
        width, height = 850.0, 420.0
    return {"fx": 334.0549853304012, "fy": 334.0549853304012, "cx": width * 0.5, "cy": height * 0.5}


def backproject_uv_with_depth(uv: np.ndarray, depth_z: np.ndarray, camera: Dict[str, float]) -> np.ndarray:
    fx, fy, cx, cy = [float(camera[k]) for k in ("fx", "fy", "cx", "cy")]
    out = np.full((uv.shape[0], uv.shape[1], 3), np.nan, dtype=np.float64)
    z = np.asarray(depth_z, dtype=np.float64)
    valid = np.isfinite(uv).all(axis=2) & np.isfinite(z) & (z > 1e-6)
    out[..., 2] = z
    out[..., 0] = (np.asarray(uv[..., 0], dtype=np.float64) - cx) * z / fx
    out[..., 1] = (np.asarray(uv[..., 1], dtype=np.float64) - cy) * z / fy
    out[~valid] = np.nan
    return out


def load_table_transform(path: Path, prefer_scaled: bool) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    key = "T_table_from_camera_scaled" if prefer_scaled and "T_table_from_camera_scaled" in payload else "T_table_from_camera"
    if key not in payload:
        raise RuntimeError(f"missing {key} in {path}")
    transform = np.asarray(payload[key], dtype=np.float64)
    if transform.shape != (4, 4):
        raise RuntimeError(f"bad table transform shape: {transform.shape}")
    return transform


def transform_points(transform: np.ndarray, points_cam: np.ndarray) -> np.ndarray:
    flat = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
    hom = np.concatenate([flat, np.ones((flat.shape[0], 1), dtype=np.float64)], axis=1)
    out = (transform @ hom.T).T[:, :3]
    out[~np.isfinite(flat).all(axis=1)] = np.nan
    return out.reshape(points_cam.shape)


def scalar_array(npz: np.lib.npyio.NpzFile, key: str, length: int, default: Any) -> np.ndarray:
    if key in npz.files:
        arr = np.asarray(npz[key])
        if arr.shape[0] == length:
            return arr
    return np.asarray([default] * length)


def string_at(arr: np.ndarray, index: int, default: str = "") -> str:
    try:
        return str(arr[index])
    except Exception:
        return default


def number_at(arr: np.ndarray, index: int, default: float = float("nan")) -> float:
    try:
        return float(arr[index])
    except Exception:
        return default


def summarize(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray([v for v in values if math.isfinite(float(v))], dtype=np.float64)
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


def compute_bounds(frames: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    pts: List[List[float]] = []
    for frame in frames:
        for cand in frame["candidates"]:
            for point in cand["points"].values():
                if all(math.isfinite(float(v)) for v in point):
                    pts.append(point)
    if not pts:
        pts = [[0.0, 0.0, 0.0]]
    arr = np.asarray(pts, dtype=np.float64)
    if arr.shape[0] >= 100:
        mn = np.nanpercentile(arr, 1.0, axis=0) - np.asarray([0.08, 0.08, 0.05], dtype=np.float64)
        mx = np.nanpercentile(arr, 99.0, axis=0) + np.asarray([0.08, 0.08, 0.07], dtype=np.float64)
    else:
        mn = np.nanmin(arr, axis=0) - np.asarray([0.08, 0.08, 0.05], dtype=np.float64)
        mx = np.nanmax(arr, axis=0) + np.asarray([0.08, 0.08, 0.07], dtype=np.float64)
    mn[2] = min(0.0, float(mn[2]))
    span = np.maximum(mx - mn, 0.05)
    return {
        "bounds": {"xMin": float(mn[0]), "xMax": float(mx[0]), "yMin": float(mn[1]), "yMax": float(mx[1]), "zMin": float(mn[2]), "zMax": float(mx[2])},
        "center": ((mn + mx) * 0.5).tolist(),
        "range": span.tolist(),
    }


def build_payload(args: argparse.Namespace) -> Dict[str, Any]:
    npz_path = Path(args.input_npz).expanduser().resolve()
    table_path = Path(args.table_frame_json).expanduser().resolve()
    if not npz_path.exists():
        raise RuntimeError(f"missing input npz: {npz_path}")
    if not table_path.exists():
        raise RuntimeError(f"missing table frame json: {table_path}")

    npz = np.load(npz_path, allow_pickle=True)
    joints_key = choose_joints_key(npz, str(args.joints_key))
    joints_cam = np.asarray(npz[joints_key], dtype=np.float64)
    if joints_cam.ndim != 3 or joints_cam.shape[1:] != (21, 3):
        raise RuntimeError(f"bad {joints_key} shape: {joints_cam.shape}")
    uv_key = ""
    geometry_mode = "camera_xyz"
    if bool(args.reproject_from_uv_depth):
        uv_key = choose_uv_key(npz, str(args.uv_key), joints_key)
        uv = np.asarray(npz[uv_key], dtype=np.float64)
        if uv.shape[:2] != joints_cam.shape[:2]:
            raise RuntimeError(f"bad {uv_key} shape: {uv.shape}, expected first dims {joints_cam.shape[:2]}")
        infilled_uv_key = choose_infilled_uv_key(npz, str(args.infilled_uv_key), uv_key)
        replaced_infilled_uv = 0
        replaced_oversize_uv = 0
        if infilled_uv_key:
            replacement_uv = np.asarray(npz[infilled_uv_key], dtype=np.float64)
            motion_infilled_arr = np.asarray(npz["motion_infilled"], dtype=np.int32).astype(bool) if "motion_infilled" in npz.files else np.zeros((joints_cam.shape[0],), dtype=bool)
            uv, replaced_infilled_uv, replaced_oversize_uv = replace_bad_or_infilled_uv(
                uv,
                replacement_uv,
                motion_infilled_arr,
                float(args.max_uv_hand_size_px),
            )
        joints_cam = backproject_uv_with_depth(uv, joints_cam[:, :, 2], load_camera(npz))
        geometry_mode = "uv_locked_phase_depth"
    else:
        infilled_uv_key = ""
        replaced_infilled_uv = 0
        replaced_oversize_uv = 0
    n = joints_cam.shape[0]
    transform = load_table_transform(table_path, bool(args.prefer_scaled_table_transform))
    joints_table = transform_points(transform, joints_cam)

    frame_index = np.asarray(npz["frame_index"], dtype=np.int64)
    elapsed = scalar_array(npz, "elapsed_sec", n, 0.0)
    labels = scalar_array(npz, "hand_label", n, "")
    ranks = scalar_array(npz, "hand_rank", n, 0)
    tracks = scalar_array(npz, "track_id", n, -1)
    conf = scalar_array(npz, "det_conf", n, float("nan"))
    motion_infilled = scalar_array(npz, "motion_infilled", n, 0)
    motion_method = scalar_array(npz, "motion_infiller_method", n, "")
    depth_smooth_qc = scalar_array(npz, "depth_smooth_qc_flag", n, "")
    motion_qc = scalar_array(npz, "motion_infiller_qc_flag", n, "")
    mano_qc = scalar_array(npz, "mano_smoothing_qc_flag", n, "")
    visibility_qc = scalar_array(npz, "mano_visibility_qc_flag", n, "")
    visibility_realign_qc = scalar_array(npz, "visibility_realign_qc_flag", n, "")
    align_rms = scalar_array(npz, "alignment_rms_m", n, float("nan"))
    mano_jump = scalar_array(npz, "mano_smooth_wrist_jump_m", n, float("nan"))
    visible_vertex_ratio = scalar_array(npz, "mano_visible_vertex_ratio", n, float("nan"))
    visible_reliable_count = scalar_array(npz, "mano_visible_reliable_joint_count", n, -1)
    if "mano_joint_visible" in npz.files and np.asarray(npz["mano_joint_visible"]).shape[:2] == (n, 21):
        joint_visible = np.asarray(npz["mano_joint_visible"]).astype(bool)
    else:
        joint_visible = np.ones((n, 21), dtype=bool)

    rows: List[int] = []
    for i, frame in enumerate(frame_index.tolist()):
        if int(args.frame_start) >= 0 and int(frame) < int(args.frame_start):
            continue
        if int(args.frame_end) >= 0 and int(frame) > int(args.frame_end):
            continue
        if int(args.stride) > 1 and (int(frame) % int(args.stride)) != 0:
            continue
        rows.append(i)
    if int(args.max_candidates) > 0:
        rows = rows[: int(args.max_candidates)]
    if not rows:
        raise RuntimeError("no candidates selected")

    by_frame: Dict[int, Dict[str, Any]] = {}
    z_values: List[float] = []
    infilled_count = 0
    for i in rows:
        frame = int(frame_index[i])
        item = by_frame.setdefault(frame, {"frame": frame, "elapsed": finite_float(elapsed[i], frame / 30.0), "candidates": []})
        pts = joints_table[i]
        point_map: Dict[str, List[float]] = {}
        for lid in range(21):
            p = pts[lid]
            if np.isfinite(p).all():
                point_map[str(lid)] = [float(p[0]), float(p[1]), float(p[2])]
                z_values.append(float(p[2]))
        visible_ids = [int(v) for v in np.where(joint_visible[i])[0].tolist()]
        if int(number_at(motion_infilled, i, 0)) != 0:
            infilled_count += 1
        item["candidates"].append({
            "row": int(i),
            "frame": frame,
            "label": string_at(labels, i),
            "rank": int(number_at(ranks, i, 0)),
            "track": int(number_at(tracks, i, -1)),
            "conf": finite_float(conf[i], 0.0),
            "motionInfilled": bool(int(number_at(motion_infilled, i, 0))),
            "motionMethod": string_at(motion_method, i),
            "depthSmoothQc": string_at(depth_smooth_qc, i),
            "motionQc": string_at(motion_qc, i),
            "manoQc": string_at(mano_qc, i),
            "visibilityQc": string_at(visibility_qc, i),
            "visibilityRealignQc": string_at(visibility_realign_qc, i),
            "alignmentRmsM": number_at(align_rms, i),
            "manoWristJumpM": number_at(mano_jump, i),
            "visibleVertexRatio": number_at(visible_vertex_ratio, i),
            "visibleReliableCount": int(number_at(visible_reliable_count, i, -1)),
            "visibleIds": visible_ids,
            "points": point_map,
        })

    frames = [by_frame[k] for k in sorted(by_frame)]
    for frame in frames:
        frame["candidates"].sort(key=lambda c: (str(c["label"]), int(c["rank"]), int(c["track"])))
    initial_index = 0
    best_score = -1
    for idx, frame in enumerate(frames):
        labels_in_frame = {str(c["label"]) for c in frame["candidates"]}
        visible_score = sum(len(c["visibleIds"]) for c in frame["candidates"])
        score = visible_score + 12 * len(labels_in_frame) + 3 * len(frame["candidates"])
        if score > best_score:
            best_score = score
            initial_index = idx

    labels_selected = [string_at(labels, i) for i in rows]
    summary = {
        "input_npz": str(npz_path),
        "table_frame_json": str(table_path),
        "joints_key": joints_key,
        "uv_key": uv_key,
        "infilled_uv_key": infilled_uv_key,
        "replaced_infilled_uv": int(replaced_infilled_uv),
        "replaced_oversize_uv": int(replaced_oversize_uv),
        "max_uv_hand_size_px": float(args.max_uv_hand_size_px),
        "geometry_mode": geometry_mode,
        "candidates_selected": len(rows),
        "frames_selected": len(frames),
        "frame_min": int(frames[0]["frame"]),
        "frame_max": int(frames[-1]["frame"]),
        "motion_infilled_candidates": int(infilled_count),
        "hand_label_counts": {label: int(labels_selected.count(label)) for label in sorted(set(labels_selected))},
        "table_z_m": summarize(z_values),
        "alignment_rms_m": summarize([number_at(align_rms, i) for i in rows]),
        "mano_visible_vertex_ratio": summarize([number_at(visible_vertex_ratio, i) for i in rows]),
    }
    title_stage = args.stage_name or npz_path.parent.name
    payload: Dict[str, Any] = {
        "title": f"{title_stage} hand skeleton in LFV table frame",
        "summary": summary,
        "frames": frames,
        "initialIndex": int(initial_index),
        "edges": HAND_EDGES,
        "names": LANDMARK_NAMES,
    }
    payload.update(compute_bounds(frames))
    video_path = Path(args.video).expanduser().resolve() if args.video else None
    if video_path and video_path.exists():
        payload["video"] = rel_or_abs(video_path, Path(args.output_html).expanduser().resolve().parent)
    else:
        payload["video"] = ""
    return payload


def html_template(payload: Dict[str, Any]) -> str:
    data_json = json.dumps(json_clean(payload), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    title = str(payload["title"])
    video_src = str(payload.get("video", ""))
    summary = payload["summary"]
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{title}</title>
<style>
:root {{ color-scheme: dark; --bg:#0d1014; --panel:#151a20; --line:#2a333e; --fg:#edf2f7; --muted:#a8b3c1; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--fg); font-family:Inter,system-ui,-apple-system,"Segoe UI",sans-serif; }}
header {{ padding:14px 18px 10px; background:#11161c; border-bottom:1px solid var(--line); }}
h1 {{ margin:0 0 6px; font-size:18px; font-weight:650; }}
.sub {{ color:var(--muted); font-size:13px; line-height:1.45; }}
main {{ display:grid; grid-template-columns:minmax(720px,1fr) 390px; gap:12px; padding:12px; min-height:calc(100vh - 72px); }}
.panel {{ min-width:0; background:var(--panel); border:1px solid var(--line); border-radius:6px; overflow:hidden; }}
.title {{ display:flex; justify-content:space-between; gap:10px; padding:10px 12px; border-bottom:1px solid var(--line); font-size:14px; font-weight:600; }}
.canvas-wrap {{ position:relative; height:calc(100vh - 190px); min-height:560px; }}
canvas {{ display:block; width:100%; height:100%; background:#07090c; }}
.hud {{ position:absolute; left:12px; top:12px; min-width:520px; max-width:760px; padding:10px 12px; border-radius:6px; background:rgba(0,0,0,.65); border:1px solid rgba(255,255,255,.14); font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; white-space:pre; }}
.controls {{ display:grid; grid-template-columns:auto 1fr auto auto auto auto; gap:10px; align-items:center; padding:10px 12px 12px; }}
button {{ color:var(--fg); background:#242c35; border:1px solid #404c5a; border-radius:5px; padding:8px 12px; cursor:pointer; font-size:14px; }}
button:hover {{ background:#2b3541; }}
input[type=range] {{ width:100%; }}
label {{ display:flex; align-items:center; gap:7px; color:var(--muted); white-space:nowrap; font-size:13px; }}
.view-controls {{ display:grid; grid-template-columns:1fr; gap:10px; padding:12px; border-top:1px solid var(--line); }}
.view-controls label {{ display:grid; grid-template-columns:52px 1fr; }}
.preset-row {{ display:grid; grid-template-columns:repeat(3,1fr); gap:8px; }}
.legend {{ display:grid; gap:8px; padding:0 12px 12px; color:var(--muted); font-size:13px; }}
.swatch {{ display:inline-block; width:11px; height:11px; border-radius:50%; margin-right:7px; vertical-align:-1px; }}
.info {{ padding:12px; color:var(--muted); font-size:13px; line-height:1.55; }}
.info b {{ color:#d9e2ec; }}
code {{ color:#d8e7ff; background:#0f1720; border-radius:3px; padding:1px 4px; overflow-wrap:anywhere; }}
video {{ display:block; width:100%; background:#050607; }}
@media (max-width:1120px) {{ main {{ grid-template-columns:1fr; }} .canvas-wrap {{ height:62vh; min-height:440px; }} }}
</style>
</head>
<body>
<header>
  <h1>{title}</h1>
  <div class="sub">Coordinate: LFV table frame. Red/orange dots are MANO z-buffer visible landmarks; gray hidden landmarks can be enabled. Coordinates are transformed from current Phase-C left-camera hand skeletons.</div>
</header>
<main>
  <section class="panel">
    <div class="title"><span>3D hand skeleton</span><span id="frameLabel"></span></div>
    <div class="canvas-wrap"><canvas id="view"></canvas><div class="hud" id="hud"></div></div>
    <div class="controls">
      <button id="play">Play</button>
      <input id="slider" type="range" min="0" max="0" value="0" />
      <button id="center">Center</button>
      <label><input id="showHidden" type="checkbox" checked /> hidden</label>
      <label><input id="showGhost" type="checkbox" /> path</label>
      <label><input id="showLabels" type="checkbox" /> labels</label>
    </div>
  </section>
  <aside class="panel">
    <div class="title"><span>view</span><span>table frame</span></div>
    <div class="view-controls">
      <div class="preset-row">
        <button id="viewOblique">Oblique</button>
        <button id="viewTop">Top</button>
        <button id="viewSide">Side</button>
      </div>
      <label>yaw <input id="yaw" type="range" min="-180" max="180" value="-38" /></label>
      <label>pitch <input id="pitch" type="range" min="0" max="88" value="58" /></label>
      <label>zoom <input id="zoom" type="range" min="45" max="260" value="120" /></label>
      <label><input id="showLeft" type="checkbox" checked /> show left</label>
      <label><input id="showRight" type="checkbox" checked /> show right</label>
      <label><input id="showInfilled" type="checkbox" checked /> show infilled</label>
    </div>
    <div class="legend">
      <div><span class="swatch" style="background:#40b6ff"></span>left hand label</div>
      <div><span class="swatch" style="background:#44d26f"></span>right hand label</div>
      <div><span class="swatch" style="background:#ff375f"></span>visible detected joints</div>
      <div><span class="swatch" style="background:#ffb240"></span>visible motion-infilled joints</div>
      <div><span class="swatch" style="background:#7d8793"></span>hidden joints if enabled</div>
    </div>
    <video controls muted loop preload="metadata" src="{video_src}"></video>
    <div class="info">
      <div><b>input</b>: <code>{summary["input_npz"]}</code></div>
      <div><b>joints key</b>: <code>{summary["joints_key"]}</code></div>
      <div><b>uv key</b>: <code>{summary["uv_key"] or "not used"}</code></div>
      <div><b>infill uv</b>: <code>{summary["infilled_uv_key"] or "not replaced"}</code></div>
      <div><b>geometry</b>: <code>{summary["geometry_mode"]}</code></div>
      <div><b>table frame</b>: <code>{summary["table_frame_json"]}</code></div>
      <div><b>frames</b>: {summary["frame_min"]}..{summary["frame_max"]}, selected {summary["frames_selected"]}</div>
      <div><b>candidates</b>: {summary["candidates_selected"]}, infilled {summary["motion_infilled_candidates"]}</div>
    </div>
  </aside>
</main>
<script>
const DATA = {data_json};
const canvas = document.getElementById('view');
const ctx = canvas.getContext('2d');
const hud = document.getElementById('hud');
const frameLabel = document.getElementById('frameLabel');
const slider = document.getElementById('slider');
const playBtn = document.getElementById('play');
const centerBtn = document.getElementById('center');
const yawEl = document.getElementById('yaw');
const pitchEl = document.getElementById('pitch');
const zoomEl = document.getElementById('zoom');
const showHidden = document.getElementById('showHidden');
const showGhost = document.getElementById('showGhost');
const showLabels = document.getElementById('showLabels');
const showLeft = document.getElementById('showLeft');
const showRight = document.getElementById('showRight');
const showInfilled = document.getElementById('showInfilled');
let index = Number.isFinite(DATA.initialIndex) ? DATA.initialIndex : 0;
let timer = null;
slider.max = Math.max(0, DATA.frames.length - 1);
slider.value = index;
function dpr() {{ return window.devicePixelRatio || 1; }}
function enabled(c) {{
  const label = String(c.label).toLowerCase();
  if(label === 'left' && !showLeft.checked) return false;
  if(label === 'right' && !showRight.checked) return false;
  if(c.motionInfilled && !showInfilled.checked) return false;
  return true;
}}
function labelColor(label) {{ return String(label).toLowerCase()==='right' ? '#44d26f' : '#40b6ff'; }}
function params() {{ return {{ yaw:Number(yawEl.value)*Math.PI/180, pitch:Number(pitchEl.value)*Math.PI/180, zoom:Number(zoomEl.value)/100 }}; }}
function rotate(p) {{
  const c=DATA.center, q=params();
  const x=p[0]-c[0], y=p[1]-c[1], z=p[2]-c[2];
  const cy=Math.cos(q.yaw), sy=Math.sin(q.yaw);
  const x1=cy*x-sy*y, y1=sy*x+cy*y;
  const cp=Math.cos(q.pitch), sp=Math.sin(q.pitch);
  return [x1, cp*y1-sp*z, sp*y1+cp*z];
}}
function project(p) {{
  const r=rotate(p);
  const range=Math.max(DATA.range[0],DATA.range[1],DATA.range[2],.1);
  const s=Math.min(canvas.width,canvas.height)/range*.58*params().zoom;
  return [canvas.width*.50+r[0]*s, canvas.height*.58-r[2]*s, r[1]];
}}
function line(a,b,color,width=2,alpha=1,dash=false) {{
  const pa=project(a), pb=project(b), k=dpr();
  ctx.save(); ctx.globalAlpha=alpha; ctx.strokeStyle=color; ctx.lineWidth=width*k; if(dash) ctx.setLineDash([7*k,7*k]);
  ctx.beginPath(); ctx.moveTo(pa[0],pa[1]); ctx.lineTo(pb[0],pb[1]); ctx.stroke(); ctx.restore();
}}
function dot(p,r,color,alpha=1,stroke=null) {{
  const pp=project(p), k=dpr();
  ctx.save(); ctx.globalAlpha=alpha; ctx.fillStyle=color; ctx.beginPath(); ctx.arc(pp[0],pp[1],r*k,0,Math.PI*2); ctx.fill();
  if(stroke) {{ ctx.strokeStyle=stroke; ctx.lineWidth=1.4*k; ctx.stroke(); }}
  ctx.restore();
}}
function text(s,p,color='#dce5ef') {{
  const pp=project(p), k=dpr();
  ctx.save(); ctx.font=`${{11*k}}px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace`; ctx.fillStyle=color;
  ctx.fillText(s, pp[0]+6*k, pp[1]-5*k); ctx.restore();
}}
function axes() {{
  const b=DATA.bounds, step=.05;
  const x0=Math.floor(b.xMin/step)*step, x1=Math.ceil(b.xMax/step)*step;
  const y0=Math.floor(b.yMin/step)*step, y1=Math.ceil(b.yMax/step)*step;
  for(let x=x0; x<=x1+1e-9; x+=step) line([x,y0,0],[x,y1,0],Math.abs(x)<1e-9?'#7f8996':'#242c36',Math.abs(x)<1e-9?1.8:1,Math.abs(x)<1e-9?.75:.55);
  for(let y=y0; y<=y1+1e-9; y+=step) line([x0,y,0],[x1,y,0],Math.abs(y)<1e-9?'#7f8996':'#242c36',Math.abs(y)<1e-9?1.8:1,Math.abs(y)<1e-9?.75:.55);
  line([0,0,0],[0.08,0,0],'#ff657a',3,.9); text('+X',[0.085,0,0],'#ff657a');
  line([0,0,0],[0,0.08,0],'#6ff093',3,.9); text('+Y',[0,0.085,0],'#6ff093');
  line([0,0,0],[0,0,0.08],'#6da8ff',3,.9); text('+Z',[0,0,0.085],'#6da8ff');
}}
function isVisible(c,lid) {{ return c.visibleIds.includes(Number(lid)); }}
function drawCandidate(c, ghost=false) {{
  if(!enabled(c)) return;
  const visibleColor = c.motionInfilled ? '#ffb240' : '#ff375f';
  const hiddenColor = '#7d8793';
  for(const e of DATA.edges) {{
    const a=String(e[0]), b=String(e[1]), pa=c.points[a], pb=c.points[b];
    if(!pa || !pb) continue;
    const av=isVisible(c,a), bv=isVisible(c,b);
    if(ghost) {{ line(pa,pb,labelColor(c.label),1.1,.13); continue; }}
    if(av && bv) line(pa,pb,visibleColor,2.5,.95);
    else if(showHidden.checked) line(pa,pb,hiddenColor,1.3,.35,true);
  }}
  if(ghost) return;
  for(const [id,p] of Object.entries(c.points)) {{
    if(isVisible(c,id)) {{
      dot(p, Number(id)===0?6:4.5, visibleColor, 1, '#ffffff');
      if(showLabels.checked) text(`${{id}} ${{DATA.names[Number(id)]}}`, p, visibleColor);
    }} else if(showHidden.checked) {{
      dot(p, Number(id)===0?4.5:3.2, hiddenColor, .45, '#222933');
      if(showLabels.checked) text(`${{id}}`, p, hiddenColor);
    }}
  }}
  const wrist=c.points['0'];
  if(wrist) text(`${{c.label}} track=${{c.track}} r${{c.rank}}${{c.motionInfilled?' infill':''}}`, wrist, labelColor(c.label));
}}
function drawGhost() {{
  if(!showGhost.checked) return;
  for(let i=0;i<DATA.frames.length;i+=10) for(const c of DATA.frames[i].candidates) drawCandidate(c,true);
}}
function hudText(fr) {{
  const active = fr.candidates.filter(enabled);
  const parts = active.map(c => `${{c.label}} tr=${{c.track}} r=${{c.rank}} conf=${{c.conf.toFixed(2)}} visible=${{c.visibleIds.length}}/21 alignRms=${{Number.isFinite(c.alignmentRmsM)?(c.alignmentRmsM*100).toFixed(1)+'cm':'NA'}} vv=${{Number.isFinite(c.visibleVertexRatio)?c.visibleVertexRatio.toFixed(2):'NA'}} method=${{c.motionMethod || 'detected'}}\\n  depth=${{c.depthSmoothQc}} motion=${{c.motionQc}} mano=${{c.manoQc}} vis=${{c.visibilityQc}}`).join('\\n');
  return `frame=${{fr.frame}}  index=${{index}}/${{DATA.frames.length-1}}\\nelapsed=${{fr.elapsed.toFixed(3)}}s\\nactive_candidates=${{active.length}}/${{fr.candidates.length}}\\n${{parts}}`;
}}
function draw() {{
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.fillStyle='#07090c'; ctx.fillRect(0,0,canvas.width,canvas.height);
  axes(); drawGhost();
  const fr=DATA.frames[index];
  for(const c of fr.candidates) drawCandidate(c,false);
  hud.textContent=hudText(fr);
  frameLabel.textContent=`frame ${{fr.frame}}`;
  slider.value=index;
}}
function setIndex(i) {{ index=Math.max(0,Math.min(DATA.frames.length-1,Number(i))); draw(); }}
function resize() {{
  const r=canvas.getBoundingClientRect(), k=dpr();
  canvas.width=Math.max(1,Math.round(r.width*k)); canvas.height=Math.max(1,Math.round(r.height*k)); draw();
}}
function setView(yaw,pitch,zoom) {{ yawEl.value=String(yaw); pitchEl.value=String(pitch); zoomEl.value=String(zoom); draw(); }}
function play() {{
  if(timer) {{ clearInterval(timer); timer=null; playBtn.textContent='Play'; return; }}
  playBtn.textContent='Pause'; timer=setInterval(()=>setIndex(index>=DATA.frames.length-1?0:index+1),70);
}}
slider.addEventListener('input',()=>setIndex(slider.value));
playBtn.addEventListener('click',play);
centerBtn.addEventListener('click',()=>setIndex(Math.floor(DATA.frames.length/2)));
document.getElementById('viewOblique').addEventListener('click',()=>setView(-38,58,120));
document.getElementById('viewTop').addEventListener('click',()=>setView(-38,88,145));
document.getElementById('viewSide').addEventListener('click',()=>setView(0,18,130));
[showHidden,showGhost,showLabels,showLeft,showRight,showInfilled,yawEl,pitchEl,zoomEl].forEach(el=>el.addEventListener('input',draw));
window.addEventListener('resize',resize);
resize();
</script>
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    output_html = Path(args.output_html).expanduser().resolve()
    payload = build_payload(args)
    output_json = Path(args.output_json).expanduser().resolve() if args.output_json else output_html.with_suffix(".json")
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html_template(payload), encoding="utf-8")
    output_json.write_text(json.dumps(json_clean(payload["summary"]), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(json_clean({**payload["summary"], "output_html": str(output_html), "output_json": str(output_json)}), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
