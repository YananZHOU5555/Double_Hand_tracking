#!/usr/bin/env python3
"""Export IK input-vs-output 3D debugger with relaxed fallback frames marked."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


HOST_ROS_ROOT = Path("/home/yannan/workspace/ros1_docker-main")
CONTAINER_ROOT = "/workspace/ros1_docker_jinhe"
DEFAULT_TCP_OFFSET = np.asarray([0.0, 0.0, 0.13149316740823477], dtype=np.float64)


def host_path(path_text: str) -> Path:
    if path_text.startswith(CONTAINER_ROOT + "/"):
        return HOST_ROS_ROOT / path_text[len(CONTAINER_ROOT) + 1 :]
    return Path(path_text).expanduser()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_float(value: Any, default: float = float("nan")) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except Exception:
        return default


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def project_so3(rot: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.asarray(rot, dtype=np.float64))
    out = u @ vt
    if np.linalg.det(out) < 0.0:
        u[:, -1] *= -1.0
        out = u @ vt
    return out


def rpy_to_rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    ry = np.asarray([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rz = np.asarray([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx


def parse_rpy(raw: Any) -> np.ndarray:
    try:
        vals = [float(v.strip()) for v in str(raw).split(",") if v.strip()]
    except Exception:
        vals = []
    if len(vals) != 3:
        return np.eye(3, dtype=np.float64)
    return rpy_to_rotation(vals[0], vals[1], vals[2])


def rot6d_to_rotation(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(3, 2, order="F")
    x = arr[:, 0]
    y = arr[:, 1]
    nx = float(np.linalg.norm(x))
    if not np.isfinite(nx) or nx < 1e-9:
        raise ValueError("invalid rot6d first axis")
    x = x / nx
    y = y - x * float(np.dot(x, y))
    ny = float(np.linalg.norm(y))
    if not np.isfinite(ny) or ny < 1e-9:
        raise ValueError("invalid rot6d second axis")
    y = y / ny
    z = np.cross(x, y)
    return project_so3(np.column_stack([x, y, z]))


def rotation_to_rpy_deg(rot: np.ndarray) -> List[float]:
    sy = math.sqrt(float(rot[0, 0] * rot[0, 0] + rot[1, 0] * rot[1, 0]))
    if sy >= 1e-9:
        roll = math.atan2(float(rot[2, 1]), float(rot[2, 2]))
        pitch = math.atan2(float(-rot[2, 0]), sy)
        yaw = math.atan2(float(rot[1, 0]), float(rot[0, 0]))
    else:
        roll = math.atan2(float(-rot[1, 2]), float(rot[1, 1]))
        pitch = math.atan2(float(-rot[2, 0]), sy)
        yaw = 0.0
    return [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]


def vec3(row: Dict[str, str], prefix: str) -> Optional[List[float]]:
    vals = [parse_float(row.get(f"{prefix}_{axis}_m")) for axis in "xyz"]
    if all(np.isfinite(vals)):
        return [float(v) for v in vals]
    return None


def make_core_index(rows: Sequence[Dict[str, str]]) -> Dict[Tuple[int, int], Dict[str, str]]:
    out: Dict[Tuple[int, int], Dict[str, str]] = {}
    for row in rows:
        try:
            key = (int(float(row["core_frame_index"])), int(float(row["frame_index"])))
        except Exception:
            continue
        out[key] = row
    return out


def load_table_transform(path: Path) -> np.ndarray:
    payload = load_json(path)
    mat = np.asarray(payload.get("T_robot_from_table", np.eye(4)), dtype=np.float64)
    if mat.shape != (4, 4):
        raise RuntimeError(f"bad T_robot_from_table in {path}")
    return mat


def load_tcp_offset(tcp_json: Path, metadata: Dict[str, Any]) -> np.ndarray:
    if isinstance(metadata.get("tcp_offset_xyz_m"), list) and len(metadata["tcp_offset_xyz_m"]) == 3:
        return np.asarray([float(v) for v in metadata["tcp_offset_xyz_m"]], dtype=np.float64)
    payload = load_json(tcp_json)
    if isinstance(payload.get("tcp_offset_xyz_in_end_pose_m"), list) and len(payload["tcp_offset_xyz_in_end_pose_m"]) == 3:
        return np.asarray([float(v) for v in payload["tcp_offset_xyz_in_end_pose_m"]], dtype=np.float64)
    return DEFAULT_TCP_OFFSET.copy()


def transform_points(mat: np.ndarray, pts: np.ndarray) -> np.ndarray:
    homo = np.concatenate([pts, np.ones((len(pts), 1), dtype=np.float64)], axis=1)
    return (mat @ homo.T).T[:, :3]


def finite_bounds(points: Iterable[Optional[Sequence[float]]]) -> Dict[str, Any]:
    arr = np.asarray([p for p in points if p is not None and np.isfinite(np.asarray(p, dtype=float)).all()], dtype=np.float64)
    if arr.size == 0:
        arr = np.asarray([[0.0, 0.0, 0.0], [0.5, 0.0, 0.4]], dtype=np.float64)
    lo = np.min(arr, axis=0)
    hi = np.max(arr, axis=0)
    span = np.maximum(hi - lo, np.asarray([0.50, 0.45, 0.35], dtype=np.float64))
    center = (lo + hi) * 0.5
    lo = center - span * 0.62
    hi = center + span * 0.62
    return {"min": lo.tolist(), "max": hi.tolist(), "center": center.tolist(), "range": span.tolist()}


def build_payload(args: argparse.Namespace) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    ik_csv = Path(args.ik_csv).expanduser().resolve()
    core_csv = Path(args.core_csv).expanduser().resolve()
    metadata_json = Path(args.ik_metadata).expanduser().resolve()
    table_json = host_path(args.table_json).resolve()
    tcp_json = Path(args.tcp_json).expanduser().resolve()
    session_dir = Path(args.session_dir).expanduser().resolve()

    metadata = load_json(metadata_json)
    tcp_offset = load_tcp_offset(tcp_json, metadata)
    table_transform = load_table_transform(table_json)
    r_robot_from_table = table_transform[:3, :3]
    r_align = parse_rpy(metadata.get("orientation_align_rpy", args.orientation_align_rpy))

    ik_rows = read_csv(ik_csv)
    core_rows = read_csv(core_csv)
    core_by_key = make_core_index(core_rows)

    frames: List[Dict[str, Any]] = []
    flat_rows: List[Dict[str, Any]] = []
    points: List[Optional[Sequence[float]]] = []
    for seq, row in enumerate(ik_rows):
        core_frame = int(parse_float(row.get("core_frame_index"), seq))
        frame_index = int(parse_float(row.get("frame_index"), core_frame))
        core = core_by_key.get((core_frame, frame_index), core_rows[min(seq, len(core_rows) - 1)] if core_rows else {})

        target_tcp = vec3(row, "target")
        raw_target_tcp = vec3(row, "raw_target")
        fk_tcp = vec3(row, "fk_tcp")
        rot6d_valid = False
        r_robot_tcp = np.eye(3, dtype=np.float64)
        try:
            rot6d = [parse_float(core.get(f"rot6d_{i}")) for i in range(6)]
            if all(np.isfinite(rot6d)):
                r_table_tcp = rot6d_to_rotation(rot6d)
                r_robot_tcp = project_so3(r_robot_from_table @ r_table_tcp @ r_align)
                rot6d_valid = True
        except Exception:
            pass

        target_ee = None
        if target_tcp is not None:
            target_ee_arr = np.asarray(target_tcp, dtype=np.float64) - r_robot_tcp @ tcp_offset
            target_ee = target_ee_arr.tolist()

        success = parse_bool(row.get("success", "0"))
        normal_success = parse_bool(row.get("normal_ik_success", "0"))
        fallback_attempted = parse_bool(row.get("relaxed_ik_attempted", "0"))
        fallback_used = parse_bool(row.get("relaxed_ik_used", "0"))
        fallback_success = parse_bool(row.get("relaxed_ik_success", "0"))
        if not success:
            solve_kind = "failed"
        elif fallback_used:
            solve_kind = "fallback"
        elif normal_success:
            solve_kind = "normal"
        else:
            solve_kind = "success_unknown"

        frame = {
            "seq": seq,
            "core": core_frame,
            "source": frame_index,
            "time": parse_float(row.get("source_elapsed_sec", row.get("sim_elapsed_sec", seq / 60.0))),
            "state": row.get("state", core.get("state", "")),
            "success": success,
            "normalSuccess": normal_success,
            "fallbackAttempted": fallback_attempted,
            "fallbackUsed": fallback_used,
            "fallbackSuccess": fallback_success,
            "solveKind": solve_kind,
            "posErr": parse_float(row.get("pos_error_m")),
            "oriErr": parse_float(row.get("ori_error_rad")),
            "normalPosErr": parse_float(row.get("normal_pos_error_m")),
            "normalOriErr": parse_float(row.get("normal_ori_error_rad")),
            "tcpDrift": parse_float(row.get("tcp_drift_m")),
            "limitMarginMinRad": parse_float(row.get("limit_margin_min_rad")),
            "limitActiveJoints": row.get("limit_active_joints", ""),
            "relaxedSeedIndex": row.get("relaxed_seed_index", ""),
            "targetTcp": target_tcp,
            "rawTargetTcp": raw_target_tcp,
            "targetEe": target_ee,
            "fkTcp": fk_tcp,
            "R": r_robot_tcp.reshape(-1, order="C").tolist(),
            "rpyDeg": rotation_to_rpy_deg(r_robot_tcp),
            "rot6dValid": rot6d_valid,
            "openWidth": parse_float(core.get("open_width_m"), parse_float(row.get("input_open_width_m"), 0.09)),
            "regularizerSegmentType": core.get("regularizer_segment_type", ""),
            "regularizerDeltaM": parse_float(core.get("regularizer_delta_m")),
        }
        frames.append(frame)
        points.extend([target_tcp, raw_target_tcp, target_ee, fk_tcp])
        flat_rows.append(
            {
                "seq": seq,
                "core_frame_index": core_frame,
                "frame_index": frame_index,
                "state": frame["state"],
                "success": int(success),
                "normal_ik_success": int(normal_success),
                "relaxed_ik_attempted": int(fallback_attempted),
                "relaxed_ik_used": int(fallback_used),
                "solve_kind": solve_kind,
                "pos_error_m": frame["posErr"],
                "ori_error_rad": frame["oriErr"],
                "normal_pos_error_m": frame["normalPosErr"],
                "normal_ori_error_rad": frame["normalOriErr"],
                "tcp_drift_m": frame["tcpDrift"],
                "limit_margin_min_rad": frame["limitMarginMinRad"],
                "target_tcp_x_m": target_tcp[0] if target_tcp else "",
                "target_tcp_y_m": target_tcp[1] if target_tcp else "",
                "target_tcp_z_m": target_tcp[2] if target_tcp else "",
                "fk_tcp_x_m": fk_tcp[0] if fk_tcp else "",
                "fk_tcp_y_m": fk_tcp[1] if fk_tcp else "",
                "fk_tcp_z_m": fk_tcp[2] if fk_tcp else "",
                "regularizer_segment_type": frame["regularizerSegmentType"],
                "regularizer_delta_m": frame["regularizerDeltaM"],
            }
        )

    table_corners_table = np.asarray(
        [[-0.45, -0.35, 0.0], [0.30, -0.35, 0.0], [0.30, 0.25, 0.0], [-0.45, 0.25, 0.0]],
        dtype=np.float64,
    )
    table_corners_robot = transform_points(table_transform, table_corners_table).tolist()
    points.extend(table_corners_robot)

    summary = {
        "frame_count": len(frames),
        "success_count": sum(1 for f in frames if f["success"]),
        "normal_success_count": sum(1 for f in frames if f["success"] and f["normalSuccess"] and not f["fallbackUsed"]),
        "fallback_used_count": sum(1 for f in frames if f["fallbackUsed"]),
        "fallback_attempted_count": sum(1 for f in frames if f["fallbackAttempted"]),
        "failed_count": sum(1 for f in frames if not f["success"]),
        "max_pos_error_m": max((float(f["posErr"]) for f in frames if np.isfinite(f["posErr"])), default=None),
        "max_tcp_drift_m": max((float(f["tcpDrift"]) for f in frames if np.isfinite(f["tcpDrift"])), default=None),
    }
    payload = {
        "title": args.title,
        "sessionDir": str(session_dir),
        "demoName": session_dir.name,
        "ikCsv": str(ik_csv),
        "ikMetadata": str(metadata_json),
        "coreCsv": str(core_csv),
        "tableJson": str(table_json),
        "tcpJson": str(tcp_json),
        "tcpOffset": tcp_offset.tolist(),
        "orientationAlignRpy": metadata.get("orientation_align_rpy", args.orientation_align_rpy),
        "summary": summary,
        "bounds": finite_bounds(points),
        "tableCornersRobot": table_corners_robot,
        "frames": frames,
    }
    return payload, flat_rows


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
html, body { margin:0; height:100%; overflow:hidden; background:#101317; color:#e9eef7; font-family:system-ui,-apple-system,Segoe UI,sans-serif; }
#wrap { display:grid; grid-template-rows:auto minmax(0,1fr) auto; height:100vh; }
#top { display:flex; align-items:center; gap:12px; padding:10px 14px; background:#1a202a; border-bottom:1px solid #334052; }
#title { font-weight:700; min-width:320px; }
button, select { background:#2b3443; color:#f2f5fb; border:1px solid #4d5b70; border-radius:6px; padding:6px 10px; }
button.active { background:#3b587b; border-color:#80aee8; }
#frame { width:min(820px,44vw); }
#status { margin-left:auto; white-space:nowrap; font-variant-numeric:tabular-nums; }
#status.fallback { color:#ffb13b; font-weight:800; }
#status.failed { color:#ff6565; font-weight:800; }
#canvas { width:100%; height:100%; display:block; background:radial-gradient(circle at 50% 45%, #252e3a 0%, #121820 62%, #080b10 100%); cursor:grab; }
#canvas.dragging { cursor:grabbing; }
#bottom { display:flex; align-items:center; gap:14px; padding:8px 14px; background:#151a22; border-top:1px solid #303847; font-size:13px; color:#b9c4d6; }
#info { font-variant-numeric:tabular-nums; white-space:nowrap; }
.dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:5px; vertical-align:-1px; }
#legend span { margin-right:12px; }
</style>
</head>
<body>
<div id="wrap">
  <div id="top">
    <div id="title"></div>
    <button id="play">Play</button>
    <button id="normalOnly">Normal</button>
    <button id="fallbackOnly">Fallback</button>
    <button id="allFrames" class="active">All</button>
    <input id="frame" type="range" min="0" max="0" value="0">
    <select id="speed"><option value="0.25">0.25x</option><option value="0.5">0.5x</option><option value="1" selected>1x</option><option value="2">2x</option></select>
    <div id="status"></div>
  </div>
  <canvas id="canvas"></canvas>
  <div id="bottom">
    <div id="legend">
      <span><i class="dot" style="background:#ffd84a"></i>IK input target TCP</span>
      <span><i class="dot" style="background:#16e0ff"></i>normal IK FK TCP</span>
      <span><i class="dot" style="background:#ff9f1a"></i>fallback FK TCP</span>
      <span><i class="dot" style="background:#ff4f5f"></i>failed</span>
      <span><i class="dot" style="background:#b06cff"></i>target EE</span>
      <span><i class="dot" style="background:#ff3030"></i>X</span>
      <span><i class="dot" style="background:#30ff64"></i>Y</span>
      <span><i class="dot" style="background:#4090ff"></i>Z</span>
    </div>
    <div id="info"></div>
  </div>
</div>
<script>
const DATA = __DATA__;
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const slider = document.getElementById('frame');
const playBtn = document.getElementById('play');
const speedSel = document.getElementById('speed');
const statusEl = document.getElementById('status');
const infoEl = document.getElementById('info');
document.getElementById('title').textContent = DATA.title;
slider.max = DATA.frames.length - 1;
let frameIndex = 0;
let playing = false;
let lastTs = 0;
let accum = 0;
let yaw = -0.92, pitch = 0.72, zoom = 1.0;
let dragging = false, dragX = 0, dragY = 0;
let mode = 'all';

function finite(p){ return p && p.length === 3 && p.every(Number.isFinite); }
function add(a,b){ return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]; }
function sub(a,b){ return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]; }
function mul(a,s){ return [a[0]*s, a[1]*s, a[2]*s]; }
function axes(R){ return [[R[0],R[3],R[6]], [R[1],R[4],R[7]], [R[2],R[5],R[8]]]; }
function resize(){
  const dpr = window.devicePixelRatio || 1;
  const r = canvas.getBoundingClientRect();
  canvas.width = Math.max(2, Math.round(r.width*dpr));
  canvas.height = Math.max(2, Math.round(r.height*dpr));
  ctx.setTransform(dpr,0,0,dpr,0,0);
  draw();
}
function project(p){
  const b = DATA.bounds;
  const c = b.center || [(b.min[0]+b.max[0])*0.5, (b.min[1]+b.max[1])*0.5, (b.min[2]+b.max[2])*0.5];
  const d = sub(p, c);
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const x1 = cy*d[0] + sy*d[1];
  const y1 = -sy*d[0] + cy*d[1];
  const z1 = d[2];
  const y2 = cp*y1 - sp*z1;
  const z2 = sp*y1 + cp*z1;
  const rect = canvas.getBoundingClientRect();
  const span = Math.max(...(b.range || [0.6,0.6,0.4]), 0.45);
  const s = Math.min(rect.width, rect.height) / span * 0.70 * zoom;
  return [rect.width*0.5 + x1*s, rect.height*0.56 - z2*s, y2];
}
function line(points, color, width=2, alpha=1, dash=null){
  ctx.save(); ctx.globalAlpha = alpha; ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineJoin='round'; ctx.lineCap='round';
  if (dash) ctx.setLineDash(dash);
  ctx.beginPath();
  let open = false;
  for (const p of points){
    if (!finite(p)) { open = false; continue; }
    const q = project(p);
    if (!open) { ctx.moveTo(q[0], q[1]); open = true; } else { ctx.lineTo(q[0], q[1]); }
  }
  ctx.stroke(); ctx.restore();
}
function point(p, color, radius=5, alpha=1, stroke=null){
  if (!finite(p)) return;
  const q = project(p);
  ctx.save(); ctx.globalAlpha = alpha;
  ctx.fillStyle = color; ctx.beginPath(); ctx.arc(q[0], q[1], radius, 0, Math.PI*2); ctx.fill();
  if (stroke) { ctx.strokeStyle=stroke; ctx.lineWidth=2; ctx.stroke(); }
  ctx.restore();
}
function label(p, text, color){
  if (!finite(p)) return;
  const q = project(p);
  ctx.fillStyle = color; ctx.font='12px system-ui'; ctx.fillText(text, q[0]+6, q[1]-6);
}
function drawGrid(){
  const c = DATA.tableCornersRobot;
  line([c[0], c[1], c[2], c[3], c[0]], '#697386', 1.5, 0.75);
  for (let i=0;i<=8;i++){
    const t=i/8;
    line([add(mul(c[0],1-t),mul(c[1],t)), add(mul(c[3],1-t),mul(c[2],t))], '#344050', 1, 0.5);
    line([add(mul(c[0],1-t),mul(c[3],t)), add(mul(c[1],1-t),mul(c[2],t))], '#344050', 1, 0.5);
  }
  line([[0,0,0],[0.16,0,0]], '#ff3030', 3); line([[0,0,0],[0,0.16,0]], '#30ff64', 3); line([[0,0,0],[0,0,0.16]], '#4090ff', 3);
  point([0,0,0], '#fff', 4); label([0,0,0], 'robot base', '#e9eef7');
}
function segmentPath(kind, key){
  const paths = [];
  let cur = [];
  for (const f of DATA.frames){
    const use = kind === 'fallback' ? f.fallbackUsed : (kind === 'normal' ? (f.success && !f.fallbackUsed) : !f.success);
    if (use && finite(f[key])) cur.push(f[key]);
    else if (cur.length) { paths.push(cur); cur = []; }
  }
  if (cur.length) paths.push(cur);
  return paths;
}
function drawAxesAt(p, R, scale=0.07){
  const a = axes(R);
  line([p, add(p, mul(a[0],scale))], '#ff3030', 4);
  line([p, add(p, mul(a[1],scale))], '#30ff64', 4);
  line([p, add(p, mul(a[2],scale))], '#4090ff', 4);
}
function drawGripper(f){
  if (!finite(f.targetTcp)) return;
  const a = axes(f.R);
  const x = a[0], y = a[1];
  const width = Math.max(0.025, Math.min(0.09, f.openWidth || 0.07));
  const back = 0.035, len = 0.085;
  const col = f.state === 'closed' ? '#ff5555' : '#36d87b';
  const l0 = add(add(f.targetTcp, mul(x, width/2)), mul(y, -back));
  const r0 = add(add(f.targetTcp, mul(x, -width/2)), mul(y, -back));
  const l1 = add(l0, mul(y, len));
  const r1 = add(r0, mul(y, len));
  line([l0,l1], col, 5); line([r0,r1], col, 5); line([l0,r0], col, 4);
}
function draw(){
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0,0,rect.width,rect.height);
  drawGrid();
  line(DATA.frames.map(f=>f.rawTargetTcp), '#a786ff', 1.5, 0.35, [6,7]);
  line(DATA.frames.map(f=>f.targetTcp), '#ffd84a', 2.5, 0.78);
  for (const p of segmentPath('normal','fkTcp')) line(p, '#16e0ff', 3, 0.92);
  for (const p of segmentPath('fallback','fkTcp')) line(p, '#ff9f1a', 4, 0.96);
  for (const p of segmentPath('failed','targetTcp')) line(p, '#ff4f5f', 5, 0.95);
  for (const f of DATA.frames){
    if (f.fallbackUsed) point(f.targetTcp, '#ff9f1a', 3.5, 0.75);
    if (!f.success) point(f.targetTcp, '#ff4f5f', 4.5, 0.9);
  }
  const f = DATA.frames[frameIndex];
  point(f.targetTcp, '#ffd84a', 8, 1, '#111');
  point(f.targetEe, '#b06cff', 6, 1, '#fff');
  line([f.targetEe, f.targetTcp], '#b06cff', 2, 0.95);
  if (finite(f.fkTcp)) point(f.fkTcp, f.fallbackUsed ? '#ff9f1a' : '#16e0ff', 8, 1, '#111');
  if (finite(f.rawTargetTcp)) point(f.rawTargetTcp, '#a786ff', 4, 0.8);
  drawAxesAt(f.targetTcp, f.R);
  drawGripper(f);
  label(f.targetTcp, 'IK input target TCP', '#ffd84a');
  if (finite(f.fkTcp)) label(f.fkTcp, f.fallbackUsed ? 'FK TCP fallback' : 'FK TCP normal', f.fallbackUsed ? '#ffb13b' : '#5ff4ff');
  slider.value = frameIndex;
  statusEl.className = f.success ? (f.fallbackUsed ? 'fallback' : '') : 'failed';
  const pos = Number.isFinite(f.posErr) ? `${(f.posErr*1000).toFixed(2)}mm` : 'n/a';
  const drift = Number.isFinite(f.tcpDrift) ? `${(f.tcpDrift*1000).toFixed(2)}mm` : 'n/a';
  const normalPos = Number.isFinite(f.normalPosErr) ? `${(f.normalPosErr*1000).toFixed(1)}mm` : 'n/a';
  statusEl.textContent = `frame ${frameIndex+1}/${DATA.frames.length} src=${f.source} ${f.solveKind} pos=${pos} drift=${drift} normal_pos=${normalPos}`;
  infoEl.textContent = `summary normal=${DATA.summary.normal_success_count} fallback=${DATA.summary.fallback_used_count} failed=${DATA.summary.failed_count}`;
}
function stepToKind(kind){
  mode = kind;
  document.getElementById('normalOnly').classList.toggle('active', kind === 'normal');
  document.getElementById('fallbackOnly').classList.toggle('active', kind === 'fallback');
  document.getElementById('allFrames').classList.toggle('active', kind === 'all');
  if (kind === 'all') return;
  const idx = DATA.frames.findIndex(f => kind === 'fallback' ? f.fallbackUsed : (f.success && !f.fallbackUsed));
  if (idx >= 0) frameIndex = idx;
  draw();
}
slider.addEventListener('input', () => { frameIndex = Number(slider.value); playing=false; playBtn.textContent='Play'; draw(); });
playBtn.addEventListener('click', () => { playing = !playing; playBtn.textContent = playing ? 'Pause' : 'Play'; });
document.getElementById('normalOnly').onclick = () => stepToKind('normal');
document.getElementById('fallbackOnly').onclick = () => stepToKind('fallback');
document.getElementById('allFrames').onclick = () => stepToKind('all');
canvas.addEventListener('mousedown', e => { dragging=true; dragX=e.clientX; dragY=e.clientY; canvas.classList.add('dragging'); });
window.addEventListener('mouseup', () => { dragging=false; canvas.classList.remove('dragging'); });
window.addEventListener('mousemove', e => {
  if (!dragging) return;
  yaw += (e.clientX - dragX) * 0.006;
  pitch += (e.clientY - dragY) * 0.004;
  pitch = Math.max(-1.3, Math.min(1.3, pitch));
  dragX=e.clientX; dragY=e.clientY; draw();
});
canvas.addEventListener('wheel', e => { e.preventDefault(); zoom *= Math.exp(-e.deltaY*0.001); zoom=Math.max(0.35,Math.min(5.0,zoom)); draw(); }, {passive:false});
window.addEventListener('keydown', e => {
  if (e.code === 'Space') { playing=!playing; playBtn.textContent=playing?'Pause':'Play'; e.preventDefault(); }
  if (e.key === 'ArrowRight') { frameIndex=Math.min(DATA.frames.length-1,frameIndex+1); draw(); }
  if (e.key === 'ArrowLeft') { frameIndex=Math.max(0,frameIndex-1); draw(); }
});
function tick(ts){
  if (!lastTs) lastTs = ts;
  const dt = (ts-lastTs)/1000; lastTs = ts;
  if (playing){
    accum += dt * Number(speedSel.value);
    while (accum > 1/60){
      frameIndex = (frameIndex + 1) % DATA.frames.length;
      accum -= 1/60;
    }
    draw();
  }
  requestAnimationFrame(tick);
}
window.addEventListener('resize', resize);
resize();
requestAnimationFrame(tick);
</script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--core-csv", required=True)
    parser.add_argument("--ik-csv", required=True)
    parser.add_argument("--ik-metadata", required=True)
    parser.add_argument("--table-json", required=True)
    parser.add_argument("--tcp-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--orientation-align-rpy", default="0,0,0")
    parser.add_argument("--title", default="QZY IK input/output fallback debugger")
    args = parser.parse_args()

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    payload, rows = build_payload(args)
    html_path = out_dir / "ik_input_output_fallback_debug.html"
    data_json = out_dir / "ik_input_output_fallback_debug_data.json"
    csv_path = out_dir / "ik_input_output_fallback_debug.csv"
    meta_path = out_dir / "ik_input_output_fallback_debug_summary.json"

    data_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(csv_path, rows)
    html = HTML_TEMPLATE.replace("__TITLE__", payload["title"]).replace("__DATA__", json.dumps(payload, separators=(",", ":"), allow_nan=True))
    html_path.write_text(html, encoding="utf-8")
    summary = {
        "html": str(html_path),
        "data_json": str(data_json),
        "csv": str(csv_path),
        "summary_json": str(meta_path),
        "summary": payload["summary"],
        "inputs": {
            "ik_csv": payload["ikCsv"],
            "ik_metadata": payload["ikMetadata"],
            "core_csv": payload["coreCsv"],
            "table_json": payload["tableJson"],
            "tcp_json": payload["tcpJson"],
        },
    }
    meta_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
