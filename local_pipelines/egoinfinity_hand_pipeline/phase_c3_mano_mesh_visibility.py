#!/usr/bin/env python3
"""Phase-C3 MANO mesh z-buffer visibility.

Input is the Phase-C2 MANO-smoothed NPZ.  For each hand candidate this stage
projects the smooth MANO mesh into the left-camera image, rasterizes a z-buffer,
and marks which MANO vertices and 21 hand joints are front-surface visible.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


RELIABLE_JOINT_IDS = [0, 5, 9, 13, 17]
MCP_JOINT_IDS = [5, 9, 13, 17]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session-dir", required=True)
    p.add_argument("--input-npz", required=True, help="Phase-C2 MANO-smoothed NPZ")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epsilon-m", type=float, default=0.010)
    p.add_argument("--surface-radius-px", type=int, default=2)
    p.add_argument("--nearest-vertices", type=int, default=18)
    p.add_argument("--joint-visible-ratio-threshold", type=float, default=0.25)
    p.add_argument("--min-visible-reliable-joints", type=int, default=2)
    p.add_argument("--zbuffer-pad-px", type=int, default=24)
    p.add_argument("--warn-vertex-visible-ratio", type=float, default=0.18)
    return p.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise RuntimeError(f"missing {label}: {path}")


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
    vals = []
    for value in values:
        try:
            f = float(value)
        except Exception:
            continue
        if math.isfinite(f):
            vals.append(f)
    arr = np.asarray(vals, dtype=np.float64)
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


def string_array(values: Sequence[str]) -> np.ndarray:
    max_len = max([1] + [len(str(v)) for v in values])
    return np.asarray([str(v) for v in values], dtype=f"<U{max_len}")


def finite_ratio(arr: np.ndarray) -> float:
    arr = np.asarray(arr)
    if arr.size == 0:
        return 0.0
    return float(np.mean(np.isfinite(arr)))


def load_camera(data: Dict[str, np.ndarray]) -> Dict[str, float]:
    if "foundation_camera_json" in data and len(data["foundation_camera_json"]):
        camera = json.loads(str(data["foundation_camera_json"][0]))
    else:
        image_size = np.asarray(data.get("image_size", [0, 0])).reshape(-1)
        width = int(image_size[0]) if image_size.size > 0 else 0
        height = int(image_size[1]) if image_size.size > 1 else 0
        focal = float(np.nanmedian(np.asarray(data.get("focal_length", [0.0]), dtype=np.float32)))
        camera = {"fx": focal, "fy": focal, "cx": width / 2.0, "cy": height / 2.0, "width": width, "height": height}
    for key in ["fx", "fy", "cx", "cy", "width", "height"]:
        if key not in camera:
            raise RuntimeError(f"camera missing {key}: {camera}")
    return camera


def project_points(points_cam: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    pts = np.asarray(points_cam, dtype=np.float32)
    uv = np.full((*pts.shape[:-1], 2), np.nan, dtype=np.float32)
    z = pts[..., 2]
    valid = np.isfinite(z) & (z > 1e-8)
    uv[..., 0][valid] = fx * pts[..., 0][valid] / z[valid] + cx
    uv[..., 1][valid] = fy * pts[..., 1][valid] / z[valid] + cy
    return uv


def rasterize_mesh_depth(
    vertices_cam: np.ndarray,
    faces: np.ndarray,
    vertices_uv: np.ndarray,
    width: int,
    height: int,
    clip_bbox_xyxy: np.ndarray,
    pad_px: int,
) -> np.ndarray:
    zbuf = np.full((height, width), np.inf, dtype=np.float32)
    if vertices_cam.size == 0 or faces.size == 0:
        return zbuf
    z = vertices_cam[:, 2].astype(np.float32)
    if clip_bbox_xyxy.shape[0] >= 4 and np.isfinite(clip_bbox_xyxy).all():
        clip_x1 = max(0, int(math.floor(float(clip_bbox_xyxy[0]))) - int(pad_px))
        clip_y1 = max(0, int(math.floor(float(clip_bbox_xyxy[1]))) - int(pad_px))
        clip_x2 = min(width - 1, int(math.ceil(float(clip_bbox_xyxy[2]))) + int(pad_px))
        clip_y2 = min(height - 1, int(math.ceil(float(clip_bbox_xyxy[3]))) + int(pad_px))
    else:
        finite_uv = vertices_uv[np.isfinite(vertices_uv).all(axis=1)]
        if finite_uv.size == 0:
            return zbuf
        clip_x1 = max(0, int(math.floor(float(np.nanmin(finite_uv[:, 0])))) - int(pad_px))
        clip_y1 = max(0, int(math.floor(float(np.nanmin(finite_uv[:, 1])))) - int(pad_px))
        clip_x2 = min(width - 1, int(math.ceil(float(np.nanmax(finite_uv[:, 0])))) + int(pad_px))
        clip_y2 = min(height - 1, int(math.ceil(float(np.nanmax(finite_uv[:, 1])))) + int(pad_px))

    for tri in faces.astype(np.int32):
        pts = vertices_uv[tri]
        zs = z[tri]
        if not np.isfinite(pts).all() or not np.isfinite(zs).all() or np.any(zs <= 1e-8):
            continue
        minx = max(clip_x1, int(math.floor(float(np.min(pts[:, 0])))))
        maxx = min(clip_x2, int(math.ceil(float(np.max(pts[:, 0])))))
        miny = max(clip_y1, int(math.floor(float(np.min(pts[:, 1])))))
        maxy = min(clip_y2, int(math.ceil(float(np.max(pts[:, 1])))))
        if maxx < minx or maxy < miny:
            continue
        x0, y0 = float(pts[0, 0]), float(pts[0, 1])
        x1, y1 = float(pts[1, 0]), float(pts[1, 1])
        x2, y2 = float(pts[2, 0]), float(pts[2, 1])
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(den) < 1e-8:
            continue
        yy, xx = np.mgrid[miny : maxy + 1, minx : maxx + 1]
        w0 = ((y1 - y2) * (xx - x2) + (x2 - x1) * (yy - y2)) / den
        w1 = ((y2 - y0) * (xx - x2) + (x0 - x2) * (yy - y2)) / den
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
        if not np.any(inside):
            continue
        tri_z = (w0 * float(zs[0]) + w1 * float(zs[1]) + w2 * float(zs[2])).astype(np.float32)
        region = zbuf[miny : maxy + 1, minx : maxx + 1]
        update = inside & (tri_z < region)
        region[update] = tri_z[update]
    return zbuf


def surface_depth_at(zbuf: np.ndarray, u: float, v: float, radius_px: int) -> float:
    if not np.isfinite(u) or not np.isfinite(v):
        return float("nan")
    h, w = zbuf.shape[:2]
    x = int(round(float(u)))
    y = int(round(float(v)))
    if x < 0 or x >= w or y < 0 or y >= h:
        return float("nan")
    x1 = max(0, x - int(radius_px))
    y1 = max(0, y - int(radius_px))
    x2 = min(w, x + int(radius_px) + 1)
    y2 = min(h, y + int(radius_px) + 1)
    patch = zbuf[y1:y2, x1:x2]
    finite = patch[np.isfinite(patch)]
    if finite.size == 0:
        return float("nan")
    return float(np.min(finite))


def vertex_visibility(
    vertices_cam: np.ndarray,
    vertices_uv: np.ndarray,
    zbuf: np.ndarray,
    radius_px: int,
    epsilon_m: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    visible = np.zeros((vertices_cam.shape[0],), dtype=np.uint8)
    surface_z = np.full((vertices_cam.shape[0],), np.nan, dtype=np.float32)
    margin = np.full((vertices_cam.shape[0],), np.nan, dtype=np.float32)
    for vid in range(vertices_cam.shape[0]):
        u, v = float(vertices_uv[vid, 0]), float(vertices_uv[vid, 1])
        vz = float(vertices_cam[vid, 2])
        surf = surface_depth_at(zbuf, u, v, int(radius_px))
        surface_z[vid] = surf
        if np.isfinite(surf) and np.isfinite(vz):
            m = vz - surf
            margin[vid] = m
            if m <= float(epsilon_m):
                visible[vid] = 1
    return visible, surface_z, margin


def joint_visibility_from_vertices(
    vertices_cam: np.ndarray,
    vertex_visible: np.ndarray,
    vertex_margin: np.ndarray,
    joints_cam: np.ndarray,
    joints_uv: np.ndarray,
    zbuf: np.ndarray,
    nearest_count: int,
    radius_px: int,
    ratio_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    joint_visible = np.zeros((joints_cam.shape[0],), dtype=np.uint8)
    joint_ratio = np.full((joints_cam.shape[0],), np.nan, dtype=np.float32)
    joint_margin = np.full((joints_cam.shape[0],), np.nan, dtype=np.float32)
    joint_surface_z = np.full((joints_cam.shape[0],), np.nan, dtype=np.float32)
    joint_tested = np.zeros((joints_cam.shape[0],), dtype=np.int32)
    for jid in range(joints_cam.shape[0]):
        joint = joints_cam[jid]
        if not np.isfinite(joint).all():
            continue
        joint_surface_z[jid] = surface_depth_at(zbuf, float(joints_uv[jid, 0]), float(joints_uv[jid, 1]), int(radius_px))
        d = np.linalg.norm(vertices_cam - joint.reshape(1, 3), axis=1)
        valid = np.isfinite(d)
        if not np.any(valid):
            continue
        valid_idx = np.where(valid)[0]
        order = valid_idx[np.argsort(d[valid])[: max(1, int(nearest_count))]]
        margins = vertex_margin[order]
        visible = vertex_visible[order]
        tested_mask = np.isfinite(margins)
        tested = int(np.sum(tested_mask))
        joint_tested[jid] = tested
        if tested == 0:
            continue
        ratio = float(np.sum(visible[tested_mask] > 0)) / float(tested)
        joint_ratio[jid] = ratio
        joint_margin[jid] = float(np.nanmedian(margins[tested_mask]))
        if ratio >= float(ratio_threshold):
            joint_visible[jid] = 1
    return joint_visible, joint_ratio, joint_margin, joint_surface_z, joint_tested


def write_joint_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fields = [
        "frame_index", "elapsed_sec", "hand_rank", "candidate_index", "track_id",
        "hand_label", "is_right", "landmark_id", "visible", "reliable_depth_joint",
        "u_px", "v_px", "cam_x_m", "cam_y_m", "cam_z_m",
        "mesh_surface_z_m", "mesh_visible_ratio", "mesh_surface_margin_m",
        "nearest_vertex_tested_count", "candidate_qc_flag",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def write_candidate_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fields = [
        "frame_index", "elapsed_sec", "hand_rank", "candidate_index", "track_id",
        "hand_label", "is_right", "visible_vertex_ratio", "visible_joint_count",
        "visible_reliable_joint_count", "visible_mcp_count", "zbuffer_pixel_count",
        "qc_flag",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def qc_flag_for_candidate(
    visible_vertex_ratio: float,
    visible_reliable_count: int,
    zbuffer_pixels: int,
    args: argparse.Namespace,
) -> str:
    flags: List[str] = []
    if zbuffer_pixels <= 0:
        flags.append("bad_empty_zbuffer")
    if visible_reliable_count < int(args.min_visible_reliable_joints):
        flags.append("bad_low_visible_reliable_joints")
    if np.isfinite(visible_vertex_ratio) and visible_vertex_ratio < float(args.warn_vertex_visible_ratio):
        flags.append("warn_low_visible_vertex_ratio")
    return "|".join(flags) if flags else "ok"


def main() -> int:
    args = parse_args()
    session_dir = Path(args.session_dir).expanduser().resolve()
    input_npz = Path(args.input_npz).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    require_file(input_npz, "Phase-C2 NPZ")
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_npz(input_npz)
    required = [
        "frame_index", "track_id", "hand_label", "is_right", "hand_rank",
        "candidate_index", "bbox_xyxy", "faces", "vertices_cam_smooth",
        "joints_cam_smooth", "joints_uv_smooth_depth_camera",
    ]
    missing = [key for key in required if key not in data]
    if missing:
        raise RuntimeError(f"input npz missing required fields: {missing}")

    camera = load_camera(data)
    fx, fy = float(camera["fx"]), float(camera["fy"])
    cx, cy = float(camera["cx"]), float(camera["cy"])
    width, height = int(camera["width"]), int(camera["height"])

    frame_index = np.asarray(data["frame_index"], dtype=np.int32)
    n = int(len(frame_index))
    faces = np.asarray(data["faces"], dtype=np.int32)
    vertices = np.asarray(data["vertices_cam_smooth"], dtype=np.float32)
    joints = np.asarray(data["joints_cam_smooth"], dtype=np.float32)
    joints_uv = np.asarray(data["joints_uv_smooth_depth_camera"], dtype=np.float32)
    bbox = np.asarray(data["bbox_xyxy"], dtype=np.float32)

    vertex_visible = np.zeros((n, vertices.shape[1]), dtype=np.uint8)
    vertex_surface_margin = np.full((n, vertices.shape[1]), np.nan, dtype=np.float32)
    joint_visible = np.zeros((n, joints.shape[1]), dtype=np.uint8)
    joint_visible_ratio = np.full((n, joints.shape[1]), np.nan, dtype=np.float32)
    joint_surface_margin = np.full((n, joints.shape[1]), np.nan, dtype=np.float32)
    joint_surface_z = np.full((n, joints.shape[1]), np.nan, dtype=np.float32)
    joint_tested_count = np.zeros((n, joints.shape[1]), dtype=np.int32)
    visible_vertex_ratio = np.full((n,), np.nan, dtype=np.float32)
    visible_joint_count = np.zeros((n,), dtype=np.int32)
    visible_reliable_count = np.zeros((n,), dtype=np.int32)
    visible_mcp_count = np.zeros((n,), dtype=np.int32)
    zbuffer_pixel_count = np.zeros((n,), dtype=np.int32)
    qc_flags: List[str] = []

    joint_rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []

    for i in range(n):
        verts_i = vertices[i]
        joints_i = joints[i]
        joints_uv_i = joints_uv[i]
        verts_uv_i = project_points(verts_i, fx, fy, cx, cy)
        zbuf = rasterize_mesh_depth(
            verts_i,
            faces,
            verts_uv_i,
            width,
            height,
            bbox[i],
            int(args.zbuffer_pad_px),
        )
        zbuffer_pixel_count[i] = int(np.count_nonzero(np.isfinite(zbuf)))
        v_visible, _, v_margin = vertex_visibility(
            verts_i,
            verts_uv_i,
            zbuf,
            int(args.surface_radius_px),
            float(args.epsilon_m),
        )
        vertex_visible[i] = v_visible
        vertex_surface_margin[i] = v_margin
        visible_vertex_ratio[i] = float(np.mean(v_visible > 0)) if v_visible.size else float("nan")
        j_visible, j_ratio, j_margin, j_surface_z, j_tested = joint_visibility_from_vertices(
            verts_i,
            v_visible,
            v_margin,
            joints_i,
            joints_uv_i,
            zbuf,
            int(args.nearest_vertices),
            int(args.surface_radius_px),
            float(args.joint_visible_ratio_threshold),
        )
        joint_visible[i] = j_visible
        joint_visible_ratio[i] = j_ratio
        joint_surface_margin[i] = j_margin
        joint_surface_z[i] = j_surface_z
        joint_tested_count[i] = j_tested
        visible_joint_count[i] = int(np.sum(j_visible > 0))
        visible_reliable_count[i] = int(np.sum(j_visible[RELIABLE_JOINT_IDS] > 0))
        visible_mcp_count[i] = int(np.sum(j_visible[MCP_JOINT_IDS] > 0))
        flag = qc_flag_for_candidate(
            float(visible_vertex_ratio[i]),
            int(visible_reliable_count[i]),
            int(zbuffer_pixel_count[i]),
            args,
        )
        qc_flags.append(flag)

        base = {
            "frame_index": int(frame_index[i]),
            "elapsed_sec": csv_float(float(data["elapsed_sec"][i])) if "elapsed_sec" in data else "",
            "hand_rank": int(data["hand_rank"][i]) if "hand_rank" in data else 0,
            "candidate_index": int(data["candidate_index"][i]) if "candidate_index" in data else i,
            "track_id": int(data["track_id"][i]),
            "hand_label": str(data["hand_label"][i]),
            "is_right": int(round(float(data["is_right"][i]))),
        }
        candidate_rows.append(
            {
                **base,
                "visible_vertex_ratio": csv_float(float(visible_vertex_ratio[i])),
                "visible_joint_count": int(visible_joint_count[i]),
                "visible_reliable_joint_count": int(visible_reliable_count[i]),
                "visible_mcp_count": int(visible_mcp_count[i]),
                "zbuffer_pixel_count": int(zbuffer_pixel_count[i]),
                "qc_flag": flag,
            }
        )
        for jid in range(joints.shape[1]):
            joint_rows.append(
                {
                    **base,
                    "landmark_id": jid,
                    "visible": int(j_visible[jid]),
                    "reliable_depth_joint": int(jid in RELIABLE_JOINT_IDS),
                    "u_px": csv_float(float(joints_uv_i[jid, 0])),
                    "v_px": csv_float(float(joints_uv_i[jid, 1])),
                    "cam_x_m": csv_float(float(joints_i[jid, 0])),
                    "cam_y_m": csv_float(float(joints_i[jid, 1])),
                    "cam_z_m": csv_float(float(joints_i[jid, 2])),
                    "mesh_surface_z_m": csv_float(float(j_surface_z[jid])),
                    "mesh_visible_ratio": csv_float(float(j_ratio[jid])),
                    "mesh_surface_margin_m": csv_float(float(j_margin[jid])),
                    "nearest_vertex_tested_count": int(j_tested[jid]),
                    "candidate_qc_flag": flag,
                }
            )

    output_npz = output_dir / "wilor_handresults_phase_c3_mesh_visibility.npz"
    joint_csv = output_dir / "mano_mesh_visibility_joints.csv"
    candidate_csv = output_dir / "mano_mesh_visibility_candidates.csv"
    summary_json = output_dir / "mano_mesh_visibility_summary.json"

    out = dict(data)
    out["mano_vertex_visible"] = vertex_visible
    out["mano_vertex_surface_margin_m"] = vertex_surface_margin.astype(np.float32)
    out["mano_joint_visible"] = joint_visible
    out["mano_joint_mesh_visible_ratio"] = joint_visible_ratio.astype(np.float32)
    out["mano_joint_mesh_surface_margin_m"] = joint_surface_margin.astype(np.float32)
    out["mano_joint_mesh_surface_z_m"] = joint_surface_z.astype(np.float32)
    out["mano_joint_nearest_vertex_tested_count"] = joint_tested_count.astype(np.int32)
    out["mano_visible_vertex_ratio"] = visible_vertex_ratio.astype(np.float32)
    out["mano_visible_joint_count"] = visible_joint_count.astype(np.int32)
    out["mano_visible_reliable_joint_count"] = visible_reliable_count.astype(np.int32)
    out["mano_visible_mcp_count"] = visible_mcp_count.astype(np.int32)
    out["mano_visibility_zbuffer_pixel_count"] = zbuffer_pixel_count.astype(np.int32)
    out["mano_visibility_qc_flag"] = string_array(qc_flags)
    np.savez_compressed(output_npz, **out)
    write_joint_csv(joint_csv, joint_rows)
    write_candidate_csv(candidate_csv, candidate_rows)

    flag_counts = Counter(qc_flags)
    hard_errors: List[str] = []
    warnings: List[str] = []
    if finite_ratio(visible_vertex_ratio) < 1.0:
        warnings.append("some_visible_vertex_ratio_non_finite")
    bad_count = sum(v for k, v in flag_counts.items() if "bad_" in str(k))
    warn_count = sum(v for k, v in flag_counts.items() if "warn_" in str(k))
    if bad_count:
        warnings.append(f"phase_c3_bad_visibility_flags:{bad_count}")
    if warn_count:
        warnings.append(f"phase_c3_warn_visibility_flags:{warn_count}")
    if np.max(zbuffer_pixel_count) <= 0:
        hard_errors.append("all_zbuffers_empty")

    summary = {
        "semantic": "LFV Phase-C3 MANO mesh z-buffer visibility in left camera frame",
        "session_dir": str(session_dir),
        "input_npz": str(input_npz),
        "output_npz": str(output_npz),
        "joint_visibility_csv": str(joint_csv),
        "candidate_visibility_csv": str(candidate_csv),
        "candidates": n,
        "camera": camera,
        "epsilon_m": float(args.epsilon_m),
        "surface_radius_px": int(args.surface_radius_px),
        "nearest_vertices": int(args.nearest_vertices),
        "joint_visible_ratio_threshold": float(args.joint_visible_ratio_threshold),
        "min_visible_reliable_joints": int(args.min_visible_reliable_joints),
        "qc_flag_counts": dict(flag_counts),
        "visible_vertex_ratio": stats(visible_vertex_ratio),
        "visible_joint_count": stats(visible_joint_count),
        "visible_reliable_joint_count": stats(visible_reliable_count),
        "visible_mcp_count": stats(visible_mcp_count),
        "zbuffer_pixel_count": stats(zbuffer_pixel_count),
        "joint_mesh_visible_ratio": stats(joint_visible_ratio.reshape(-1)),
        "joint_mesh_surface_margin_m": stats(joint_surface_margin.reshape(-1)),
        "smooth_geometry_finite_ratio": {
            "vertices_cam_smooth": finite_ratio(vertices),
            "joints_cam_smooth": finite_ratio(joints),
        },
        "hard_errors": hard_errors,
        "warnings": warnings,
        "ok": len(hard_errors) == 0,
    }
    summary_json.write_text(json.dumps(json_clean(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(json_clean(summary), indent=2, ensure_ascii=False))
    return 0 if len(hard_errors) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
