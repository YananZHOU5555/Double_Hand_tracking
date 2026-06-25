#!/usr/bin/env python3
"""Diagnostic Phase-C1a: silhouette-guided MANO XY alignment.

This stage is an experimental alternative to the current sparse-joint median
translation.  It keeps the WiLoR/MANO pose fixed, chooses one semantic depth
anchor, estimates that anchor depth from FoundationStereo inside a hand mask,
and searches camera-frame X/Y translation so the projected MANO silhouette
matches the hand segmentation mask.

It writes a full-length NPZ with additional fields.  Rows outside
--process-frames are copied from the input Phase-C geometry.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from diagnose_phase_c_mask_gated_depth import (  # noqa: E402
    SamHandSegmenter,
    bbox_mask,
    bbox_xyxy_to_xywh,
    clean_mask,
    grabcut_mask,
    json_clean,
    load_camera,
    load_npz,
    parse_frame_list,
    read_depth_frame_csv,
    stats,
)


HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
RELIABLE_IDS = [0, 5, 9, 13, 17]
ANCHORS = [
    ("palm_center", RELIABLE_IDS),
    ("middle_mcp", [9]),
    ("wrist", [0]),
    ("index_mcp", [5]),
    ("ring_mcp", [13]),
    ("pinky_mcp", [17]),
]

RAW_COLOR = (255, 0, 255)
OLD_COLOR = (45, 45, 255)
NEW_COLOR = (60, 255, 60)
ANCHOR_COLOR = (0, 255, 255)
WHITE = (245, 245, 245)
BLACK = (0, 0, 0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session-dir", required=True)
    p.add_argument("--source-pipeline", default="egoinfinity_hand_pipeline")
    p.add_argument("--input-npz", default="")
    p.add_argument("--depth-summary-json", default="")
    p.add_argument("--output-dir", default="")
    p.add_argument("--video", default="")
    p.add_argument("--segmenter", choices=["sam", "grabcut", "bbox"], default="sam")
    p.add_argument("--sam-checkpoint", default="/home/yannan/workspace/learning-from-video/models/sam/sam_vit_b_01ec64.pth")
    p.add_argument("--sam-model-type", default="vit_b")
    p.add_argument("--sam-device", default="cuda")
    p.add_argument("--grabcut-iters", type=int, default=1)
    p.add_argument("--bbox-pad-ratio", type=float, default=0.15)
    p.add_argument("--mask-dilate-px", type=int, default=2)
    p.add_argument("--anchor-patch-radius", type=int, default=2)
    p.add_argument("--anchor-min-pixels", type=int, default=5)
    p.add_argument("--anchor-max-depth-span-m", type=float, default=0.12)
    p.add_argument("--xy-search-px", type=float, default=34.0)
    p.add_argument("--xy-step-px", type=float, default=6.0)
    p.add_argument("--xy-refine-px", type=float, default=10.0)
    p.add_argument("--xy-refine-step-px", type=float, default=2.0)
    p.add_argument("--forearm-weight", type=float, default=0.20)
    p.add_argument("--precision-weight", type=float, default=0.35)
    p.add_argument("--anchor-shift-penalty", type=float, default=0.0015)
    p.add_argument("--min-silhouette-area", type=int, default=80)
    p.add_argument("--process-frames", default="590-630")
    p.add_argument("--frames", default="590-630")
    p.add_argument("--worst-count", type=int, default=16)
    p.add_argument("--scale", type=float, default=0.70)
    return p.parse_args()


def load_depth_summary(source_root: Path, requested: str = "") -> Tuple[Path, Dict[int, Path]]:
    if requested:
        summary_path = Path(requested).expanduser().resolve()
    else:
        stable = source_root / "foundationstereo_depth_stabilized" / "foundationstereo_depth_stabilized_summary.json"
        raw = source_root / "foundationstereo_depth" / "foundationstereo_depth_summary.json"
        summary_path = stable if stable.exists() else raw
    if not summary_path.exists():
        raise RuntimeError(f"missing depth summary: {summary_path}")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    frame_csv = Path(payload["outputs"]["frame_csv"])
    return summary_path, read_depth_frame_csv(frame_csv)


def project(points_cam: np.ndarray, cam: Dict[str, float]) -> np.ndarray:
    pts = np.asarray(points_cam, dtype=np.float64)
    uv = np.full((pts.shape[0], 2), np.nan, dtype=np.float64)
    z = pts[:, 2]
    valid = np.isfinite(pts).all(axis=1) & (z > 1e-8)
    uv[valid, 0] = float(cam["fx"]) * pts[valid, 0] / z[valid] + float(cam["cx"])
    uv[valid, 1] = float(cam["fy"]) * pts[valid, 1] / z[valid] + float(cam["cy"])
    return uv


def backproject(u: float, v: float, z: float, cam: Dict[str, float]) -> np.ndarray:
    return np.asarray([
        (float(u) - float(cam["cx"])) * float(z) / float(cam["fx"]),
        (float(v) - float(cam["cy"])) * float(z) / float(cam["fy"]),
        float(z),
    ], dtype=np.float64)


def uv_rms(stage: np.ndarray, ref: np.ndarray) -> float:
    diff = np.asarray(stage, dtype=np.float64) - np.asarray(ref, dtype=np.float64)
    valid = np.isfinite(diff).all(axis=1)
    if not np.any(valid):
        return float("nan")
    dist = np.linalg.norm(diff[valid], axis=1)
    return float(np.sqrt(np.mean(dist * dist)))


def read_video_frame(cap: cv2.VideoCapture, frame_index: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f"failed to read video frame {frame_index}")
    return frame


def anchor_rel_and_uv(joints_rel: np.ndarray, joints_uv: np.ndarray, ids: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    idx = [int(v) for v in ids]
    rel = np.asarray(joints_rel, dtype=np.float64)[idx]
    uv = np.asarray(joints_uv, dtype=np.float64)[idx]
    return np.nanmean(rel, axis=0), np.nanmean(uv, axis=0)


def sample_depth_masked(
    depth: np.ndarray,
    mask: np.ndarray,
    u: float,
    v: float,
    radius: int,
    min_pixels: int,
    max_span_m: float,
) -> Tuple[float, int, float, float, str]:
    if not np.isfinite([u, v]).all():
        return float("nan"), 0, float("nan"), float("nan"), "nonfinite_uv"
    h, w = depth.shape[:2]
    x = int(round(float(u)))
    y = int(round(float(v)))
    if x < 0 or x >= w or y < 0 or y >= h:
        return float("nan"), 0, float("nan"), float("nan"), "outside_image"
    r = int(radius)
    x1, x2 = max(0, x - r), min(w, x + r + 1)
    y1, y2 = max(0, y - r), min(h, y + r + 1)
    d_patch = np.asarray(depth[y1:y2, x1:x2], dtype=np.float32)
    m_patch = np.asarray(mask[y1:y2, x1:x2] > 0)
    vals = d_patch[np.isfinite(d_patch) & (d_patch > 0.01) & m_patch]
    if vals.size < int(min_pixels):
        return float("nan"), int(vals.size), float("nan"), float("nan"), "no_masked_depth"
    span = float(np.percentile(vals, 90) - np.percentile(vals, 10)) if vals.size >= 2 else 0.0
    std = float(np.std(vals))
    if math.isfinite(max_span_m) and span > float(max_span_m):
        return float("nan"), int(vals.size), span, std, "depth_span_high"
    return float(np.median(vals)), int(vals.size), span, std, "ok"


def choose_anchor(
    joints_rel: np.ndarray,
    joints_uv: np.ndarray,
    depth: np.ndarray,
    hand_mask: np.ndarray,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    for name, ids in ANCHORS:
        rel, uv = anchor_rel_and_uv(joints_rel, joints_uv, ids)
        z, count, span, std, status = sample_depth_masked(
            depth,
            hand_mask,
            float(uv[0]),
            float(uv[1]),
            int(args.anchor_patch_radius),
            int(args.anchor_min_pixels),
            float(args.anchor_max_depth_span_m),
        )
        if status == "ok":
            return {
                "anchor_name": name,
                "anchor_ids": ",".join(str(v) for v in ids),
                "anchor_rel": rel,
                "anchor_uv_initial": uv,
                "anchor_depth_m_initial": float(z),
                "anchor_valid_pixels_initial": int(count),
                "anchor_depth_span_m_initial": float(span),
                "anchor_depth_std_m_initial": float(std),
                "anchor_status": "ok",
            }
    return {
        "anchor_name": "",
        "anchor_ids": "",
        "anchor_rel": np.full(3, np.nan, dtype=np.float64),
        "anchor_uv_initial": np.full(2, np.nan, dtype=np.float64),
        "anchor_depth_m_initial": float("nan"),
        "anchor_valid_pixels_initial": 0,
        "anchor_depth_span_m_initial": float("nan"),
        "anchor_depth_std_m_initial": float("nan"),
        "anchor_status": "no_valid_anchor_depth",
    }


def make_target_weight(mask: np.ndarray, joints_uv: np.ndarray, forearm_weight: float) -> np.ndarray:
    weight = (mask > 0).astype(np.float32)
    if not np.isfinite(joints_uv[[0, 5, 9, 13, 17]]).all():
        return weight
    wrist = np.asarray(joints_uv[0], dtype=np.float32)
    mcp_center = np.mean(np.asarray(joints_uv[[5, 9, 13, 17]], dtype=np.float32), axis=0)
    direction = wrist - mcp_center
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return weight
    direction /= norm
    ys, xs = np.nonzero(weight > 0)
    if xs.size == 0:
        return weight
    pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    proj = (pts - wrist.reshape(1, 2)) @ direction.reshape(2, 1)
    forearm = proj[:, 0] > 0.0
    weight[ys[forearm], xs[forearm]] *= float(forearm_weight)
    return weight


def rasterize_mesh_silhouette(
    vertices_cam: np.ndarray,
    faces: np.ndarray,
    cam: Dict[str, float],
    shape: Tuple[int, int],
) -> np.ndarray:
    h, w = shape
    uv = project(vertices_cam, cam)
    z = vertices_cam[:, 2]
    out = np.zeros((h, w), dtype=np.uint8)
    if uv.size == 0:
        return out
    for tri in np.asarray(faces, dtype=np.int32):
        if tri.shape[0] != 3:
            continue
        if not (np.isfinite(uv[tri]).all() and np.isfinite(z[tri]).all() and np.all(z[tri] > 1e-5)):
            continue
        pts = np.rint(uv[tri]).astype(np.int32)
        if (
            np.max(pts[:, 0]) < 0 or np.min(pts[:, 0]) >= w or
            np.max(pts[:, 1]) < 0 or np.min(pts[:, 1]) >= h
        ):
            continue
        cv2.fillConvexPoly(out, pts, 255, lineType=cv2.LINE_AA)
    return out


def silhouette_score(
    sil: np.ndarray,
    target_weight: np.ndarray,
    anchor_shift_px: float,
    args: argparse.Namespace,
) -> Tuple[float, float, float, int]:
    sil_bool = sil > 0
    sil_area = int(np.count_nonzero(sil_bool))
    if sil_area < int(args.min_silhouette_area):
        return -1e9, 0.0, 0.0, sil_area
    target_sum = float(np.sum(target_weight))
    if target_sum <= 1e-6:
        return -1e9, 0.0, 0.0, sil_area
    overlap_weight = float(np.sum(target_weight[sil_bool]))
    coverage = overlap_weight / target_sum
    precision = overlap_weight / float(max(1, sil_area))
    score = (1.0 - float(args.precision_weight)) * coverage + float(args.precision_weight) * precision
    score -= float(args.anchor_shift_penalty) * float(anchor_shift_px)
    return float(score), float(coverage), float(precision), sil_area


def search_xy(
    vertices_rel: np.ndarray,
    faces: np.ndarray,
    anchor_rel: np.ndarray,
    anchor_uv_target: np.ndarray,
    cam_t_seed: np.ndarray,
    cam: Dict[str, float],
    target_weight: np.ndarray,
    args: argparse.Namespace,
    search_px: float,
    step_px: float,
    center_delta_px: Tuple[float, float] = (0.0, 0.0),
) -> Dict[str, Any]:
    z_anchor = float(anchor_rel[2] + cam_t_seed[2])
    if not math.isfinite(z_anchor) or z_anchor <= 1e-6:
        return {"ok": False, "reason": "bad_anchor_z"}
    offsets = np.arange(-float(search_px), float(search_px) + 1e-6, float(step_px), dtype=np.float64)
    best: Dict[str, Any] = {"score": -1e18, "ok": False, "reason": "no_candidate"}
    shape = target_weight.shape[:2]
    for du in offsets + float(center_delta_px[0]):
        for dv in offsets + float(center_delta_px[1]):
            tx = ((float(anchor_uv_target[0]) + du - float(cam["cx"])) * z_anchor / float(cam["fx"])) - float(anchor_rel[0])
            ty = ((float(anchor_uv_target[1]) + dv - float(cam["cy"])) * z_anchor / float(cam["fy"])) - float(anchor_rel[1])
            cam_t = np.asarray([tx, ty, float(cam_t_seed[2])], dtype=np.float64)
            verts_cam = np.asarray(vertices_rel, dtype=np.float64) + cam_t.reshape(1, 3)
            sil = rasterize_mesh_silhouette(verts_cam, faces, cam, shape)
            score, coverage, precision, area = silhouette_score(
                sil,
                target_weight,
                math.hypot(float(du), float(dv)),
                args,
            )
            if score > float(best["score"]):
                best = {
                    "ok": True,
                    "score": float(score),
                    "coverage": float(coverage),
                    "precision": float(precision),
                    "silhouette_area": int(area),
                    "cam_t": cam_t,
                    "delta_u_px": float(du),
                    "delta_v_px": float(dv),
                    "silhouette": sil,
                }
    return best


def refine_depth_at_moved_anchor(
    anchor_rel: np.ndarray,
    cam_t: np.ndarray,
    depth: np.ndarray,
    hand_mask: np.ndarray,
    cam: Dict[str, float],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    uv = project((anchor_rel + cam_t).reshape(1, 3), cam)[0]
    z, count, span, std, status = sample_depth_masked(
        depth,
        hand_mask,
        float(uv[0]),
        float(uv[1]),
        int(args.anchor_patch_radius),
        int(args.anchor_min_pixels),
        float(args.anchor_max_depth_span_m),
    )
    if status != "ok":
        return {
            "ok": False,
            "status": status,
            "anchor_uv_refined": uv,
            "anchor_depth_m_refined": float("nan"),
            "anchor_valid_pixels_refined": int(count),
            "anchor_depth_span_m_refined": float(span),
            "anchor_depth_std_m_refined": float(std),
        }
    anchor_cam = backproject(float(uv[0]), float(uv[1]), float(z), cam)
    refined_cam_t = anchor_cam - anchor_rel
    return {
        "ok": True,
        "status": "ok",
        "anchor_uv_refined": uv,
        "anchor_depth_m_refined": float(z),
        "anchor_valid_pixels_refined": int(count),
        "anchor_depth_span_m_refined": float(span),
        "anchor_depth_std_m_refined": float(std),
        "cam_t": refined_cam_t,
    }


def draw_text(img: np.ndarray, text: str, org: Tuple[int, int], color: Tuple[int, int, int] = WHITE, scale: float = 0.42) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_skeleton(img: np.ndarray, uv: np.ndarray, color: Tuple[int, int, int], thickness: int = 2, radius: int = 3, alpha: float = 0.8) -> None:
    overlay = img.copy()
    h, w = img.shape[:2]

    def ok(p: np.ndarray) -> bool:
        return np.isfinite(p).all() and -80 <= float(p[0]) <= w + 80 and -80 <= float(p[1]) <= h + 80

    for a, b in HAND_EDGES:
        if ok(uv[a]) and ok(uv[b]):
            cv2.line(
                overlay,
                (int(round(float(uv[a, 0]))), int(round(float(uv[a, 1])))),
                (int(round(float(uv[b, 0]))), int(round(float(uv[b, 1])))),
                color,
                thickness,
                cv2.LINE_AA,
            )
    for i in range(min(21, uv.shape[0])):
        if ok(uv[i]):
            cv2.circle(overlay, (int(round(float(uv[i, 0]))), int(round(float(uv[i, 1])))), radius, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0.0, dst=img)


def make_mask_overlay(image: np.ndarray, mask: np.ndarray, silhouette: Optional[np.ndarray] = None) -> np.ndarray:
    out = image.copy()
    green = np.zeros_like(out)
    green[:, :, 1] = 190
    hit = mask > 0
    out[hit] = cv2.addWeighted(out, 0.60, green, 0.40, 0.0)[hit]
    if silhouette is not None:
        red = np.zeros_like(out)
        red[:, :, 2] = 220
        s = silhouette > 0
        out[s] = cv2.addWeighted(out, 0.60, red, 0.40, 0.0)[s]
    return out


def tile_images(images: Sequence[np.ndarray], columns: int, scale: float) -> np.ndarray:
    if not images:
        raise RuntimeError("no images to tile")
    h, w = images[0].shape[:2]
    rows = int(math.ceil(len(images) / float(columns)))
    canvas = np.zeros((rows * h, columns * w, 3), dtype=np.uint8)
    for i, img in enumerate(images):
        y = (i // columns) * h
        x = (i % columns) * w
        canvas[y:y + h, x:x + w] = img
    if scale != 1.0:
        canvas = cv2.resize(canvas, (max(1, int(canvas.shape[1] * scale)), max(1, int(canvas.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    return canvas


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fields = [
        "frame_index", "row", "hand_label", "track_id", "hand_rank", "status",
        "anchor_name", "anchor_ids", "anchor_depth_m_initial", "anchor_valid_pixels_initial",
        "anchor_depth_span_m_initial", "anchor_depth_std_m_initial", "anchor_depth_m_refined",
        "anchor_valid_pixels_refined", "anchor_depth_span_m_refined", "anchor_depth_std_m_refined",
        "old_rms_px", "new_rms_px", "delta_rms_px", "old_mask_score", "new_mask_score",
        "old_mask_coverage", "new_mask_coverage", "old_mask_precision", "new_mask_precision",
        "new_delta_u_px", "new_delta_v_px", "new_silhouette_area", "old_cam_t_x",
        "old_cam_t_y", "old_cam_t_z", "new_cam_t_x", "new_cam_t_y", "new_cam_t_z",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def main() -> int:
    args = parse_args()
    session = Path(args.session_dir).expanduser().resolve()
    source_root = session / "quality" / str(args.source_pipeline) / "stages"
    input_npz = Path(args.input_npz).expanduser().resolve() if args.input_npz else source_root / "phase_c_depth_align" / "wilor_handresults_phase_c_depth_aligned.npz"
    video = Path(args.video).expanduser().resolve() if args.video else session / "processed_topcam" / "left_table.mp4"
    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else session / "quality" / "egoinfinity_hand_alignment_pipeline" / "quality_check" / "phase_c1a_silhouette_xy_align"
    out_dir.mkdir(parents=True, exist_ok=True)
    depth_summary_path, depth_paths = load_depth_summary(source_root, args.depth_summary_json)

    data = load_npz(input_npz)
    cam = load_camera(data)
    faces = np.asarray(data["faces"], dtype=np.int32)
    frames_filter = set(parse_frame_list(str(args.process_frames))) if str(args.process_frames).strip() else None
    selected_frames = set(parse_frame_list(str(args.frames))) if str(args.frames).strip() else set()

    n = int(len(data["frame_index"]))
    cam_t_sil = np.asarray(data["cam_t_depth"], dtype=np.float32).copy()
    joints_cam_sil = np.asarray(data["joints_cam_depth"], dtype=np.float32).copy()
    vertices_cam_sil = np.asarray(data["vertices_cam_depth"], dtype=np.float32).copy()
    joints_uv_sil = np.stack([project(joints_cam_sil[i], cam) for i in range(n)], axis=0).astype(np.float32)
    status_arr = np.full(n, "not_processed", dtype="<U64")
    anchor_arr = np.full(n, "", dtype="<U32")
    mask_score_arr = np.full(n, np.nan, dtype=np.float32)
    mask_coverage_arr = np.full(n, np.nan, dtype=np.float32)
    mask_precision_arr = np.full(n, np.nan, dtype=np.float32)
    anchor_depth_arr = np.full(n, np.nan, dtype=np.float32)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sam: Optional[SamHandSegmenter] = None
    if args.segmenter == "sam":
        sam_ckpt = Path(args.sam_checkpoint).expanduser().resolve()
        if not sam_ckpt.exists():
            raise RuntimeError(f"missing SAM checkpoint: {sam_ckpt}")
        sam = SamHandSegmenter(sam_ckpt, str(args.sam_model_type), str(args.sam_device))

    frame_cache: Dict[int, np.ndarray] = {}
    depth_cache: Dict[int, np.ndarray] = {}
    rows: List[Dict[str, Any]] = []
    viz_rows: List[Dict[str, Any]] = []

    for i in range(n):
        frame = int(data["frame_index"][i])
        if frames_filter is not None and frame not in frames_filter:
            continue
        if frame not in depth_paths:
            status_arr[i] = "missing_depth"
            continue
        if frame not in frame_cache:
            frame_cache[frame] = read_video_frame(cap, frame)
        if frame not in depth_cache:
            depth_cache[frame] = np.load(depth_paths[frame]).astype(np.float32)
        image = frame_cache[frame]
        depth = depth_cache[frame]
        bbox_xywh = bbox_xyxy_to_xywh(np.asarray(data["bbox_xyxy"][i], dtype=np.float64), width, height, float(args.bbox_pad_ratio))
        if args.segmenter == "bbox":
            hand_mask, seg_score = bbox_mask(image.shape[:2], bbox_xywh)
        elif args.segmenter == "grabcut":
            hand_mask, seg_score = grabcut_mask(image, bbox_xywh, int(args.grabcut_iters))
        else:
            assert sam is not None
            hand_mask, seg_score = sam.segment(image, bbox_xywh, np.asarray(data["joints_uv"][i], dtype=np.float32))
        if int(args.mask_dilate_px) > 0:
            k = int(args.mask_dilate_px) * 2 + 1
            hand_mask = cv2.dilate(hand_mask, np.ones((k, k), dtype=np.uint8), iterations=1)
            hand_mask = clean_mask(hand_mask, keep_largest=True)
        target_weight = make_target_weight(hand_mask, np.asarray(data["joints_uv"][i], dtype=np.float64), float(args.forearm_weight))

        joints_rel = np.asarray(data["joints_3d_rel"][i], dtype=np.float64)
        vertices_rel = np.asarray(data["vertices_rel"][i], dtype=np.float64)
        joints_uv_raw = np.asarray(data["joints_uv"][i], dtype=np.float64)
        old_cam_t = np.asarray(data["cam_t_depth"][i], dtype=np.float64)
        old_uv = project(np.asarray(data["joints_cam_depth"][i], dtype=np.float64), cam)
        old_sil = rasterize_mesh_silhouette(vertices_rel + old_cam_t.reshape(1, 3), faces, cam, hand_mask.shape[:2])
        old_score, old_cov, old_prec, _old_area = silhouette_score(old_sil, target_weight, 0.0, args)
        old_rms = uv_rms(old_uv, joints_uv_raw)

        anchor = choose_anchor(joints_rel, joints_uv_raw, depth, hand_mask, args)
        if anchor["anchor_status"] != "ok":
            status_arr[i] = anchor["anchor_status"]
            row = {
                "frame_index": frame, "row": i, "hand_label": str(data["hand_label"][i]),
                "track_id": int(data["track_id"][i]), "hand_rank": int(data["hand_rank"][i]),
                "status": str(anchor["anchor_status"]), "old_rms_px": f"{old_rms:.6f}",
                "old_mask_score": f"{old_score:.9f}", "old_mask_coverage": f"{old_cov:.9f}",
                "old_mask_precision": f"{old_prec:.9f}",
            }
            rows.append(row)
            continue

        anchor_rel = np.asarray(anchor["anchor_rel"], dtype=np.float64)
        anchor_uv = np.asarray(anchor["anchor_uv_initial"], dtype=np.float64)
        anchor_cam = backproject(float(anchor_uv[0]), float(anchor_uv[1]), float(anchor["anchor_depth_m_initial"]), cam)
        seed_cam_t = anchor_cam - anchor_rel
        coarse = search_xy(vertices_rel, faces, anchor_rel, anchor_uv, seed_cam_t, cam, target_weight, args, float(args.xy_search_px), float(args.xy_step_px))
        if not coarse.get("ok"):
            status_arr[i] = str(coarse.get("reason", "xy_search_failed"))
            continue
        refine_depth = refine_depth_at_moved_anchor(anchor_rel, np.asarray(coarse["cam_t"], dtype=np.float64), depth, hand_mask, cam, args)
        depth_cam_t = np.asarray(refine_depth["cam_t"], dtype=np.float64) if refine_depth.get("ok") else np.asarray(coarse["cam_t"], dtype=np.float64)
        fine = search_xy(
            vertices_rel,
            faces,
            anchor_rel,
            np.asarray(refine_depth.get("anchor_uv_refined", project((anchor_rel + depth_cam_t).reshape(1, 3), cam)[0]), dtype=np.float64),
            depth_cam_t,
            cam,
            target_weight,
            args,
            float(args.xy_refine_px),
            float(args.xy_refine_step_px),
            center_delta_px=(0.0, 0.0),
        )
        best = fine if fine.get("ok") else coarse
        final_depth = refine_depth_at_moved_anchor(anchor_rel, np.asarray(best["cam_t"], dtype=np.float64), depth, hand_mask, cam, args)
        final_cam_t = np.asarray(final_depth["cam_t"], dtype=np.float64) if final_depth.get("ok") else np.asarray(best["cam_t"], dtype=np.float64)
        final_joints = joints_rel + final_cam_t.reshape(1, 3)
        final_vertices = vertices_rel + final_cam_t.reshape(1, 3)
        final_uv = project(final_joints, cam)
        final_sil = rasterize_mesh_silhouette(final_vertices, faces, cam, hand_mask.shape[:2])
        final_score, final_cov, final_prec, final_area = silhouette_score(final_sil, target_weight, math.hypot(float(best["delta_u_px"]), float(best["delta_v_px"])), args)
        final_rms = uv_rms(final_uv, joints_uv_raw)

        cam_t_sil[i] = final_cam_t.astype(np.float32)
        joints_cam_sil[i] = final_joints.astype(np.float32)
        vertices_cam_sil[i] = final_vertices.astype(np.float32)
        joints_uv_sil[i] = final_uv.astype(np.float32)
        status_arr[i] = "ok" if final_depth.get("ok") else "ok_depth_refine_failed"
        anchor_arr[i] = str(anchor["anchor_name"])
        mask_score_arr[i] = float(final_score)
        mask_coverage_arr[i] = float(final_cov)
        mask_precision_arr[i] = float(final_prec)
        anchor_depth_arr[i] = float(final_depth.get("anchor_depth_m_refined", anchor["anchor_depth_m_initial"]))

        row = {
            "frame_index": frame,
            "row": i,
            "hand_label": str(data["hand_label"][i]),
            "track_id": int(data["track_id"][i]),
            "hand_rank": int(data["hand_rank"][i]),
            "status": str(status_arr[i]),
            "anchor_name": str(anchor["anchor_name"]),
            "anchor_ids": str(anchor["anchor_ids"]),
            "anchor_depth_m_initial": f"{float(anchor['anchor_depth_m_initial']):.9f}",
            "anchor_valid_pixels_initial": int(anchor["anchor_valid_pixels_initial"]),
            "anchor_depth_span_m_initial": f"{float(anchor['anchor_depth_span_m_initial']):.9f}",
            "anchor_depth_std_m_initial": f"{float(anchor['anchor_depth_std_m_initial']):.9f}",
            "anchor_depth_m_refined": f"{float(final_depth.get('anchor_depth_m_refined', float('nan'))):.9f}" if math.isfinite(float(final_depth.get("anchor_depth_m_refined", float("nan")))) else "",
            "anchor_valid_pixels_refined": int(final_depth.get("anchor_valid_pixels_refined", 0)),
            "anchor_depth_span_m_refined": f"{float(final_depth.get('anchor_depth_span_m_refined', float('nan'))):.9f}" if math.isfinite(float(final_depth.get("anchor_depth_span_m_refined", float("nan")))) else "",
            "anchor_depth_std_m_refined": f"{float(final_depth.get('anchor_depth_std_m_refined', float('nan'))):.9f}" if math.isfinite(float(final_depth.get("anchor_depth_std_m_refined", float("nan")))) else "",
            "old_rms_px": f"{old_rms:.6f}",
            "new_rms_px": f"{final_rms:.6f}",
            "delta_rms_px": f"{final_rms - old_rms:.6f}",
            "old_mask_score": f"{old_score:.9f}",
            "new_mask_score": f"{final_score:.9f}",
            "old_mask_coverage": f"{old_cov:.9f}",
            "new_mask_coverage": f"{final_cov:.9f}",
            "old_mask_precision": f"{old_prec:.9f}",
            "new_mask_precision": f"{final_prec:.9f}",
            "new_delta_u_px": f"{float(best['delta_u_px']):.6f}",
            "new_delta_v_px": f"{float(best['delta_v_px']):.6f}",
            "new_silhouette_area": int(final_area),
            "old_cam_t_x": f"{old_cam_t[0]:.9f}",
            "old_cam_t_y": f"{old_cam_t[1]:.9f}",
            "old_cam_t_z": f"{old_cam_t[2]:.9f}",
            "new_cam_t_x": f"{final_cam_t[0]:.9f}",
            "new_cam_t_y": f"{final_cam_t[1]:.9f}",
            "new_cam_t_z": f"{final_cam_t[2]:.9f}",
            "_image": image,
            "_mask": hand_mask,
            "_old_sil": old_sil,
            "_new_sil": final_sil,
            "_raw_uv": joints_uv_raw,
            "_old_uv": old_uv,
            "_new_uv": final_uv,
            "_anchor_uv_initial": anchor_uv,
            "_anchor_uv_refined": np.asarray(final_depth.get("anchor_uv_refined", anchor_uv), dtype=np.float64),
        }
        rows.append(row)
        if frame in selected_frames or len(viz_rows) < int(args.worst_count):
            viz_rows.append(row)

    cap.release()

    out = dict(data)
    out["cam_t_silhouette"] = cam_t_sil
    out["joints_cam_silhouette"] = joints_cam_sil
    out["vertices_cam_silhouette"] = vertices_cam_sil
    out["joints_uv_silhouette"] = joints_uv_sil
    out["silhouette_status"] = status_arr
    out["silhouette_anchor_name"] = anchor_arr
    out["silhouette_mask_score"] = mask_score_arr
    out["silhouette_mask_coverage"] = mask_coverage_arr
    out["silhouette_mask_precision"] = mask_precision_arr
    out["silhouette_anchor_depth_m"] = anchor_depth_arr
    output_npz = out_dir / "wilor_handresults_phase_c1a_silhouette_xy_aligned.npz"
    np.savez_compressed(output_npz, **out)

    csv_rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    output_csv = out_dir / "phase_c1a_silhouette_xy_align_candidates.csv"
    write_csv(output_csv, csv_rows)

    def make_panel(row: Dict[str, Any]) -> np.ndarray:
        image = make_mask_overlay(row["_image"], row["_mask"], row["_new_sil"])
        draw_skeleton(image, row["_raw_uv"], RAW_COLOR, 1, 2, 0.38)
        draw_skeleton(image, row["_old_uv"], OLD_COLOR, 2, 3, 0.62)
        draw_skeleton(image, row["_new_uv"], NEW_COLOR, 2, 3, 0.84)
        au = row["_anchor_uv_refined"]
        if np.isfinite(au).all():
            cv2.circle(image, (int(round(float(au[0]))), int(round(float(au[1])))), 8, ANCHOR_COLOR, 2, cv2.LINE_AA)
        draw_text(image, f"f={row['frame_index']} {row['hand_label']} tr={row['track_id']} {row['anchor_name']} {row['status']}", (8, 18), WHITE, 0.42)
        draw_text(image, f"old red={row['old_rms_px']} new green={row['new_rms_px']} score {row['old_mask_score']}->{row['new_mask_score']}", (8, 38), WHITE, 0.34)
        return image

    selected = [r for r in rows if int(r.get("frame_index", -1)) in selected_frames and r.get("status", "").startswith("ok")]
    if not selected:
        selected = [r for r in rows if r.get("status", "").startswith("ok")][: int(args.worst_count)]
    if selected:
        cv2.imwrite(str(out_dir / "phase_c1a_silhouette_selected_frames.jpg"), tile_images([make_panel(r) for r in selected], 4, float(args.scale)))
    ok_rows = [r for r in rows if r.get("status", "").startswith("ok") and "new_rms_px" in r]
    worst_new = sorted(ok_rows, key=lambda r: float(r["new_rms_px"]), reverse=True)[: int(args.worst_count)]
    best_improve = sorted(ok_rows, key=lambda r: float(r["delta_rms_px"]))[: int(args.worst_count)]
    if worst_new:
        cv2.imwrite(str(out_dir / "phase_c1a_silhouette_worst_new.jpg"), tile_images([make_panel(r) for r in worst_new], 4, float(args.scale)))
    if best_improve:
        cv2.imwrite(str(out_dir / "phase_c1a_silhouette_best_improve.jpg"), tile_images([make_panel(r) for r in best_improve], 4, float(args.scale)))

    paired = [r for r in ok_rows if "old_rms_px" in r and "new_rms_px" in r]
    summary = {
        "session_dir": str(session),
        "source_pipeline": str(args.source_pipeline),
        "input_npz": str(input_npz),
        "depth_summary": str(depth_summary_path),
        "video": str(video),
        "output_dir": str(out_dir),
        "output_npz": str(output_npz),
        "candidate_csv": str(output_csv),
        "segmenter": str(args.segmenter),
        "process_frames": str(args.process_frames),
        "rows_processed": int(len(rows)),
        "status_counts": dict(Counter(str(r.get("status", "")) for r in rows)),
        "anchor_counts": dict(Counter(str(r.get("anchor_name", "")) for r in ok_rows)),
        "old_rms_px": stats(float(r["old_rms_px"]) for r in paired),
        "new_rms_px": stats(float(r["new_rms_px"]) for r in paired),
        "delta_rms_px_new_minus_old": stats(float(r["delta_rms_px"]) for r in paired),
        "old_mask_score": stats(float(r["old_mask_score"]) for r in paired),
        "new_mask_score": stats(float(r["new_mask_score"]) for r in paired),
        "anchor_depth_m_refined": stats(float(r["anchor_depth_m_refined"]) for r in paired if str(r.get("anchor_depth_m_refined", ""))),
        "contact_sheets": {
            "selected": str(out_dir / "phase_c1a_silhouette_selected_frames.jpg"),
            "worst_new": str(out_dir / "phase_c1a_silhouette_worst_new.jpg"),
            "best_improve": str(out_dir / "phase_c1a_silhouette_best_improve.jpg"),
        },
    }
    summary_path = out_dir / "phase_c1a_silhouette_xy_align_summary.json"
    summary_path.write_text(json.dumps(json_clean(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(json_clean(summary), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
