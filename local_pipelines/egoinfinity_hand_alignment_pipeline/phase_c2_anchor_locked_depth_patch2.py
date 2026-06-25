#!/usr/bin/env python3
"""Phase-C2a anchor-locked depth projection with temporal depth smoothing.

This experimental stage keeps the Phase-C2 smoothed MANO pose/shape, selects
one robust raw WiLoR 2D anchor, samples FoundationStereo depth in a small
hand-mask-gated patch around that anchor, then translates the whole MANO hand
so the same semantic MANO anchor projects back to the raw 2D anchor.

Unlike ``phase_c1a_silhouette_xy_align.py``, this stage does not run silhouette
search.  X/Y is determined by the selected raw 2D anchor; Z is determined by a
2px-radius masked FoundationStereo median depth and then smoothed per track.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from phase_c1a_silhouette_xy_align import (  # noqa: E402
    HAND_EDGES,
    SamHandSegmenter,
    backproject,
    bbox_mask,
    bbox_xyxy_to_xywh,
    choose_anchor,
    clean_mask,
    grabcut_mask,
    load_depth_summary,
    project,
    read_video_frame,
)
from phase_c1b_depth_smooth import (  # noqa: E402
    csv_float,
    frame_to_frame_jump,
    gaussian_smooth_weighted,
    json_clean,
    stats,
    string_array,
    temporal_outlier_reject,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session-dir", required=True)
    p.add_argument("--input-npz", default="")
    p.add_argument("--depth-summary-json", default="")
    p.add_argument("--output-dir", default="")
    p.add_argument("--video", default="")
    p.add_argument("--segmenter", choices=["sam", "grabcut", "bbox"], default="sam")
    p.add_argument("--sam-checkpoint", default="/home/yannan/workspace/learning-from-video/models/sam/sam_vit_b_01ec64.pth")
    p.add_argument("--sam-model-type", default="vit_b")
    p.add_argument("--sam-device", default="cuda")
    p.add_argument("--bbox-pad-ratio", type=float, default=0.15)
    p.add_argument("--grabcut-iters", type=int, default=1)
    p.add_argument("--mask-dilate-px", type=int, default=2)
    p.add_argument("--anchor-patch-radius", type=int, default=2)
    p.add_argument("--anchor-min-pixels", type=int, default=5)
    p.add_argument("--anchor-max-depth-span-m", type=float, default=0.12)
    p.add_argument("--smooth-sigma-z", type=float, default=5.0)
    p.add_argument("--smooth-min-track-anchors", type=int, default=4)
    p.add_argument("--temporal-mad-window", type=int, default=5)
    p.add_argument("--temporal-mad-factor", type=float, default=5.0)
    p.add_argument("--max-smooth-delta-z-m", type=float, default=0.30)
    p.add_argument("--warn-smooth-delta-z-m", type=float, default=0.080)
    p.add_argument("--bad-smooth-delta-z-m", type=float, default=0.160)
    p.add_argument("--warn-wrist-jump-m", type=float, default=0.090)
    p.add_argument("--bad-wrist-jump-m", type=float, default=0.160)
    p.add_argument(
        "--skip-motion-infilled-anchor-lock",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Do not re-depth-anchor transformer/neighbor infilled rows; keep their MANO/depth geometry.",
    )
    p.add_argument(
        "--max-anchor-lock-rms-px",
        type=float,
        default=120.0,
        help="Reject anchor-lock candidates whose projected skeleton is this far from raw 2D landmarks.",
    )
    p.add_argument("--render-overlay", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--contact-stride", type=int, default=60)
    p.add_argument("--contact-extra-frames", default="590,595,600,605,610,615,620,625,630")
    p.add_argument("--progress-every", type=int, default=100)
    return p.parse_args()


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def parse_int_set(text: str) -> set[int]:
    out: set[int] = set()
    for part in str(text).replace(";", ",").split(","):
        part = part.strip()
        if part:
            out.add(int(part))
    return out


def finite_stats(values: Iterable[float]) -> Dict[str, Any]:
    return stats(values)


def uv_rms(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    valid = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    if int(np.sum(valid)) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.sum((a[valid] - b[valid]) ** 2, axis=1))))


def get_geometry(data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    if "joints_3d_rel_smooth" in data and "vertices_rel_smooth" in data:
        rel_j = np.asarray(data["joints_3d_rel_smooth"], dtype=np.float32)
        rel_v = np.asarray(data["vertices_rel_smooth"], dtype=np.float32)
        source = "phase_c2_smooth_relative"
    else:
        rel_j = np.asarray(data["joints_3d_rel"], dtype=np.float32)
        rel_v = np.asarray(data["vertices_rel"], dtype=np.float32)
        source = "raw_relative"

    if "cam_t_smooth" in data:
        cam_t = np.asarray(data["cam_t_smooth"], dtype=np.float32)
    elif "cam_t_depth" in data:
        cam_t = np.asarray(data["cam_t_depth"], dtype=np.float32)
    else:
        cam_t = np.asarray(data["cam_t"], dtype=np.float32)

    if "joints_cam_smooth" in data:
        joints_cam = np.asarray(data["joints_cam_smooth"], dtype=np.float32)
    elif "joints_cam_depth" in data:
        joints_cam = np.asarray(data["joints_cam_depth"], dtype=np.float32)
    else:
        joints_cam = rel_j + cam_t[:, None, :]
    return rel_j, rel_v, cam_t, joints_cam, source


def choose_default_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path, Path]:
    session = Path(args.session_dir).expanduser().resolve()
    input_npz = Path(args.input_npz).expanduser().resolve() if args.input_npz else (
        session / "quality" / "egoinfinity_hand_pipeline" / "stages" / "phase_c_mano_smooth" / "wilor_handresults_phase_c2_mano_smooth.npz"
    )
    stages_root = session / "quality" / "egoinfinity_hand_pipeline" / "stages"
    if args.depth_summary_json:
        depth_summary = Path(args.depth_summary_json).expanduser().resolve()
    else:
        stable = stages_root / "foundationstereo_depth_stabilized" / "foundationstereo_depth_stabilized_summary.json"
        raw = stages_root / "foundationstereo_depth" / "foundationstereo_depth_summary.json"
        depth_summary = stable if stable.exists() else raw
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (
        session / "quality" / "egoinfinity_hand_alignment_pipeline" / "stages" / "phase_c2_anchor_locked_depth_patch2"
    )
    video = Path(args.video).expanduser().resolve() if args.video else session / "processed_topcam" / "left_table.mp4"
    return input_npz, depth_summary, output_dir, video


def make_mask(
    image: np.ndarray,
    bbox_xywh: Tuple[int, int, int, int],
    joints_uv: np.ndarray,
    args: argparse.Namespace,
    sam: SamHandSegmenter | None,
) -> Tuple[np.ndarray, float]:
    if args.segmenter == "bbox":
        mask, score = bbox_mask(image.shape[:2], bbox_xywh)
    elif args.segmenter == "grabcut":
        mask, score = grabcut_mask(image, bbox_xywh, int(args.grabcut_iters))
    else:
        if sam is None:
            raise RuntimeError("SAM segmenter requested but not initialized")
        mask, score = sam.segment(image, bbox_xywh, np.asarray(joints_uv, dtype=np.float32))
    if int(args.mask_dilate_px) > 0:
        k = int(args.mask_dilate_px) * 2 + 1
        mask = cv2.dilate(mask, np.ones((k, k), dtype=np.uint8), iterations=1)
        mask = clean_mask(mask, keep_largest=True)
    return mask, float(score)


def draw_text(img: np.ndarray, text: str, org: Tuple[int, int], color: Tuple[int, int, int], scale: float = 0.52, thick: int = 1) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_skeleton(img: np.ndarray, uv: np.ndarray, color: Tuple[int, int, int], thickness: int = 2, radius: int = 3, alpha: float = 1.0) -> None:
    layer = img.copy()
    uv = np.asarray(uv, dtype=np.float64)
    for a, b in HAND_EDGES:
        if np.isfinite(uv[[a, b]]).all():
            p1 = tuple(np.round(uv[a]).astype(int))
            p2 = tuple(np.round(uv[b]).astype(int))
            cv2.line(layer, p1, p2, color, thickness, cv2.LINE_AA)
    for p in uv:
        if np.isfinite(p).all():
            q = tuple(np.round(p).astype(int))
            cv2.circle(layer, q, radius, color, -1, cv2.LINE_AA)
            cv2.circle(layer, q, radius + 1, (0, 0, 0), 1, cv2.LINE_AA)
    if alpha < 1.0:
        cv2.addWeighted(layer, alpha, img, 1.0 - alpha, 0, img)
    else:
        img[:] = layer


def tile_images(images: Sequence[np.ndarray], cols: int = 4) -> np.ndarray | None:
    if not images:
        return None
    h = max(im.shape[0] for im in images)
    w = max(im.shape[1] for im in images)
    rows = int(math.ceil(len(images) / float(cols)))
    canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for idx, im in enumerate(images):
        y = (idx // cols) * h
        x = (idx % cols) * w
        canvas[y:y + im.shape[0], x:x + im.shape[1]] = im
    return canvas


def write_quality_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fields = [
        "frame_index", "elapsed_sec", "row", "candidate_index", "track_id", "hand_label", "hand_rank",
        "status", "qc_flag", "anchor_name", "anchor_ids", "anchor_depth_m", "anchor_depth_smooth_m",
        "anchor_valid_pixels", "anchor_depth_span_m", "anchor_depth_std_m", "anchor_smooth_delta_z_m",
        "anchor_trust", "seg_score", "orig_rms_px", "locked_rms_px", "locked_smooth_rms_px",
        "before_wrist_z_jump_m", "after_wrist_z_jump_m",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def write_track_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fields = [
        "track_key", "track_id", "hand_label", "frame_count", "frame_min", "frame_max",
        "valid_anchor_count", "smoothed", "before_wrist_z_jump_p95_m",
        "after_wrist_z_jump_p95_m", "smooth_delta_z_p95_m", "smooth_delta_z_max_m",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def qc_flag(status: str, delta_z: float, after_jump: float, args: argparse.Namespace) -> str:
    flags: List[str] = []
    if status != "anchor_lock_smooth_ok":
        flags.append(status)
    abs_delta = abs(float(delta_z)) if math.isfinite(float(delta_z)) else float("nan")
    if math.isfinite(abs_delta):
        if abs_delta > float(args.bad_smooth_delta_z_m):
            flags.append("bad_large_anchor_depth_smooth_delta")
        elif abs_delta > float(args.warn_smooth_delta_z_m):
            flags.append("warn_large_anchor_depth_smooth_delta")
    if math.isfinite(float(after_jump)):
        if float(after_jump) > float(args.bad_wrist_jump_m):
            flags.append("bad_after_anchor_lock_wrist_z_jump")
        elif float(after_jump) > float(args.warn_wrist_jump_m):
            flags.append("warn_after_anchor_lock_wrist_z_jump")
    return "|".join(flags) if flags else "ok"


def main() -> int:
    args = parse_args()
    session_dir = Path(args.session_dir).expanduser().resolve()
    input_npz, depth_summary_json, output_dir, video = choose_default_paths(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_npz.exists():
        raise RuntimeError(f"missing input npz: {input_npz}")
    if not depth_summary_json.exists():
        raise RuntimeError(f"missing depth summary json: {depth_summary_json}")
    if not video.exists():
        raise RuntimeError(f"missing video: {video}")

    data = load_npz(input_npz)
    rel_j, rel_v, orig_cam_t, orig_joints_cam, geometry_source = get_geometry(data)
    n = int(len(data["frame_index"]))
    frame_index = np.asarray(data["frame_index"], dtype=np.int32)
    track_id = np.asarray(data["track_id"], dtype=np.int32)
    hand_label = np.asarray(data["hand_label"]).astype(str)
    raw_uv = np.asarray(data["joints_uv"], dtype=np.float32)
    bbox_xyxy = np.asarray(data["bbox_xyxy"], dtype=np.float32)
    motion_infilled = np.asarray(data.get("motion_infilled", np.zeros((n,), dtype=np.int32)), dtype=np.int32)
    cam = json.loads(str(data["foundation_camera_json"][0]))
    for key in ("fx", "fy", "cx", "cy"):
        cam[key] = float(cam[key])

    stages_root = session_dir / "quality" / "egoinfinity_hand_pipeline" / "stages"
    _depth_summary_path, depth_paths = load_depth_summary(stages_root, str(depth_summary_json))

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

    sam: SamHandSegmenter | None = None
    if args.segmenter == "sam":
        sam_ckpt = Path(args.sam_checkpoint).expanduser().resolve()
        if not sam_ckpt.exists():
            raise RuntimeError(f"missing SAM checkpoint: {sam_ckpt}")
        sam = SamHandSegmenter(sam_ckpt, str(args.sam_model_type), str(args.sam_device))

    cam_t_locked = orig_cam_t.copy().astype(np.float32)
    joints_locked = orig_joints_cam.copy().astype(np.float32)
    verts_locked = (rel_v + orig_cam_t[:, None, :]).astype(np.float32)
    cam_t_smooth = cam_t_locked.copy()
    joints_smooth = joints_locked.copy()
    verts_smooth = verts_locked.copy()
    uv_locked = project(joints_locked.reshape(-1, 3), cam).reshape(n, 21, 2).astype(np.float32)
    uv_smooth = uv_locked.copy()

    status = np.asarray(["missing_depth_frame"] * n, dtype=object)
    anchor_name = np.asarray([""] * n, dtype=object)
    anchor_ids_text = np.asarray([""] * n, dtype=object)
    anchor_depth = np.full((n,), np.nan, dtype=np.float32)
    anchor_depth_smooth = np.full((n,), np.nan, dtype=np.float32)
    anchor_valid_pixels = np.zeros((n,), dtype=np.int32)
    anchor_span = np.full((n,), np.nan, dtype=np.float32)
    anchor_std = np.full((n,), np.nan, dtype=np.float32)
    anchor_uv = np.full((n, 2), np.nan, dtype=np.float32)
    anchor_rel = np.full((n, 3), np.nan, dtype=np.float32)
    seg_score = np.full((n,), np.nan, dtype=np.float32)
    trust = np.zeros((n,), dtype=np.float32)
    delta_z = np.zeros((n,), dtype=np.float32)
    orig_rms = np.full((n,), np.nan, dtype=np.float32)
    locked_rms = np.full((n,), np.nan, dtype=np.float32)
    smooth_rms = np.full((n,), np.nan, dtype=np.float32)

    rows_by_frame: Dict[int, List[int]] = defaultdict(list)
    for i, frame in enumerate(frame_index.tolist()):
        rows_by_frame[int(frame)].append(i)

    frame_cache: Dict[int, np.ndarray] = {}
    depth_cache: Dict[int, np.ndarray] = {}
    t0 = time.time()
    for i in range(n):
        frame = int(frame_index[i])
        if bool(motion_infilled[i]) and bool(args.skip_motion_infilled_anchor_lock):
            status[i] = "motion_infilled_skip_anchor_lock"
            orig_rms[i] = uv_rms(project(orig_joints_cam[i], cam), raw_uv[i])
            smooth_rms[i] = orig_rms[i]
            continue
        if frame not in depth_paths:
            status[i] = "missing_depth_frame"
            continue
        if frame not in frame_cache:
            frame_cache[frame] = read_video_frame(cap, frame)
        if frame not in depth_cache:
            depth_cache[frame] = np.load(depth_paths[frame]).astype(np.float32)
        image = frame_cache[frame]
        depth = depth_cache[frame]
        bbox_xywh = bbox_xyxy_to_xywh(bbox_xyxy[i], width, height, float(args.bbox_pad_ratio))
        mask, score = make_mask(image, bbox_xywh, raw_uv[i], args, sam)
        seg_score[i] = float(score)
        anchor = choose_anchor(rel_j[i], raw_uv[i], depth, mask, args)
        status[i] = str(anchor["anchor_status"])
        orig_uv_i = project(orig_joints_cam[i], cam)
        orig_rms[i] = uv_rms(orig_uv_i, raw_uv[i])
        if status[i] != "ok":
            continue
        ids = [int(v) for v in str(anchor["anchor_ids"]).split(",") if v != ""]
        anchor_name[i] = str(anchor["anchor_name"])
        anchor_ids_text[i] = str(anchor["anchor_ids"])
        anchor_depth[i] = float(anchor["anchor_depth_m_initial"])
        anchor_depth_smooth[i] = anchor_depth[i]
        anchor_valid_pixels[i] = int(anchor["anchor_valid_pixels_initial"])
        anchor_span[i] = float(anchor["anchor_depth_span_m_initial"])
        anchor_std[i] = float(anchor["anchor_depth_std_m_initial"])
        anchor_uv[i] = np.asarray(anchor["anchor_uv_initial"], dtype=np.float32)
        anchor_rel[i] = np.asarray(anchor["anchor_rel"], dtype=np.float32)
        target = backproject(float(anchor_uv[i, 0]), float(anchor_uv[i, 1]), float(anchor_depth[i]), cam)
        cam_t = np.asarray(target, dtype=np.float32) - anchor_rel[i]
        cam_t_locked[i] = cam_t
        joints_locked[i] = rel_j[i] + cam_t[None, :]
        verts_locked[i] = rel_v[i] + cam_t[None, :]
        uv_locked[i] = project(joints_locked[i], cam).astype(np.float32)
        locked_rms[i] = uv_rms(uv_locked[i], raw_uv[i])
        if math.isfinite(float(locked_rms[i])) and float(locked_rms[i]) > float(args.max_anchor_lock_rms_px):
            status[i] = "anchor_lock_bad_2d_rms"
            cam_t_locked[i] = orig_cam_t[i]
            joints_locked[i] = orig_joints_cam[i]
            verts_locked[i] = rel_v[i] + orig_cam_t[i][None, :]
            uv_locked[i] = project(joints_locked[i], cam).astype(np.float32)
            continue

    groups: Dict[Tuple[str, int], List[int]] = defaultdict(list)
    for i in range(n):
        groups[(str(hand_label[i]), int(track_id[i]))].append(i)

    before_jump = np.full((n,), np.nan, dtype=np.float32)
    after_jump = np.full((n,), np.nan, dtype=np.float32)
    track_rows: List[Dict[str, Any]] = []
    for key, indices_unsorted in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        indices = sorted(indices_unsorted, key=lambda j: (int(frame_index[j]), int(data["candidate_index"][j]) if "candidate_index" in data else j))
        valid = np.asarray([status[j] == "ok" and math.isfinite(float(anchor_depth[j])) for j in indices], dtype=bool)
        valid_count = int(np.sum(valid))
        current_z = np.asarray([float(anchor_depth[j]) if valid[k] else float("nan") for k, j in enumerate(indices)], dtype=np.float64)
        before_jump[indices] = frame_to_frame_jump(np.asarray([float(cam_t_locked[j, 2]) for j in indices], dtype=np.float64))
        smoothed = valid_count >= int(args.smooth_min_track_anchors)
        if smoothed:
            frame_vals = np.asarray([int(frame_index[j]) for j in indices], dtype=np.float64)
            valid_frames = frame_vals[valid]
            valid_z = current_z[valid]
            filled = np.interp(frame_vals, valid_frames, valid_z)
            raw_trust = valid.astype(np.float64)
            final_trust = temporal_outlier_reject(
                current_z,
                raw_trust,
                int(args.temporal_mad_window),
                float(args.temporal_mad_factor),
            )
            smooth_z = gaussian_smooth_weighted(filled, final_trust, float(args.smooth_sigma_z))
            for local_idx, j in enumerate(indices):
                if status[j] != "ok":
                    continue
                raw_delta = float(smooth_z[local_idx] - float(anchor_depth[j]))
                clipped_delta = float(np.clip(raw_delta, -float(args.max_smooth_delta_z_m), float(args.max_smooth_delta_z_m)))
                z_new = float(anchor_depth[j]) + clipped_delta
                anchor_depth_smooth[j] = z_new
                delta_z[j] = clipped_delta
                trust[j] = float(final_trust[local_idx])
                if trust[j] <= 0:
                    status[j] = "anchor_lock_temporal_outlier"
                    continue
                target = backproject(float(anchor_uv[j, 0]), float(anchor_uv[j, 1]), z_new, cam)
                cam_t = np.asarray(target, dtype=np.float32) - anchor_rel[j]
                cam_t_smooth[j] = cam_t
                joints_smooth[j] = rel_j[j] + cam_t[None, :]
                verts_smooth[j] = rel_v[j] + cam_t[None, :]
                uv_smooth[j] = project(joints_smooth[j], cam).astype(np.float32)
                smooth_rms[j] = uv_rms(uv_smooth[j], raw_uv[j])
                if math.isfinite(float(smooth_rms[j])) and float(smooth_rms[j]) > float(args.max_anchor_lock_rms_px):
                    status[j] = "anchor_lock_smooth_bad_2d_rms"
                    cam_t_smooth[j] = orig_cam_t[j]
                    joints_smooth[j] = orig_joints_cam[j]
                    verts_smooth[j] = rel_v[j] + orig_cam_t[j][None, :]
                    uv_smooth[j] = project(joints_smooth[j], cam).astype(np.float32)
                    trust[j] = 0.0
                    continue
                status[j] = "anchor_lock_smooth_ok"
        else:
            for j in indices:
                if status[j] == "ok":
                    status[j] = "track_skipped_low_anchor_count"
                    anchor_depth_smooth[j] = anchor_depth[j]
                    smooth_rms[j] = locked_rms[j]
                    trust[j] = 1.0
                    cam_t_smooth[j] = cam_t_locked[j]
                    joints_smooth[j] = joints_locked[j]
                    verts_smooth[j] = verts_locked[j]
                    uv_smooth[j] = uv_locked[j]
        after_jump[indices] = frame_to_frame_jump(np.asarray([float(cam_t_smooth[j, 2]) for j in indices], dtype=np.float64))
        track_rows.append(
            {
                "track_key": f"{key[0]}:{key[1]}",
                "track_id": key[1],
                "hand_label": key[0],
                "frame_count": len(indices),
                "frame_min": int(frame_index[indices[0]]),
                "frame_max": int(frame_index[indices[-1]]),
                "valid_anchor_count": valid_count,
                "smoothed": int(smoothed),
                "before_wrist_z_jump_p95_m": csv_float(float(np.nanpercentile(before_jump[indices], 95))),
                "after_wrist_z_jump_p95_m": csv_float(float(np.nanpercentile(after_jump[indices], 95))),
                "smooth_delta_z_p95_m": csv_float(float(np.nanpercentile(np.abs(delta_z[indices]), 95))),
                "smooth_delta_z_max_m": csv_float(float(np.nanmax(np.abs(delta_z[indices])))),
            }
        )

    qc_flags: List[str] = []
    quality_rows: List[Dict[str, Any]] = []
    for i in range(n):
        flag = qc_flag(str(status[i]), float(delta_z[i]), float(after_jump[i]), args)
        qc_flags.append(flag)
        quality_rows.append(
            {
                "frame_index": int(frame_index[i]),
                "elapsed_sec": csv_float(float(data["elapsed_sec"][i])) if "elapsed_sec" in data else "",
                "row": i,
                "candidate_index": int(data["candidate_index"][i]) if "candidate_index" in data else i,
                "track_id": int(track_id[i]),
                "hand_label": str(hand_label[i]),
                "hand_rank": int(data["hand_rank"][i]) if "hand_rank" in data else 0,
                "status": str(status[i]),
                "qc_flag": flag,
                "anchor_name": str(anchor_name[i]),
                "anchor_ids": str(anchor_ids_text[i]),
                "anchor_depth_m": csv_float(float(anchor_depth[i])),
                "anchor_depth_smooth_m": csv_float(float(anchor_depth_smooth[i])),
                "anchor_valid_pixels": int(anchor_valid_pixels[i]),
                "anchor_depth_span_m": csv_float(float(anchor_span[i])),
                "anchor_depth_std_m": csv_float(float(anchor_std[i])),
                "anchor_smooth_delta_z_m": csv_float(float(delta_z[i])),
                "anchor_trust": csv_float(float(trust[i])),
                "seg_score": csv_float(float(seg_score[i])),
                "orig_rms_px": csv_float(float(orig_rms[i])),
                "locked_rms_px": csv_float(float(locked_rms[i])),
                "locked_smooth_rms_px": csv_float(float(smooth_rms[i])),
                "before_wrist_z_jump_m": csv_float(float(before_jump[i])),
                "after_wrist_z_jump_m": csv_float(float(after_jump[i])),
            }
        )

    output_npz = output_dir / "wilor_handresults_phase_c2_anchor_locked_depth_patch2_smooth.npz"
    quality_csv = output_dir / "phase_c2_anchor_locked_depth_patch2_quality.csv"
    track_csv = output_dir / "phase_c2_anchor_locked_depth_patch2_track_summary.csv"
    summary_json = output_dir / "phase_c2_anchor_locked_depth_patch2_summary.json"
    overlay_mp4 = output_dir / "phase_c2_anchor_locked_depth_patch2_overlay.mp4"
    contact_jpg = output_dir / "phase_c2_anchor_locked_depth_patch2_contact.jpg"

    out = dict(data)
    out["cam_t_anchor_locked"] = cam_t_locked.astype(np.float32)
    out["joints_cam_anchor_locked"] = joints_locked.astype(np.float32)
    out["vertices_cam_anchor_locked"] = verts_locked.astype(np.float32)
    out["joints_uv_anchor_locked"] = uv_locked.astype(np.float32)
    out["cam_t_anchor_locked_smooth"] = cam_t_smooth.astype(np.float32)
    out["joints_cam_anchor_locked_smooth"] = joints_smooth.astype(np.float32)
    out["vertices_cam_anchor_locked_smooth"] = verts_smooth.astype(np.float32)
    out["joints_uv_anchor_locked_smooth"] = uv_smooth.astype(np.float32)
    out["anchor_lock_status"] = string_array([str(v) for v in status])
    out["anchor_lock_qc_flag"] = string_array(qc_flags)
    out["anchor_lock_anchor_name"] = string_array([str(v) for v in anchor_name])
    out["anchor_lock_anchor_ids"] = string_array([str(v) for v in anchor_ids_text])
    out["anchor_lock_anchor_uv"] = anchor_uv.astype(np.float32)
    out["anchor_lock_anchor_depth_m"] = anchor_depth.astype(np.float32)
    out["anchor_lock_anchor_depth_smooth_m"] = anchor_depth_smooth.astype(np.float32)
    out["anchor_lock_anchor_valid_pixels"] = anchor_valid_pixels.astype(np.int32)
    out["anchor_lock_anchor_depth_span_m"] = anchor_span.astype(np.float32)
    out["anchor_lock_anchor_depth_std_m"] = anchor_std.astype(np.float32)
    out["anchor_lock_smooth_delta_z_m"] = delta_z.astype(np.float32)
    out["anchor_lock_trust"] = trust.astype(np.float32)
    out["anchor_lock_orig_rms_px"] = orig_rms.astype(np.float32)
    out["anchor_lock_locked_rms_px"] = locked_rms.astype(np.float32)
    out["anchor_lock_locked_smooth_rms_px"] = smooth_rms.astype(np.float32)
    np.savez_compressed(output_npz, **out)
    write_quality_csv(quality_csv, quality_rows)
    write_track_csv(track_csv, track_rows)

    if bool(args.render_overlay):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        writer = cv2.VideoWriter(str(overlay_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        contact_frames = set(range(int(np.min(frame_index)), int(np.max(frame_index)) + 1, max(1, int(args.contact_stride))))
        contact_frames.update(parse_int_set(args.contact_extra_frames))
        contact_images: List[np.ndarray] = []
        colors = {"left": (255, 210, 40), "right": (60, 230, 80)}
        raw_anchor_color = (0, 255, 255)
        locked_anchor_color = (255, 255, 0)
        faint_color = (155, 155, 155)
        for frame in range(frame_count):
            ok, image = cap.read()
            if not ok:
                break
            img = image.copy()
            row_ids = rows_by_frame.get(frame, [])
            draw_text(img, f"phase_c2_anchor_locked_depth_patch2_smooth | frame={frame} | hands={len(row_ids)}", (10, 28), (0, 255, 255), 0.60, 2)
            draw_text(img, "yellow circle=raw anchor; cyan cross + bright skeleton=smoothed anchor-lock; faint=original C2", (10, 54), (255, 255, 255), 0.43, 1)
            for i in row_ids:
                color = colors.get(str(hand_label[i]), (220, 220, 220))
                draw_skeleton(img, project(orig_joints_cam[i], cam), faint_color, thickness=1, radius=2, alpha=0.22)
                if str(status[i]) in ("anchor_lock_smooth_ok", "track_skipped_low_anchor_count", "anchor_lock_temporal_outlier"):
                    draw_skeleton(img, uv_smooth[i], color, thickness=2, radius=3, alpha=0.95)
                if np.isfinite(anchor_uv[i]).all():
                    p = tuple(np.round(anchor_uv[i]).astype(int))
                    cv2.circle(img, p, 8, raw_anchor_color, -1, cv2.LINE_AA)
                    cv2.circle(img, p, 9, (0, 0, 0), 2, cv2.LINE_AA)
                    ids = [int(v) for v in str(anchor_ids_text[i]).split(",") if v != ""]
                    if ids:
                        q_uv = np.nanmean(uv_smooth[i, ids], axis=0)
                        if np.isfinite(q_uv).all():
                            q = tuple(np.round(q_uv).astype(int))
                            cv2.drawMarker(img, q, locked_anchor_color, markerType=cv2.MARKER_CROSS, markerSize=22, thickness=3, line_type=cv2.LINE_AA)
                    label = f"{hand_label[i]} t{int(track_id[i])} {anchor_name[i]} z={anchor_depth_smooth[i]:.3f} dz={delta_z[i]:+.3f} rms {orig_rms[i]:.1f}->{smooth_rms[i]:.1f}"
                    x = max(2, min(width - 300, int(anchor_uv[i, 0]) - 80))
                    y = max(78, min(height - 8, int(anchor_uv[i, 1]) - 16))
                    draw_text(img, label, (x, y), color, 0.36, 1)
            writer.write(img)
            if frame in contact_frames:
                contact_images.append(cv2.resize(img, (width // 2, height // 2)))
        writer.release()
        if contact_images:
            tiled = tile_images(contact_images, cols=4)
            if tiled is not None:
                cv2.imwrite(str(contact_jpg), tiled)
    cap.release()

    status_counts = Counter(str(v) for v in status)
    qc_counts = Counter(qc_flags)
    anchor_counts = Counter(str(v) for v in anchor_name if str(v))
    hard_errors: List[str] = []
    warnings: List[str] = []
    if not np.isfinite(joints_smooth).all() or not np.isfinite(verts_smooth).all():
        hard_errors.append("non_finite_anchor_locked_smooth_geometry")
    bad_count = sum(v for k, v in qc_counts.items() if "bad_" in str(k))
    warn_count = sum(v for k, v in qc_counts.items() if "warn_" in str(k))
    if bad_count:
        warnings.append(f"anchor_locked_depth_bad_flags:{bad_count}")
    if warn_count:
        warnings.append(f"anchor_locked_depth_warn_flags:{warn_count}")

    summary = {
        "semantic": "Phase-C2a anchor-locked 2px FoundationStereo depth with per-track depth smoothing",
        "version_name": "phase_c2_anchor_locked_depth_patch2_smooth",
        "session_dir": str(session_dir),
        "input_npz": str(input_npz),
        "geometry_source": geometry_source,
        "depth_summary_json": str(depth_summary_json),
        "video": str(video),
        "output_dir": str(output_dir),
        "output_npz": str(output_npz),
        "quality_csv": str(quality_csv),
        "track_summary_csv": str(track_csv),
        "overlay_mp4": str(overlay_mp4) if bool(args.render_overlay) else "",
        "contact_sheet": str(contact_jpg) if bool(args.render_overlay) else "",
        "candidate_count": n,
        "status_counts": dict(status_counts),
        "qc_flag_counts": dict(qc_counts),
        "anchor_counts": dict(anchor_counts),
        "anchor_patch_radius_px": int(args.anchor_patch_radius),
        "anchor_patch_window": f"{2 * int(args.anchor_patch_radius) + 1}x{2 * int(args.anchor_patch_radius) + 1}",
        "anchor_min_pixels": int(args.anchor_min_pixels),
        "anchor_max_depth_span_m": float(args.anchor_max_depth_span_m),
        "smooth_sigma_z": float(args.smooth_sigma_z),
        "smooth_min_track_anchors": int(args.smooth_min_track_anchors),
        "max_smooth_delta_z_m": float(args.max_smooth_delta_z_m),
        "orig_rms_px": finite_stats(orig_rms),
        "locked_rms_px": finite_stats(locked_rms),
        "locked_smooth_rms_px": finite_stats(smooth_rms),
        "anchor_depth_m": finite_stats(anchor_depth),
        "anchor_depth_smooth_m": finite_stats(anchor_depth_smooth),
        "smooth_delta_z_m": finite_stats(delta_z),
        "anchor_valid_pixels": finite_stats(anchor_valid_pixels),
        "before_wrist_z_jump_m": finite_stats(before_jump),
        "after_wrist_z_jump_m": finite_stats(after_jump),
        "elapsed_sec": float(time.time() - t0),
        "hard_errors": hard_errors,
        "warnings": warnings,
        "ok": len(hard_errors) == 0,
    }
    summary_json.write_text(json.dumps(json_clean(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(json_clean(summary), indent=2, ensure_ascii=False))
    return 0 if len(hard_errors) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
