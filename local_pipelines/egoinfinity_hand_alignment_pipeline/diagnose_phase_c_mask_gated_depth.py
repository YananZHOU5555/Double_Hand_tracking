#!/usr/bin/env python3
"""Diagnose whether hand-mask gating stabilizes Phase-C depth alignment.

The current Phase-C stage samples FoundationStereo depth at WiLoR 2D joints.
This diagnostic adds a segmentation gate before depth sampling:

  * build a per-candidate hand mask from the image and WiLoR bbox/joints
  * keep only reliable joints (wrist + MCPs by default) that fall inside mask
  * sample depth only from pixels inside the hand mask patch
  * re-estimate the camera-space translation and compare against current Phase-C

It intentionally does not replace the main pipeline output.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
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
RELIABLE_IDS = [0, 5, 9, 13, 17]
ALL_IDS = list(range(21))

RAW_COLOR = (255, 0, 255)
OLD_COLOR = (45, 45, 255)
NEW_COLOR = (60, 255, 60)
REJECT_COLOR = (0, 220, 255)
WHITE = (245, 245, 245)
BLACK = (0, 0, 0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session-dir", required=True)
    p.add_argument("--source-pipeline", default="egoinfinity_hand_pipeline")
    p.add_argument("--output-dir", default="")
    p.add_argument("--segmenter", choices=["grabcut", "bbox", "sam"], default="grabcut")
    p.add_argument("--process-frames", default="", help="Optional frame list/ranges to process, e.g. 590-630,212.")
    p.add_argument("--grabcut-iters", type=int, default=1)
    p.add_argument("--sam-checkpoint", default="/home/yannan/workspace/learning-from-video/models/sam/sam_vit_b_01ec64.pth")
    p.add_argument("--sam-model-type", default="vit_b")
    p.add_argument("--sam-device", default="cuda")
    p.add_argument("--bbox-pad-ratio", type=float, default=0.15)
    p.add_argument("--mask-hit-radius", type=int, default=3)
    p.add_argument("--mask-dilate-px", type=int, default=3)
    p.add_argument("--patch-size", type=int, default=7)
    p.add_argument("--min-patch-pixels", type=int, default=3)
    p.add_argument("--min-reliable-joints", type=int, default=2)
    p.add_argument("--allow-all-joints-fallback", action="store_true")
    p.add_argument("--frames", default="94,212,540,562,563,579,590,591,596,616,619,626")
    p.add_argument("--worst-count", type=int, default=16)
    p.add_argument("--scale", type=float, default=0.60)
    p.add_argument("--max-frames", type=int, default=0, help="Optional debug limit by unique frame count; 0 means full video.")
    return p.parse_args()


def parse_frame_list(text: str) -> List[int]:
    out: List[int] = []
    for item in str(text).replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            a, b = item.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(item))
    return sorted(dict.fromkeys(out))


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
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
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


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def load_camera(data: Dict[str, np.ndarray]) -> Dict[str, float]:
    if "foundation_camera_json" in data:
        raw = str(np.asarray(data["foundation_camera_json"]).reshape(-1)[0])
        cam = json.loads(raw)
        return {k: float(cam[k]) for k in ("fx", "fy", "cx", "cy")}
    size = np.asarray(data.get("image_size", [850, 420])).reshape(-1)
    width = float(size[0]) if size.size >= 1 else 850.0
    height = float(size[1]) if size.size >= 2 else 420.0
    focal = 334.0549853304012
    return {"fx": focal, "fy": focal, "cx": width * 0.5, "cy": height * 0.5}


def read_depth_frame_csv(path: Path) -> Dict[int, Path]:
    out: Dict[int, Path] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            depth_path = Path(row["depth_npy"])
            if depth_path.exists():
                out[int(row["frame_index"])] = depth_path
    return out


def load_depth_summary(source_root: Path) -> Tuple[Path, Dict[int, Path]]:
    stable_summary = source_root / "foundationstereo_depth_stabilized" / "foundationstereo_depth_stabilized_summary.json"
    raw_summary = source_root / "foundationstereo_depth" / "foundationstereo_depth_summary.json"
    summary_path = stable_summary if stable_summary.exists() else raw_summary
    if not summary_path.exists():
        raise RuntimeError(f"missing FoundationStereo depth summary under {source_root}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    frame_csv = Path(summary["outputs"]["frame_csv"])
    return summary_path, read_depth_frame_csv(frame_csv)


def project(points_cam: np.ndarray, cam: Dict[str, float]) -> np.ndarray:
    pts = np.asarray(points_cam, dtype=np.float64)
    uv = np.full((pts.shape[0], 2), np.nan, dtype=np.float64)
    z = pts[:, 2]
    valid = np.isfinite(pts).all(axis=1) & (z > 1e-8)
    uv[valid, 0] = float(cam["fx"]) * pts[valid, 0] / z[valid] + float(cam["cx"])
    uv[valid, 1] = float(cam["fy"]) * pts[valid, 1] / z[valid] + float(cam["cy"])
    return uv


def uv_rms(stage: np.ndarray, ref: np.ndarray) -> float:
    d = np.asarray(stage, dtype=np.float64) - np.asarray(ref, dtype=np.float64)
    valid = np.isfinite(d).all(axis=1)
    if not np.any(valid):
        return float("nan")
    dist = np.linalg.norm(d[valid], axis=1)
    return float(np.sqrt(np.mean(dist * dist)))


def clean_mask(mask: np.ndarray, keep_largest: bool = True) -> np.ndarray:
    out = (mask > 0).astype(np.uint8) * 255
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    if keep_largest:
        n, labels, comp_stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
        if n > 1:
            best = 1 + int(np.argmax(comp_stats[1:, cv2.CC_STAT_AREA]))
            out = (labels == best).astype(np.uint8) * 255
    return out


def bbox_xyxy_to_xywh(
    bbox: Sequence[float],
    width: int,
    height: int,
    pad_ratio: float,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    pad = float(pad_ratio) * max(bw, bh)
    x1 = max(0.0, x1 - pad)
    y1 = max(0.0, y1 - pad)
    x2 = min(float(width), x2 + pad)
    y2 = min(float(height), y2 + pad)
    xi1 = int(math.floor(x1))
    yi1 = int(math.floor(y1))
    xi2 = max(xi1 + 1, int(math.ceil(x2)))
    yi2 = max(yi1 + 1, int(math.ceil(y2)))
    return xi1, yi1, min(width - xi1, xi2 - xi1), min(height - yi1, yi2 - yi1)


def bbox_mask(shape: Tuple[int, int], bbox_xywh: Tuple[int, int, int, int]) -> Tuple[np.ndarray, float]:
    x, y, w, h = bbox_xywh
    mask = np.zeros(shape, dtype=np.uint8)
    mask[y:y + h, x:x + w] = 255
    return mask, 0.0


def grabcut_mask(frame_bgr: np.ndarray, bbox_xywh: Tuple[int, int, int, int], iters: int) -> Tuple[np.ndarray, float]:
    x, y, w, h = bbox_xywh
    mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    if w < 5 or h < 5:
        return mask, 0.0
    gc_mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    bgd = np.zeros((1, 65), dtype=np.float64)
    fgd = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(frame_bgr, gc_mask, (x, y, w, h), bgd, fgd, max(1, int(iters)), cv2.GC_INIT_WITH_RECT)
        mask[(gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)] = 255
    except Exception:
        mask[y:y + h, x:x + w] = 255
    if int(np.count_nonzero(mask)) < 40:
        mask[y:y + h, x:x + w] = 255
    return clean_mask(mask), 0.0


class SamHandSegmenter:
    def __init__(self, checkpoint: Path, model_type: str, device: str):
        from segment_anything import SamPredictor, sam_model_registry  # type: ignore

        model = sam_model_registry[model_type](checkpoint=str(checkpoint))
        model.to(device=device)
        model.eval()
        self.predictor = SamPredictor(model)

    def segment(self, frame_bgr: np.ndarray, bbox_xywh: Tuple[int, int, int, int], joints_uv: np.ndarray) -> Tuple[np.ndarray, float]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(rgb)
        x, y, w, h = bbox_xywh
        box = np.asarray([x, y, x + w, y + h], dtype=np.float32)
        finite = joints_uv[np.isfinite(joints_uv).all(axis=1)]
        point_coords = finite.astype(np.float32) if finite.size else None
        point_labels = np.ones((len(finite),), dtype=np.int32) if finite.size else None
        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=True,
        )
        box_area = float(max(1, w * h))
        best_i = 0
        best_score = -1e9
        for i, mask in enumerate(masks):
            mask_u8 = mask.astype(np.uint8) * 255
            hit = 0
            for uv in finite:
                px = int(round(float(uv[0])))
                py = int(round(float(uv[1])))
                if 0 <= py < mask_u8.shape[0] and 0 <= px < mask_u8.shape[1] and mask_u8[py, px] > 0:
                    hit += 1
            hit_ratio = float(hit) / float(max(1, len(finite)))
            area_ratio = float(np.count_nonzero(mask_u8)) / box_area
            area_penalty = abs(math.log(max(0.05, min(4.0, area_ratio))))
            score = float(scores[i]) + 0.65 * hit_ratio - 0.08 * area_penalty
            if score > best_score:
                best_score = score
                best_i = i
        return clean_mask(masks[best_i].astype(np.uint8) * 255), float(scores[best_i])


def mask_hit(mask: np.ndarray, u: float, v: float, radius: int) -> bool:
    if not np.isfinite([u, v]).all():
        return False
    x = int(round(float(u)))
    y = int(round(float(v)))
    h, w = mask.shape[:2]
    if x < 0 or x >= w or y < 0 or y >= h:
        return False
    x1 = max(0, x - radius)
    y1 = max(0, y - radius)
    x2 = min(w, x + radius + 1)
    y2 = min(h, y + radius + 1)
    return bool(np.any(mask[y1:y2, x1:x2] > 0))


def sample_masked_depth_patch(
    depth: np.ndarray,
    mask: np.ndarray,
    u: float,
    v: float,
    patch_size: int,
    min_pixels: int,
) -> Tuple[float, int, str]:
    if not np.isfinite([u, v]).all():
        return float("nan"), 0, "nonfinite_uv"
    h, w = depth.shape[:2]
    x = int(round(float(u)))
    y = int(round(float(v)))
    if x < 0 or x >= w or y < 0 or y >= h:
        return float("nan"), 0, "outside_image"
    half = int(patch_size) // 2
    x1 = max(0, x - half)
    y1 = max(0, y - half)
    x2 = min(w, x + half + 1)
    y2 = min(h, y + half + 1)
    d_patch = depth[y1:y2, x1:x2]
    m_patch = mask[y1:y2, x1:x2] > 0
    valid = d_patch[(d_patch > 0.01) & np.isfinite(d_patch) & m_patch]
    if valid.size < int(min_pixels):
        return float("nan"), int(valid.size), "no_masked_depth"
    return float(np.median(valid)), int(valid.size), "ok"


def estimate_mask_gated_translation(
    joints_rel: np.ndarray,
    joints_uv: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    cam: Dict[str, float],
    args: argparse.Namespace,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    joint_ids = list(RELIABLE_IDS)
    used_ids: List[int] = []
    used_depths: List[float] = []
    used_pixels: List[int] = []
    rejected: Dict[int, str] = {}

    def try_ids(ids: Sequence[int], source_name: str) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        used_ids.clear()
        used_depths.clear()
        used_pixels.clear()
        rejected.clear()
        translations: List[np.ndarray] = []
        for jid in ids:
            u, v = joints_uv[int(jid)]
            if not mask_hit(mask, float(u), float(v), int(args.mask_hit_radius)):
                rejected[int(jid)] = "outside_hand_mask"
                continue
            d, px_count, reason = sample_masked_depth_patch(
                depth,
                mask,
                float(u),
                float(v),
                int(args.patch_size),
                int(args.min_patch_pixels),
            )
            if not math.isfinite(d):
                rejected[int(jid)] = reason
                continue
            point = np.asarray([
                (float(u) - float(cam["cx"])) * d / float(cam["fx"]),
                (float(v) - float(cam["cy"])) * d / float(cam["fy"]),
                d,
            ], dtype=np.float64)
            translations.append(point - joints_rel[int(jid)])
            used_ids.append(int(jid))
            used_depths.append(float(d))
            used_pixels.append(int(px_count))
        if len(translations) < int(args.min_reliable_joints):
            return None, {
                "source": f"{source_name}_insufficient",
                "joint_ids": list(used_ids),
                "sampled_depths_m": list(used_depths),
                "patch_pixel_counts": list(used_pixels),
                "rejected": dict(rejected),
                "valid_joint_count": int(len(used_ids)),
                "rms_m": float("nan"),
                "max_residual_m": float("nan"),
                "tz_spread_m": float("nan"),
            }
        trans = np.stack(translations, axis=0)
        cam_t = np.median(trans, axis=0)
        residuals = np.linalg.norm(trans - cam_t.reshape(1, 3), axis=1)
        tz_spread = float(np.percentile(trans[:, 2], 90) - np.percentile(trans[:, 2], 10)) if trans.shape[0] >= 2 else 0.0
        return cam_t, {
            "source": source_name,
            "joint_ids": list(used_ids),
            "sampled_depths_m": list(used_depths),
            "patch_pixel_counts": list(used_pixels),
            "rejected": dict(rejected),
            "valid_joint_count": int(len(used_ids)),
            "rms_m": float(np.sqrt(np.mean(residuals * residuals))),
            "max_residual_m": float(np.max(residuals)),
            "tz_spread_m": tz_spread,
        }

    cam_t, info = try_ids(joint_ids, "mask_reliable_joints")
    if cam_t is not None or not args.allow_all_joints_fallback:
        return cam_t, info
    cam_t, info = try_ids(ALL_IDS, "mask_all_joints")
    return cam_t, info


def read_video_frame(cap: cv2.VideoCapture, frame_index: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f"failed to read video frame {frame_index}")
    return frame


def draw_text(img: np.ndarray, s: str, org: Tuple[int, int], color: Tuple[int, int, int] = WHITE, scale: float = 0.42) -> None:
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_skeleton(img: np.ndarray, uv: np.ndarray, color: Tuple[int, int, int], thickness: int = 2, radius: int = 3, alpha: float = 0.80) -> None:
    overlay = img.copy()
    h, w = img.shape[:2]

    def ok(p: np.ndarray) -> bool:
        return np.isfinite(p).all() and -80 <= float(p[0]) <= w + 80 and -80 <= float(p[1]) <= h + 80

    for a, b in HAND_EDGES:
        if ok(uv[a]) and ok(uv[b]):
            pa = (int(round(float(uv[a, 0]))), int(round(float(uv[a, 1]))))
            pb = (int(round(float(uv[b, 0]))), int(round(float(uv[b, 1]))))
            cv2.line(overlay, pa, pb, color, int(thickness), cv2.LINE_AA)
    for jid in range(min(21, uv.shape[0])):
        if ok(uv[jid]):
            p = (int(round(float(uv[jid, 0]))), int(round(float(uv[jid, 1]))))
            cv2.circle(overlay, p, radius + (1 if jid in RELIABLE_IDS else 0), color, -1, cv2.LINE_AA)
            cv2.circle(overlay, p, radius + 1, WHITE, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0.0, dst=img)


def make_mask_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = image.copy()
    color = np.zeros_like(out)
    color[:, :, 1] = 200
    hit = mask > 0
    out[hit] = cv2.addWeighted(out, 0.55, color, 0.45, 0.0)[hit]
    return out


def tile_images(images: Sequence[np.ndarray], columns: int, scale: float) -> np.ndarray:
    if not images:
        raise RuntimeError("no images to tile")
    h, w = images[0].shape[:2]
    rows = int(math.ceil(len(images) / float(columns)))
    canvas = np.zeros((rows * h, columns * w, 3), dtype=np.uint8)
    for idx, img in enumerate(images):
        y = (idx // columns) * h
        x = (idx % columns) * w
        canvas[y:y + h, x:x + w] = img
    if scale != 1.0:
        canvas = cv2.resize(canvas, (max(1, int(canvas.shape[1] * scale)), max(1, int(canvas.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    return canvas


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fields = [
        "frame_index", "row", "hand_label", "track_id", "hand_rank", "det_conf",
        "old_alignment_source", "old_alignment_joint_ids", "old_alignment_rms_m",
        "mask_source", "mask_valid_joint_count", "mask_joint_ids", "mask_depths_m",
        "mask_patch_pixel_counts", "mask_rejected_joint_ids", "mask_rejected_reasons",
        "mask_area_ratio", "mask_score", "old_phase_c_rms_px", "mask_gated_rms_px",
        "delta_rms_px", "mask_alignment_rms_m", "mask_alignment_max_residual_m",
        "mask_tz_spread_m", "old_cam_t_x", "old_cam_t_y", "old_cam_t_z",
        "mask_cam_t_x", "mask_cam_t_y", "mask_cam_t_z",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def main() -> int:
    args = parse_args()
    session_dir = Path(args.session_dir).expanduser().resolve()
    source_root = session_dir / "quality" / str(args.source_pipeline) / "stages"
    phase_c_npz = source_root / "phase_c_depth_align" / "wilor_handresults_phase_c_depth_aligned.npz"
    video_path = session_dir / "processed_topcam" / "left_table.mp4"
    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (
        session_dir / "quality" / "egoinfinity_hand_alignment_pipeline" / "quality_check" / f"phase_c_mask_gated_depth_{args.segmenter}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_npz(phase_c_npz)
    cam = load_camera(data)
    depth_summary_path, depth_paths = load_depth_summary(source_root)
    if not video_path.exists():
        raise RuntimeError(f"missing left-table video: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    sam_segmenter: Optional[SamHandSegmenter] = None
    if args.segmenter == "sam":
        sam_checkpoint = Path(args.sam_checkpoint).expanduser().resolve()
        if not sam_checkpoint.exists():
            raise RuntimeError(f"missing SAM checkpoint: {sam_checkpoint}")
        sam_segmenter = SamHandSegmenter(sam_checkpoint, str(args.sam_model_type), str(args.sam_device))

    unique_frames = sorted(set(int(v) for v in data["frame_index"].tolist()))
    if str(args.process_frames).strip():
        allowed = set(parse_frame_list(str(args.process_frames)))
        row_indices = [i for i, f in enumerate(data["frame_index"].tolist()) if int(f) in allowed]
    elif int(args.max_frames) > 0:
        allowed = set(unique_frames[: int(args.max_frames)])
        row_indices = [i for i, f in enumerate(data["frame_index"].tolist()) if int(f) in allowed]
    else:
        row_indices = list(range(len(data["frame_index"])))

    depth_cache: Dict[int, np.ndarray] = {}
    frame_cache: Dict[int, np.ndarray] = {}
    mask_cache: Dict[int, Tuple[np.ndarray, float]] = {}

    rows: List[Dict[str, Any]] = []
    selected_frames = set(parse_frame_list(args.frames))
    selected_panels: List[np.ndarray] = []

    for i in row_indices:
        frame_index = int(data["frame_index"][i])
        if frame_index not in depth_paths:
            continue
        if frame_index not in depth_cache:
            depth_cache[frame_index] = np.load(depth_paths[frame_index]).astype(np.float32)
        if frame_index not in frame_cache:
            frame_cache[frame_index] = read_video_frame(cap, frame_index)

        image = frame_cache[frame_index]
        bbox_xywh = bbox_xyxy_to_xywh(np.asarray(data["bbox_xyxy"][i], dtype=np.float64), width, height, float(args.bbox_pad_ratio))
        cache_key = hash((frame_index, i, args.segmenter, bbox_xywh))
        if cache_key not in mask_cache:
            if args.segmenter == "bbox":
                mask, score = bbox_mask(image.shape[:2], bbox_xywh)
            elif args.segmenter == "grabcut":
                mask, score = grabcut_mask(image, bbox_xywh, int(args.grabcut_iters))
            elif args.segmenter == "sam":
                assert sam_segmenter is not None
                mask, score = sam_segmenter.segment(image, bbox_xywh, np.asarray(data["joints_uv"][i], dtype=np.float32))
            else:
                raise RuntimeError(f"unknown segmenter: {args.segmenter}")
            if int(args.mask_dilate_px) > 0:
                k = int(args.mask_dilate_px) * 2 + 1
                mask = cv2.dilate(mask, np.ones((k, k), dtype=np.uint8), iterations=1)
            mask_cache[cache_key] = (mask, float(score))
        mask, score = mask_cache[cache_key]

        joints_uv = np.asarray(data["joints_uv"][i], dtype=np.float64)
        joints_rel = np.asarray(data["joints_3d_rel"][i], dtype=np.float64)
        old_cam_t = np.asarray(data["cam_t_depth"][i], dtype=np.float64)
        old_uv = project(np.asarray(data["joints_cam_depth"][i], dtype=np.float64), cam)
        old_rms = uv_rms(old_uv, joints_uv)

        cam_t_mask, info = estimate_mask_gated_translation(
            joints_rel,
            joints_uv,
            depth_cache[frame_index],
            mask,
            cam,
            args,
        )
        if cam_t_mask is None:
            new_uv = np.full_like(joints_uv, np.nan)
            new_rms = float("nan")
            cam_t_out = np.asarray([np.nan, np.nan, np.nan], dtype=np.float64)
        else:
            cam_t_out = np.asarray(cam_t_mask, dtype=np.float64)
            new_uv = project(joints_rel + cam_t_out.reshape(1, 3), cam)
            new_rms = uv_rms(new_uv, joints_uv)

        rejected = info.get("rejected", {})
        reject_ids = ",".join(str(k) for k in sorted(rejected))
        reject_reasons = "|".join(f"{k}:{rejected[k]}" for k in sorted(rejected))
        used_ids = [int(v) for v in info.get("joint_ids", [])]
        used_depths = [float(v) for v in info.get("sampled_depths_m", [])]
        patch_counts = [int(v) for v in info.get("patch_pixel_counts", [])]
        mask_area_ratio = float(np.count_nonzero(mask)) / float(max(1, width * height))
        row = {
            "frame_index": frame_index,
            "row": int(i),
            "hand_label": str(data["hand_label"][i]),
            "track_id": int(data["track_id"][i]),
            "hand_rank": int(data["hand_rank"][i]),
            "det_conf": f"{float(data['det_conf'][i]):.6f}",
            "old_alignment_source": str(data["alignment_source"][i]),
            "old_alignment_joint_ids": str(data["alignment_joint_ids"][i]),
            "old_alignment_rms_m": f"{float(data['alignment_rms_m'][i]):.9f}",
            "mask_source": str(info.get("source", "unknown")),
            "mask_valid_joint_count": int(info.get("valid_joint_count", 0)),
            "mask_joint_ids": ",".join(str(v) for v in used_ids),
            "mask_depths_m": ",".join(f"{v:.9f}" for v in used_depths),
            "mask_patch_pixel_counts": ",".join(str(v) for v in patch_counts),
            "mask_rejected_joint_ids": reject_ids,
            "mask_rejected_reasons": reject_reasons,
            "mask_area_ratio": f"{mask_area_ratio:.9f}",
            "mask_score": f"{score:.9f}",
            "old_phase_c_rms_px": f"{old_rms:.6f}",
            "mask_gated_rms_px": f"{new_rms:.6f}" if math.isfinite(new_rms) else "",
            "delta_rms_px": f"{new_rms - old_rms:.6f}" if math.isfinite(new_rms) and math.isfinite(old_rms) else "",
            "mask_alignment_rms_m": f"{float(info.get('rms_m', float('nan'))):.9f}" if math.isfinite(float(info.get("rms_m", float("nan")))) else "",
            "mask_alignment_max_residual_m": f"{float(info.get('max_residual_m', float('nan'))):.9f}" if math.isfinite(float(info.get("max_residual_m", float("nan")))) else "",
            "mask_tz_spread_m": f"{float(info.get('tz_spread_m', float('nan'))):.9f}" if math.isfinite(float(info.get("tz_spread_m", float("nan")))) else "",
            "old_cam_t_x": f"{old_cam_t[0]:.9f}",
            "old_cam_t_y": f"{old_cam_t[1]:.9f}",
            "old_cam_t_z": f"{old_cam_t[2]:.9f}",
            "mask_cam_t_x": f"{cam_t_out[0]:.9f}" if math.isfinite(float(cam_t_out[0])) else "",
            "mask_cam_t_y": f"{cam_t_out[1]:.9f}" if math.isfinite(float(cam_t_out[1])) else "",
            "mask_cam_t_z": f"{cam_t_out[2]:.9f}" if math.isfinite(float(cam_t_out[2])) else "",
            "_old_rms": old_rms,
            "_new_rms": new_rms,
            "_mask": mask,
            "_old_uv": old_uv,
            "_new_uv": new_uv,
            "_raw_uv": joints_uv,
            "_image": image,
            "_used_ids": used_ids,
            "_rejected": rejected,
        }
        rows.append(row)

    cap.release()
    csv_rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    csv_path = out_dir / "phase_c_mask_gated_depth_candidates.csv"
    write_csv(csv_path, csv_rows)

    def make_panel(row: Dict[str, Any]) -> np.ndarray:
        img = make_mask_overlay(row["_image"], row["_mask"])
        draw_skeleton(img, row["_raw_uv"], RAW_COLOR, 1, 2, 0.45)
        draw_skeleton(img, row["_old_uv"], OLD_COLOR, 2, 3, 0.72)
        if math.isfinite(float(row["_new_rms"])):
            draw_skeleton(img, row["_new_uv"], NEW_COLOR, 2, 3, 0.84)
        for jid in row["_used_ids"]:
            u, v = row["_raw_uv"][int(jid)]
            if np.isfinite([u, v]).all():
                cv2.circle(img, (int(round(float(u))), int(round(float(v)))), 6, NEW_COLOR, 2, cv2.LINE_AA)
        for jid in row["_rejected"]:
            u, v = row["_raw_uv"][int(jid)]
            if np.isfinite([u, v]).all():
                cv2.drawMarker(img, (int(round(float(u))), int(round(float(v)))), REJECT_COLOR, cv2.MARKER_TILTED_CROSS, 13, 2)
        draw_text(img, f"f={row['frame_index']} {row['hand_label']} tr={row['track_id']} src={row['mask_source']}", (8, 18), WHITE, 0.44)
        draw_text(img, f"old red={row['old_phase_c_rms_px']}px new green={row['mask_gated_rms_px'] or 'NA'}px used={row['mask_joint_ids']}", (8, 38), WHITE, 0.36)
        draw_text(img, "mask green fill | raw magenta | rejected yellow x", (8, 58), WHITE, 0.34)
        return img

    selected_rows = [r for r in rows if int(r["frame_index"]) in selected_frames]
    selected_rows = sorted(selected_rows, key=lambda r: (int(r["frame_index"]), str(r["hand_label"]), int(r["hand_rank"])))
    worst_old = sorted(rows, key=lambda r: float(r["_old_rms"]), reverse=True)[: int(args.worst_count)]
    worst_new = sorted(
        [r for r in rows if math.isfinite(float(r["_new_rms"]))],
        key=lambda r: float(r["_new_rms"]),
        reverse=True,
    )[: int(args.worst_count)]
    if selected_rows:
        selected_panels = [make_panel(r) for r in selected_rows[: max(1, int(args.worst_count) * 2)]]
        cv2.imwrite(str(out_dir / "phase_c_mask_gated_selected_frames.jpg"), tile_images(selected_panels, 4, float(args.scale)))
    if worst_old:
        cv2.imwrite(str(out_dir / "phase_c_mask_gated_worst_old.jpg"), tile_images([make_panel(r) for r in worst_old], 4, float(args.scale)))
    if worst_new:
        cv2.imwrite(str(out_dir / "phase_c_mask_gated_worst_new.jpg"), tile_images([make_panel(r) for r in worst_new], 4, float(args.scale)))

    old_vals = [float(r["_old_rms"]) for r in rows]
    new_vals = [float(r["_new_rms"]) for r in rows if math.isfinite(float(r["_new_rms"]))]
    paired = [r for r in rows if math.isfinite(float(r["_old_rms"])) and math.isfinite(float(r["_new_rms"]))]
    improved = sum(1 for r in paired if float(r["_new_rms"]) < float(r["_old_rms"]))
    worse = sum(1 for r in paired if float(r["_new_rms"]) > float(r["_old_rms"]))
    source_counts = Counter(str(r["mask_source"]) for r in rows)
    old_source_counts = Counter(str(r["old_alignment_source"]) for r in rows)
    reject_counts = Counter()
    for r in rows:
        for item in str(r["mask_rejected_reasons"]).split("|"):
            if ":" in item:
                reject_counts[item.split(":", 1)[1]] += 1
    summary = {
        "session_dir": str(session_dir),
        "source_pipeline": str(args.source_pipeline),
        "phase_c_npz": str(phase_c_npz),
        "depth_summary": str(depth_summary_path),
        "video": str(video_path),
        "output_dir": str(out_dir),
        "csv": str(csv_path),
        "segmenter": str(args.segmenter),
        "process_frames": str(args.process_frames),
        "grabcut_iters": int(args.grabcut_iters),
        "candidate_count": int(len(rows)),
        "estimated_count": int(len(new_vals)),
        "no_estimate_count": int(len(rows) - len(new_vals)),
        "estimated_ratio": float(len(new_vals) / max(1, len(rows))),
        "paired_count": int(len(paired)),
        "improved_count": int(improved),
        "worse_count": int(worse),
        "unchanged_count": int(len(paired) - improved - worse),
        "improved_ratio": float(improved / max(1, len(paired))),
        "old_alignment_source_counts": dict(old_source_counts),
        "mask_source_counts": dict(source_counts),
        "mask_reject_reason_counts": dict(reject_counts),
        "old_phase_c_rms_px": stats(old_vals),
        "mask_gated_rms_px": stats(new_vals),
        "delta_rms_px_new_minus_old": stats(float(r["_new_rms"]) - float(r["_old_rms"]) for r in paired),
        "mask_valid_joint_count": stats(int(r["mask_valid_joint_count"]) for r in rows),
        "mask_alignment_rms_m": stats(float(r["mask_alignment_rms_m"]) for r in rows if str(r["mask_alignment_rms_m"])),
        "mask_tz_spread_m": stats(float(r["mask_tz_spread_m"]) for r in rows if str(r["mask_tz_spread_m"])),
        "contact_sheets": {
            "selected": str(out_dir / "phase_c_mask_gated_selected_frames.jpg"),
            "worst_old": str(out_dir / "phase_c_mask_gated_worst_old.jpg"),
            "worst_new": str(out_dir / "phase_c_mask_gated_worst_new.jpg"),
        },
        "note": "New RMS is only computed for candidates with enough hand-mask-gated depth samples. No main pipeline files were modified.",
    }
    summary_path = out_dir / "phase_c_mask_gated_depth_summary.json"
    summary_path.write_text(json.dumps(json_clean(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(json_clean(summary), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
