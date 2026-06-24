#!/usr/bin/env python3
"""Export a depth-sorted Phase-C hand skeleton overlay video."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import cv2
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

# BGR for OpenCV.
VISIBLE_COLOR = (45, 45, 255)
INFILLED_COLOR = (0, 215, 255)
HIDDEN_COLOR = (255, 120, 20)
LEFT_TEXT = (255, 170, 60)
RIGHT_TEXT = (80, 220, 120)
WHITE = (245, 245, 245)
BLACK = (0, 0, 0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-video", required=True)
    p.add_argument("--input-npz", required=True)
    p.add_argument("--output-video", required=True)
    p.add_argument("--output-summary", default="")
    p.add_argument("--joints-key", default="auto")
    p.add_argument("--uv-key", default="auto")
    p.add_argument(
        "--infilled-uv-key",
        default="auto",
        help="UV source for motion-infilled or oversized rows. Use none/off to keep --uv-key unchanged.",
    )
    p.add_argument("--max-uv-hand-size-px", type=float, default=900.0)
    p.add_argument("--frame-start", type=int, default=-1)
    p.add_argument("--frame-end", type=int, default=-1)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--show-hidden", action="store_true", default=True)
    p.add_argument("--hide-hidden", action="store_true")
    p.add_argument("--draw-labels", action="store_true")
    p.add_argument("--draw-bbox", action="store_true")
    p.add_argument("--line-thickness", type=int, default=2)
    p.add_argument("--point-radius", type=int, default=4)
    p.add_argument("--font-scale", type=float, default=0.48)
    p.add_argument("--alpha", type=float, default=0.78)
    p.add_argument("--codec", default="mp4v")
    return p.parse_args()


def choose_key(npz: np.lib.npyio.NpzFile, requested: str, candidates: Iterable[str]) -> str:
    if requested and requested != "auto":
        if requested not in npz.files:
            raise RuntimeError(f"missing requested key: {requested}")
        return requested
    for key in candidates:
        if key in npz.files:
            return key
    raise RuntimeError(f"missing any key from: {', '.join(candidates)}")


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
    if repl.shape[:2] != uv.shape[:2]:
        return uv, replaced_infilled, replaced_oversize
    for i in range(uv.shape[0]):
        size = uv_hand_size(uv[i])
        use_replacement = False
        if bool(motion_infilled[i]):
            use_replacement = True
            replaced_infilled += 1
        elif math.isfinite(size) and size > float(max_size_px):
            use_replacement = True
            replaced_oversize += 1
        if use_replacement and np.isfinite(repl[i]).any():
            uv[i] = repl[i]
    return uv, replaced_infilled, replaced_oversize


def finite_point_2d(p: np.ndarray, width: int, height: int, margin: int = 80) -> bool:
    return (
        np.isfinite(p).all()
        and float(p[0]) >= -margin
        and float(p[0]) <= width + margin
        and float(p[1]) >= -margin
        and float(p[1]) <= height + margin
    )


def finite_z(z: float) -> bool:
    return math.isfinite(float(z)) and float(z) > 0.0


def text_with_outline(img: np.ndarray, text: str, org: Tuple[int, int], color: Tuple[int, int, int], scale: float, thickness: int = 1) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def blend_line(img: np.ndarray, a: Tuple[int, int], b: Tuple[int, int], color: Tuple[int, int, int], thickness: int, alpha: float) -> None:
    overlay = img.copy()
    cv2.line(overlay, a, b, color, thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0.0, dst=img)


def blend_circle(img: np.ndarray, center: Tuple[int, int], radius: int, color: Tuple[int, int, int], alpha: float, outline: bool = True) -> None:
    overlay = img.copy()
    cv2.circle(overlay, center, radius, color, -1, cv2.LINE_AA)
    if outline:
        cv2.circle(overlay, center, radius + 1, WHITE, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0.0, dst=img)


def project_from_camera(points_cam: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    out = np.full((points_cam.shape[0], 2), np.nan, dtype=np.float32)
    z = points_cam[:, 2]
    valid = np.isfinite(points_cam).all(axis=1) & (z > 1e-6)
    out[valid, 0] = (points_cam[valid, 0] * fx / z[valid]) + cx
    out[valid, 1] = (points_cam[valid, 1] * fy / z[valid]) + cy
    return out


def load_camera(npz: np.lib.npyio.NpzFile) -> Dict[str, float]:
    if "foundation_camera_json" in npz.files:
        raw = str(np.asarray(npz["foundation_camera_json"]).reshape(-1)[0])
        try:
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


def scalar_array(npz: np.lib.npyio.NpzFile, key: str, n: int, default: Any) -> np.ndarray:
    if key in npz.files:
        arr = np.asarray(npz[key])
        if arr.shape[0] == n:
            return arr
    return np.asarray([default] * n)


def str_at(arr: np.ndarray, i: int, default: str = "") -> str:
    try:
        return str(arr[i])
    except Exception:
        return default


def num_at(arr: np.ndarray, i: int, default: float = float("nan")) -> float:
    try:
        return float(arr[i])
    except Exception:
        return default


def build_frame_map(args: argparse.Namespace, width: int, height: int) -> Tuple[Dict[int, List[Dict[str, Any]]], Dict[str, Any]]:
    npz_path = Path(args.input_npz).expanduser().resolve()
    if not npz_path.exists():
        raise RuntimeError(f"missing input npz: {npz_path}")
    npz = np.load(npz_path, allow_pickle=True)
    joints_key = choose_key(npz, str(args.joints_key), ("joints_cam_visibility_depth", "joints_cam_smooth", "joints_cam_depth_smooth", "joints_cam_depth", "joints_cam"))
    joints_cam = np.asarray(npz[joints_key], dtype=np.float32)
    if joints_cam.ndim != 3 or joints_cam.shape[1:] != (21, 3):
        raise RuntimeError(f"bad {joints_key} shape: {joints_cam.shape}")
    n = joints_cam.shape[0]
    motion_infilled = scalar_array(npz, "motion_infilled", n, 0).astype(bool)
    uv_key = ""
    infilled_uv_key = ""
    replaced_infilled_uv = 0
    replaced_oversize_uv = 0
    if args.uv_key != "camera_projection":
        try:
            if args.uv_key == "auto":
                if joints_key == "joints_cam":
                    uv_candidates = ("joints_uv", "joints_uv_smooth_depth_camera")
                else:
                    uv_candidates = ("joints_uv_smooth_depth_camera", "joints_uv")
            else:
                uv_candidates = (str(args.uv_key),)
            uv_key = choose_key(npz, str(args.uv_key), uv_candidates)
            uv = np.asarray(npz[uv_key], dtype=np.float32)
            if uv.shape[:2] != (n, 21):
                uv_key = ""
            elif uv_key != "camera_projection":
                infilled_uv_key = choose_infilled_uv_key(npz, str(args.infilled_uv_key), uv_key)
                if infilled_uv_key:
                    uv, replaced_infilled_uv, replaced_oversize_uv = replace_bad_or_infilled_uv(
                        uv,
                        np.asarray(npz[infilled_uv_key], dtype=np.float32),
                        motion_infilled,
                        float(args.max_uv_hand_size_px),
                    )
        except RuntimeError:
            uv_key = ""
    if not uv_key:
        cam = load_camera(npz)
        uv = np.stack([project_from_camera(joints_cam[i], cam["fx"], cam["fy"], cam["cx"], cam["cy"]) for i in range(n)], axis=0)
        uv_key = "camera_projection"

    frame_index = np.asarray(npz["frame_index"], dtype=np.int64)
    labels = scalar_array(npz, "hand_label", n, "")
    ranks = scalar_array(npz, "hand_rank", n, 0)
    tracks = scalar_array(npz, "track_id", n, -1)
    conf = scalar_array(npz, "det_conf", n, float("nan"))
    bbox = np.asarray(npz["bbox_xyxy"], dtype=np.float32) if "bbox_xyxy" in npz.files and np.asarray(npz["bbox_xyxy"]).shape == (n, 4) else np.full((n, 4), np.nan, dtype=np.float32)
    motion_method = scalar_array(npz, "motion_infiller_method", n, "")
    if "mano_joint_visible" in npz.files and np.asarray(npz["mano_joint_visible"]).shape[:2] == (n, 21):
        joint_visible = np.asarray(npz["mano_joint_visible"]).astype(bool)
    else:
        joint_visible = np.ones((n, 21), dtype=bool)

    by_frame: Dict[int, List[Dict[str, Any]]] = {}
    selected = 0
    hidden_points = 0
    visible_points = 0
    infilled_candidates = 0
    for i in range(n):
        frame = int(frame_index[i])
        if int(args.frame_start) >= 0 and frame < int(args.frame_start):
            continue
        if int(args.frame_end) >= 0 and frame > int(args.frame_end):
            continue
        selected += 1
        infilled = bool(int(num_at(motion_infilled, i, 0)) != 0)
        if infilled:
            infilled_candidates += 1
        visible_ids = [int(v) for v in np.where(joint_visible[i])[0].tolist()]
        visible_points += len(visible_ids)
        hidden_points += 21 - len(visible_ids)
        by_frame.setdefault(frame, []).append({
            "row": i,
            "label": str_at(labels, i),
            "rank": int(num_at(ranks, i, 0)),
            "track": int(num_at(tracks, i, -1)),
            "conf": num_at(conf, i, 0.0),
            "bbox": bbox[i].astype(float),
            "uv": uv[i].astype(float),
            "z": joints_cam[i, :, 2].astype(float),
            "visible": joint_visible[i],
            "infilled": infilled,
            "method": str_at(motion_method, i),
        })
    summary = {
        "input_npz": str(npz_path),
        "joints_key": joints_key,
        "uv_key": uv_key,
        "infilled_uv_key": infilled_uv_key,
        "replaced_infilled_uv": int(replaced_infilled_uv),
        "replaced_oversize_uv": int(replaced_oversize_uv),
        "max_uv_hand_size_px": float(args.max_uv_hand_size_px),
        "selected_candidates": int(selected),
        "selected_frames": int(len(by_frame)),
        "visible_points": int(visible_points),
        "hidden_points": int(hidden_points),
        "motion_infilled_candidates": int(infilled_candidates),
        "colors": {
            "visible_detected": "red",
            "motion_infilled": "yellow",
            "hidden_joints": "blue",
        },
        "depth_sort": "draw primitives by descending mean camera z, so smaller z/closer camera primitives are drawn last",
    }
    return by_frame, summary


def primitive_color(is_hidden: bool, is_infilled: bool) -> Tuple[int, int, int]:
    if is_hidden:
        return HIDDEN_COLOR
    if is_infilled:
        return INFILLED_COLOR
    return VISIBLE_COLOR


def add_overlay_for_frame(
    frame: np.ndarray,
    candidates: List[Dict[str, Any]],
    args: argparse.Namespace,
    frame_index: int,
) -> Tuple[np.ndarray, int]:
    h, w = frame.shape[:2]
    primitives: List[Tuple[float, str, Dict[str, Any]]] = []
    show_hidden = bool(args.show_hidden) and not bool(args.hide_hidden)
    for cand in candidates:
        uv = cand["uv"]
        z = cand["z"]
        visible = cand["visible"]
        infilled = bool(cand["infilled"])
        for a, b in HAND_EDGES:
            if not finite_point_2d(uv[a], w, h) or not finite_point_2d(uv[b], w, h):
                continue
            if not finite_z(z[a]) or not finite_z(z[b]):
                continue
            hidden = not (bool(visible[a]) and bool(visible[b]))
            if hidden and not show_hidden:
                continue
            primitives.append((float((z[a] + z[b]) * 0.5), "line", {
                "a": (int(round(float(uv[a, 0]))), int(round(float(uv[a, 1])))),
                "b": (int(round(float(uv[b, 0]))), int(round(float(uv[b, 1])))),
                "color": primitive_color(hidden, infilled),
                "hidden": hidden,
                "infilled": infilled,
            }))
        for lid in range(21):
            if not finite_point_2d(uv[lid], w, h) or not finite_z(z[lid]):
                continue
            hidden = not bool(visible[lid])
            if hidden and not show_hidden:
                continue
            primitives.append((float(z[lid]), "point", {
                "center": (int(round(float(uv[lid, 0]))), int(round(float(uv[lid, 1])))),
                "radius": int(args.point_radius) + (1 if lid == 0 else 0),
                "color": primitive_color(hidden, infilled),
                "hidden": hidden,
                "infilled": infilled,
                "lid": lid,
            }))
        if bool(args.draw_bbox):
            box = cand["bbox"]
            if np.isfinite(box).all():
                zbox = float(np.nanmedian(z[np.isfinite(z)])) if np.isfinite(z).any() else 999.0
                primitives.append((zbox, "bbox", {
                    "box": tuple(int(round(float(v))) for v in box),
                    "color": primitive_color(False, infilled),
                }))

    # In a standard camera frame, larger z is farther from camera. Draw far first,
    # then near last, so close hands naturally occlude far hands in the overlay.
    primitives.sort(key=lambda item: item[0], reverse=True)
    out = frame.copy()
    for _, kind, item in primitives:
        if kind == "line":
            thickness = max(1, int(args.line_thickness) - (1 if item["hidden"] else 0))
            blend_line(out, item["a"], item["b"], item["color"], thickness, float(args.alpha) * (0.75 if item["hidden"] else 1.0))
        elif kind == "point":
            blend_circle(out, item["center"], int(item["radius"]), item["color"], float(args.alpha) * (0.80 if item["hidden"] else 1.0))
            if bool(args.draw_labels):
                text_with_outline(out, str(item["lid"]), (item["center"][0] + 5, item["center"][1] - 5), item["color"], float(args.font_scale) * 0.78, 1)
        elif kind == "bbox":
            x1, y1, x2, y2 = item["box"]
            cv2.rectangle(out, (x1, y1), (x2, y2), item["color"], 1, cv2.LINE_AA)

    header = f"Phase-C hand overlay | frame={frame_index} | red=visible yellow=infilled blue=hidden | depth sorted: near camera on top"
    text_with_outline(out, header, (12, 24), WHITE, float(args.font_scale), 1)
    y = 46
    for cand in sorted(candidates, key=lambda c: (str(c["label"]), int(c["rank"]), int(c["track"])))[:4]:
        label = str(cand["label"])
        color = RIGHT_TEXT if label == "right" else LEFT_TEXT
        visible_count = int(np.sum(cand["visible"]))
        info = f"{label} tr={cand['track']} r={cand['rank']} conf={cand['conf']:.2f} visible={visible_count}/21 method={cand['method'] or ('infilled' if cand['infilled'] else 'detected')}"
        text_with_outline(out, info, (12, y), color, float(args.font_scale) * 0.88, 1)
        y += 19
    return out, len(primitives)


def main() -> int:
    args = parse_args()
    input_video = Path(args.input_video).expanduser().resolve()
    output_video = Path(args.output_video).expanduser().resolve()
    if not input_video.exists():
        raise RuntimeError(f"missing input video: {input_video}")
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {input_video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    by_frame, summary = build_frame_map(args, width, height)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*str(args.codec)[:4]), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open writer: {output_video}")

    frame_id = 0
    written = 0
    overlay_frames = 0
    primitive_counts: List[int] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if int(args.frame_start) >= 0 and frame_id < int(args.frame_start):
            frame_id += 1
            continue
        if int(args.frame_end) >= 0 and frame_id > int(args.frame_end):
            break
        candidates = by_frame.get(frame_id, [])
        if candidates:
            frame, primitive_count = add_overlay_for_frame(frame, candidates, args, frame_id)
            overlay_frames += 1
            primitive_counts.append(int(primitive_count))
        else:
            text_with_outline(frame, f"Phase-C hand overlay | frame={frame_id} | no candidate", (12, 24), WHITE, float(args.font_scale), 1)
        writer.write(frame)
        written += 1
        frame_id += 1
        if int(args.max_frames) > 0 and written >= int(args.max_frames):
            break
    cap.release()
    writer.release()
    summary.update({
        "input_video": str(input_video),
        "output_video": str(output_video),
        "video_width": int(width),
        "video_height": int(height),
        "fps": float(fps),
        "source_total_frames": int(total_frames),
        "written_frames": int(written),
        "overlay_frames": int(overlay_frames),
        "primitive_count_per_overlay_frame": {
            "min": int(min(primitive_counts)) if primitive_counts else 0,
            "median": float(np.median(primitive_counts)) if primitive_counts else 0.0,
            "max": int(max(primitive_counts)) if primitive_counts else 0,
        },
    })
    output_summary = Path(args.output_summary).expanduser().resolve() if args.output_summary else output_video.with_suffix(".json")
    output_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
