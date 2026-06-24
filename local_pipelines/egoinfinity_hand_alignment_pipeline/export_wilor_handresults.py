#!/usr/bin/env python3
"""Run WiLoR and export EgoInfinity-style HandResult arrays for LFV videos."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


LANDMARK_COUNT = 21
LIGHT_PURPLE = (0.25098039, 0.274117647, 0.65882353)


def csv_float(value: float) -> str:
    return f"{float(value):.9f}" if np.isfinite(float(value)) else ""


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def assign_egoinfinity_track_ids(candidates: List[Dict[str, Any]], prev_dets: List[Dict[str, Any]], next_track_id: int) -> int:
    """Match EgoInfinity HandDetector: greedy IoU to previous frame, threshold 0.2."""
    if not candidates:
        return next_track_id
    if not prev_dets:
        for cand in candidates:
            cand["track_id"] = next_track_id
            next_track_id += 1
        return next_track_id

    pairs = []
    for pi, prev in enumerate(prev_dets):
        for ci, cand in enumerate(candidates):
            pairs.append((bbox_iou(prev["bbox"], cand["bbox"]), pi, ci))
    pairs.sort(reverse=True)

    used_prev = set()
    used_curr = set()
    for iou_val, pi, ci in pairs:
        if iou_val < 0.2:
            break
        if pi in used_prev or ci in used_curr:
            continue
        candidates[ci]["track_id"] = int(prev_dets[pi]["track_id"])
        used_prev.add(pi)
        used_curr.add(ci)

    for ci, cand in enumerate(candidates):
        if ci not in used_curr:
            cand["track_id"] = next_track_id
            next_track_id += 1
    return next_track_id


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise RuntimeError(f"missing {label}: {path}")


def patch_torch_load_for_legacy_ultralytics(torch_mod: Any) -> None:
    original_load = torch_mod.load
    if getattr(original_load, "_lfv_legacy_ultralytics_patch", False):
        return

    def load_compat(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    load_compat._lfv_legacy_ultralytics_patch = True  # type: ignore[attr-defined]
    torch_mod.load = load_compat


def project_full_img(points: Any, cam_trans: Any, focal_length: float, img_res: Any) -> np.ndarray:
    import torch

    points_t = torch.from_numpy(points.astype(np.float32)) if isinstance(points, np.ndarray) else points.float().detach().cpu()
    cam_t = torch.from_numpy(cam_trans.astype(np.float32)) if isinstance(cam_trans, np.ndarray) else cam_trans.float().detach().cpu()
    if isinstance(img_res, np.ndarray):
        w, h = float(img_res[0]), float(img_res[1])
    else:
        w, h = float(img_res[0].item()), float(img_res[1].item())

    k = torch.eye(3, dtype=torch.float32)
    k[0, 0] = float(focal_length)
    k[1, 1] = float(focal_length)
    k[0, 2] = w / 2.0
    k[1, 2] = h / 2.0
    pts = points_t + cam_t.reshape(1, 3)
    pts = pts / pts[..., -1:]
    projected = (k @ pts.T).T
    return projected[..., :-1].numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--wilor-root", default="/home/yannan/workspace/learning-from-video/WiLor")
    parser.add_argument("--video", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--detector", default="")
    parser.add_argument("--model-config", default="")
    parser.add_argument("--rescale-factor", type=float, default=2.0)
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int, default=-1)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--hand", default="best", choices=["right", "left", "best"])
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--save-overlay", action="store_true")
    return parser.parse_args()


def empty_npz(output_npz: Path, faces: np.ndarray, width: int, height: int) -> None:
    np.savez_compressed(
        output_npz,
        schema_version=np.asarray([1], dtype=np.int32),
        frame_index=np.zeros((0,), dtype=np.int32),
        elapsed_sec=np.zeros((0,), dtype=np.float32),
        hand_rank=np.zeros((0,), dtype=np.int32),
        candidate_index=np.zeros((0,), dtype=np.int32),
        track_id=np.zeros((0,), dtype=np.int32),
        hand_label=np.asarray([], dtype="<U5"),
        is_right=np.zeros((0,), dtype=np.float32),
        det_conf=np.zeros((0,), dtype=np.float32),
        bbox_xyxy=np.zeros((0, 4), dtype=np.float32),
        cam_t=np.zeros((0, 3), dtype=np.float32),
        pred_cam=np.zeros((0, 3), dtype=np.float32),
        focal_length=np.zeros((0,), dtype=np.float32),
        global_orient=np.zeros((0, 1, 3, 3), dtype=np.float32),
        hand_pose=np.zeros((0, 15, 3, 3), dtype=np.float32),
        betas=np.zeros((0, 10), dtype=np.float32),
        joints_3d_rel=np.zeros((0, LANDMARK_COUNT, 3), dtype=np.float32),
        vertices_rel=np.zeros((0, 778, 3), dtype=np.float32),
        joints_cam=np.zeros((0, LANDMARK_COUNT, 3), dtype=np.float32),
        vertices_cam=np.zeros((0, 778, 3), dtype=np.float32),
        joints_uv=np.zeros((0, LANDMARK_COUNT, 2), dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int32),
        image_size=np.asarray([width, height], dtype=np.int32),
    )


def main() -> int:
    args = parse_args()
    session_dir = Path(args.session_dir).expanduser().resolve()
    wilor_root = Path(args.wilor_root).expanduser().resolve()
    video_path = Path(args.video).expanduser().resolve() if args.video else session_dir / "processed_topcam" / "left_table.mp4"
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else session_dir / "quality" / "egoinfinity_hand_alignment_pipeline" / "stages" / "raw_wilor_handresults"
    checkpoint = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else wilor_root / "pretrained_models" / "wilor_final.ckpt"
    detector_path = Path(args.detector).expanduser().resolve() if args.detector else wilor_root / "pretrained_models" / "detector.pt"
    model_config = Path(args.model_config).expanduser().resolve() if args.model_config else wilor_root / "pretrained_models" / "model_config.yaml"

    require_file(video_path, "LFV left/table video")
    require_file(checkpoint, "WiLoR checkpoint")
    require_file(detector_path, "WiLoR detector")
    require_file(model_config, "WiLoR model config")
    require_file(wilor_root / "mano_data" / "MANO_RIGHT.pkl", "MANO_RIGHT.pkl")
    require_file(wilor_root / "mano_data" / "mano_mean_params.npz", "mano_mean_params.npz")

    sys.path.insert(0, str(wilor_root))
    os.chdir(str(wilor_root))

    try:
        import cv2
        import torch
        from ultralytics import YOLO
        from wilor.datasets.vitdet_dataset import ViTDetDataset
        from wilor.models import load_wilor
        from wilor.utils import recursive_to
        from wilor.utils.renderer import cam_crop_to_full
    except Exception as exc:
        raise RuntimeError(f"WiLoR dependencies are not ready: {exc!r}") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    output_npz = output_dir / "wilor_handresults_raw.npz"
    predictions_csv = output_dir / "wilor_predictions_raw.csv"
    detections_csv = output_dir / "wilor_detections_raw.csv"
    summary_json = output_dir / "wilor_handresults_raw_summary.json"
    overlay_mp4 = output_dir / "wilor_handresults_raw_overlay.mp4"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_cfg = load_wilor(checkpoint_path=str(checkpoint), cfg_path=str(model_config))
    if args.fast:
        torch.set_float32_matmul_precision("high")
        model = model.half()
        model.backbone.skip_blocks = True
    model = model.to(device).eval()
    patch_torch_load_for_legacy_ultralytics(torch)
    detector = YOLO(str(detector_path)).to(device)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 60.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.save_overlay:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(overlay_mp4), fourcc, fps / max(1, int(args.frame_stride)), (width, height))

    pred_fields = [
        "frame_index", "elapsed_sec", "hand_rank", "candidate_index", "track_id",
        "hand_label", "is_right", "det_conf", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        "cam_t_x", "cam_t_y", "cam_t_z", "scaled_focal_length", "landmark_id",
        "local_x_m", "local_y_m", "local_z_m", "cam_x_m", "cam_y_m", "cam_z_m", "u_px", "v_px",
    ]
    det_fields = [
        "frame_index", "elapsed_sec", "hand_rank", "track_id", "hand_label", "is_right", "det_conf",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "accepted",
    ]

    frame_index: List[int] = []
    elapsed_sec: List[float] = []
    hand_rank: List[int] = []
    candidate_index: List[int] = []
    track_id_arr: List[int] = []
    hand_label: List[str] = []
    is_right_arr: List[float] = []
    det_conf: List[float] = []
    bbox_xyxy: List[np.ndarray] = []
    cam_t_arr: List[np.ndarray] = []
    pred_cam_arr: List[np.ndarray] = []
    focal_arr: List[float] = []
    global_orient_arr: List[np.ndarray] = []
    hand_pose_arr: List[np.ndarray] = []
    betas_arr: List[np.ndarray] = []
    joints_rel_arr: List[np.ndarray] = []
    vertices_rel_arr: List[np.ndarray] = []
    joints_cam_arr: List[np.ndarray] = []
    vertices_cam_arr: List[np.ndarray] = []
    joints_uv_arr: List[np.ndarray] = []

    frames_seen = 0
    frames_processed = 0
    frames_with_hand = 0
    pred_rows = 0
    det_rows = 0
    global_candidate_index = 0
    prev_tracked_dets: List[Dict[str, Any]] = []
    next_track_id = 0

    with predictions_csv.open("w", newline="", encoding="utf-8") as pred_f, detections_csv.open("w", newline="", encoding="utf-8") as det_f:
        pred_writer = csv.DictWriter(pred_f, fieldnames=pred_fields)
        det_writer = csv.DictWriter(det_f, fieldnames=det_fields)
        pred_writer.writeheader()
        det_writer.writeheader()

        while True:
            ok, img_bgr = cap.read()
            if not ok:
                break
            fidx = frames_seen
            frames_seen += 1
            if fidx < int(args.frame_start):
                continue
            if int(args.frame_end) >= 0 and fidx > int(args.frame_end):
                break
            if fidx % max(1, int(args.frame_stride)) != 0:
                continue
            if args.max_frames > 0 and frames_processed >= int(args.max_frames):
                break
            elapsed = fidx / fps if fps > 1e-6 else float("nan")
            frames_processed += 1

            detections = detector(img_bgr, conf=float(args.conf), verbose=False)[0]
            candidates: List[Dict[str, Any]] = []
            for det in detections:
                box_data = det.boxes.data.cpu().detach().squeeze().numpy()
                if box_data.ndim == 0 or box_data.shape[0] < 5:
                    continue
                cls = float(det.boxes.cls.cpu().detach().squeeze().item())
                conf = float(box_data[4])
                label = "right" if cls >= 0.5 else "left"
                candidates.append(
                    {
                        "bbox": np.asarray(box_data[:4], dtype=np.float32),
                        "is_right": cls,
                        "conf": conf,
                        "label": label,
                        "accepted": args.hand == "best" or args.hand == label,
                    }
                )

            next_track_id = assign_egoinfinity_track_ids(candidates, prev_tracked_dets, next_track_id)
            prev_tracked_dets = [
                {"bbox": item["bbox"].copy(), "track_id": int(item["track_id"])}
                for item in candidates
            ]
            candidates.sort(key=lambda item: item["conf"], reverse=True)
            accepted = [c for c in candidates if c["accepted"]]
            if accepted:
                frames_with_hand += 1

            for rank, item in enumerate(candidates):
                x1, y1, x2, y2 = [float(v) for v in item["bbox"]]
                det_writer.writerow(
                    {
                        "frame_index": fidx,
                        "elapsed_sec": csv_float(elapsed),
                        "hand_rank": rank,
                        "track_id": int(item["track_id"]),
                        "hand_label": item["label"],
                        "is_right": csv_float(float(item["is_right"])),
                        "det_conf": csv_float(float(item["conf"])),
                        "bbox_x1": csv_float(x1),
                        "bbox_y1": csv_float(y1),
                        "bbox_x2": csv_float(x2),
                        "bbox_y2": csv_float(y2),
                        "accepted": int(bool(item["accepted"])),
                    }
                )
                det_rows += 1

            draw = img_bgr.copy()
            if not accepted:
                if writer is not None:
                    writer.write(draw)
                continue

            boxes = np.stack([c["bbox"] for c in accepted], axis=0)
            right = np.stack([float(c["is_right"]) for c in accepted], axis=0)
            dataset = ViTDetDataset(model_cfg, img_bgr, boxes, right, rescale_factor=float(args.rescale_factor), fp16=bool(args.fast))
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=max(1, int(args.batch_size)), shuffle=False, num_workers=0)

            global_rank = 0
            for batch in dataloader:
                batch = recursive_to(batch, device)
                with torch.no_grad():
                    out = model(batch)
                multiplier = 2 * batch["right"] - 1
                pred_cam = out["pred_cam"]
                pred_cam[:, 1] = multiplier * pred_cam[:, 1]
                box_center = batch["box_center"].float()
                box_size = batch["box_size"].float()
                img_size = batch["img_size"].float()
                scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
                pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()
                mano_params = out["pred_mano_params"]

                for n in range(int(batch["img"].shape[0])):
                    cand = accepted[global_rank]
                    hand_rank_n = global_rank
                    global_rank += 1
                    is_right = float(batch["right"][n].detach().cpu().item())
                    verts = out["pred_vertices"][n].detach().float().cpu().numpy()
                    joints = out["pred_keypoints_3d"][n].detach().float().cpu().numpy()
                    verts[:, 0] = (2 * is_right - 1) * verts[:, 0]
                    joints[:, 0] = (2 * is_right - 1) * joints[:, 0]
                    cam_t = pred_cam_t_full[n].astype(np.float32)
                    img_size_np = img_size[n].detach().cpu().numpy()
                    kpts_2d = project_full_img(joints, cam_t, float(scaled_focal_length), img_size_np).astype(np.float32)
                    cam_joints = joints + cam_t.reshape(1, 3)
                    cam_verts = verts + cam_t.reshape(1, 3)
                    x1, y1, x2, y2 = [float(v) for v in cand["bbox"]]
                    cidx = global_candidate_index
                    global_candidate_index += 1

                    frame_index.append(int(fidx))
                    elapsed_sec.append(float(elapsed))
                    hand_rank.append(int(hand_rank_n))
                    candidate_index.append(int(cidx))
                    track_id_arr.append(int(cand["track_id"]))
                    hand_label.append(str(cand["label"]))
                    is_right_arr.append(float(is_right))
                    det_conf.append(float(cand["conf"]))
                    bbox_xyxy.append(np.asarray([x1, y1, x2, y2], dtype=np.float32))
                    cam_t_arr.append(cam_t.astype(np.float32))
                    pred_cam_arr.append(pred_cam[n].detach().float().cpu().numpy().astype(np.float32))
                    focal_arr.append(float(scaled_focal_length))
                    global_orient_arr.append(mano_params["global_orient"][n].detach().float().cpu().numpy().astype(np.float32))
                    hand_pose_arr.append(mano_params["hand_pose"][n].detach().float().cpu().numpy().astype(np.float32))
                    betas_arr.append(mano_params["betas"][n].detach().float().cpu().numpy().astype(np.float32))
                    joints_rel_arr.append(joints.astype(np.float32))
                    vertices_rel_arr.append(verts.astype(np.float32))
                    joints_cam_arr.append(cam_joints.astype(np.float32))
                    vertices_cam_arr.append(cam_verts.astype(np.float32))
                    joints_uv_arr.append(kpts_2d.astype(np.float32))

                    if writer is not None:
                        color = (0, 200, 255) if cand["label"] == "right" else (255, 180, 0)
                        cv2.rectangle(draw, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        cv2.putText(draw, f"{cand['label']} {cand['conf']:.2f}", (int(x1), max(20, int(y1) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
                        for uv in kpts_2d:
                            cv2.circle(draw, (int(round(uv[0])), int(round(uv[1]))), 2, (0, 0, 255), -1)

                    for landmark_id in range(LANDMARK_COUNT):
                        pred_writer.writerow(
                            {
                                "frame_index": fidx,
                                "elapsed_sec": csv_float(elapsed),
                                "hand_rank": hand_rank_n,
                                "candidate_index": cidx,
                                "track_id": int(cand["track_id"]),
                                "hand_label": cand["label"],
                                "is_right": csv_float(is_right),
                                "det_conf": csv_float(float(cand["conf"])),
                                "bbox_x1": csv_float(x1),
                                "bbox_y1": csv_float(y1),
                                "bbox_x2": csv_float(x2),
                                "bbox_y2": csv_float(y2),
                                "cam_t_x": csv_float(float(cam_t[0])),
                                "cam_t_y": csv_float(float(cam_t[1])),
                                "cam_t_z": csv_float(float(cam_t[2])),
                                "scaled_focal_length": csv_float(float(scaled_focal_length)),
                                "landmark_id": landmark_id,
                                "local_x_m": csv_float(float(joints[landmark_id, 0])),
                                "local_y_m": csv_float(float(joints[landmark_id, 1])),
                                "local_z_m": csv_float(float(joints[landmark_id, 2])),
                                "cam_x_m": csv_float(float(cam_joints[landmark_id, 0])),
                                "cam_y_m": csv_float(float(cam_joints[landmark_id, 1])),
                                "cam_z_m": csv_float(float(cam_joints[landmark_id, 2])),
                                "u_px": csv_float(float(kpts_2d[landmark_id, 0])),
                                "v_px": csv_float(float(kpts_2d[landmark_id, 1])),
                            }
                        )
                        pred_rows += 1
            if writer is not None:
                writer.write(draw)

    cap.release()
    if writer is not None:
        writer.release()

    faces = np.asarray(model.mano.faces, dtype=np.int32)
    if frame_index:
        np.savez_compressed(
            output_npz,
            schema_version=np.asarray([1], dtype=np.int32),
            frame_index=np.asarray(frame_index, dtype=np.int32),
            elapsed_sec=np.asarray(elapsed_sec, dtype=np.float32),
            hand_rank=np.asarray(hand_rank, dtype=np.int32),
            candidate_index=np.asarray(candidate_index, dtype=np.int32),
            track_id=np.asarray(track_id_arr, dtype=np.int32),
            hand_label=np.asarray(hand_label),
            is_right=np.asarray(is_right_arr, dtype=np.float32),
            det_conf=np.asarray(det_conf, dtype=np.float32),
            bbox_xyxy=np.stack(bbox_xyxy).astype(np.float32),
            cam_t=np.stack(cam_t_arr).astype(np.float32),
            pred_cam=np.stack(pred_cam_arr).astype(np.float32),
            focal_length=np.asarray(focal_arr, dtype=np.float32),
            global_orient=np.stack(global_orient_arr).astype(np.float32),
            hand_pose=np.stack(hand_pose_arr).astype(np.float32),
            betas=np.stack(betas_arr).astype(np.float32),
            joints_3d_rel=np.stack(joints_rel_arr).astype(np.float32),
            vertices_rel=np.stack(vertices_rel_arr).astype(np.float32),
            joints_cam=np.stack(joints_cam_arr).astype(np.float32),
            vertices_cam=np.stack(vertices_cam_arr).astype(np.float32),
            joints_uv=np.stack(joints_uv_arr).astype(np.float32),
            faces=faces,
            image_size=np.asarray([width, height], dtype=np.int32),
        )
    else:
        empty_npz(output_npz, faces, width, height)

    summary = {
        "semantic": "Raw WiLoR full MANO HandResult-like export for EgoInfinity-style LFV hand pipeline.",
        "session_dir": str(session_dir),
        "video": str(video_path),
        "output_dir": str(output_dir),
        "handresults_npz": str(output_npz),
        "predictions_csv": str(predictions_csv),
        "detections_csv": str(detections_csv),
        "overlay_mp4": str(overlay_mp4 if args.save_overlay else ""),
        "wilor_root": str(wilor_root),
        "checkpoint": str(checkpoint),
        "detector": str(detector_path),
        "model_config": str(model_config),
        "device": str(device),
        "fps": float(fps),
        "image_size": [int(width), int(height)],
        "frames_seen": int(frames_seen),
        "frames_processed": int(frames_processed),
        "frames_with_accepted_hand": int(frames_with_hand),
        "accepted_hand_coverage": float(frames_with_hand / frames_processed) if frames_processed else 0.0,
        "candidates": int(len(frame_index)),
        "prediction_rows": int(pred_rows),
        "detection_rows": int(det_rows),
        "hand_filter": str(args.hand),
        "conf": float(args.conf),
        "rescale_factor": float(args.rescale_factor),
        "contains_mano_params": True,
        "npz_fields": [
            "global_orient", "hand_pose", "betas", "vertices_rel", "joints_3d_rel",
            "vertices_cam", "joints_cam", "cam_t", "joints_uv",
        ],
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
