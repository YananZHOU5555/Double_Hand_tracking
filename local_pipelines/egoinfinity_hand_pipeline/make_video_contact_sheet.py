#!/usr/bin/env python3
"""Create a simple contact sheet from selected frames of a video."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--frames", default="94,212,300,596,616,626,772")
    p.add_argument("--cols", type=int, default=3)
    p.add_argument("--scale", type=float, default=0.55)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    video = Path(args.video).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    frames = [int(v.strip()) for v in str(args.frames).split(",") if v.strip()]
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open {video}")
    tiles = []
    for frame_id in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = cap.read()
        if not ok:
            continue
        cv2.putText(frame, f"frame={frame_id}", (12, frame.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, f"frame={frame_id}", (12, frame.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        if float(args.scale) != 1.0:
            frame = cv2.resize(frame, None, fx=float(args.scale), fy=float(args.scale), interpolation=cv2.INTER_AREA)
        tiles.append(frame)
    cap.release()
    if not tiles:
        raise RuntimeError("no frames extracted")
    h, w = tiles[0].shape[:2]
    cols = max(1, int(args.cols))
    rows = int(np.ceil(len(tiles) / cols))
    sheet = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = tile
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), sheet)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
