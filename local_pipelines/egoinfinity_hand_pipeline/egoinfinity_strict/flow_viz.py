"""Small local flow visualization helpers for strict MEMFOF QA."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def flow_to_color(flow: np.ndarray) -> np.ndarray:
    """Convert optical flow `(H,W,2)` to a BGR HSV visualization."""
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = np.asarray(ang * 180.0 / np.pi / 2.0, dtype=np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag / max(float(np.nanpercentile(mag, 95)), 1e-6) * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def dump_flow_field(cache_dir: Path, index: int, flow: np.ndarray, rgb_a: Optional[np.ndarray] = None) -> None:
    """Write a debug flow image; compatible with EgoInfinity's call site."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    color = flow_to_color(flow)
    if rgb_a is not None:
        base = cv2.cvtColor(rgb_a, cv2.COLOR_RGB2BGR) if rgb_a.ndim == 3 else cv2.cvtColor(rgb_a, cv2.COLOR_GRAY2BGR)
        color = cv2.addWeighted(base, 0.45, color, 0.55, 0.0)
    cv2.imwrite(str(cache_dir / f"flow_{index:08d}.jpg"), color)
