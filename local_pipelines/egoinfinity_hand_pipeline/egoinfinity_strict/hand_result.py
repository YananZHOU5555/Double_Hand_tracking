"""Minimal HandResult dataclass for the local EgoInfinity Phase-C snapshot."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HandResult:
    """Per-hand MANO reconstruction in camera frame.

    This mirrors EgoInfinity's `hand_reconstructor.HandResult` fields without
    importing WiLoR at module import time.  The LFV pipeline already exports
    these arrays to NPZ in `export_wilor_handresults.py`.
    """

    global_orient: np.ndarray
    hand_pose: np.ndarray
    betas: np.ndarray
    joints_3d: np.ndarray
    joints_3d_rel: np.ndarray
    vertices: np.ndarray
    cam_t: np.ndarray
    joints_2d: np.ndarray
    scaled_focal: float
    is_right: bool
    bbox: np.ndarray
    confidence: float
    track_id: int
