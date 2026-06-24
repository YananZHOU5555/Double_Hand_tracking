# EgoInfinity-style Hand Alignment Pipeline

This folder is an isolated fork of the current best
`egoinfinity_hand_pipeline` state. Use it for hand / MANO / depth alignment
experiments. Keep the original `local_pipelines/egoinfinity_hand_pipeline`
available for gripper-mapping work in the parallel debugging thread.

Current implementation state is tracked in:

```text
local_pipelines/egoinfinity_hand_alignment_pipeline/PIPELINE_IMPLEMENTATION.md
```

This is an experimental, isolated hand-tracking pipeline. In this fork, changes
should stay focused on hand alignment and diagnostic visualization rather than
gripper mapping or robot IK.

## Goal

Move from the current CSV-only WiLoR outputs to a HandResult-like representation
that can support the EgoInfinity hand post-processing stages:

1. WiLoR full MANO export.
2. EgoInfinity Phase-B-compatible track-level handedness / dedup / bad-track filtering.
3. FoundationStereo depth alignment.
4. Missing-frame infill.
5. Biomechanical constraints.
6. Translation, joint, and MANO-parameter smoothing.
7. Export LFV CSV/NPZ/HTML diagnostics for alignment inspection.

Current migration implements raw WiLoR MANO export, Phase-B track/label/dedup,
FoundationStereo depth alignment, independent EgoInfinity-style depth
stabilization, EgoInfinity-style hand depth smoothing, EgoInfinity
MotionInfiller, MANO temporal smoothing/forward, MANO mesh visibility, and an
optional visibility-aware depth re-align experiment.

## Run

```bash
cd /home/yannan/workspace/learning-from-video
MAX_FRAMES=0 REBUILD_EGO_HAND=false RUN_PHASE_C=true \
RUN_PHASE_C_DEPTH_STABILIZE=true RUN_PHASE_C_DEPTH_SMOOTH=true \
RUN_MOTION_INFILL=true RUN_PHASE_C2=true RUN_PHASE_C3=true \
  bash local_pipelines/egoinfinity_hand_alignment_pipeline/run_egoinfinity_hand_alignment_pipeline.sh \
  /home/yannan/workspace/ros1_docker-main/rosbag_data/human_teaching_videos/bag_20260622_1548_001
```

Useful overrides:

```bash
HAND=best              # best, left, or right. best keeps both detected hands.
MAX_FRAMES=120         # quick smoke test
FRAME_START=580
FRAME_END=640
REBUILD_EGO_HAND=true
WILOR_CONF=0.30
RUN_PHASE_C_DEPTH_STABILIZE=true
RUN_MOTION_INFILL=true
RUN_VISIBILITY_REALIGN=true   # experimental branch, default off
```

## Outputs

All outputs live under:

```text
<demo>/quality/egoinfinity_hand_alignment_pipeline/
```

Current stage outputs:

```text
stages/raw_wilor_handresults/
  wilor_handresults_raw.npz
  wilor_predictions_raw.csv
  wilor_detections_raw.csv
  wilor_handresults_raw_summary.json

stages/phase_b_track_postprocess/
  wilor_handresults_phase_b.npz
  wilor_predictions_phase_b.csv
  wilor_phase_b_events.csv
  wilor_phase_b_summary.json

stages/foundationstereo_depth/
stages/foundationstereo_depth_stabilized/  # when RUN_PHASE_C_DEPTH_STABILIZE=true
stages/phase_c_depth_align/
stages/phase_c_depth_smooth/
stages/phase_c_motion_infiller/            # when RUN_MOTION_INFILL=true
stages/phase_c_mano_smooth/
stages/phase_c_mesh_visibility/
stages/phase_c_visibility_depth_realign/   # only when RUN_VISIBILITY_REALIGN=true
```

The NPZ files contain MANO parameters:

```text
global_orient        (N, 1, 3, 3)
hand_pose            (N, 15, 3, 3)
betas                (N, 10)
vertices_rel         (N, 778, 3)
joints_3d_rel        (N, 21, 3)
vertices_cam         (N, 778, 3)
joints_cam           (N, 21, 3)
cam_t                (N, 3)
joints_uv            (N, 21, 2)
```

## Current Limitations

- Independent depth stabilization and MotionInfiller are implemented but default
  off, because both change downstream behavior and should be compared per demo.
- Visibility-aware re-align is experimental and off by default because it can
  make results worse when visibility or depth is wrong.
- Phase-B follows the EgoInfinity implementation level: detector-style previous-frame
  IoU track IDs, per-track handedness majority vote, duplicate removal, and bad-track
  filtering. EgoInfinity also mirrors geometry at this stage when correcting
  handedness; MANO forward is done later in the Phase-C smoothing stage.

## Strict EgoInfinity Components

Strict hand Phase-C uses a local snapshot under:

```text
egoinfinity_strict/
  depth_align.py
  depth_stabilize.py
  biomech_constraints.py
  mano_smoothing.py
  motion_infiller.py
  pose_tracker/memfof_flow.py
  infiller_utils/
```

This snapshot is copied from `/home/yannan/workspace/EgoInfinity` and lightly
adapted for LFV:

- `depth_align.py` keeps the EgoInfinity translation-from-depth logic and adds
  an LFV `fx/fy/cx/cy` backprojection path for FoundationStereo.
- `depth_stabilize.py` imports the local MEMFOF wrapper instead of the external
  EgoInfinity package path.
- `hand_result.py` is a minimal dataclass so Phase-C can use NPZ-exported
  HandResult fields without importing WiLoR at module import time.

Download/check strict components:

```bash
cd /home/yannan/workspace/learning-from-video
INSTALL_MEMFOF=true PREFETCH_MEMFOF=true \
  bash local_pipelines/egoinfinity_hand_alignment_pipeline/setup_egoinfinity_strict_components.sh
```

Status check only:

```bash
cd /home/yannan/workspace/learning-from-video
/home/yannan/miniforge3/envs/wilor_lfv/bin/python \
  local_pipelines/egoinfinity_hand_alignment_pipeline/check_egoinfinity_strict_components.py \
  --egoinfinity-root /home/yannan/workspace/EgoInfinity \
  --checkpoint-dir /home/yannan/workspace/EgoInfinity/pretrained_models
```

Current verified strict status is saved at:

```text
local_pipelines/egoinfinity_hand_alignment_pipeline/strict_components_status.json
```

## Node Quality Gates

Node-level QA entry point:

```bash
cd /home/yannan/workspace/learning-from-video
/home/yannan/miniforge3/envs/wilor_lfv/bin/python \
  local_pipelines/egoinfinity_hand_alignment_pipeline/phase_c_quality_gates.py \
  --session-dir /home/yannan/workspace/ros1_docker-main/rosbag_data/human_teaching_videos/bag_20260622_1548_001
```

It currently checks:

1. Processed topcam preflight.
2. Phase-B HandResult NPZ integrity.
3. FoundationStereo depth summary.
4. Phase-C0b independent depth stabilization when present.
5. Phase-C depth alignment.
6. Phase-C1b depth smoothing when present.
7. Phase-C1c MotionInfiller when present.
8. Phase-C2 MANO temporal smoothing when present.
9. Phase-C3 MANO mesh visibility when present.
10. Phase-C4 visibility-aware depth re-align when present.
