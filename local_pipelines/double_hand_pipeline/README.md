# Double Hand Pipeline

This folder is a working copy of `local_pipelines/qzy_wilor_fallback` for the
new dual-hand teaching videos under:

```text
/home/yannan/workspace/ros1_docker-main/rosbag_data/human_teaching_videos
```

Current status:

- `run_double_hand_pipeline.sh` runs one explicitly mapped hand/arm lane.
- `run_double_hand_lanes.sh` is a thin multi-lane launcher that consumes an
  upstream mapping such as `HAND_ARM_MAP='left:left,right:right'`.
- The scripts do not infer the hand identity or arm mapping by default. This is
  intentional because the new videos can contain two hands or alternating
  single/double-hand segments.

Each lane writes outputs into a hand/arm-specific folder:

```text
<session>/quality/double_hand_pipeline/<hand>_hand_<arm>_arm/
```

Examples:

```bash
cd /home/yannan/workspace/learning-from-video

# Right hand -> right arm.
HAND=right ARM=right \
bash local_pipelines/double_hand_pipeline/run_double_hand_pipeline.sh BAG_20260622_1804_001

# Left hand -> left arm.
HAND=left ARM=left \
bash local_pipelines/double_hand_pipeline/run_double_hand_pipeline.sh BAG_20260622_1804_001

# Run both lanes from an explicit upstream mapping.
HAND_ARM_MAP='left:left,right:right' \
bash local_pipelines/double_hand_pipeline/run_double_hand_lanes.sh BAG_20260622_1804_001

# After both lanes have IK outputs, build one synchronized dual-arm replay bag.
HAND_ARM_MAP='left:left,right:right' \
bash local_pipelines/double_hand_pipeline/build_double_arm_dense_replay_bag.sh BAG_20260622_1804_001

# Inspect the replay bag before touching the robot.
DRY_RUN=true \
bash local_pipelines/double_hand_pipeline/play_double_arm_dense_replay_bag_enter.sh BAG_20260622_1804_001
```

Default paths and required labels:

- `HOST_SESSION_ROOT=/home/yannan/workspace/ros1_docker-main/rosbag_data/human_teaching_videos`
- `PROCESS_TOPCAM=auto`, so `processed_topcam/` is generated from `episode_0.bag` if missing
- `HAND` is required for single-lane runs: `left` or `right`
- `ARM` is required for single-lane runs: `left` or `right`
- `HAND_ARM_MAP` is required for multi-lane runs, using `<hand>:<arm>` entries separated by commas
- `ARM=right` uses `right_arm_table_latest.json` and `right_arm_touch_tcp_xr2_latest.json`
- `ARM=left` uses `left_arm_table_latest.json` and `left_arm_touch_tcp_zr2_latest.json`

Easy parts implemented in this copy:

- right and left robot frame selection based on explicit `ARM`
- hand-specific output folders under `quality/double_hand_pipeline/`
- lane metadata at `<lane>/lane_config.json`, recording the upstream labels and robot calibration files
- pre-gripper WiLoR segmentation confidence audit:
  `wilor_segmentation_occlusion_confidence/`
- a multi-lane launcher that runs independent left/right lanes without coordination logic
- synchronized dual-arm replay bag packaging from the two per-arm IK outputs
- a safety-gated dual-arm rosbag replay entry point
- diagnostic WiLoR candidate continuity checks before gripper mapping:
  bbox-track label flips, palm-normal flips, short missing gaps, and
  same-location duplicate hands
- temporal WiLoR stabilizer before gripper mapping. WiLoR is run with
  `--hand best` so all candidates are retained, then the stabilizer emits one
  stable target-hand CSV for the explicit `HAND` lane.

The pre-gripper confidence audit is intentionally diagnostic only. The WiLoR
stage now also saves `wilor_mesh_geometry.npz`, so the audit can use the MANO
mesh in camera frame instead of relying only on 2D masks. It writes per-frame
and per-landmark confidence before gripper mapping. In the double-hand pipeline
this audit defaults to all WiLoR candidates, because filtering by raw WiLoR
left/right labels would hide the frames where the detector label is wrong. It
downweights:

- WiLoR landmarks projected outside the segmented visible hand.
- Landmarks overlapping an optional object mask, when one is supplied.
- Landmarks whose nearby MANO surface patch is hidden behind the front-most
  MANO mesh surface in the current camera view.
- Distal finger landmarks that project inside the palm area but are deeper than
  the palm mean, as a fallback when mesh geometry is not available.

Before using mesh self-occlusion as a confidence penalty, the audit gates the
MANO model itself. A candidate is marked `model_unreliable` when detector
confidence is very low, the palm/core hand points jump together, many landmarks
jump together, or the projected MANO silhouette does not overlap the segmented
hand mask. In that case mesh self-occlusion is not trusted.

For dual-hand videos, the audit also builds a lightweight physical-hand track
from bbox IoU / center distance, independent of WiLoR's own left/right label.
This catches the common failure where the same image-space hand disappears for
one or two frames, reappears with the opposite label, or receives two overlapping
MANO candidates. These rows are marked with `track_label_flip`,
`track_palm_normal_flip`, `track_reacquired_gap`, and
`duplicate_overlap_drop`.

The stabilizer applies the same idea as an input gate for hand-to-gripper
mapping:

- `identity_hold_ms=400`: a left/right label change must remain stable for
  about 12 frames at 30fps before the physical track changes identity.
- `missing_bridge_ms=200`: a target hand missing for up to about 6 frames is
  bridged by holding the last good MANO pose.
- `palm_flip_window_ms=300`, `palm_flip_angle_deg=100`, and
  `hard_palm_speed_deg_s=900`: sudden palm-normal flips are held instead of
  passed into gripper mapping.
- same-location duplicate candidates keep only the higher-confidence candidate.

The stabilizer does not edit the original WiLoR CSV. It writes a new
`wilor_predictions_stabilized_<hand>.csv`, and downstream constrained hand/TCP
/ orientation stages consume that stable CSV.

In the overlay, solid dot color is the final confidence, while a red ring/x
marks a MANO mesh self-occluded landmark. This is the cue for the case where a
curled finger lies inside the 2D hand mask but is not actually visible from the
camera.

Useful outputs:

```text
<lane>/wilor_segmentation_occlusion_confidence/wilor_segmentation_occlusion_confidence.json
<lane>/wilor_segmentation_occlusion_confidence/wilor_frame_segmentation_confidence.csv
<lane>/wilor_segmentation_occlusion_confidence/wilor_landmark_segmentation_confidence.csv
<lane>/wilor_segmentation_occlusion_confidence/wilor_track_consistency_events.csv
<lane>/wilor_segmentation_occlusion_confidence/wilor_segmentation_occlusion_confidence_overlay.mp4
<lane>/wilor_segmentation_occlusion_confidence/wilor_segmentation_occlusion_worst_contact_sheet.jpg
<lane>/stages/wilor_temporal_stabilized_<hand>/wilor_predictions_stabilized_<hand>.csv
<lane>/stages/wilor_temporal_stabilized_<hand>/wilor_predictions_stabilized_<hand>_events.csv
<lane>/stages/wilor_temporal_stabilized_<hand>/wilor_predictions_stabilized_<hand>.json
```

Known missing pieces for robust dual-hand replay:

- simultaneous left/right hand tracking association and conflict handling
- evaluating whether hold-only bad-candidate handling should become
  interpolation for longer occlusions
- hand-model-assisted recovery when MediaPipe/WiLoR temporarily swaps or loses one hand
