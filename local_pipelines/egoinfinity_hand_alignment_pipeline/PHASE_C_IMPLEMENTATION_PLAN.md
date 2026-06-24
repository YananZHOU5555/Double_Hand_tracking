# EgoInfinity + FoundationStereo Hand Phase-C Implementation Plan

This document is the detailed implementation plan for adapting the EgoInfinity
hand post-processing stack to the LFV pipeline.  The main goal is to replace the
current sparse/visible-point hand-depth prototype with a full HandResult-level
pipeline:

```text
processed_topcam
  -> WiLoR full MANO HandResult
  -> EgoInfinity Phase-B track/label/dedup
  -> FoundationStereo metric depth
  -> depth-aligned hand cam_t
  -> temporal smoothing / biomech / MANO forward
  -> QA + overlay + 3D table-frame export
  -> gripper mapping / IK input
```

The core rule is: do not silently replace EgoInfinity logic with a cheaper
approximation.  If a strict EgoInfinity component cannot run, the output must say
that explicitly in the summary and in the per-frame QA.

## 0. Current State

### Existing LFV files

```text
local_pipelines/egoinfinity_hand_alignment_pipeline/
  export_wilor_handresults.py
  postprocess_handresults.py
  quality_check_handresults.py
  run_egoinfinity_hand_alignment_pipeline.sh
```

Current implemented stages:

1. Raw WiLoR full MANO export.
2. EgoInfinity Phase-B-style track / label / dedup postprocess.
3. Basic Phase-B quality report.

Current missing stages:

1. FoundationStereo depth generation inside this pipeline.
2. Depth stabilization.
3. Depth-guided `cam_t` alignment.
4. Motion infiller.
5. Biomechanical constraints.
6. MANO parameter smoothing + MANO forward.
7. Phase-C quality report and visual outputs.

### Strict component snapshot

The strict hand Phase-C source snapshot now lives under:

```text
local_pipelines/egoinfinity_hand_alignment_pipeline/egoinfinity_strict/
```

Copied/adapted components:

```text
depth_align.py
depth_stabilize.py
biomech_constraints.py
mano_smoothing.py
motion_infiller.py
pose_tracker/memfof_flow.py
infiller_utils/
```

The local adaptation is intentionally small:

1. `depth_align.py` adds LFV `fx/fy/cx/cy` backprojection for rectified/cropped
   FoundationStereo depth.
2. `depth_stabilize.py` imports the local MEMFOF wrapper.
3. `hand_result.py` provides a minimal HandResult dataclass backed by NPZ data.

Strict component setup/check scripts:

```text
setup_egoinfinity_strict_components.sh
check_egoinfinity_strict_components.py
phase_c_quality_gates.py
```

Verified local components:

```text
/home/yannan/workspace/EgoInfinity/pretrained_models/detector.pt
/home/yannan/workspace/EgoInfinity/pretrained_models/wilor_final.ckpt
/home/yannan/workspace/EgoInfinity/pretrained_models/infiller.pt
/home/yannan/workspace/EgoInfinity/pretrained_models/sam2.1_hiera_small.pt
/home/yannan/workspace/EgoInfinity/third_party/wilor/mano_data/MANO_RIGHT.pkl
memfof package + MEMFOF checkpoint cache
```

### Existing LFV FoundationStereo prototype

The old prototype lives mostly under:

```text
local_pipelines/double_hand_pipeline/run_foundationstereo_hand_model_trim20.sh
scripts/run_lfv_foundationstereo_disparity.py
scripts/export_wilor_visible_foundation_depth_table_html.py
scripts/export_wilor_visible_hand_skeleton_table_html.py
```

This prototype samples FoundationStereo depth only at visible WiLoR landmarks and
then fits a local hand skeleton to those metric points.  This is useful for
diagnostics, but it is not equivalent to EgoInfinity Phase-C because it does not
update the full MANO HandResult state.

## 1. Target Output Layout

For a session:

```text
<session>/quality/egoinfinity_hand_alignment_pipeline/
  stages/
    raw_wilor_handresults/
    phase_b_track_postprocess/
    foundationstereo_depth/
    phase_c_depth_align/
    phase_c_mano_smooth/
  quality_check/
    handresults_quality_summary.json
    phase_c_depth_quality_summary.json
    phase_c_alignment_quality.csv
    phase_c_track_quality.csv
    phase_c_contact_sheet.jpg
    phase_b_vs_phase_c_overlay.mp4
    phase_c_hand_table.html
```

The main Phase-C NPZ should be:

```text
stages/phase_c_mano_smooth/wilor_handresults_phase_c_foundation.npz
```

Required new NPZ fields:

```text
frame_index                         (N,)
track_id                            (N,)
hand_label                          (N,)
is_right                            (N,)
bbox_xyxy                           (N, 4)
det_conf                            (N,)

global_orient_raw                   (N, 1, 3, 3)
hand_pose_raw                       (N, 15, 3, 3)
betas_raw                           (N, 10)
cam_t_wilor                         (N, 3)
joints_3d_rel_raw                   (N, 21, 3)
vertices_rel_raw                    (N, 778, 3)
joints_uv                           (N, 21, 2)

cam_t_depth                         (N, 3)
cam_t_smooth                        (N, 3)
global_orient_smooth                (N, 1, 3, 3)
hand_pose_smooth                    (N, 15, 3, 3)
betas_track_median                  (N, 10)
joints_cam_depth                    (N, 21, 3)
joints_cam_smooth                   (N, 21, 3)
vertices_cam_smooth                 (N, 778, 3)

alignment_source                    (N,) string
alignment_valid_joint_count         (N,)
alignment_joint_ids                 (N,) object/string
alignment_rms_m                     (N,)
alignment_max_residual_m            (N,)
stabilize_scale                     (num_depth_frames,)
stabilize_offset_m                  (num_depth_frames,)
qc_flag                             (N,) string
faces                               (F, 3)
image_size                          (2,)
foundation_camera                   object/json
```

`alignment_source` must be one of:

```text
depth_reliable_joints
depth_all_joints
wilor_focal_rescale_fallback
missing_depth
not_evaluated
```

## 2. Pipeline Nodes And Quality Checks

Each node must write both data artifacts and QA artifacts.  The runner should
fail on hard errors and continue with warnings only when the output is explicitly
marked as degraded.

### Node A: Processed Topcam Preflight

Inputs:

```text
processed_topcam/left_table.mp4
processed_topcam/right_table.mp4
processed_topcam/processing_metadata.json
config/stereo_calibration_fisheye.json
```

Implementation:

1. Confirm left/right table videos exist.
2. Confirm frame counts match.
3. Confirm frame dimensions match `processing_metadata.crop.width/height`.
4. Confirm calibration and crop metadata are from current latest calibration.

Quality check:

```text
QC-A1 image_size_match
QC-A2 left_right_frame_count_match
QC-A3 metadata_crop_match
QC-A4 calibration_file_exists
QC-A5 table_frame_latest_exists
```

Hard fail:

1. Missing left/right videos.
2. Mismatched frame count.
3. Video size does not match crop metadata.

Warning:

1. Missing table-frame JSON only blocks table-frame export, not camera-frame
   Phase-C.

### Node B: Raw WiLoR Full MANO Export

Existing script:

```text
export_wilor_handresults.py
```

Inputs:

```text
processed_topcam/left_table.mp4
WiLoR checkpoint
MANO_RIGHT.pkl
```

Outputs:

```text
raw_wilor_handresults/wilor_handresults_raw.npz
raw_wilor_handresults/wilor_predictions_raw.csv
raw_wilor_handresults/wilor_detections_raw.csv
raw_wilor_handresults/wilor_handresults_raw_overlay.mp4 optional
raw_wilor_handresults/wilor_handresults_raw_summary.json
```

Quality check:

```text
QC-B1 candidates_count > 0
QC-B2 finite_mano_params
QC-B3 finite_cam_t
QC-B4 joints_uv_inside_or_near_image
QC-B5 bbox_inside_image
QC-B6 per_frame_detection_count
QC-B7 per_side_detection_coverage
QC-B8 raw_track_id_present
```

Initial thresholds:

```text
accepted_hand_coverage >= 0.20                  warning below
nan_ratio(global_orient/hand_pose/betas) == 0   hard fail
nan_ratio(joints_uv/cam_t) == 0                 hard fail
uv outside image by > 30 px                     warning
bbox outside image by > 10 px                   warning
```

Notes:

1. This stage should keep both hands when `HAND=best`.
2. It should not decide robot arm mapping.
3. It should preserve WiLoR/MANO outputs as raw evidence.

### Node C: EgoInfinity Phase-B Track / Label / Dedup

Existing script:

```text
postprocess_handresults.py
```

Implementation target:

1. Use EgoInfinity-style previous-frame greedy bbox IoU tracking.
2. Use track-level handedness majority vote.
3. Mirror geometry when handedness is corrected, matching EgoInfinity behavior.
4. Drop same-frame duplicate bboxes using IoU threshold.
5. Remove bad tracks using short-track, position-outlier, and size-outlier rules.

Outputs:

```text
phase_b_track_postprocess/wilor_handresults_phase_b.npz
phase_b_track_postprocess/wilor_predictions_phase_b.csv
phase_b_track_postprocess/wilor_phase_b_events.csv
phase_b_track_postprocess/wilor_phase_b_summary.json
quality_check/handresults_track_timeline.png
quality_check/handresults_phase_b_contact_sheet.jpg
```

Quality check:

```text
QC-C1 track_count_by_side
QC-C2 dominant_track_coverage_by_side
QC-C3 label_flip_events
QC-C4 duplicate_drop_count
QC-C5 remaining_same_side_overlap
QC-C6 short_track_removed_count
QC-C7 track_gap_count
QC-C8 sudden_side_switch_count
```

Initial thresholds:

```text
same-frame duplicate IoU > 0.30                 drop lower confidence
track length < 5 frames                         drop
same-side secondary track far > 0.50 m          drop if short
size ratio > 3.0 or < 0.33                      drop if short
remaining same-side IoU > 0.30 after dedup      hard fail
label flips on a track                          warning, must be logged
```

Important:

EgoInfinity's tracker does not bridge long gaps.  If a physical hand disappears
and returns, a new track is expected.  Do not merge tracks silently at Phase-B.
Gap bridging belongs to the infiller / identity-continuity layer.

### Node D: FoundationStereo Depth

Existing script:

```text
scripts/run_lfv_foundationstereo_disparity.py
```

Inputs:

```text
processed_topcam/left_table.mp4
processed_topcam/right_table.mp4
processed_topcam/processing_metadata.json
config/stereo_calibration_fisheye.json
```

Outputs:

```text
foundationstereo_depth/disparity/disparity_%08d.npy
foundationstereo_depth/depth/depth_%08d.npy
foundationstereo_depth/lr_mask/lr_mask_%08d.npy
foundationstereo_depth/foundationstereo_depth_frames.csv
foundationstereo_depth/foundationstereo_depth_summary.json
foundationstereo_depth/depth_preview_contact_sheet.jpg
```

Coordinate convention:

```text
depth_m is z in the left rectified/cropped camera frame.
pixel coordinates must match processed_topcam/left_table.mp4.
```

Quality check:

```text
QC-D1 depth_frame_count_matches_requested_frames
QC-D2 depth_image_size_matches_wilor_image_size
QC-D3 finite_depth_ratio_per_frame
QC-D4 hand_bbox_depth_valid_ratio
QC-D5 reliable_joint_depth_valid_count
QC-D6 median_depth_range
QC-D7 lr_mask_valid_ratio
```

Initial thresholds:

```text
global valid_depth_ratio median < 0.15          warning
global valid_depth_ratio median < 0.05          hard fail
hand bbox valid ratio < 0.05                    frame warning
median depth outside [0.15, 2.50] m             frame warning
depth frame missing                             hard fail
image size mismatch                             hard fail
```

Note:

FoundationStereo replaces EgoInfinity's MoGe-2 depth source.  This is acceptable
because downstream EgoInfinity Phase-C only needs a metric depth map plus camera
intrinsics.  The LFV implementation must use `fx, fy, cx, cy`, not a single
assumed center focal unless `fx == fy` is verified.

### Node E: Depth Stabilization

EgoInfinity source:

```text
EgoInfinity/egoinfinity/pipeline/depth_stabilize.py
```

Implementation target:

1. Build dynamic masks from hand bboxes.
2. Optionally merge optical-flow masks in strict mode.
3. Build static background template by temporal median.
4. Estimate per-frame scale and offset on background pixels.
5. Apply correction to the full depth map.

Outputs:

```text
phase_c_depth_align/depth_stable/depth_%08d.npy
phase_c_depth_align/depth_stabilization_summary.json
phase_c_depth_align/background_depth_template.npy
phase_c_depth_align/dynamic_mask_preview.jpg
```

Quality check:

```text
QC-E1 background_template_valid_ratio
QC-E2 dynamic_mask_bbox_coverage
QC-E3 stabilize_scale_distribution
QC-E4 stabilize_offset_distribution
QC-E5 background_residual_before_after
QC-E6 optical_flow_mask_status
```

Initial thresholds:

```text
template valid ratio < 0.20                     warning
scale outside [0.70, 1.30]                      frame hard fail
scale outside [0.85, 1.15]                      frame warning
abs(offset) > 0.20 m                            frame hard fail
abs(offset) > 0.08 m                            frame warning
background residual not improved                warning
```

Strictness:

1. Bbox-only stabilization is allowed as a first implementation but must be
   labeled `flow_mask_status=bbox_only`.
2. Full EgoInfinity parity requires MEMFOF optical flow masks.  If MEMFOF is not
   available, do not claim strict Phase-C parity.

### Node F: Depth-Guided Hand `cam_t` Alignment

EgoInfinity source:

```text
EgoInfinity/egoinfinity/pipeline/depth_align.py
```

LFV adaptation:

Use FoundationStereo depth with separate intrinsics:

```text
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
Z = depth_m
```

Alignment rule:

1. Try reliable joint IDs:

```text
[0, 5, 9, 13, 17]  # wrist + index/middle/ring/pinky MCP
```

2. Sample median depth patch around each joint.
3. Back-project valid joints into camera frame.
4. Compute per-joint translation:

```text
t_i = P_depth_i - joints_3d_rel_i
```

5. Use robust median translation as `cam_t_depth`.
6. If fewer than 2 reliable joints are valid, try all 21 joints.
7. If no valid joints exist, fallback to WiLoR focal rescale and mark it.

Outputs:

```text
phase_c_depth_align/wilor_handresults_phase_c_depth_aligned.npz
phase_c_depth_align/phase_c_alignment_quality.csv
phase_c_depth_align/phase_c_alignment_summary.json
```

Quality check:

```text
QC-F1 valid_reliable_joint_count
QC-F2 valid_all_joint_count
QC-F3 alignment_source_ratio
QC-F4 alignment_rms_m
QC-F5 alignment_max_residual_m
QC-F6 cam_t_depth_finite
QC-F7 cam_t_depth_jump
QC-F8 depth_vs_wilor_tz_delta
```

Initial thresholds:

```text
valid reliable joints >= 2                      good
valid reliable joints == 1 and all joints >= 2  warning, use all joints
valid all joints == 0                           fallback, frame warning
alignment RMS <= 0.035 m                        good
alignment RMS in (0.035, 0.080] m               warning
alignment RMS > 0.080 m                         frame bad
cam_t jump > 0.08 m/frame at 30 FPS             warning
cam_t jump > 0.15 m/frame at 30 FPS             hard frame flag
fallback ratio > 0.20 for a track               track warning
fallback ratio > 0.50 for a track               track bad
```

No silent fallback:

Every fallback must write:

```text
alignment_source
alignment_reason
valid_joint_ids
sampled_depths_m
```

### Node G: Translation And Joint Smoothing

EgoInfinity source:

```text
smooth_translations
reject_spikes
smooth_joints_savgol
```

Implementation:

1. Group candidates by `track_id`.
2. Build `track_cam_ts[track_id][frame] = cam_t_depth`.
3. Apply median smoothing with window 5.
4. Build joints:

```text
joints_cam_depth = joints_3d_rel + cam_t_smooth
```

5. Run spike rejection.
6. Run Savitzky-Golay smoothing on joints with window 7 and polyorder 2.

Outputs:

```text
phase_c_mano_smooth/phase_c_track_smoothing_quality.csv
```

Quality check:

```text
QC-G1 pre_smooth_velocity_distribution
QC-G2 post_smooth_velocity_distribution
QC-G3 pre_smooth_acceleration_distribution
QC-G4 post_smooth_acceleration_distribution
QC-G5 spike_removed_count
QC-G6 smoothing_displacement_from_aligned_cam_t
QC-G7 short_track_smoothing_skipped_count
```

Initial thresholds:

```text
median smoothing displacement > 0.05 m          warning
max smoothing displacement > 0.12 m             track warning
spike removed count > 5% of track length        track warning
short track < 7 frames                          skip SavGol, log it
```

Important:

Smoothing should reduce obvious single-frame jumps but must not erase meaningful
grasp/contact motion.  Therefore the QA must compare before/after displacement
and not only report that the curve is smoother.

### Node H: Biomechanical Constraints

EgoInfinity source:

```text
biomech_constraints.py
```

Implementation:

1. Convert `hand_pose` rotation matrices to axis-angle.
2. Clamp each MANO joint to plausible swing/twist ranges.
3. Convert back to rotation matrices.
4. If MANO model is loaded, recompute local joints and vertices.

Outputs:

```text
phase_c_mano_smooth/biomech_events.csv
```

Quality check:

```text
QC-H1 clamped_candidate_count
QC-H2 clamped_joint_histogram
QC-H3 max_pose_delta_after_clamp
QC-H4 mano_forward_success_after_clamp
```

Initial thresholds:

```text
clamped candidates > 30% of track               warning
MANO forward failure                            hard fail for MANO smooth stage
non-finite joints/vertices after clamp          hard fail
```

### Node I: MANO Parameter Smoothing + MANO Forward

EgoInfinity source:

```text
mano_smoothing.py
```

Implementation:

1. Group by `track_id`.
2. Convert `global_orient` and `hand_pose` rotation matrices to quaternions.
3. Enforce quaternion sign continuity.
4. Apply SavGol smoothing.
5. Convert smoothed quaternions to axis-angle.
6. Use per-track median `betas`.
7. Apply biomechanical clamp again.
8. Run MANO forward.
9. Add `cam_t_smooth` to get camera-frame joints and vertices.
10. Flip x for left hands exactly as WiLoR/EgoInfinity does.

Outputs:

```text
phase_c_mano_smooth/wilor_handresults_phase_c_foundation.npz
phase_c_mano_smooth/mano_smoothing_summary.json
phase_c_mano_smooth/mano_smoothing_events.csv
```

Quality check:

```text
QC-I1 quaternion_norm_close_to_1
QC-I2 quaternion_sign_flip_count
QC-I3 smoothed_rotation_delta
QC-I4 betas_track_std_before
QC-I5 vertices_finite
QC-I6 joints_finite
QC-I7 projected_joints_match_original_bbox
QC-I8 mesh_temporal_jump
```

Initial thresholds:

```text
quaternion norm error > 1e-3                    hard fail
non-finite vertices/joints                      hard fail
median beta std per track > 0.10                warning
projected wrist/MCP drift > 20 px median        warning
mesh jump > 0.15 m/frame                        frame bad
```

Notes:

1. This stage requires loading the same MANO model used by WiLoR.
2. If the MANO model cannot be loaded, the stage must stop or explicitly output
   `mano_forward_status=missing_model`; it must not emit fake smoothed mesh.

### Node J: Motion Infiller

EgoInfinity source:

```text
motion_infiller.py
```

Dependency:

```text
HaWoR / EgoInfinity Transformer checkpoint
```

Strict implementation:

1. Use the original `MotionInfiller`.
2. Input aligned `cam_t`, MANO orientation, hand pose, betas.
3. Fill missing frames only inside active range.
4. Inject filled HandResult objects with `confidence=0.0`.

Outputs:

```text
phase_c_mano_smooth/infill_events.csv
phase_c_mano_smooth/infill_summary.json
```

Quality check:

```text
QC-J1 checkpoint_exists
QC-J2 dual_valid_ratio
QC-J3 filled_frame_count
QC-J4 boundary_position_continuity
QC-J5 boundary_rotation_continuity
QC-J6 no_fill_outside_active_range
```

Policy:

1. If checkpoint is missing, skip and mark `infiller_status=checkpoint_missing`.
2. Do not use linear interpolation as the default replacement.
3. If a deterministic interpolation fallback is later added for debugging, output
   must be clearly named `phase_c_interp_debug`, not the main Phase-C result.

### Node K: Visualization And Export

Required outputs:

```text
quality_check/phase_b_vs_phase_c_overlay.mp4
quality_check/phase_c_hand_table.html
quality_check/phase_c_contact_sheet.jpg
quality_check/phase_c_depth_alignment_contact_sheet.jpg
quality_check/phase_c_summary.md
```

Visualization requirements:

1. Overlay should show raw WiLoR mesh and Phase-C smoothed mesh with different
   colors.
2. Overlay must show frame index.
3. 3D HTML must support toggles:

```text
show left hand
show right hand
show raw WiLoR
show depth-aligned
show smoothed MANO
show FoundationStereo sampled reliable joints
show track labels
show bad/fallback frames
```

4. Table-frame HTML should use `table_frame_latest.json` only at export time.
   The core Phase-C state stays in left camera frame.

Quality check:

```text
QC-K1 overlay_video_written
QC-K2 html_written
QC-K3 table_transform_exists
QC-K4 camera_to_table_points_finite
QC-K5 per_track_bounds_reasonable
QC-K6 bad_frames_visible_in_html
```

Initial thresholds:

```text
table-frame hand z outside [-0.20, 1.20] m      warning
track spatial extent > 1.50 m                   warning
missing overlay/html                            hard fail for release run
```

### Node L: Gripper Mapping Interface

This is not implemented in Phase-C first pass, but the output schema must be
ready for it.

Required export:

```text
phase_c_mano_smooth/wilor_phase_c_gripper_input.csv
```

Candidate fields:

```text
frame_index
elapsed_sec
track_id
hand_label
hand_to_robot_arm_label
wrist_table_xyz
index_mcp_table_xyz
middle_mcp_table_xyz
thumb_mcp_table_xyz
tcp_proxy_table_xyz
hand_basis_x_table
hand_basis_y_table
hand_basis_z_table
alignment_source
alignment_rms_m
phase_c_qc_flag
```

Quality check:

```text
QC-L1 required_landmarks_finite
QC-L2 basis_orthonormality
QC-L3 tcp_proxy_finite
QC-L4 arm_mapping_label_present
QC-L5 bad_alignment_frames_excluded_or_flagged
```

## 3. Runner Design

Extend:

```text
local_pipelines/egoinfinity_hand_alignment_pipeline/run_egoinfinity_hand_alignment_pipeline.sh
```

New environment variables:

```text
RUN_PHASE_C=false
REBUILD_DEPTH=false
REBUILD_PHASE_C=false
STRICT_EGOINFINITY=false
DEPTH_STABILIZE=true
USE_FLOW_MASK=false
RUN_INFILLER=false
INFILLER_CHECKPOINT=
FOUNDATIONSTEREO_ROOT=/home/yannan/workspace/external/FoundationStereo
STEREO_MODEL=/home/yannan/workspace/external/FoundationStereo/pretrained_models/23-51-11/model_best_bp2.pth
VALID_ITERS=16
LR_CHECK=true
MAX_DEPTH_M=2.5
ALIGN_PATCH_SIZE=7
ALIGN_MIN_RELIABLE_JOINTS=2
SMOOTH_TRANSLATION_WINDOW=5
SMOOTH_JOINT_WINDOW=7
SMOOTH_JOINT_POLYORDER=2
```

Expected command:

```bash
cd /home/yannan/workspace/learning-from-video
RUN_PHASE_C=true REBUILD_PHASE_C=true \
  bash local_pipelines/egoinfinity_hand_alignment_pipeline/run_egoinfinity_hand_alignment_pipeline.sh \
  /home/yannan/workspace/ros1_docker-main/rosbag_data/human_teaching_videos/bag_20260622_1548_001
```

The runner must print:

```text
raw_handresults_npz
phase_b_handresults_npz
foundation_depth_summary
phase_c_depth_aligned_npz
phase_c_final_npz
phase_c_quality_summary
phase_c_overlay_mp4
phase_c_hand_table_html
```

## 4. Implementation Milestones

### Milestone 1: Data adapter and short-window depth alignment

Target frames:

```text
FRAME_START=580
FRAME_END=640
```

Work:

1. Write `phase_c_foundation_depth_align.py`.
2. Load Phase-B NPZ.
3. Load FoundationStereo depth maps.
4. Align `cam_t_depth`.
5. Emit alignment CSV/JSON.
6. Emit camera-frame 3D HTML or contact sheet for this window.

Acceptance:

```text
phase_c_alignment_quality.csv exists
fallback ratio is reported
cam_t_depth is finite for most detected hands
right/left tracks no longer overlap due to missing metric depth
```

### Milestone 2: Full-sequence depth alignment QA

Work:

1. Run full baseline sequence.
2. Add per-track quality summary.
3. Add track timeline with fallback/bad frames.

Acceptance:

```text
phase_c_alignment_summary.json reports hard_errors=0
dominant tracks have depth alignment source for most frames
all fallback frames are visible in contact sheet/html
```

### Milestone 3: Depth stabilization

Work:

1. Port bbox-based stabilization.
2. Save stable depth maps.
3. Add stabilization QA.

Acceptance:

```text
background residual improves or stays stable
scale/offset distributions are reasonable
hand-depth alignment does not get worse compared with raw depth
```

### Milestone 4: Translation and joint smoothing

Work:

1. Add median `cam_t` smoothing.
2. Add spike rejection.
3. Add SavGol joint smoothing.

Acceptance:

```text
single-frame jumps around frames 599-626 are reduced or explicitly flagged
smoothing displacement stays below warning threshold for normal frames
no useful contact motion is erased in overlay
```

### Milestone 5: MANO smoothing and forward

Work:

1. Load WiLoR MANO model.
2. Port quaternion smoothing.
3. Apply median betas.
4. Apply biomech constraints.
5. Re-run MANO forward.

Acceptance:

```text
phase_c final NPZ has vertices_cam_smooth and joints_cam_smooth
overlay MP4 shows smoother mesh without hand-label flips
MANO forward has no non-finite outputs
```

### Milestone 6: Strict EgoInfinity optional components

Work:

1. Add MEMFOF optical-flow masks if dependencies are available.
2. Add MotionInfiller if checkpoint exists.

Acceptance:

```text
summary explicitly says strict components enabled/disabled
no checkpoint-missing stage is reported as successful strict parity
```

### Milestone 7: Export for gripper mapping

Work:

1. Convert Phase-C camera-frame hand state to table frame.
2. Export landmark and basis CSV.
3. Add robot-arm mapping labels later from upstream.

Acceptance:

```text
gripper input CSV has finite wrist/index/middle/thumb landmarks
bad/fallback frames are marked and can be filtered downstream
```

## 5. Main Risks

### Coordinate mismatch

Risk:

WiLoR `joints_uv` and FoundationStereo depth must refer to the same
`processed_topcam/left_table.mp4` pixel coordinate system.

Mitigation:

`QC-D2` and `QC-F` must hard-fail on image size mismatch or impossible sampled
depth.

### Occlusion and object contact

Risk:

FoundationStereo depth at fingertips can be invalid or object depth, not hand
depth.

Mitigation:

Default depth alignment uses wrist + MCPs, not fingertips.  Fingertips are only
used as all-joint fallback and must be marked.

### WiLoR hand flips

Risk:

If WiLoR produces a wrong left/right hand or palm flip, depth cannot fully fix
it.

Mitigation:

Phase-B track majority vote and Phase-C rotation/jump QA must flag these frames.
They should be visible in contact sheets.

### Motion infiller dependency

Risk:

EgoInfinity's infiller requires a checkpoint not guaranteed to exist locally.

Mitigation:

Do not approximate it in the main result.  Skip and mark status unless the real
checkpoint is available.

### Runtime

Risk:

FoundationStereo and MANO smoothing on full sequences can be slow.

Mitigation:

Support `FRAME_START/FRAME_END/MAX_FRAMES` for short-window tests, then full run.

## 6. Immediate Next Code Changes

Recommended first code patch:

```text
local_pipelines/egoinfinity_hand_alignment_pipeline/
  phase_c_foundation_depth_align.py
  phase_c_quality_check.py
```

Then update:

```text
run_egoinfinity_hand_alignment_pipeline.sh
README.md
```

First test command:

```bash
cd /home/yannan/workspace/learning-from-video
RUN_PHASE_C=true REBUILD_PHASE_C=true FRAME_START=580 FRAME_END=640 \
  bash local_pipelines/egoinfinity_hand_alignment_pipeline/run_egoinfinity_hand_alignment_pipeline.sh \
  /home/yannan/workspace/ros1_docker-main/rosbag_data/human_teaching_videos/bag_20260622_1548_001
```

First full test command:

```bash
cd /home/yannan/workspace/learning-from-video
RUN_PHASE_C=true REBUILD_PHASE_C=true MAX_FRAMES=0 \
  bash local_pipelines/egoinfinity_hand_alignment_pipeline/run_egoinfinity_hand_alignment_pipeline.sh \
  /home/yannan/workspace/ros1_docker-main/rosbag_data/human_teaching_videos/bag_20260622_1548_001
```
