# EgoInfinity Hand Pipeline Implementation

This file records what is actually implemented in the local LFV
`egoinfinity_hand_pipeline`.  It is meant to be the debugging map for the
current pipeline, not a paper-level design document.

## Entry Point

```bash
cd /home/yannan/workspace/learning-from-video
RUN_PHASE_C=true RUN_PHASE_C_DEPTH_STABILIZE=true RUN_PHASE_C_DEPTH_SMOOTH=true \
RUN_MOTION_INFILL=true RUN_PHASE_C2=true RUN_PHASE_C3=true \
  bash local_pipelines/egoinfinity_hand_pipeline/run_egoinfinity_hand_pipeline.sh \
  /home/yannan/workspace/ros1_docker-main/rosbag_data/human_teaching_videos/bag_20260622_1548_001
```

Optional branches are deliberately off unless requested:

```bash
RUN_VISIBILITY_REALIGN=true
DEPTH_STABILIZE_USE_FLOW_MASK=true
```

Output root:

```text
<demo>/quality/egoinfinity_hand_pipeline/
```

## Stage A: Processed Topcam Preflight

Input:

```text
<demo>/processed_topcam/left_table.mp4
<demo>/processed_topcam/right_table.mp4
<demo>/processed_topcam/processing_metadata.json
```

If `processed_topcam` is missing and `PROCESS_TOPCAM=auto`, the runner calls:

```text
scripts/process_lfv_demo_topcam.sh
```

QC:

```text
quality_check/phase_c_node_quality_gates.json
```

Checks: left/right video readable, same frame count, same size, crop metadata
matches video size, calibration/table-frame paths exist.

## Stage B0: Raw WiLoR Full MANO Export

Script:

```text
local_pipelines/egoinfinity_hand_pipeline/export_wilor_handresults.py
```

Input:

```text
processed_topcam/left_table.mp4
WiLor/pretrained_models/wilor_final.ckpt
WiLor/pretrained_models/detector.pt
WiLor/pretrained_models/model_config.yaml
WiLor/mano_data/MANO_RIGHT.pkl
```

Outputs:

```text
stages/raw_wilor_handresults/wilor_handresults_raw.npz
stages/raw_wilor_handresults/wilor_predictions_raw.csv
stages/raw_wilor_handresults/wilor_detections_raw.csv
stages/raw_wilor_handresults/wilor_handresults_raw_summary.json
```

Important NPZ fields:

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
faces                (1538, 3)
```

Status: implemented.

## Stage B1: Phase-B Track / Label / Dedup

Script:

```text
local_pipelines/egoinfinity_hand_pipeline/postprocess_handresults.py
```

Purpose:

- Greedy previous-frame bbox IoU tracking.
- Per-track hand-label majority correction.
- Same-frame duplicate removal.
- Short/bad track filtering.

Outputs:

```text
stages/phase_b_track_postprocess/wilor_handresults_phase_b.npz
stages/phase_b_track_postprocess/wilor_predictions_phase_b.csv
stages/phase_b_track_postprocess/wilor_phase_b_events.csv
stages/phase_b_track_postprocess/wilor_phase_b_summary.json
```

Status: implemented.  This is an EgoInfinity-style local adapter, not a byte-for-byte
reuse of the whole EgoInfinity runtime.

## Stage B-QC: Raw / Phase-B HandResult QC

Script:

```text
local_pipelines/egoinfinity_hand_pipeline/quality_check_handresults.py
```

Outputs:

```text
quality_check/handresults_quality_summary.json
quality_check/handresults_quality_per_candidate.csv
quality_check/handresults_track_timeline.png
quality_check/handresults_phase_b_contact_sheet.jpg
```

Status: implemented.

## Stage C0: FoundationStereo Depth

Script:

```text
scripts/run_lfv_foundationstereo_disparity.py
```

Enabled by:

```bash
RUN_PHASE_C=true
```

Outputs:

```text
stages/foundationstereo_depth/foundationstereo_depth_summary.json
stages/foundationstereo_depth/foundationstereo_depth_frames.csv
stages/foundationstereo_depth/depth_*.npy
```

Status: implemented and used by Phase-C.

QC: full-video coverage, stride, valid-depth ratio, missing depth files.

## Stage C0b: EgoInfinity Static-Camera Depth Stabilization

Script:

```text
local_pipelines/egoinfinity_hand_pipeline/phase_c0b_depth_stabilize.py
```

Enabled by:

```bash
RUN_PHASE_C_DEPTH_STABILIZE=true
```

Purpose:

- Wrap `EgoInfinity/egoinfinity/pipeline/depth_stabilize.py` as a standalone
  LFV node.
- Build dynamic masks from Phase-B hand bboxes.
- Compute a static background depth template from FoundationStereo depth.
- Estimate per-frame scale/offset from background pixels.
- Write corrected depth maps and a new depth summary JSON.
- Let Phase-C1/C1b/C4 consume the stabilized depth summary as a drop-in
  replacement for the raw FoundationStereo summary.

Optional:

```bash
DEPTH_STABILIZE_USE_FLOW_MASK=true
DEPTH_STABILIZE_WRITE_DYNAMIC_MASKS=true
```

Outputs:

```text
stages/foundationstereo_depth_stabilized/foundationstereo_depth_stabilized_summary.json
stages/foundationstereo_depth_stabilized/foundationstereo_depth_stabilized_frames.csv
stages/foundationstereo_depth_stabilized/depth_stabilize_corrections.csv
stages/foundationstereo_depth_stabilized/depth_stabilized/depth_stabilized_*.npy
stages/foundationstereo_depth_stabilized/background_depth_template.npy
```

Current baseline result on `bag_20260622_1548_001`:

```text
frames_exported: 843 / 843
template_valid_ratio: 0.9350
scale median: 0.9997, range: 0.9937..1.0037
offset median: 0.24 mm, range: -1.61..18.50 mm
qc: 843 ok, hard_errors: 0
```

Status: implemented and QC-gated.  Default is off so raw and stabilized depth
can be compared explicitly.

## Stage C1: Phase-C FoundationStereo Depth Alignment

Script:

```text
local_pipelines/egoinfinity_hand_pipeline/phase_c_foundation_depth_align.py
```

Purpose:

- Keep WiLoR/MANO root-relative hand structure.
- Replace absolute camera translation with FoundationStereo metric depth.
- Sample wrist + MCP joints first: `[0, 5, 9, 13, 17]`.
- Fallback to all joints only if too few reliable joints have valid depth.
- Record whether alignment came from reliable joints or fallback.

Outputs:

```text
stages/phase_c_depth_align/wilor_handresults_phase_c_depth_aligned.npz
stages/phase_c_depth_align/phase_c_alignment_quality.csv
stages/phase_c_depth_align/phase_c_alignment_summary.json
```

Important fields:

```text
cam_t_wilor
cam_t_depth
joints_cam_depth
vertices_cam_depth
alignment_source
alignment_valid_joint_count
alignment_rms_m
diagnosis_category
qc_issue_tags
foundation_camera_json
```

Depth stabilization:

- The preferred path is the independent Stage C0b depth summary.
- The older in-stage `DEPTH_STABILIZE=true` switch still exists for debugging,
  but it should not be used together with C0b unless testing double correction.

Status: implemented.

## Stage C1b: EgoInfinity Depth Smooth

Script:

```text
local_pipelines/egoinfinity_hand_pipeline/phase_c1b_depth_smooth.py
```

Enabled by:

```bash
RUN_PHASE_C_DEPTH_SMOOTH=true
```

Purpose:

- Adapt the hand-Z part of `EgoInfinity/egoinfinity/pipeline/post_tracking/depth_smooth.py`.
- Sample FoundationStereo depth at wrist + MCP joints: `[0, 5, 9, 13, 17]`.
- Use per-frame MAD filtering on sampled joint depths.
- Use temporal MAD rejection to remove single-frame depth spikes.
- Apply weighted Gaussian smoothing per physical hand track.
- Translate the whole hand along camera Z by `delta_z`.
- Optionally smooth mesh vertex mean-Z with `DEPTH_SMOOTH_VERTEX_MEAN_Z=true`.

Important behavior:

- The output NPZ overwrites `cam_t_depth`, `joints_cam_depth`, and
  `vertices_cam_depth` so Phase-C2 can consume it directly.
- The previous translation is preserved in `cam_t_depth_before_depth_smooth`.
- If this stage is enabled, the runner passes this NPZ into Phase-C2.
- The runner detects input mismatch, so toggling this stage on/off forces Phase-C2
  to rebuild instead of reusing the wrong output.

Outputs:

```text
stages/phase_c_depth_smooth/wilor_handresults_phase_c1b_depth_smooth.npz
stages/phase_c_depth_smooth/depth_smooth_quality.csv
stages/phase_c_depth_smooth/depth_smooth_track_summary.csv
stages/phase_c_depth_smooth/depth_smooth_summary.json
```

Important fields:

```text
cam_t_depth_before_depth_smooth
cam_t_depth_smooth
joints_cam_depth_smooth
vertices_cam_depth_smooth
depth_smooth_anchor_z_m
depth_smooth_anchor_z_smoothed_m
depth_smooth_delta_z_m
depth_smooth_vertex_mean_delta_z_m
depth_smooth_trust
depth_smooth_valid_sample_count
depth_smooth_status
depth_smooth_qc_flag
```

Current baseline result on `bag_20260622_1548_001`:

```text
candidates: 1323
tracks: 3
hard_errors: 0
before_wrist_z_jump_mean: 5.47 mm
after_wrist_z_jump_mean: 2.07 mm
before_wrist_z_jump_max: 347.1 mm
after_wrist_z_jump_max: 21.6 mm
qc: 1136 ok, 20 temporal outlier anchors, 1 bad large delta, 13 warn flags
```

Status: implemented and QC-gated.

## Stage C1c: EgoInfinity MotionInfiller

Script:

```text
local_pipelines/egoinfinity_hand_pipeline/phase_c1c_motion_infiller.py
```

Enabled by:

```bash
RUN_MOTION_INFILL=true
```

Source:

```text
EgoInfinity/egoinfinity/pipeline/motion_infiller.py
EgoInfinity/pretrained_models/infiller.pt
```

Purpose:

- Convert the current LFV NPZ candidate table into EgoInfinity `HandResult`
  lists.
- Run the original HaWoR/EgoInfinity transformer MotionInfiller when enough
  dual-hand context exists.
- Keep detected rows unchanged.
- Append synthetic rows only for missing `(frame, hand_label)` inside the active
  range.
- Re-run MANO forward for filled frames through the original infiller path, not
  by only translating an old mesh.

Important behavior:

- Filled rows are marked with `motion_infilled=1`.
- `motion_infiller_method` is `transformer`, `slerp_fallback`, or
  `neighbor_interp`.
- If enabled, the runner passes this NPZ into Stage C2.
- The runner detects input mismatch, so toggling this stage forces Stage C2 to
  rebuild.

Outputs:

```text
stages/phase_c_motion_infiller/wilor_handresults_phase_c1c_motion_infilled.npz
stages/phase_c_motion_infiller/motion_infiller_quality.csv
stages/phase_c_motion_infiller/motion_infiller_summary.json
```

Important fields:

```text
motion_infilled
motion_infiller_method
motion_infiller_gap_len
motion_infiller_nearest_detected_frame
motion_infiller_reference_index
motion_infiller_cam_t_jump_m
motion_infiller_wrist_jump_m
motion_infiller_qc_flag
```

Current baseline result on `bag_20260622_1548_001`:

```text
candidates_in: 1323
candidates_out: 1379
motion_infilled_candidates: 56
dual_valid_ratio: 0.906
long_gap_method: transformer
qc: 1378 ok, 1 bad_wrist_jump
bad frame: frame 772, right hand, detected row, not synthetic
```

Status: implemented and QC-gated.  Default is off because it changes candidate
count and should be compared visually before becoming the main path.

## Stage C2: MANO Temporal Smoothing + Biomech + MANO Forward

Script:

```text
local_pipelines/egoinfinity_hand_pipeline/phase_c2_mano_temporal_smooth.py
```

Enabled by:

```bash
RUN_PHASE_C2=true
```

Purpose:

- Group candidates by physical track.
- Convert `global_orient` and `hand_pose` rotation matrices to quaternions.
- Enforce quaternion sign continuity.
- Apply Savitzky-Golay smoothing per track.
- Convert back to axis-angle.
- Apply EgoInfinity strict biomechanical clamp.
- Use per-track median `betas`.
- Re-run MANO forward to generate a new smooth mesh and joints.

This is not a mesh-translation approximation.

Outputs:

```text
stages/phase_c_mano_smooth/wilor_handresults_phase_c2_mano_smooth.npz
stages/phase_c_mano_smooth/mano_smoothing_quality.csv
stages/phase_c_mano_smooth/mano_smoothing_track_summary.csv
stages/phase_c_mano_smooth/mano_smoothing_summary.json
```

Important fields:

```text
cam_t_smooth
joints_cam_smooth
vertices_cam_smooth
joints_3d_rel_smooth
vertices_rel_smooth
global_orient_smooth
hand_pose_smooth
betas_smooth
joints_uv_smooth_depth_camera
mano_smoothing_status
mano_smoothing_qc_flag
```

Current baseline result on `bag_20260622_1548_001`:

```text
candidates: 1379 after MotionInfiller
tracks: 3
hard_errors: 0
smooth_joints_finite_ratio: 1.0
smooth_vertices_finite_ratio: 1.0
qc after C1b + C1c input: 1377 ok, 1 warn_smooth_wrist_jump, 1 bad_smooth_wrist_jump
```

Note: all candidates currently trigger biomech clamp. This may mean the strict
joint limits are too aggressive for the raw WiLoR rotations, so this needs visual
review before treating clamp as a quality improvement.

Status: implemented.

## Stage C3: MANO Mesh Z-Buffer Visibility

Script:

```text
local_pipelines/egoinfinity_hand_pipeline/phase_c3_mano_mesh_visibility.py
```

Enabled by:

```bash
RUN_PHASE_C3=true
```

Purpose:

- Project the Phase-C2 smooth MANO mesh into the left-table camera frame.
- Rasterize a per-hand z-buffer.
- Mark front-surface-visible MANO vertices.
- Infer 21 joint visibility from nearby MANO vertices.
- Report visible wrist/MCP count for later depth re-alignment.

Outputs:

```text
stages/phase_c_mesh_visibility/wilor_handresults_phase_c3_mesh_visibility.npz
stages/phase_c_mesh_visibility/mano_mesh_visibility_joints.csv
stages/phase_c_mesh_visibility/mano_mesh_visibility_candidates.csv
stages/phase_c_mesh_visibility/mano_mesh_visibility_summary.json
```

Important fields:

```text
mano_vertex_visible
mano_vertex_surface_margin_m
mano_joint_visible
mano_joint_mesh_visible_ratio
mano_joint_mesh_surface_margin_m
mano_joint_mesh_surface_z_m
mano_visible_vertex_ratio
mano_visible_joint_count
mano_visible_reliable_joint_count
mano_visible_mcp_count
mano_visibility_qc_flag
```

Status: implemented as a visibility/audit/filter-output node. It feeds the
optional Stage C4 branch but does not change the main C2 output by itself.

## Stage C4: Visibility-Aware Depth Re-Alignment

Script:

```text
local_pipelines/egoinfinity_hand_pipeline/phase_c4_visibility_depth_realign.py
```

Enabled by:

```bash
RUN_VISIBILITY_REALIGN=true
```

Purpose:

- Experimental LFV-only stage; this is not directly from EgoInfinity.
- Use Stage C3 MANO mesh visibility to choose which joints may touch depth.
- Prefer visible wrist/MCP joints.
- Fall back to visible stable non-tip joints.
- Do not use all visible joints unless `VIS_REALIGN_ENABLE_ALL_VISIBLE_FALLBACK=true`.
- Reject noisy local depth patches using patch spread/MAD.
- Estimate a new camera translation from backprojected FoundationStereo samples.
- Keep previous `cam_t_smooth` when visible samples are insufficient or RMS is too high.

Important behavior:

- This stage is off by default.
- It does not overwrite `cam_t_smooth`.
- It writes `cam_t_visibility_depth` and related geometry as a separate branch.
- `VIS_REALIGN_KEEP_PREVIOUS_ON_BAD_RMS=true` by default because this stage can
  make the pipeline worse when visibility or depth is wrong.

Outputs:

```text
stages/phase_c_visibility_depth_realign/wilor_handresults_phase_c4_visibility_depth_realign.npz
stages/phase_c_visibility_depth_realign/visibility_depth_realign_quality.csv
stages/phase_c_visibility_depth_realign/visibility_depth_realign_summary.json
```

Important fields:

```text
cam_t_visibility_depth_candidate
cam_t_visibility_depth
joints_cam_visibility_depth
vertices_cam_visibility_depth
visibility_realign_source
visibility_realign_selected_joint_count
visibility_realign_selected_joint_ids
visibility_realign_sampled_depths_m
visibility_realign_rms_m
visibility_realign_delta_m
visibility_realign_candidate_delta_m
visibility_realign_rejected_bad_rms
visibility_realign_qc_flag
```

Current baseline result on `bag_20260622_1548_001`:

```text
candidates: 1379
hard_errors: 0
source: 544 visible_reliable, 381 visible_stable, 169 keep_previous_low_visible_depth
rejected_bad_rms_keep_previous: 285
kept_previous_total: 454
median_delta: 9.0 mm
p95_delta: 55.6 mm
qc: 712 ok, 289 bad flags, 213 warn flags
```

Status: implemented and QC-gated, but experimental. Do not feed this into
gripper mapping until visual review shows it helps.

## Node-Level QC

Script:

```text
local_pipelines/egoinfinity_hand_pipeline/phase_c_quality_gates.py
```

Output:

```text
quality_check/phase_c_node_quality_gates.json
```

Currently checks:

- Stage A processed topcam.
- Stage B Phase-B HandResult NPZ.
- Stage C0 FoundationStereo depth.
- Stage C0b independent depth stabilization when present.
- Stage C1 depth alignment.
- Stage C1b EgoInfinity depth smooth.
- Stage C1c EgoInfinity MotionInfiller when present.
- Stage C2 MANO smoothing.
- Stage C3 mesh visibility.
- Stage C4 visibility-aware depth re-align when present.

## Remaining Hand-Stack Work

- Visual review of C1c MotionInfiller around the one detected-row wrist jump.
- Decide whether Stage C0b should become default after comparing raw vs
  stabilized depth on several demos.
- Decide whether Stage C4 visibility-aware re-align helps enough to feed later
  gripper mapping.  It remains experimental and off by default.
