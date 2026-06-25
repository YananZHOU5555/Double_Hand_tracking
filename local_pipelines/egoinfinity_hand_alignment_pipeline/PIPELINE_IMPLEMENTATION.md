# EgoInfinity Hand Pipeline Implementation

This file records what is actually implemented in the local LFV
`egoinfinity_hand_alignment_pipeline`.  It is meant to be the debugging map for the
current pipeline, not a paper-level design document.

## Entry Point

```bash
cd /home/yannan/workspace/learning-from-video
RUN_PHASE_C=true RUN_PHASE_C_DEPTH_STABILIZE=true RUN_PHASE_C_DEPTH_SMOOTH=true \
RUN_MOTION_INFILL=true RUN_PHASE_C2=true RUN_PHASE_C3=true \
  bash local_pipelines/egoinfinity_hand_alignment_pipeline/run_egoinfinity_hand_alignment_pipeline.sh \
  /home/yannan/workspace/ros1_docker-main/rosbag_data/human_teaching_videos/bag_20260622_1548_001
```

Optional branches are deliberately off unless requested:

```bash
RUN_VISIBILITY_REALIGN=true
DEPTH_STABILIZE_USE_FLOW_MASK=true
```

Output root:

```text
<demo>/quality/egoinfinity_hand_alignment_pipeline/
```

Every runner invocation also writes stage timing:

```text
pipeline_timing.jsonl
pipeline_timing_summary.json
```

The JSONL file has one record per stage with `stage`, `status`, `duration_sec`,
and `exit_code`.  Status is one of `ok`, `reuse`, `skip`, or `failed`.  The
summary JSON aggregates the same records and is also written on failure through
the runner exit trap.

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
local_pipelines/egoinfinity_hand_alignment_pipeline/export_wilor_handresults.py
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
local_pipelines/egoinfinity_hand_alignment_pipeline/postprocess_handresults.py
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
local_pipelines/egoinfinity_hand_alignment_pipeline/quality_check_handresults.py
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
local_pipelines/egoinfinity_hand_alignment_pipeline/phase_c0b_depth_stabilize.py
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
local_pipelines/egoinfinity_hand_alignment_pipeline/phase_c_foundation_depth_align.py
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

## Stage C1-Diagnostic: Mask-Gated Depth Sampling

Script:

```text
local_pipelines/egoinfinity_hand_alignment_pipeline/diagnose_phase_c_mask_gated_depth.py
```

Purpose:

- Test whether Phase-C depth alignment becomes more stable if FoundationStereo
  depth is sampled only inside a hand segmentation mask.
- Keep WiLoR `joints_uv` unchanged.
- Use SAM, GrabCut, or bbox masks as a gate for depth sampling.
- By default, use only reliable joints `[0, 5, 9, 13, 17]` and refuse the
  all-joint fallback.

Example:

```bash
cd /home/yannan/workspace/learning-from-video
/home/yannan/workspace/learning-from-video/.venv-dinosam/bin/python \
  local_pipelines/egoinfinity_hand_alignment_pipeline/diagnose_phase_c_mask_gated_depth.py \
  --session-dir /home/yannan/workspace/ros1_docker-main/rosbag_data/human_teaching_videos/bag_20260622_1548_001 \
  --source-pipeline egoinfinity_hand_pipeline \
  --segmenter sam \
  --sam-device cuda
```

Outputs:

```text
quality_check/phase_c_mask_gated_depth_sam/phase_c_mask_gated_depth_summary.json
quality_check/phase_c_mask_gated_depth_sam/phase_c_mask_gated_depth_candidates.csv
quality_check/phase_c_mask_gated_depth_sam/phase_c_mask_gated_selected_frames.jpg
quality_check/phase_c_mask_gated_depth_sam/phase_c_mask_gated_worst_old.jpg
quality_check/phase_c_mask_gated_depth_sam/phase_c_mask_gated_worst_new.jpg
```

Current baseline result:

```text
full video, SAM:
  old Phase-C median RMS: 19.54 px
  mask-gated median RMS: 19.29 px
  old Phase-C p90 RMS: 33.89 px
  mask-gated p90 RMS: 31.75 px
  old max RMS: 80.56 px
  mask-gated max RMS: 60.20 px
  all 37 depth_all_joints fallback rows are rejected as insufficient reliable mask depth

problem interval 590-630, SAM:
  candidates: 53
  candidates with estimate: 34
  median alignment residual: 17.5 mm
  median z spread: 26.4 mm
  median overlay RMS on estimated subset: 50.69 px
```

Status: implemented as a diagnostic/QC branch only.  It is useful for rejecting
low-confidence depth fallback rows, but it does not solve the main projection
offset by itself.

## Stage C1a-Diagnostic: Silhouette XY + Anchor Depth

Script:

```text
local_pipelines/egoinfinity_hand_alignment_pipeline/phase_c1a_silhouette_xy_align.py
```

Purpose:

- Test a projection-preserving alternative to Stage C1 sparse multi-joint
  translation.
- Keep WiLoR/MANO pose, shape, and root-relative hand mesh fixed.
- Estimate camera-frame X/Y by searching for the MANO projected silhouette that
  overlaps the hand segmentation mask.
- Downweight the forearm side below the wrist so the optimizer is biased toward
  palm/finger overlap instead of sleeve/arm overlap.
- Estimate Z from one robust semantic anchor and a local masked FoundationStereo
  depth patch.  Candidate anchors are palm center, wrist, and MCP joints.
- Keep this as a diagnostic branch because mask quality can pull the hand away
  from the raw WiLoR 2D joints.

Outputs:

```text
quality_check/phase_c1a_silhouette_xy_align/wilor_handresults_phase_c1a_silhouette_xy_aligned.npz
quality_check/phase_c1a_silhouette_xy_align/phase_c1a_silhouette_xy_align_candidates.csv
quality_check/phase_c1a_silhouette_xy_align/phase_c1a_silhouette_xy_align_summary.json
quality_check/phase_c1a_silhouette_xy_align/phase_c1a_silhouette_selected_frames.jpg
quality_check/phase_c1a_silhouette_xy_align/phase_c1a_silhouette_worst_new.jpg
quality_check/phase_c1a_silhouette_xy_align/phase_c1a_silhouette_best_improve.jpg
```

Important fields:

```text
cam_t_silhouette
joints_cam_silhouette
vertices_cam_silhouette
joints_uv_silhouette
silhouette_status
silhouette_anchor_name
silhouette_mask_score
silhouette_mask_coverage
silhouette_mask_precision
silhouette_anchor_depth_m
```

Problem-interval test result on `bag_20260622_1548_001`, frames `590-630`,
SAM masks:

```text
candidates: 53
status: 29 ok, 24 ok_depth_refine_failed
old Phase-C median RMS: 39.78 px
silhouette median RMS: 40.86 px
old Phase-C p90 RMS: 55.98 px
silhouette p90 RMS: 44.24 px
old max RMS: 60.20 px
silhouette max RMS: 53.02 px
old mask score median: 0.722
silhouette mask score median: 0.854
```

Interpretation:

- This method reduces the worst projection outliers and improves silhouette
  overlap.
- It does not yet improve the median frame, and it slightly hurts some right-hand
  frames where the original WiLoR 2D projection was already good.
- `ok_depth_refine_failed` means the second anchor-depth sample after the XY
  move failed the local mask/depth consistency checks, so the stage kept the
  first anchor depth.
- The dominant remaining issue is segmentation quality: SAM often includes
  forearm/sleeve, so silhouette overlap can chase the wrong region even with
  wrist-below-forearm downweighting.

Status: implemented as an experimental diagnostic node only.  Do not make it a
main pipeline stage until visual review shows it improves both 2D overlay and
table-frame 3D skeleton.

## Stage C1b: EgoInfinity Depth Smooth

Script:

```text
local_pipelines/egoinfinity_hand_alignment_pipeline/phase_c1b_depth_smooth.py
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
local_pipelines/egoinfinity_hand_alignment_pipeline/phase_c1c_motion_infiller.py
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
local_pipelines/egoinfinity_hand_alignment_pipeline/phase_c2_mano_temporal_smooth.py
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
- Flag occlusion-related wrist orientation flips using raw frame-to-frame MANO
  global-orientation jumps and raw-to-smoothed global-orientation deltas.

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
mano_raw_global_rot_jump_deg
mano_smooth_global_rot_jump_deg
mano_global_rot_delta_deg
mano_orientation_flip_core
mano_orientation_flip_neighbor
```

Current baseline result on `bag_20260622_1548_001`:

```text
candidates: 1379 after MotionInfiller
tracks: 3
hard_errors: 0
smooth_joints_finite_ratio: 1.0
smooth_vertices_finite_ratio: 1.0
orientation flip test: 3 core candidates, 4 neighbor candidates
problem range: frame 625/626/629 right hand are core bad; frame 627/628 are
  in or near the occlusion interval; frame 624/630 are boundary neighbor warn.
qc after C1b + C1c input: mostly ok, with wrist-orientation flip warnings/bad
  flags isolated to a small number of candidates.
```

Note: all candidates currently trigger biomech clamp. This may mean the strict
joint limits are too aggressive for the raw WiLoR rotations, so this needs visual
review before treating clamp as a quality improvement.

Status: implemented.

## Stage C2b: Bad-Pose Repair

Script:

```text
local_pipelines/egoinfinity_hand_alignment_pipeline/phase_c2b_bad_pose_repair.py
```

Enabled by:

```bash
RUN_PHASE_C2B_REPAIR=true
```

Purpose:

- Repair detected-but-bad MANO pose frames after Phase-C2.
- Target short occlusion-induced failures such as the right hand around frames
  `625-629`, and isolated infiller jumps such as frame `579`.
- Use Phase-C2 diagnostics to flag suspicious rows:
  `mano_orientation_flip_core`, large global rotation delta, raw/smooth global
  rotation jumps, and large motion-infilled wrist jumps.
- Bridge tiny good gaps inside a bad span with `PHASE_C2B_BRIDGE_GOOD_GAP_FRAMES`.
- Interpolate `cam_t_smooth`, `global_orient_smooth`, `hand_pose_smooth`, and
  `betas_smooth` from the nearest same-track trusted neighbors.
- Require both trusted neighbors to be within `PHASE_C2B_NEIGHBOR_WINDOW_FRAMES`.
- Re-run MANO forward for repaired rows instead of moving the mesh approximately.

Outputs:

```text
stages/phase_c2b_bad_pose_repair/wilor_handresults_phase_c2b_bad_pose_repaired.npz
stages/phase_c2b_bad_pose_repair/bad_pose_repair_quality.csv
stages/phase_c2b_bad_pose_repair/bad_pose_repair_summary.json
```

Important fields:

```text
pose_repair_bad_pose
pose_repair_repaired
pose_repair_reason
pose_repair_method
pose_repair_prev_frame
pose_repair_next_frame
pose_repair_alpha
pose_repair_qc_flag
cam_t_smooth_before_pose_repair
joints_cam_smooth_before_pose_repair
vertices_cam_smooth_before_pose_repair
```

When enabled, Phase-C3 consumes this repaired NPZ instead of the raw Phase-C2
NPZ.  When disabled, the pipeline keeps the previous Phase-C2 -> Phase-C3 path.

## Stage C3: MANO Mesh Z-Buffer Visibility

Script:

```text
local_pipelines/egoinfinity_hand_alignment_pipeline/phase_c3_mano_mesh_visibility.py
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

Anchor-locked patch2 branch:

The same script can now be run on the Stage C2a geometry by selecting fields:

```bash
python local_pipelines/egoinfinity_hand_alignment_pipeline/phase_c3_mano_mesh_visibility.py \
  --input-npz stages/phase_c2_anchor_locked_depth_patch2/wilor_handresults_phase_c2_anchor_locked_depth_patch2_smooth.npz \
  --vertices-cam-field vertices_cam_anchor_locked_smooth \
  --joints-cam-field joints_cam_anchor_locked_smooth \
  --joints-uv-field joints_uv_anchor_locked_smooth \
  --output-dir stages/phase_c3_mesh_visibility_anchor_locked_patch2
```

Current anchor-locked result on `bag_20260622_1548_001`:

```text
candidates: 1379
ok: 690
bad_low_visible_reliable_joints: 400
bad_low_visible_reliable_joints + warn_low_visible_vertex_ratio: 186
warn_low_visible_vertex_ratio: 101
visible_joint_count median: 11
visible_reliable_joint_count median: 2
```

This is expected to be strict: low visible reliable count should be treated as a
filtering/QC signal, not a hard failure of the whole pipeline.

## Stage C2a: Anchor-Locked Patch2 Depth Smooth

Script:

```text
local_pipelines/egoinfinity_hand_alignment_pipeline/phase_c2_anchor_locked_depth_patch2.py
```

Purpose:

- Keep the Phase-C2 smoothed MANO pose/shape.
- Select one raw WiLoR 2D anchor per hand candidate.
- Sample FoundationStereo depth in a `2px` radius hand-mask-gated patch around
  that anchor.
- Translate the C2 MANO hand so the same semantic MANO anchor projects to the
  raw 2D anchor.
- Smooth the selected anchor depth per physical hand track while preserving the
  raw anchor 2D projection.
- Skip anchor-lock for rows with `motion_infilled=1`; those rows are already
  hallucinated by the motion infiller, so their 2D anchors are not trusted for
  a second depth re-alignment.
- Reject any anchor-lock result whose projected skeleton is more than `120 px`
  RMS away from raw WiLoR 2D landmarks.

Anchor policy:

```text
palm_center -> middle_mcp -> wrist -> index_mcp -> ring_mcp -> pinky_mcp
```

The first anchor whose `5x5` local depth patch has enough valid SAM-mask pixels
and acceptable depth span is used.  This stage intentionally skips silhouette
search; X/Y is controlled by the raw selected anchor, and Z is controlled by the
masked local FoundationStereo depth.

Outputs:

```text
stages/phase_c2_anchor_locked_depth_patch2/wilor_handresults_phase_c2_anchor_locked_depth_patch2_smooth.npz
stages/phase_c2_anchor_locked_depth_patch2/phase_c2_anchor_locked_depth_patch2_quality.csv
stages/phase_c2_anchor_locked_depth_patch2/phase_c2_anchor_locked_depth_patch2_track_summary.csv
stages/phase_c2_anchor_locked_depth_patch2/phase_c2_anchor_locked_depth_patch2_summary.json
stages/phase_c2_anchor_locked_depth_patch2/phase_c2_anchor_locked_depth_patch2_overlay.mp4
stages/phase_c2_anchor_locked_depth_patch2/phase_c2_anchor_locked_depth_patch2_contact.jpg
```

Important fields:

```text
cam_t_anchor_locked
joints_cam_anchor_locked
vertices_cam_anchor_locked
joints_uv_anchor_locked
cam_t_anchor_locked_smooth
joints_cam_anchor_locked_smooth
vertices_cam_anchor_locked_smooth
joints_uv_anchor_locked_smooth
anchor_lock_status
anchor_lock_qc_flag
anchor_lock_anchor_name
anchor_lock_anchor_ids
anchor_lock_anchor_uv
anchor_lock_anchor_depth_m
anchor_lock_anchor_depth_smooth_m
anchor_lock_smooth_delta_z_m
anchor_lock_trust
```

Current result on `bag_20260622_1548_001`:

```text
candidates: 1379
status: 1296 anchor_lock_smooth_ok, 20 temporal_outlier, 7 no_valid_anchor_depth, 56 motion_infilled_skip_anchor_lock
anchor selection: 1233 palm_center, 34 middle_mcp, 28 index_mcp, 21 wrist
orig RMS median: 31.97 px
anchor-locked smooth RMS median: 20.35 px
before wrist-Z jump median/p90: 2.8 mm / 11.0 mm
after wrist-Z jump median/p90: 1.4 mm / 6.6 mm
```

Interpretation:

- This fixes a large part of the 2D/depth projection offset.
- It reduces depth jitter, but it does not fix MANO orientation flips.
- The `563-618` right-hand missing segment remains a motion-infiller segment;
  Stage C2a now leaves that segment in the infiller geometry instead of
  re-projecting it from uncertain 2D anchors.
- The `625-629` right-hand bag-occlusion issue is a pose/orientation problem, not
  an anchor-depth problem.  It is handled by optional Stage C2b before C3.

## Stage C4: Visibility-Aware Depth Re-Alignment

Script:

```text
local_pipelines/egoinfinity_hand_alignment_pipeline/phase_c4_visibility_depth_realign.py
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

Anchor-locked patch2 branch result on `bag_20260622_1548_001`:

```text
input: stages/phase_c3_mesh_visibility_anchor_locked_patch2/wilor_handresults_phase_c3_mesh_visibility.npz
output: stages/phase_c4_visibility_depth_realign_anchor_locked_patch2/
candidates: 1379
source: 599 visible_reliable, 517 visible_stable, 187 keep_previous_low_visible_depth
bad_rms_keep_previous: 76
kept_previous_total: 263
median_delta: 15.9 mm
p95_delta: 53.0 mm
qc: 922 ok, 77 bad flags, 193 warn flags
```

This remains an experimental branch.  It should not replace the Stage C2a
anchor-locked output until its overlay/table-frame behavior is visually better
than the simpler anchor-lock result.

## Node-Level QC

Script:

```text
local_pipelines/egoinfinity_hand_alignment_pipeline/phase_c_quality_gates.py
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

## C3 Downstream: Gripper Mapping And IK

C3 is the current boundary between hand tracking/alignment and robot retargeting.
The downstream stack should consume the repaired C3 NPZ, not the experimental
C4 branch, unless explicitly testing C4.

Current downstream entry point:

```bash
cd /home/yannan/workspace/learning-from-video
bash local_pipelines/egoinfinity_hand_alignment_pipeline/run_alignment_gripper_mapping_ik_backends.sh \
  /home/yannan/workspace/ros1_docker-main/rosbag_data/human_teaching_videos/bag_20260622_1548_001
```

Backward-compatible old entry point:

```text
local_pipelines/egoinfinity_hand_alignment_pipeline/run_alignment_gripper_mapping_pinocchio_ik.sh
```

Default C3 input:

```text
stages/phase_c3_mesh_visibility_anchor_locked_patch2_pose_repaired/
  wilor_handresults_phase_c3_mesh_visibility.npz
```

Default C3 fields consumed:

```text
joints_cam_anchor_locked_smooth
joints_uv_anchor_locked_smooth
```

Overall downstream structure:

```text
C3 repaired hand tracking
  -> Phase-D EgoInfinity pre-IK
  -> Phase-E gripper-base core CSV
  -> IK backend A: Pinocchio DLS
  -> IK backend B: PyRoki
  -> MuJoCo / 3D debug / replay consumers
```

### Phase-D EgoInfinity Pre-IK

Script:

```text
local_pipelines/egoinfinity_hand_pipeline/phase_d_egoinfinity_preik/build_egoinfinity_preik.py
```

Purpose:

- Build dense left/right hand trajectories from C3 MANO joints.
- Construct wrist 6D pose and smooth it.
- Transform camera-frame hand targets to table frame.
- Export weak gripper/contact/grasp/push-tuck labels.

Output:

```text
stages/phase_d_preik_anchor_locked_patch2_pose_repaired/
  preik_targets.npz
  hand_features.npz
  phase_labels.csv
  phase_labels.html
```

### Phase-E Piper Gripper-Base CSV

Script:

```text
local_pipelines/egoinfinity_hand_pipeline/phase_e_piper_gripper_base_ik/export_phase_d_to_piper_core_csv.py
```

Purpose:

- Convert Phase-D wrist targets into the legacy Piper core CSV schema.
- Treat mapped wrist position as the robot `gripper_base` origin.
- Keep `TCP_OFFSET_XYZ=0,0,0` by default, so this is not the old pinch/TCP
  offset target.

Output:

```text
stages/phase_e_piper_gripper_base_ik_input_anchor_locked_patch2_pose_repaired/
  phase_d_right_gripper_base_core.csv
  quality_check/phase_d_right_preik_wrist6d_table_3d.html
  quality_check/phase_d_right_preik_wrist6d_robot_3d.html
```

This CSV is the shared interface for both IK backends.

### Phase-E Backend A: Pinocchio DLS

Enabled by default:

```bash
RUN_PINOCCHIO_IK=true
```

Script called inside `ros1_noetic`:

```text
/home/yannan/workspace/ros1_docker-main/workspaces/scripts/lfv_simulate_piper_core_gripper_pinocchio.py
```

Recommended current baseline:

```bash
IK_MODE=position ORIENTATION_SOURCE=none
```

Pose mode is wired but should be treated as orientation-debug until the human
wrist frame to Piper gripper-base frame mapping is stable:

```bash
IK_MODE=pose ORIENTATION_SOURCE=rot6d
```

### Phase-F Backend B: PyRoki

Enabled explicitly:

```bash
RUN_PYROKI_IK=true
```

Script:

```text
local_pipelines/egoinfinity_hand_pipeline/phase_f_pyroki_ik_backend/solve_phase_e_pyroki_ik.py
```

Setup:

```bash
local_pipelines/egoinfinity_hand_pipeline/phase_f_pyroki_ik_backend/install_pyroki_env.sh \
  /home/yannan/workspace/.venvs/pyroki
```

PyRoki consumes the same Phase-E CSV as Pinocchio.  It is currently an
independent comparison backend, not the default real-robot backend.

Run PyRoki only:

```bash
RUN_PINOCCHIO_IK=false RUN_PYROKI_IK=true \
IK_MODE=position ORIENTATION_SOURCE=none \
bash local_pipelines/egoinfinity_hand_alignment_pipeline/run_alignment_gripper_mapping_ik_backends.sh <session>
```

Short PyRoki smoke:

```bash
RUN_PINOCCHIO_IK=false RUN_PYROKI_IK=true \
PYROKI_MAX_FRAMES=5 PYROKI_STAGE_NAME=phase_f_pyroki_ik_smoke_5f_position_none \
IK_MODE=position ORIENTATION_SOURCE=none \
bash local_pipelines/egoinfinity_hand_alignment_pipeline/run_alignment_gripper_mapping_ik_backends.sh <session>
```

Run both backends on the same targets:

```bash
RUN_PINOCCHIO_IK=true RUN_PYROKI_IK=true \
IK_MODE=position ORIENTATION_SOURCE=none \
bash local_pipelines/egoinfinity_hand_alignment_pipeline/run_alignment_gripper_mapping_ik_backends.sh <session>
```

Status: integrated as a downstream backend switch.  The hand-stack C0-C3 runner
does not run IK directly; this separation is intentional so C3 results can be
frozen and compared across multiple IK backends.

Validated full PyRoki position-only run on `bag_20260622_1548_001`:

```text
output: stages/phase_f_pyroki_ik_right_gripper_base_absolute_s1_position_none
frames: 685 / 685 success
max_pos_error_m: 0.000906
phase_f_pyroki_ik time: 16.82s
```

Validated Pinocchio position-only runner check on the same Phase-E CSV:

```text
output: stages/phase_e_pinocchio_ik_runner_check_position_none_no_render
frames: 685 / 685 success
max_pos_error_m: 0.000100
phase_e_pinocchio_ik time: 0.92s
```
