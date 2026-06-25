# Hand Overlay Alignment Diagnostic

This file records the current diagnosis for the hand-skeleton offset in the
alignment fork.  The baseline demo used here is:

`/home/yannan/workspace/ros1_docker-main/rosbag_data/human_teaching_videos/bag_20260622_1548_001`

The diagnostic compares every stage against raw WiLoR image-space joints
(`joints_uv`) in the same candidate row.  In the contact sheets:

- magenta = raw WiLoR `joints_uv`
- green = stage UV being displayed or projected

## Diagnostic Outputs

Main comparison:

`/home/yannan/workspace/ros1_docker-main/rosbag_data/human_teaching_videos/bag_20260622_1548_001/quality/egoinfinity_hand_alignment_pipeline/quality_check/stage_overlay_alignment/stage_overlay_alignment_selected_frames.jpg`

Summary:

`/home/yannan/workspace/ros1_docker-main/rosbag_data/human_teaching_videos/bag_20260622_1548_001/quality/egoinfinity_hand_alignment_pipeline/quality_check/stage_overlay_alignment/stage_overlay_alignment_summary.json`

Right-hand focused comparison:

`/home/yannan/workspace/ros1_docker-main/rosbag_data/human_teaching_videos/bag_20260622_1548_001/quality/egoinfinity_hand_alignment_pipeline/quality_check/stage_overlay_alignment_right_focus/stage_overlay_alignment_selected_frames.jpg`

CSV with all per-candidate measurements:

`/home/yannan/workspace/ros1_docker-main/rosbag_data/human_teaching_videos/bag_20260622_1548_001/quality/egoinfinity_hand_alignment_pipeline/quality_check/stage_overlay_alignment/stage_overlay_alignment_all_candidates.csv`

## How The Current Skeleton Is Produced

Raw WiLoR:

- Source file: `stages/raw_wilor_handresults/wilor_handresults_raw.npz`
- Display field: `joints_uv`
- Meaning: WiLoR MANO prediction projected by WiLoR's own image-space camera.
- This is the tightest 2D overlay reference in the current pipeline.

Phase-B:

- Source file: `stages/phase_b_track_postprocess/wilor_handresults_phase_b.npz`
- Display field: `joints_uv`
- Meaning: track/label/dedup filtering only.  It should not change geometry.

Phase-C depth align:

- Source file: `stages/phase_c_depth_align/wilor_handresults_phase_c_depth_aligned.npz`
- Display field in this diagnostic: projection of `joints_cam_depth`
- Meaning: raw WiLoR `joints_uv` is used to sample FoundationStereo depth.
  Then each sampled 2D+depth point is backprojected into the left camera frame.
  The stage estimates a new metric translation:
  `cam_t_depth = median(backprojected_point - MANO_root_relative_joint)`.
- This is the first stage where the displayed skeleton is no longer WiLoR's
  original 2D overlay.

Phase-C1b depth smooth:

- Source file: `stages/phase_c_depth_smooth/wilor_handresults_phase_c1b_depth_smooth.npz`
- Display field in this diagnostic: projection of `joints_cam_depth_smooth`
- Meaning: EgoInfinity-style hand-Z smoothing.  The implementation changes the
  depth/Z component of `cam_t_depth`, `joints_cam_depth`, and
  `vertices_cam_depth`.  It does not recompute x/y from the original 2D
  observations after changing z.
- This can alter projected hand size and finger/MCP positions even if the wrist
  becomes more stable.

Phase-C1c motion infill:

- Source file: `stages/phase_c_motion_infiller/wilor_handresults_phase_c1c_motion_infilled.npz`
- Display field in this diagnostic: projection of `joints_cam_depth_smooth`
- Meaning: adds missing candidates.  For already detected rows, current numbers
  match C1b.

Phase-C2 MANO smooth:

- Source file: `stages/phase_c_mano_smooth/wilor_handresults_phase_c2_mano_smooth.npz`
- Display field: `joints_uv_smooth_depth_camera`
- Meaning: smooth MANO pose/global orientation/betas and run MANO forward again.
  The output is then projected with the FoundationStereo/left-table camera.
- C2 inherits the C1b depth-projection offset and adds only a small extra
  smoothing delta.

Phase-C2b bad-pose repair:

- Source file: `stages/phase_c2b_bad_pose_repair/wilor_handresults_phase_c2b_bad_pose_repaired.npz`
- Display field: `joints_uv_smooth_depth_camera`
- Meaning: optional repair for detected-but-bad MANO pose rows.  It flags
  orientation flips / large rotation jumps / large infilled wrist jumps, then
  interpolates MANO pose and camera translation from same-track trusted
  neighbors and re-runs MANO forward.  It is meant to fix local occlusion spans
  such as frame `625-629`, not the global Phase-C depth projection offset.

Phase-C3 mesh visibility:

- Source file: `stages/phase_c_mesh_visibility/wilor_handresults_phase_c3_mesh_visibility.npz`
- Display field: `joints_uv_smooth_depth_camera`
- Meaning: MANO mesh z-buffer visibility.  It adds `mano_joint_visible`; it does
  not move the skeleton.

Phase-C3 locked-fixed overlay:

- Source file: `stages/phase_c_mesh_visibility/wilor_handresults_phase_c3_mesh_visibility.npz`
- Display field: normally raw `joints_uv`; for motion-infilled or oversized rows,
  fallback to `joints_uv_smooth_depth_camera`.
- Meaning: this is a display mode, not a geometry fix.  It looks tighter because
  detected frames are locked to raw WiLoR 2D.

Phase-C4 visibility depth realign:

- Source file:
  `stages/phase_c_visibility_depth_realign/wilor_handresults_phase_c4_visibility_depth_realign.npz`
- Display field in this diagnostic: projection of `joints_cam_visibility_depth`
- Meaning: optional visibility-aware depth re-align.  It moves the skeleton only
  slightly relative to C3 in this run.

## Current Numeric Result

Median UV RMS against raw WiLoR `joints_uv`:

| Stage | Median RMS px | P90 RMS px | Median wrist px | Median MCP RMS px |
|---|---:|---:|---:|---:|
| Raw WiLoR | 0.000 | 0.000 | 0.000 | 0.000 |
| Phase-B | 0.000 | 0.000 | 0.000 | 0.000 |
| Phase-C depth project | 19.541 | 33.892 | 16.587 | 10.033 |
| Phase-C1b depth smooth project | 31.413 | 45.623 | 4.984 | 26.327 |
| Phase-C1c motion project | 31.413 | 45.623 | 4.984 | 26.327 |
| Phase-C2 MANO smooth project | 31.543 | 45.996 | 5.055 | 27.069 |
| Phase-C3 mesh visibility project | 31.543 | 45.996 | 5.055 | 27.069 |
| Phase-C3 locked-fixed display | 0.000 | 0.000 | 0.000 | 0.000 |
| Phase-C4 visibility realign project | 31.455 | 46.314 | 5.408 | 27.095 |

Stage-to-stage movement:

| Comparison | Median RMS px | P90 RMS px | Median wrist px | Median MCP RMS px |
|---|---:|---:|---:|---:|
| C1b -> C2 | 4.364 | 11.602 | 0.189 | 0.890 |
| C2 -> C3 | 0.000 | 0.000 | 0.000 | 0.000 |
| C3 -> C4 projected | 1.233 | 4.349 | 1.241 | 0.923 |

## Diagnosis

The major image-space offset is introduced before C2/C3.

The first visible offset appears at Phase-C, when raw WiLoR 2D is converted into
metric camera-frame MANO geometry using FoundationStereo depth.  The offset then
becomes larger at Phase-C1b depth smoothing.  C2 MANO smoothing only adds a small
extra movement, and C3 mesh visibility does not move the skeleton at all.

The current likely failure mode is:

1. WiLoR `joints_uv` is good as a 2D overlay.
2. Phase-C samples FoundationStereo depth at those 2D points and estimates a
   metric `cam_t_depth`.
3. C1b changes z/depth over time, but x/y remains tied to the previous metric
   translation.
4. Reprojecting that camera-frame geometry changes scale and shifts MCP/finger
   joints relative to the original image.

This explains why the "C3 locked-fixed" video looks much more aligned: it is
mostly displaying raw WiLoR 2D for detected frames instead of reprojecting the
depth-aligned/smoothed camera-frame hand.

## Next Alignment Debug Targets

1. Compare WiLoR's original camera model against the left-table/FoundationStereo
   camera model.  A focal/principal-point/crop mismatch will appear exactly as a
   projection offset after Phase-C.
2. Add a projection-preserving depth mode:
   keep raw `joints_uv` as the image-space constraint and use FoundationStereo
   only for z/depth.  After z smoothing, recompute x/y by backprojecting the
   desired raw UV with the smoothed z instead of keeping old x/y.
3. Separate display UV from geometry UV:
   keep raw WiLoR 2D for visual overlay on detected frames, but keep camera-frame
   geometry for 3D/table-frame analysis.  Do not let a good-looking locked 2D
   overlay hide a bad 3D geometry.
4. For C1b, test whether smoothing only the wrist/root z is enough, and whether
   applying the same z correction to every joint is causing MCP/finger drift.
5. Re-run the stage diagnostic after each change and compare this file's table.

## Phase-C Depth Project Diagnosis

Added a focused diagnostic:

`local_pipelines/egoinfinity_hand_alignment_pipeline/diagnose_phase_c_depth_project.py`

Outputs for the baseline run:

`/home/yannan/workspace/ros1_docker-main/rosbag_data/human_teaching_videos/bag_20260622_1548_001/quality/egoinfinity_hand_alignment_pipeline/quality_check/phase_c_depth_project_diagnostic/`

Important files:

- `phase_c_depth_project_diagnostic_summary.json`
- `phase_c_depth_project_diagnostic_all_candidates.csv`
- `phase_c_depth_project_selected_frames.jpg`
- `phase_c_depth_project_worst_frames.jpg`
- `phase_c_depth_project_2d_tz_fit.csv`

The diagnostic compares four variants:

| Variant | Meaning |
|---|---|
| raw WiLoR 2D | original `joints_uv`, used as visual reference |
| `wilor_foundation` | `joints_3d_rel + cam_t_wilor`, projected with the Foundation/left-table camera |
| `phase_c` | `joints_cam_depth`, projected with the Foundation/left-table camera |
| `xy_locked_depth` | keep Phase-C depth `z`, but solve x/y translation from raw `joints_uv` |

Current result:

| Metric | Median | P90 | Notes |
|---|---:|---:|---|
| `phase_c_rms_px` | 19.54 px | 33.89 px | current Phase-C offset |
| `wilor_foundation_rms_px` | 217.98 px | 300.71 px | WiLoR `cam_t` is not directly compatible with Foundation camera |
| `xy_locked_depth_rms_px` | 18.28 px | 29.42 px | solving x/y helps only a little |
| `phase_c_minus_xy_locked_rms_px` | 12.97 px | 27.31 px | x/y median translation contributes part of the error |
| `translation_tx_spread_m` | 0.0125 m | 0.0389 m | per-joint x translation estimates disagree |
| `translation_ty_spread_m` | 0.0072 m | 0.0176 m | per-joint y translation estimates disagree |
| `translation_tz_spread_m` | 0.0487 m | 0.0813 m | per-joint sampled depths disagree |
| `alignment_rms_m` | 0.0246 m | 0.0486 m | depth alignment residual across sampled joints |

The extra 2D-fit check searches the best camera `z` that makes the MANO relative
skeleton match raw `joints_uv` under the Foundation camera:

| Metric | Median | P10 | P90 |
|---|---:|---:|---:|
| Phase-C depth `z` | 0.327 m | 0.217 m | 0.447 m |
| best 2D-fit `z` | 0.397 m | 0.266 m | 0.657 m |
| `depth_z / best_2d_z` | 0.812 | 0.549 | 0.985 |

This means Phase-C usually places the MANO hand closer to the camera than the
raw 2D hand size would imply, so the reprojected hand becomes too large.  Some
bad frames also have inconsistent sampled depths across joints, for example
finger joints sampling object/table depth while wrist/MCP samples hand depth.

Current interpretation:

1. WiLoR's original `cam_t` is not in the same projection system as the
   Foundation/left-table camera, so it cannot be reused directly.
2. FoundationStereo depth gives a useful metric anchor, but per-joint depth
   samples are not always mutually consistent.
3. Phase-C estimates one median translation from those samples.  If sampled
   depths disagree, the median translation can be pulled away from the raw 2D
   hand.
4. Even when x/y is solved from raw 2D after fixing Phase-C `z`, the hand is
   still offset because the depth `z` is often too close for the raw 2D hand
   size.

The next likely fix is to make Phase-C projection-preserving:

- keep raw WiLoR `joints_uv` as a 2D constraint for detected frames;
- use FoundationStereo depth only to estimate/smooth metric `z`;
- after choosing `z`, recompute x/y from raw `joints_uv` instead of keeping the
  median x/y translation from depth samples;
- reject or downweight depth samples when the per-joint translation spread is
  large, especially large `translation_tz_spread_m`.

## Phase-C Mask-Gated Depth Diagnosis

Added a mask-gated depth diagnostic:

`local_pipelines/egoinfinity_hand_alignment_pipeline/diagnose_phase_c_mask_gated_depth.py`

Purpose:

- Keep WiLoR `joints_uv` as the 2D observation.
- Build a hand mask from the left-table frame and the WiLoR bbox/joints.
- Sample FoundationStereo depth only where the joint and local patch are inside
  the hand mask.
- Use only reliable joints `[0, 5, 9, 13, 17]` by default.
- Do not fall back to all 21 joints unless explicitly requested.

Full-video SAM run:

```bash
cd /home/yannan/workspace/learning-from-video
/home/yannan/workspace/learning-from-video/.venv-dinosam/bin/python \
  local_pipelines/egoinfinity_hand_alignment_pipeline/diagnose_phase_c_mask_gated_depth.py \
  --session-dir /home/yannan/workspace/ros1_docker-main/rosbag_data/human_teaching_videos/bag_20260622_1548_001 \
  --source-pipeline egoinfinity_hand_pipeline \
  --segmenter sam \
  --sam-device cuda \
  --scale 0.50
```

Outputs:

`/home/yannan/workspace/ros1_docker-main/rosbag_data/human_teaching_videos/bag_20260622_1548_001/quality/egoinfinity_hand_alignment_pipeline/quality_check/phase_c_mask_gated_depth_sam/`

Important files:

- `phase_c_mask_gated_depth_summary.json`
- `phase_c_mask_gated_depth_candidates.csv`
- `phase_c_mask_gated_selected_frames.jpg`
- `phase_c_mask_gated_worst_old.jpg`
- `phase_c_mask_gated_worst_new.jpg`

Full-video result:

| Metric | Current Phase-C | SAM mask-gated reliable joints |
|---|---:|---:|
| candidates | 1323 | 1323 |
| candidates with estimate | 1323 | 1285 |
| median overlay RMS | 19.54 px | 19.29 px |
| p90 overlay RMS | 33.89 px | 31.75 px |
| max overlay RMS | 80.56 px | 60.20 px |
| median alignment residual | 24.6 mm | 24.1 mm |
| median sampled-depth z spread | 48.7 mm | 47.7 mm |

Problem-frame range `590-630`:

| Metric | Current Phase-C | SAM mask-gated reliable joints |
|---|---:|---:|
| candidates | 53 | 53 |
| candidates with estimate | 53 | 34 |
| median overlay RMS | 39.78 px | 50.69 px on estimated subset |
| median alignment residual | high/mixed | 17.5 mm |
| median sampled-depth z spread | high/mixed | 26.4 mm |

Interpretation:

1. The hand mask gate correctly rejects weak fallback cases.  All 37 rows that
   previously used `depth_all_joints` become `mask_reliable_joints_insufficient`.
   This prevents low-confidence fingertip/table/object depth from entering the
   later trajectory.
2. The mask gate alone does not fix the systematic image-space offset.  For
   normal `depth_reliable_joints` rows, improvements and regressions are roughly
   balanced.
3. On the bad `590-630` interval, SAM often finds a hand/forearm mask, but many
   reliable joints still have `no_masked_depth`.  That means the dominant issue
   is not only table contamination; FoundationStereo depth around hand/object
   boundaries is missing or inconsistent, and the MANO scale/depth fit remains
   underconstrained.
4. This gate is still useful as a QC/rejection mechanism.  It should be added to
   Phase-C as an optional switch, but it should not be treated as the main
   alignment fix.

## Phase-C1a Silhouette XY + Anchor Depth Diagnosis

Added a projection-preserving alignment experiment:

`local_pipelines/egoinfinity_hand_alignment_pipeline/phase_c1a_silhouette_xy_align.py`

Purpose:

- Keep the WiLoR/MANO relative hand pose and shape fixed.
- Use segmentation-mask overlap to solve camera-frame X/Y translation.
- Use one robust anchor point plus local masked FoundationStereo depth to solve
  Z, instead of taking a componentwise median translation over several joints.
- Downweight the wrist-to-forearm side of the mask because the sleeve/forearm
  can dominate the projected silhouette area.

Focused run:

```bash
cd /home/yannan/workspace/learning-from-video
/home/yannan/workspace/learning-from-video/.venv-dinosam/bin/python \
  local_pipelines/egoinfinity_hand_alignment_pipeline/phase_c1a_silhouette_xy_align.py \
  --session-dir /home/yannan/workspace/ros1_docker-main/rosbag_data/human_teaching_videos/bag_20260622_1548_001 \
  --source-pipeline egoinfinity_hand_pipeline \
  --segmenter sam \
  --sam-device cuda \
  --process-frames 590-630 \
  --frames 590-630 \
  --xy-search-px 30 \
  --xy-step-px 8 \
  --xy-refine-px 8 \
  --xy-refine-step-px 2 \
  --scale 0.70
```

Outputs:

`/home/yannan/workspace/ros1_docker-main/rosbag_data/human_teaching_videos/bag_20260622_1548_001/quality/egoinfinity_hand_alignment_pipeline/quality_check/phase_c1a_silhouette_xy_align/`

Important files:

- `wilor_handresults_phase_c1a_silhouette_xy_aligned.npz`
- `phase_c1a_silhouette_xy_align_summary.json`
- `phase_c1a_silhouette_xy_align_candidates.csv`
- `phase_c1a_silhouette_selected_frames.jpg`
- `phase_c1a_silhouette_worst_new.jpg`
- `phase_c1a_silhouette_best_improve.jpg`
- `phase_c1a_silhouette_table_skeleton_590_630.html`
- `phase_c1a_silhouette_overlay_590_630.mp4`

Problem-frame range `590-630`:

| Metric | Current Phase-C | Silhouette XY + anchor depth |
|---|---:|---:|
| candidates | 53 | 53 |
| median overlay RMS | 39.78 px | 40.86 px |
| mean overlay RMS | 40.75 px | 37.82 px |
| p90 overlay RMS | 55.98 px | 44.24 px |
| max overlay RMS | 60.20 px | 53.02 px |
| median mask score | 0.722 | 0.854 |

Grouped observations:

- Left-hand track 0 improves overall: median RMS `44.75 -> 41.97 px`.
- Right-hand track 8 slightly worsens: median RMS `15.11 -> 18.14 px`.
- `middle_mcp` and `palm_center` anchors are usually better than `index_mcp`.
- The method reduces large failures, but can hurt frames where WiLoR already
  projected cleanly.

Interpretation:

This validates the idea that the main offset is caused by the depth-projection
translation stage, not MANO pose itself.  However, the current silhouette
objective is still too dependent on segmentation quality.  SAM frequently
includes forearm/sleeve and sometimes object-contact regions.  Therefore this
stage is a useful diagnostic and possible future alignment path, but should stay
off the main gripper-mapping pipeline until the hand-only mask and anchor policy
are stronger.

## Anchor-Locked Patch2 Depth Smooth

Implemented as:

```text
local_pipelines/egoinfinity_hand_alignment_pipeline/phase_c2_anchor_locked_depth_patch2.py
```

This is the strict version of the current alignment idea:

- Do not run silhouette search.
- Choose one raw WiLoR 2D anchor by the fixed priority
  `palm_center -> middle_mcp -> wrist -> index_mcp -> ring_mcp -> pinky_mcp`.
- Sample FoundationStereo depth at that anchor using a `2px` radius, `5x5`
  SAM-mask-gated patch.
- Move the C2 MANO hand so the same semantic MANO anchor lands on that raw
  2D anchor at the sampled depth.
- Smooth the selected anchor depth per `(hand_label, track_id)` and recompute
  camera-frame translation so the raw 2D anchor remains locked.
- Rows marked `motion_infilled=1` are now skipped by this anchor-lock stage.
  These rows do not have a trustworthy observed 2D anchor, so re-sampling
  FoundationStereo depth around their predicted landmarks can create very large
  jumps.
- Anchor-lock candidates with projected 2D skeleton RMS above `120 px` are
  rejected and kept at the incoming C2 geometry.

Output for `bag_20260622_1548_001`:

```text
/home/yannan/workspace/ros1_docker-main/rosbag_data/human_teaching_videos/bag_20260622_1548_001/quality/egoinfinity_hand_alignment_pipeline/stages/phase_c2_anchor_locked_depth_patch2/
```

Key files:

```text
wilor_handresults_phase_c2_anchor_locked_depth_patch2_smooth.npz
phase_c2_anchor_locked_depth_patch2_quality.csv
phase_c2_anchor_locked_depth_patch2_track_summary.csv
phase_c2_anchor_locked_depth_patch2_summary.json
phase_c2_anchor_locked_depth_patch2_overlay.mp4
phase_c2_anchor_locked_depth_patch2_contact.jpg
```

Current full-video result:

| Metric | Value |
|---|---:|
| candidates | 1379 |
| smooth ok | 1296 |
| temporal outlier | 20 |
| no valid anchor depth | 7 |
| motion-infilled skipped | 56 |
| orig RMS median | 31.97 px |
| anchor-locked smooth RMS median | 20.35 px |
| before wrist-Z jump median | 2.8 mm |
| after wrist-Z jump median | 1.4 mm |
| before wrist-Z jump p90 | 11.0 mm |
| after wrist-Z jump p90 | 6.6 mm |

The `625-629` bag-occlusion segment still has correct depth-lock execution but
does not become geometrically correct, because its dominant failure is MANO
orientation/pose under occlusion.  That segment should be handled by
MotionInfiller or explicit pose-window replacement next.

## Anchor-Locked C3/C4 Completion

After Stage C2a, the remaining visibility stages were run on the anchor-locked
geometry rather than the old C2 geometry.

Stage C3 command used the field selection:

```text
vertices_cam_anchor_locked_smooth
joints_cam_anchor_locked_smooth
joints_uv_anchor_locked_smooth
```

Output:

```text
/home/yannan/workspace/ros1_docker-main/rosbag_data/human_teaching_videos/bag_20260622_1548_001/quality/egoinfinity_hand_alignment_pipeline/stages/phase_c3_mesh_visibility_anchor_locked_patch2/
```

Result:

| Metric | Value |
|---|---:|
| candidates | 1379 |
| ok | 690 |
| visible joint count median | 11 |
| visible reliable joint count median | 2 |
| bad visibility flags | 588 |
| warn visibility flags | 289 |

Stage C4 then used:

```text
cam_t_anchor_locked_smooth
joints_3d_rel_smooth
vertices_rel_smooth
joints_uv_anchor_locked_smooth
```

Output:

```text
/home/yannan/workspace/ros1_docker-main/rosbag_data/human_teaching_videos/bag_20260622_1548_001/quality/egoinfinity_hand_alignment_pipeline/stages/phase_c4_visibility_depth_realign_anchor_locked_patch2/
```

Result:

| Metric | Value |
|---|---:|
| candidates | 1379 |
| ok QC | 922 |
| bad flags | 77 |
| warn flags | 193 |
| keep previous total | 263 |
| rejected bad RMS | 76 |
| median delta | 15.9 mm |
| p95 delta | 53.0 mm |

Interpretation:

The C3/C4 branch now runs to completion on the new anchor-locked geometry.  C4
still moves the hand by a nontrivial amount and should remain experimental until
its overlay/table-frame output is checked.  The safer current hand-position
baseline is still Stage C2a.

## Wrist Flip Occlusion Flag

Frame range `625-629` has a separate problem from depth projection: the right
hand enters the bag and WiLoR/MANO wrist orientation flips under severe
occlusion.

Implemented in:

`local_pipelines/egoinfinity_hand_alignment_pipeline/phase_c2_mano_temporal_smooth.py`

Signals:

- `mano_raw_global_rot_jump_deg`: frame-to-frame jump of raw WiLoR MANO
  `global_orient` inside one physical track.
- `mano_smooth_global_rot_jump_deg`: frame-to-frame jump after temporal
  smoothing/MANO forward.
- `mano_global_rot_delta_deg`: how much the smoothed global orientation differs
  from raw WiLoR for that candidate.
- `mano_orientation_flip_core`: hard occlusion/flip candidate.
- `mano_orientation_flip_neighbor`: same-track candidate within the configured
  neighbor frame window of a core flip.

Test result on `bag_20260622_1548_001` using the C1c MotionInfiller input:

| Frame | Hand | Result |
|---:|---|---|
| 625 | right | `bad_global_rot_delta`, core flip |
| 626 | right | `bad_raw_global_rot_jump`, core flip |
| 627 | right | neighbor warning |
| 628 | right | `warn_global_rot_delta` + neighbor warning |
| 629 | right | `bad_global_rot_delta` + `bad_raw_global_rot_jump`, core flip |

Boundary frames `624` and `630` are marked as neighbor warnings only.  This keeps
the exact flip frames bad while allowing downstream stages to decide whether to
drop or downweight the whole occlusion window.

## Re-run Command

```bash
cd /home/yannan/workspace/learning-from-video
/home/yannan/miniforge3/envs/wilor_lfv/bin/python \
  local_pipelines/egoinfinity_hand_alignment_pipeline/diagnose_stage_overlay_alignment.py \
  --session-dir /home/yannan/workspace/ros1_docker-main/rosbag_data/human_teaching_videos/bag_20260622_1548_001 \
  --source-pipeline egoinfinity_hand_pipeline \
  --frames 94,212,540,562,563,579,590,591,596,616,619,626 \
  --worst-stage phase_c2_mano_smooth_project \
  --worst-count 12 \
  --scale 0.50
```

Phase-C focused command:

```bash
cd /home/yannan/workspace/learning-from-video
/home/yannan/miniforge3/envs/wilor_lfv/bin/python \
  local_pipelines/egoinfinity_hand_alignment_pipeline/diagnose_phase_c_depth_project.py \
  --session-dir /home/yannan/workspace/ros1_docker-main/rosbag_data/human_teaching_videos/bag_20260622_1548_001 \
  --source-pipeline egoinfinity_hand_pipeline \
  --frames 94,212,540,562,563,579,590,591,596,616,619,626 \
  --worst-count 16 \
  --scale 0.50
```
