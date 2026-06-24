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
