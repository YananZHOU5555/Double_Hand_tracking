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
