# Double Hand Tracking

This repository is a code snapshot of the current LFV double-hand / WiLoR /
EgoInfinity-style hand-tracking experiments.

The current best version for hand-skeleton stability is the Phase-C3 branch in:

```text
local_pipelines/egoinfinity_hand_pipeline/
```

Specifically, the best visual/debug path is the C3 MANO mesh visibility output
with WiLoR 2D locked projection and corrected infill UV:

```text
phase_c3_wilor2d_locked_depth_infill_projected
```

That version keeps raw WiLoR 2D landmarks for real detections, but replaces
motion-infilled rows with `joints_uv_smooth_depth_camera` before drawing or
backprojecting. This avoids the failure where infilled hands become extremely
large because `joints_uv` has no real observation for those frames.

## Repository Contents

```text
local_pipelines/
  egoinfinity_hand_pipeline/   Current C3 hand-model pipeline.
  double_hand_pipeline/        Dual-hand / dual-arm lane wrapper.
  qzy_wilor_fallback/          Previous best single-arm WiLoR fallback pipeline.

docs/
  qzy_readme.md                Running notes and frozen LFV pipeline decisions.
```

Large generated outputs are intentionally not committed: rosbag files, mp4/html
visualizations, NPZ/NPY depth products, model checkpoints, and calibration data
stay in the local workspace.

## Expected Workspace

These scripts are still designed to run inside the existing LFV workspace:

```text
/home/yannan/workspace/learning-from-video
/home/yannan/workspace/ros1_docker-main
/home/yannan/workspace/external/FoundationStereo
/home/yannan/workspace/EgoInfinity
```

The code calls existing LFV helper scripts under `learning-from-video/scripts/`
and uses local model/checkpoint paths. This repository is therefore a versioned
snapshot of the working pipeline code, not yet a fully standalone package.

## Current C3 Run

Example full run on the baseline demo:

```bash
cd /home/yannan/workspace/learning-from-video

MAX_FRAMES=0 REBUILD_EGO_HAND=false RUN_PHASE_C=true \
RUN_PHASE_C_DEPTH_STABILIZE=true RUN_PHASE_C_DEPTH_SMOOTH=true \
RUN_MOTION_INFILL=true RUN_PHASE_C2=true RUN_PHASE_C3=true \
bash local_pipelines/egoinfinity_hand_pipeline/run_egoinfinity_hand_pipeline.sh \
  /home/yannan/workspace/ros1_docker-main/rosbag_data/human_teaching_videos/bag_20260622_1548_001
```

Generate the currently preferred locked C3 overlay:

```bash
cd /home/yannan/workspace/learning-from-video

/home/yannan/miniforge3/envs/wilor_lfv/bin/python \
  local_pipelines/egoinfinity_hand_pipeline/export_phase_c_video_overlay.py \
  --input-video /home/yannan/workspace/ros1_docker-main/rosbag_data/human_teaching_videos/bag_20260622_1548_001/processed_topcam/left_table.mp4 \
  --input-npz /home/yannan/workspace/ros1_docker-main/rosbag_data/human_teaching_videos/bag_20260622_1548_001/quality/egoinfinity_hand_pipeline/stages/phase_c_mesh_visibility/wilor_handresults_phase_c3_mesh_visibility.npz \
  --joints-key joints_cam_smooth \
  --uv-key joints_uv \
  --infilled-uv-key joints_uv_smooth_depth_camera \
  --output-video /home/yannan/workspace/ros1_docker-main/rosbag_data/human_teaching_videos/bag_20260622_1548_001/quality/egoinfinity_hand_pipeline/quality_check/video_overlay/phase_c3_wilor2d_locked_depth_infill_projected_overlay.mp4 \
  --draw-bbox
```

Generate the matching LFV table-frame 3D HTML:

```bash
cd /home/yannan/workspace/learning-from-video

python3 local_pipelines/egoinfinity_hand_pipeline/export_phase_c_table_skeleton_html.py \
  --input-npz /home/yannan/workspace/ros1_docker-main/rosbag_data/human_teaching_videos/bag_20260622_1548_001/quality/egoinfinity_hand_pipeline/stages/phase_c_mesh_visibility/wilor_handresults_phase_c3_mesh_visibility.npz \
  --table-frame-json /home/yannan/workspace/ros1_docker-main/data/lfv_calibration/table_frame_latest.json \
  --output-html /home/yannan/workspace/ros1_docker-main/rosbag_data/human_teaching_videos/bag_20260622_1548_001/quality/egoinfinity_hand_pipeline/quality_check/table_frame_hand_skeleton/phase_c3_wilor2d_locked_depth_infill_projected_table_skeleton.html \
  --video /home/yannan/workspace/ros1_docker-main/rosbag_data/human_teaching_videos/bag_20260622_1548_001/processed_topcam/left_table.mp4 \
  --stage-name phase_c3_wilor2d_locked_depth_infill_projected \
  --joints-key joints_cam_smooth \
  --uv-key joints_uv \
  --infilled-uv-key joints_uv_smooth_depth_camera \
  --reproject-from-uv-depth
```

## Notes

- Visibility-aware depth re-align is experimental and should stay switchable.
- Phase-C3 is currently preferred over C4 for skeleton stability.
- Motion infill is useful, but raw `joints_uv` must not be trusted for infilled
  rows.
- Dual-hand arm mapping is explicit. The pipeline does not yet solve high-level
  coordination logic between the two robot arms.
