# Hand Alignment Version

This folder is a fork of the current best `egoinfinity_hand_pipeline` state.

Purpose:

- Keep the existing `local_pipelines/egoinfinity_hand_pipeline` available for
  gripper-mapping work in another session.
- Use this folder for hand / MANO / depth alignment experiments only.
- Preserve the current best C3 behavior as the baseline before changing
  alignment logic.

## Output Isolation

This version writes to:

```text
<demo>/quality/egoinfinity_hand_alignment_pipeline/
```

The original pipeline writes to:

```text
<demo>/quality/egoinfinity_hand_pipeline/
```

That separation is intentional. Do not reuse the original output directory from
this alignment branch, otherwise the two debugging threads will overwrite each
other.

## Current Baseline

The best current skeleton-stability branch is:

```text
Phase-C3 MANO mesh visibility
+ WiLoR raw 2D locked projection for detected rows
+ joints_uv_smooth_depth_camera fallback for motion-infilled rows
```

Preferred debug artifacts:

```text
phase_c3_wilor2d_locked_depth_infill_projected_overlay.mp4
phase_c3_wilor2d_locked_depth_infill_projected_table_skeleton.html
```

The key fix in this baseline is that raw `joints_uv` is not trusted for
motion-infilled rows. Infilled rows have no real 2D observation and can otherwise
produce hand sizes of several thousand pixels.

## Next Alignment Work

The next changes in this folder should focus on:

- measuring 2D WiLoR overlay error against the real hand silhouette;
- measuring table-frame hand offset over time;
- separating global hand translation error from MANO pose/shape error;
- testing camera/table-depth alignment changes without changing gripper mapping;
- keeping visibility-aware re-align switchable because it can improve or worsen
  results depending on the frame.

## Entry Point

```bash
cd /home/yannan/workspace/learning-from-video
MAX_FRAMES=0 REBUILD_EGO_HAND=false RUN_PHASE_C=true \
RUN_PHASE_C_DEPTH_STABILIZE=true RUN_PHASE_C_DEPTH_SMOOTH=true \
RUN_MOTION_INFILL=true RUN_PHASE_C2=true RUN_PHASE_C3=true \
bash local_pipelines/egoinfinity_hand_alignment_pipeline/run_egoinfinity_hand_alignment_pipeline.sh \
  /home/yannan/workspace/ros1_docker-main/rosbag_data/human_teaching_videos/bag_20260622_1548_001
```
