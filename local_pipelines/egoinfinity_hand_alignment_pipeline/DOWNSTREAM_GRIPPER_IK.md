# Alignment Downstream Gripper Mapping + IK Backends

This branch consumes the repaired alignment C3 output and then reuses the
existing EgoInfinity-style Phase-D/Phase-E mapping.  Phase-E writes one shared
Piper gripper-base core CSV, then one or both IK backends can consume that same
CSV:

```text
C3 repaired hand tracking
  -> Phase-D EgoInfinity pre-IK
  -> Phase-E gripper-base core CSV
  -> IK backend A: Pinocchio DLS
  -> IK backend B: PyRoki
  -> MuJoCo / 3D debug / replay consumers
```

## Current Upstream Input

Default input:

```text
<session>/quality/egoinfinity_hand_alignment_pipeline/stages/
  phase_c3_mesh_visibility_anchor_locked_patch2_pose_repaired/
    wilor_handresults_phase_c3_mesh_visibility.npz
```

The downstream runner explicitly reads:

```text
joints_cam_anchor_locked_smooth
joints_uv_anchor_locked_smooth
```

Those are the repaired anchor-locked fields from the current upstream alignment
pipeline. This avoids silently falling back to the older `joints_cam_smooth`
trajectory.

## Run

```bash
cd /home/yannan/workspace/learning-from-video
bash local_pipelines/egoinfinity_hand_alignment_pipeline/run_alignment_gripper_mapping_ik_backends.sh \
  /home/yannan/workspace/ros1_docker-main/rosbag_data/human_teaching_videos/bag_20260622_1548_001
```

To stop before IK and inspect only gripper mapping / pre-IK 6D wrist pose:

```bash
RUN_PINOCCHIO_IK=false RUN_PYROKI_IK=false \
bash local_pipelines/egoinfinity_hand_alignment_pipeline/run_alignment_gripper_mapping_ik_backends.sh \
  /home/yannan/workspace/ros1_docker-main/rosbag_data/human_teaching_videos/bag_20260622_1548_001
```

The older script name still works for compatibility:

```text
local_pipelines/egoinfinity_hand_alignment_pipeline/run_alignment_gripper_mapping_pinocchio_ik.sh
```

## Outputs

Phase-D pre-IK targets:

```text
<session>/quality/egoinfinity_hand_alignment_pipeline/stages/
  phase_d_preik_anchor_locked_patch2_pose_repaired/
    preik_targets.npz
    hand_features.npz
    phase_labels.csv
```

Phase-E Piper gripper-base IK input:

```text
<session>/quality/egoinfinity_hand_alignment_pipeline/stages/
  phase_e_piper_gripper_base_ik_input_anchor_locked_patch2_pose_repaired/
    phase_d_right_gripper_base_core.csv
    quality_check/phase_d_right_preik_wrist6d_table_3d.html
    quality_check/phase_d_right_preik_wrist6d_robot_3d.html
```

Original Pinocchio IK output:

```text
<session>/quality/egoinfinity_hand_alignment_pipeline/stages/
  phase_e_pinocchio_ik_right_gripper_base_<mode>_s<scale>_<ik_mode>_<orientation_source>/
    piper_pinocchio_core_ik.csv
    piper_pinocchio_core_ik_metadata.json
    piper_pinocchio_core_ik_preview.png
    piper_urdf_core_sim.mp4
```

PyRoki IK output:

```text
<session>/quality/egoinfinity_hand_alignment_pipeline/stages/
  phase_f_pyroki_ik_right_gripper_base_<mode>_s<scale>_<ik_mode>_<orientation_source>/
    piper_pyroki_core_ik.csv
    piper_pyroki_core_ik_metadata.json
    piper_pyroki_core_ik_preview.png
```

## Target Semantics

The mapped wrist position is currently treated as the robot gripper-base origin,
not the pinch center and not the old TCP offset point. The runner sets:

```text
TCP_OFFSET_XYZ=0,0,0
```

If a later stage wants pinch-center or task contact-point retargeting, that
should be added as a separate explicit transform after this baseline is stable.

## Useful Switches

The default IK run is position-only. This is the stable baseline for checking
whether the upstream gripper-base positions are reachable:

```bash
IK_MODE=position ORIENTATION_SOURCE=none bash .../run_alignment_gripper_mapping_ik_backends.sh <session>
```

The alignment branch default is `TARGET_MODE=absolute SCALE=1.0`. The older
`anchored_delta SCALE=0.25` mode was only a temporary visual sanity check for
unaligned upstream trajectories.

For the current right-arm runs, use:

```text
/home/yannan/workspace/ros1_docker-main/data/lfv_calibration/right_arm_table_latest.json
```

Do not use `right_arm_table_axis_projected_latest.json` as the default for this
alignment branch; in the current table frame it places the robot base near the
left-arm side and makes right-arm absolute IK appear incorrectly unreachable.

Pose IK using the exported wrist `rot6d` should be treated as an explicit
orientation-debug run until the human wrist frame is aligned to the Piper
gripper-base frame:

```bash
IK_MODE=pose ORIENTATION_SOURCE=rot6d bash .../run_alignment_gripper_mapping_ik_backends.sh <session>
```

Pose IK with a gripper-frame rotation adjustment:

```bash
IK_MODE=pose ORIENTATION_SOURCE=rot6d ORIENTATION_ALIGN_RPY=0,0,1.570796 \
  bash .../run_alignment_gripper_mapping_ik_backends.sh <session>
```

Disable IK render for faster numeric output:

```bash
RENDER_STRIDE=999999 bash .../run_alignment_gripper_mapping_ik_backends.sh <session>
```

The default IK path enables the existing DLS stability options in the original
Pinocchio module:

```text
NORMAL_LIMIT_MARGIN_RAD=0.035
NORMAL_SMOOTH_WEIGHT=0.01
NORMAL_LIMIT_WEIGHT=0.02
FAILED_FRAME_STRATEGY=interpolate
RELAXED_IK_FALLBACK=tcp_priority
```

Run PyRoki only on the same Phase-E CSV:

```bash
RUN_PINOCCHIO_IK=false RUN_PYROKI_IK=true \
IK_MODE=position ORIENTATION_SOURCE=none \
bash local_pipelines/egoinfinity_hand_alignment_pipeline/run_alignment_gripper_mapping_ik_backends.sh \
  /home/yannan/workspace/ros1_docker-main/rosbag_data/human_teaching_videos/bag_20260622_1548_001
```

Fast PyRoki smoke on a short slice:

```bash
RUN_PINOCCHIO_IK=false RUN_PYROKI_IK=true \
PYROKI_MAX_FRAMES=5 PYROKI_STAGE_NAME=phase_f_pyroki_ik_smoke_5f_position_none \
IK_MODE=position ORIENTATION_SOURCE=none \
bash local_pipelines/egoinfinity_hand_alignment_pipeline/run_alignment_gripper_mapping_ik_backends.sh <session>
```

Run both backends for a direct comparison:

```bash
RUN_PINOCCHIO_IK=true RUN_PYROKI_IK=true \
IK_MODE=position ORIENTATION_SOURCE=none \
bash local_pipelines/egoinfinity_hand_alignment_pipeline/run_alignment_gripper_mapping_ik_backends.sh <session>
```

Current validated PyRoki position-only full run on
`bag_20260622_1548_001`:

```text
output:
  /home/yannan/workspace/ros1_docker-main/rosbag_data/human_teaching_videos/bag_20260622_1548_001/quality/egoinfinity_hand_alignment_pipeline/stages/phase_f_pyroki_ik_right_gripper_base_absolute_s1_position_none
frames: 685 / 685 success
max_pos_error_m: 0.000906
target_mode: absolute
scale: 1.0
ik_mode: position
orientation_source: none
target_link: gripper_base
```

Timing for that run:

```text
phase_d_egoinfinity_preik: 1.46s
phase_e_piper_gripper_base_csv: 0.11s
phase_f_pyroki_ik: 16.82s
```

Current validated Pinocchio position-only no-render runner check on the same
Phase-E CSV:

```text
output:
  /home/yannan/workspace/ros1_docker-main/rosbag_data/human_teaching_videos/bag_20260622_1548_001/quality/egoinfinity_hand_alignment_pipeline/stages/phase_e_pinocchio_ik_runner_check_position_none_no_render
frames: 685 / 685 success
max_pos_error_m: 0.000100
target_mode: absolute
scale: 1.0
ik_mode: position
orientation_source: none
```

## Current Axis-Perm Pose IK Trajectory Audit

Audited file:

```text
/home/yannan/workspace/ros1_docker-main/rosbag_data/human_teaching_videos/bag_20260622_1548_001/quality/egoinfinity_hand_alignment_pipeline/stages/phase_e_pinocchio_ik_right_gripper_base_absolute_s1_pose_rot6d_axisperm_hz_hx_hy/piper_pinocchio_core_ik.csv
```

This trajectory is part of the current C3 downstream pipeline.  The verified
chain is:

```text
phase_c3_mesh_visibility_anchor_locked_patch2_pose_repaired/
  wilor_handresults_phase_c3_mesh_visibility.npz
  keys:
    joints_cam_anchor_locked_smooth
    joints_uv_anchor_locked_smooth
  input:
    phase_c2_anchor_locked_depth_patch2_pose_repaired/

-> phase_d_preik_anchor_locked_patch2_pose_repaired/
     preik_targets.npz
     metrics.json source_npz = C3 repaired NPZ

-> phase_e_piper_gripper_base_ik_input_anchor_locked_patch2_pose_repaired/
     phase_d_right_gripper_base_core.csv
     metadata source_preik_npz = Phase-D preik_targets.npz

-> phase_e_pinocchio_ik_right_gripper_base_absolute_s1_pose_rot6d_axisperm_hz_hx_hy/
     piper_pinocchio_core_ik.csv
     metadata gripper_core_csv = Phase-E right gripper-base core CSV
```

IK settings from metadata:

```text
target_mode: absolute
scale: 1.0
ik_mode: pose
orientation_source: rot6d
orientation_align_rpy: 0,-1.57079632679,-1.57079632679
robot_table_json: right_arm_table_latest.json
normal_success_count: 286 / 685
relaxed fallback used: 399 / 685
success_count: 685 / 685
max_pos_error_m: 0.002857
```

The Phase-E table-frame target positions were re-transformed with
`right_arm_table_latest.json`; the transformed positions match the IK CSV
`raw_abs_*` / `target_*` columns to numerical precision.
