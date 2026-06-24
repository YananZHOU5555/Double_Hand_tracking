# QZY WiLoR Fallback Pipeline

This folder wraps the locally iterated LFV pipeline that was last used as the
best non-MPS branch:

```text
processed_topcam
  -> MediaPipe stereo hand21
  -> V0 active-trim gripper state
  -> WiLoR predictions
  -> WiLoR V2 stereo-constrained hand
  -> WiLoR TCP core
  -> index-middle MCP + thumb-index YZ tilt orientation core
  -> V0 state + WiLoR TCP/orientation fused table
  -> open-segment IK regularizer
  -> Piper pose rot6d IK with tcp_priority relaxed fallback
  -> MuJoCo HTML replay
```

It intentionally does not use the V2.2 MPS-like / FoundationStereo branch.

Example:

```bash
cd /home/yannan/workspace/learning-from-video
bash local_pipelines/qzy_wilor_fallback/run_qzy_wilor_fallback_pipeline.sh \
  /home/yannan/workspace/ros1_docker-main/rosbag_data/LFV_demos_for_new_calib/soap_grasp_001
```

Key defaults:

- `TRIM_FRAMES=20`
- `THUMB_YZ_TILT_GAIN=-1.0`
- `ORIENTATION_ALIGN_RPY=0,1.570796326795,1.570796326795`
- `ROBOT_TABLE_JSON=/workspace/ros1_docker_jinhe/data/lfv_calibration/right_arm_table_latest.json`
- `TCP_PIVOT_JSON=/home/yannan/workspace/ros1_docker-main/data/lfv_calibration/right_arm_touch_tcp_xr2_latest.json`
- `RELAXED_IK_FALLBACK=tcp_priority`
- `SAVE_HAND21_OVERLAY=true`
- `SAVE_WILOR_OVERLAY=true`

The final terminal summary prints absolute paths for the main debug outputs:
MediaPipe left/right tracking overlays, MediaPipe hand21 core overlay, WiLoR
overlay, hand detection timeline/statistics, constrained hand HTML, 6D pose
HTML, regularized trajectory HTML, IK input/output fallback HTML, and MuJoCo
replay HTML.
