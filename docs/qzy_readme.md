# 回0脚本

cd /home/yannan/workspace/learning-from-video
export LFV_ALLOW_ROBOT_MOTION=1

bash scripts/run_right_arm_slow_mit_home_zero.sh

---

# 当前 LFV Pipeline 冻结说明

这部分记录当前最可信的 LFV pipeline 状态。核心原则是：**轨迹/TCP 和姿态/orientation 分开维护**。不要把 hand tracking、TCP position、orientation、gripper open/close 四条线混在一起判断。

## 1. 当前冻结结论

- **轨迹 / TCP position 当前优先用 WiLoR**：`soap_grasp_high_4` 里 WiLoR V2 model-only active-trim20 的 TCP 轨迹目视和数值都比 MediaPipe V0 active-trim20 更稳。
- **姿态 / orientation 当前继续用 MediaPipe/stereo 观测 + 手模约束链路**：不要用 V0 脚本在 WiLoR 21 点上重新算 `mcp_3d` 当最终姿态；当前更可信的是手模约束后输出的 SO(3)。最新实验里，手局部坐标系的 X 轴优先试 `index_mcp -> middle_mcp`，比原来的 `thumb_mcp -> index_mcp` 更稳定。
- 当前最合理拆分：
  - `TCP position / trajectory`: WiLoR V2 MCP-X smoothed model trajectory。
  - `orientation`: MediaPipe/stereo 观测约束手模后的 SO(3)，当前候选优先 `index_middle_mcp` 手局部轴。
  - `gripper open/close`: 沿用 V0 pinch 二分类。

## 2. 输入层

标准输入来自固定顶视双目：

```text
top stereo videos / table frame calibration
  -> MediaPipe stereo hand21
  -> table-frame right_hand_21_landmarks_table.csv
  -> hand21_quality_report.json
```

`hand21_quality_report.json` 里两个 segment 的含义：

- `detected_active_segment`: 双目都检测到手的最长连续段，允许短 gap。
- `core_interaction_segment`: 在 detected active segment 基础上，去掉首尾边界/手太小等不稳定帧后的核心段。

以 `soap_grasp_high_4` 为例：

```text
detected_active_segment = 115..632
core_interaction_segment = 125..613
```

## 3. Hand Tracking

当前 hand tracking 有两条用途不同的输入：

```text
raw stereo MediaPipe hand21
  -> stereo triangulation
  -> table-frame 21 keypoints

WiLoR V2 constrained hand21
  -> hand-model constrained 21 keypoints
  -> table-frame 21 keypoints / model trajectory
```

用途：

- MediaPipe/stereo hand21 给手模约束和 QA 提供观测基础。
- WiLoR V2 constrained hand21 当前更适合作为 TCP trajectory 的来源。
- 给 QA 提供 hand detection segment。
- 不要只看 V2.0 gate 后点数来判断 raw hand detection 是否丢点。

## 4. Pose / Orientation Pipeline（当前冻结）

当前冻结的姿态链路：

```text
MediaPipe/stereo hand21 observation
  -> WiLoR / hand-model constrained fitting
  -> trim20 / scale guard
  -> MCP-row / wrist / finger MCP geometry
  -> SO(3) local smoothing + angular velocity limiting
  -> rot6d
```

冻结原则：

- 姿态使用手模约束后的 SO(3)，不是 raw 21 点直接重算出的 V0 `mcp_3d`。
- 原始 baseline 是 `thumb_mcp -> index_mcp` 定义 hand local X。
- 当前更推荐继续试 `index_mcp -> middle_mcp` 定义 hand local X；TCP 中心和夹爪开合状态仍然使用 `thumb_tip / index_tip`，只替换 orientation frame 的横向轴。
- 不使用 V0 `mcp_3d` 在 WiLoR 21 点上重新算出来的姿态作为冻结姿态。
- 不做多 demo orientation 平均。
- 坏姿态帧只允许 anchor 内部短 gap 插值或 hold。
- 长 gap 不用其他 demo 补姿态，也不编造姿态。

当前相关目录：

```text
/home/yannan/workspace/ros1_docker-main/rosbag_data/lfv_demos/soap_grasp_high_4/quality/wilor_v2_constrained_trim20_scale_guard_max1p35/
/home/yannan/workspace/ros1_docker-main/rosbag_data/lfv_demos/soap_grasp_high_4/quality/wilor_v2_mcp_x_smooth_core/
/home/yannan/workspace/ros1_docker-main/rosbag_data/lfv_demos/soap_grasp_high_4/quality/wilor_v2_index_middle_mcp_x_smooth_core/
```

`index_middle_mcp` 版本的手模姿态 HTML：

```text
/home/yannan/workspace/ros1_docker-main/rosbag_data/lfv_demos/soap_grasp_high_4/quality/wilor_v2_index_middle_mcp_x_smooth_core/wilor_v2_index_middle_mcp_x_smoothed_core_6d_pose_over_frames.html
```

thumb-index vs index-middle 当前对比：

```text
old thumb_index raw rot step:
  p95 = 6.528 deg
  max = 178.048 deg
  angular speed limiter adjusted = 30 frames
  max correction = 2.785 rad

new index_middle raw rot step:
  p95 = 6.304 deg
  max = 14.248 deg
  angular speed limiter adjusted = 29 frames
  max correction = 0.103 rad

smoothed rot step:
  old p95/max = 5.497 / 6.875 deg
  new p95/max = 5.495 / 6.875 deg
```

结论：`index_mcp -> middle_mcp` 不会直接解决 Piper full 6D IK，但它明显去掉了 thumb-index 版本里的 raw orientation 大翻转，是当前更干净的手模姿态输入。

## 5. TCP Position / Trajectory Pipeline（当前优先 WiLoR）

当前 TCP trajectory 优先看 WiLoR V2 model-only active-trim20：

```text
WiLoR V2 constrained hand
  -> MCP-X HumanEgo-style TCP
  -> xyz triangular smoothing
  -> active detected segment trim20
```

`soap_grasp_high_4` 对比结果：

```text
active trim20: 135..612

MediaPipe V0 active-trim20 TCP:
  step p95 = 6.37 mm
  step max = 11.04 mm

WiLoR V2 model-only active-trim20 TCP:
  step p95 = 5.72 mm
  step max = 9.33 mm
```

当前推荐看的 WiLoR 轨迹文件：

```text
/home/yannan/workspace/ros1_docker-main/rosbag_data/lfv_demos/soap_grasp_high_4/quality/wilor_v2_model_only_active_trim20/
```

注意：轨迹用 WiLoR 更好，不代表姿态要改成 V0 的 WiLoR-21点 `mcp_3d` 重算姿态。姿态仍按上一节的手模约束 SO(3) 链路维护。

## 6. Gripper Open / Close

夹爪开合继续沿用 V0 逻辑，不是训练出来的分类器：

```text
thumb_tip / index_tip pinch distance
  -> hysteresis open/closed
  -> centered vote
  -> closed gap bridge
  -> grasp_binary
  -> piper_joint7_m / piper_joint8_m
```

默认判断逻辑：

- pinch 小于 close 阈值 -> `closed`
- pinch 大于 open 阈值 -> `open`
- 中间区域保持上一帧状态

## 7. Open Segment IK-friendly Regularizer（第一版已实现）

这个模块是 **IK solver 的输入预处理**，放在 gripper 6D pose/action table 出来之后、任何 IK 求解之前：

```text
WiLoR TCP trajectory
  + MediaPipe/stereo observation + hand-model orientation
  + V0 gripper open/close state
  -> raw table-frame gripper pose/action table
  -> open-segment IK-friendly regularizer
  -> IK-ready table-frame gripper pose/action table
  -> table-to-robot calibration / TCP offset compensation
  -> Piper IK
  -> dense replay plan
```

设计原则：

- `closed` 状态表示夹爪已经和物品交互，这段是真正需要忠实保留的动作。
- `open` 状态下没有稳定物体交互，移动过程可以更强地做平滑、直线化、降抖和 IK 可达性约束。
- 不覆盖 raw prediction，始终另存一份 IK-ready CSV，方便对比是预测问题、regularizer 问题还是 IK 问题。
- 这个模块不放在 IK 之后；IK 只消费 regularized 后的 table-frame target。

处理方式：

```text
raw gripper pose/action table
  -> 根据 state/grasp_binary 找连续 open / closed segments
  -> closed segments: 原始 TCP + orientation 尽量保持不变
  -> open segments: 用 IK-friendly 轨迹重写
```

open segment 的默认策略：

- `xyz`: 在相邻 protected anchors 之间做直线或 minimum-jerk 插值。
- `orientation`: 保持邻近闭合段姿态，或在两个 anchor 之间做 SO(3) 平滑插值。
- `gripper`: open segment 内保持 open command，状态切换点附近保留短 guard window。
- `velocity / acceleration`: 对 open segment 使用更严格的步长、速度、角速度限制，优先帮助 IK 连续求解。

segment anchor 规则：

- 第一次闭合前的 open 段：从起点平滑到第一次稳定闭合帧。
- 两段闭合之间的 open 段：从上一段释放点平滑到下一段闭合点。
- 最后一次打开后的 open 段：从释放点平滑到结束点或安全 retreat 点。

`soap_grasp_high_4` 当前状态切换参考：

```text
V0 gripper state:
  first stable closed: frame 219, t = 3.648s
  closed -> open: frame 488, t = 8.132s

WiLoR model-only state:
  first stable closed: frame 207, t = 3.448s
  closed -> open: frame 484, t = 8.064s
```

当前冻结策略里 gripper open/close 沿用 V0，所以 open segment regularizer 的分段边界优先使用最终融合后的 V0 state，而不是 WiLoR 文件自己的 state。

当前第一版实现：

```text
/home/yannan/workspace/learning-from-video/scripts/lfv_regularize_open_segments_for_ik.py
```

`soap_grasp_high_4` 第一版输出：

```text
/home/yannan/workspace/ros1_docker-main/rosbag_data/lfv_demos/soap_grasp_high_4/quality/wilor_tcp_v0_state_open_regularized_ik/
```

输出文件：

```text
gripper_pose_table_core_wilor_tcp_v0_state_open_regularized_ik_raw_fused.csv
gripper_pose_table_core_wilor_tcp_v0_state_open_regularized_ik.csv
wilor_tcp_v0_state_open_regularized_ik_summary.json
wilor_tcp_v0_state_open_regularized_ik_compare_3d.html
wilor_tcp_v0_state_open_regularized_ik_regularized_3d.html
```

这版使用：

```text
TCP: WiLoR V2 model-only active-trim20
orientation: v0_active_trim20_tcp_model_orientation 里的手模 rot6d
state/open_width/piper_joint7/8: V0 gripper state
```

`soap_grasp_high_4` 第一版统计：

```text
rows = 478
guard_frames = 8
closed_frames = 269
protected_frames = 285
regularized_open_frames = 193

raw step p95 = 5.72 mm
regularized step p95 = 4.22 mm

raw step max = 9.33 mm
regularized step max = 8.81 mm
```

IK 测试结果：

```text
position IK + orientation_source=none:
  status = pass
  success = 478 / 478
  max_pos_error = 0.10 mm
  dense replay samples = 985
  dense replay status = pass

pose IK + orientation_source=rot6d:
  status = warning
  success = 0 / 478
  max_pos_error = 797.39 mm
```

当前结论：regularized 版本可以作为 **position IK** 输入；full 6D `pose IK + rot6d` 仍不可直接用于当前 Piper 求解。

IK 输出目录：

```text
/home/yannan/workspace/ros1_docker-main/rosbag_data/lfv_demos/soap_grasp_high_4/quality/urdf_sim/piper_pinocchio_core_absolute_scale_0p25_core_gripper_pose_table_core_wilor_tcp_v0_state_open_regularized_ik/
```

Pose IK 姿态对齐/offset 诊断：

```text
未做 hand->gripper 轴对齐:
  pose IK + rot6d success = 0 / 478

只做 hand->gripper 轴对齐:
  hand X -> gripper_base Y
  hand Y -> gripper_base Z
  hand Z -> gripper_base X
  success = 292 / 478

在轴对齐后额外加 local gripper Z offset:
  coarse sweep: -120, -90, -60, -30, 0, +30, +60, +90, +120 deg
  fine sweep around -85 deg
  best current: local Z offset = -83 deg
  success = 305 / 478
```

当前 offset sweep 结论：

- local Z offset 有改善，但只是小幅改善，不能彻底解决 full 6D pose IK。
- `-83 deg` 比不加 offset 多解出 13 帧。
- 失败仍集中在 wrist：`joint5`、`joint4`、`joint6` 贴上限。
- `joint2/joint3` 在轴对齐后不再是主要问题。

offset sweep 输出：

```text
/home/yannan/workspace/ros1_docker-main/rosbag_data/lfv_demos/soap_grasp_high_4/quality/urdf_sim/pose_rot6d_local_z_offset_sweep/
/home/yannan/workspace/ros1_docker-main/rosbag_data/lfv_demos/soap_grasp_high_4/quality/urdf_sim/piper_pose_rot6d_axisalign_local_z_m83deg/
```

`local Z = -83 deg` MuJoCo:

```text
/home/yannan/workspace/ros1_docker-main/rosbag_data/lfv_demos/soap_grasp_high_4/quality/urdf_sim/piper_pose_rot6d_axisalign_local_z_m83deg/mujoco_replay/piper_pose_ik_mujoco_replay.html
/home/yannan/workspace/ros1_docker-main/rosbag_data/lfv_demos/soap_grasp_high_4/quality/urdf_sim/piper_pose_rot6d_axisalign_local_z_m83deg/mujoco_replay/piper_pose_ik_local_z_m83_contact_sheet.png
```

## 8. 当前关键本地文件

V0 HumanEgo-style TCP / gripper action：

```text
/home/yannan/workspace/ros1_docker-main/rosbag_data/lfv_demos/soap_grasp_high_4/quality/humanego_gripper_action/gripper_pose_table_core_humanego_mit.csv
```

当前 WiLoR trajectory 版本：

```text
/home/yannan/workspace/ros1_docker-main/rosbag_data/lfv_demos/soap_grasp_high_4/quality/wilor_v2_model_only_active_trim20/gripper_pose_table_core_wilor_v2_model_only_active_trim20.csv
/home/yannan/workspace/ros1_docker-main/rosbag_data/lfv_demos/soap_grasp_high_4/quality/wilor_v2_model_only_active_trim20/wilor_v2_model_only_active_trim20_full.html
/home/yannan/workspace/ros1_docker-main/rosbag_data/lfv_demos/soap_grasp_high_4/quality/wilor_v2_model_only_active_trim20/wilor_v2_model_only_active_trim20_closeview.html
```

当前 MediaPipe/stereo 观测 + 手模姿态版本：

```text
/home/yannan/workspace/ros1_docker-main/rosbag_data/lfv_demos/soap_grasp_high_4/quality/v0_active_trim20_tcp_model_orientation/gripper_pose_table_core_v0_tcp_wilor_v2_model_orientation.csv
/home/yannan/workspace/ros1_docker-main/rosbag_data/lfv_demos/soap_grasp_high_4/quality/v0_active_trim20_tcp_model_orientation/v0_tcp_wilor_v2_model_orientation_robust_all.html
/home/yannan/workspace/ros1_docker-main/rosbag_data/lfv_demos/soap_grasp_high_4/quality/v0_active_trim20_tcp_model_orientation/v0_tcp_wilor_v2_model_orientation_closeview.html
```

当前 index-middle MCP 手模姿态实验版本：

```text
/home/yannan/workspace/ros1_docker-main/rosbag_data/lfv_demos/soap_grasp_high_4/quality/wilor_v2_index_middle_mcp_x_smooth_core/gripper_pose_table_core_wilor_v2_mcp_x_smooth.csv
/home/yannan/workspace/ros1_docker-main/rosbag_data/lfv_demos/soap_grasp_high_4/quality/wilor_v2_index_middle_mcp_x_smooth_core/wilor_v2_index_middle_mcp_x_smoothed_core_6d_pose_over_frames.html
```

## 9. IK / Piper Replay Pipeline

```text
gripper_pose_table_core_*.csv
  -> open-segment IK-friendly regularizer
  -> IK-ready table-frame TCP / rot6d / gripper command
  -> table-to-robot calibration
  -> TCP offset compensation
  -> Piper URDF + Pinocchio IK
  -> dense 60Hz joint action chunk
  -> MIT replay / replay bag / QA
```

IK 输入是 gripper core CSV，不是 raw hand21：

```text
center_x/y/z       # table frame TCP target
rot6d_0..5         # optional 6D orientation
state              # open / closed
piper_joint7/8     # gripper command
```

qzy baseline 优先接：

```text
/home/yannan/workspace/ros1_docker-main/rosbag_data/lfv_demos/soap_grasp_high_4/quality/wilor_v2_model_only_active_trim20/gripper_pose_table_core_wilor_v2_model_only_active_trim20.csv
```

诊断版：

```text
/home/yannan/workspace/ros1_docker-main/rosbag_data/lfv_demos/soap_grasp_high_4/quality/v0_active_trim20_tcp_model_orientation/gripper_pose_table_core_v0_tcp_wilor_v2_model_orientation.csv
```

标定输入：

```text
/home/yannan/workspace/ros1_docker-main/data/lfv_calibration/right_arm_table_latest.json
/home/yannan/workspace/ros1_docker-main/data/lfv_calibration/right_arm_tcp_pivot_latest.json
```

IK / dense plan：

```text
/home/yannan/workspace/learning-from-video/scripts/simulate_lfv_piper_urdf_core.sh
/home/yannan/workspace/learning-from-video/scripts/prepare_right_arm_urdf_core_dense_60hz_plan.sh
/home/yannan/workspace/ros1_docker-main/workspaces/scripts/lfv_simulate_piper_core_gripper_pinocchio.py
/home/yannan/workspace/ros1_docker-main/workspaces/scripts/lfv_build_right_arm_dense_replay_plan.py
```

输出：

```text
<demo>/quality/urdf_sim/<ik_dir>/piper_pinocchio_core_ik.csv
<demo>/quality/urdf_sim/<ik_dir>/piper_pinocchio_core_ik_metadata.json
<demo>/quality/urdf_sim/<ik_dir>/ik_pose_debugger/ik_target_pose_debugger.html
<demo>/quality/urdf_sim/<ik_dir>/dense_replay_60hz_timescale_2/right_arm_dense_replay_plan.csv
<demo>/quality/urdf_sim/<ik_dir>/dense_replay_60hz_timescale_2/right_arm_dense_replay_plan_preview.png
```

常用参数：

```text
TARGET_MODE=absolute
MIN_ROBOT_Z_M=0.010
OUTPUT_HZ=60
REPLAY_TIME_SCALE=2.0
MAX_JOINT_STEP_RAD=0.012
MAX_GRIPPER_STEP=0.002
IK_MODE=position              # 当前更稳
ORIENTATION_SOURCE=none       # pose IK 仍作为调试项
```

- `position IK` 只强约束 TCP 位置，当前更适合 MIT 真机验证。
- `pose IK + rot6d` 先只用于 `ik_target_pose_debugger.html` 审计。

当前 `soap_grasp_high_4` 轻裁剪版本：

```text
crop source frames: 150..590

regularized gripper table:
/home/yannan/workspace/ros1_docker-main/rosbag_data/lfv_demos/soap_grasp_high_4/quality/wilor_tcp_v0_state_open_regularized_button_up_trim150_590/gripper_pose_table_core_wilor_tcp_v0_state_open_regularized_button_up_trim150_590.csv

button-up source -> gripper_base axis alignment:
source -x -> gripper_base +Y   # 夹爪开合方向
source +y -> gripper_base +Z   # 夹爪向外 / TCP 前向
source -z -> gripper_base +X   # 剩余右手系方向

matrix:
[[ 0, -1,  0],
 [ 0,  0,  1],
 [-1,  0,  0]]

script RPY:
--orientation-align-rpy=0.000000000000,1.570796326795,1.570796326795
```

当前 IK 结果：

```text
position IK:
  output = /home/yannan/workspace/ros1_docker-main/rosbag_data/lfv_demos/soap_grasp_high_4/quality/urdf_sim/piper_position_regularized_trim150_590/
  success = 441 / 441
  max_pos_error = 0.10 mm

full 6D pose IK:
  output = /home/yannan/workspace/ros1_docker-main/rosbag_data/lfv_demos/soap_grasp_high_4/quality/urdf_sim/piper_pose_button_up_regularized_trim150_590/
  success = 202 / 441
  max_pos_error = 121.13 mm
```

index-middle MCP orientation 实验：

```text
input fused + regularized table:
/home/yannan/workspace/ros1_docker-main/rosbag_data/lfv_demos/soap_grasp_high_4/quality/wilor_tcp_v0_state_open_regularized_index_middle_button_up_trim150_590/gripper_pose_table_core_wilor_tcp_v0_state_open_regularized_index_middle_button_up_trim150_590.csv

regularizer compare HTML:
/home/yannan/workspace/ros1_docker-main/rosbag_data/lfv_demos/soap_grasp_high_4/quality/wilor_tcp_v0_state_open_regularized_index_middle_button_up_trim150_590/wilor_tcp_v0_state_open_regularized_index_middle_button_up_trim150_590_compare_3d.html

rows = 441
closed_frames = 269
protected_frames = 285
regularized_open_frames = 156

full 6D pose IK:
  output = /home/yannan/workspace/ros1_docker-main/rosbag_data/lfv_demos/soap_grasp_high_4/quality/urdf_sim/piper_pose_index_middle_button_up_regularized_trim150_590/
  success = 240 / 441
  median_pos_error = 0.092 mm
  p95_pos_error = 183.04 mm
  max_pos_error = 213.91 mm

MuJoCo home-ramp replay:
/home/yannan/workspace/ros1_docker-main/rosbag_data/lfv_demos/soap_grasp_high_4/quality/urdf_sim/piper_pose_index_middle_button_up_regularized_trim150_590/mujoco_replay_home_ramp_2s/piper_pose_ik_mujoco_replay.html
```

失败段：

```text
idx 0-81    source_frame 150-231  state open->closed    frames 82  max_pos 175.62 mm  max_ori 43.19 deg
idx 99-184  source_frame 249-334  state closed->closed  frames 86  max_pos 213.91 mm  max_ori 52.51 deg
idx 206-225 source_frame 356-375  state closed->closed  frames 20  max_pos 82.32 mm   max_ori 20.35 deg
idx 294-306 source_frame 444-456  state closed->closed  frames 13  max_pos 16.69 mm   max_ori 4.12 deg
```

index-middle 24 个 signed axis permutation sweep：

```text
summary:
/home/yannan/workspace/ros1_docker-main/rosbag_data/lfv_demos/soap_grasp_high_4/quality/urdf_sim/piper_pose_index_middle_axis_align_sweep_trim150_590/axis_align_sweep_summary.json

best label = Xg_mz_Yg_mx_Zg_py
best matrix:
[[ 0, -1,  0],
 [ 0,  0,  1],
 [-1,  0,  0]]

best success = 240 / 441
second best success = 238 / 441
```

结论：index-middle MCP 作为 hand orientation frame 的输入更稳定，full pose IK 成功帧从 `202/441` 提到 `240/441`。但 p95/max position error 变大，且失败集中在前段和几个 closed block，所以它是更好的姿态输入候选，不是最终 IK 解法。当前 button-up 矩阵仍然是 axis sweep 里最好的固定轴对齐。

index-middle base + thumb-index MCP YZ tilt 实验：

这个版本按新的 hand-frame 设计：

```text
base frame:
  x0 = index_mcp -> middle_mcp
  y0 = wrist -> (index_mcp + middle_mcp) / 2, projected orthogonal to x0
  z0 = x0 cross y0

thumb cue:
  v = thumb_mcp -> index_mcp
  v_base = R_base^T * v
  project v_base to base YZ plane
  rotate base Y/Z around x0 toward this YZ angle
```

实现入口：

```text
/home/yannan/workspace/learning-from-video/WiLor/lfv_adapter/lfv_export_wilor_v2_mcp_x_smooth_core.py

new mode:
--orientation-x-axis index_middle_thumb_index_yz_tilt
--thumb-yz-tilt-gain <gain>
--thumb-yz-tilt-clamp-deg 45
```

gain sweep summary：

```text
/home/yannan/workspace/ros1_docker-main/rosbag_data/lfv_demos/soap_grasp_high_4/quality/urdf_sim/piper_pose_index_middle_thumb_index_yz_tilt_gain_sweep_trim150_590/summary.json
```

结果：

```text
index_middle only:
  success = 240 / 441
  median_pos_error = 0.092 mm
  p95_pos_error = 183.04 mm
  max_pos_error = 213.91 mm

thumb-index YZ tilt gain +1.0:
  observed tilt = +27.5..+39.7 deg
  success = 0 / 441

thumb-index YZ tilt gain +0.5:
  observed tilt = +13.8..+19.9 deg
  success = 65 / 441

thumb-index YZ tilt gain -0.5:
  observed tilt = -19.9..-13.8 deg
  success = 190 / 441

thumb-index YZ tilt gain -1.0:
  observed tilt = -39.7..-27.5 deg
  success = 354 / 441
  median_pos_error = 0.079 mm
  p95_pos_error = 61.41 mm
  max_pos_error = 86.03 mm
```

当前最好实验版本：

```text
hand pose HTML:
/home/yannan/workspace/ros1_docker-main/rosbag_data/lfv_demos/soap_grasp_high_4/quality/wilor_v2_index_middle_thumb_index_yz_tilt_gm1p0_smooth_core/wilor_v2_index_middle_thumb_index_yz_tilt_gm1p0_smoothed_core_6d_pose_over_frames.html

regularized compare HTML:
/home/yannan/workspace/ros1_docker-main/rosbag_data/lfv_demos/soap_grasp_high_4/quality/wilor_tcp_v0_state_open_regularized_index_middle_thumb_index_yz_tilt_gm1p0_button_up_trim150_590/wilor_tcp_v0_state_open_regularized_index_middle_thumb_index_yz_tilt_gm1p0_button_up_trim150_590_compare_3d.html

IK output:
/home/yannan/workspace/ros1_docker-main/rosbag_data/lfv_demos/soap_grasp_high_4/quality/urdf_sim/piper_pose_index_middle_thumb_index_yz_tilt_gm1p0_button_up_regularized_trim150_590/

MuJoCo home-ramp replay:
/home/yannan/workspace/ros1_docker-main/rosbag_data/lfv_demos/soap_grasp_high_4/quality/urdf_sim/piper_pose_index_middle_thumb_index_yz_tilt_gm1p0_button_up_regularized_trim150_590/mujoco_replay_home_ramp_2s/piper_pose_ik_mujoco_replay.html
```

TCP-priority relaxed IK fallback：

这个 fallback 放在 Piper URDF + Pinocchio IK 内部，只在 normal full 6D pose IK 失败时触发。它不是新的 hand pose 预测方法，而是 IK 后端的可达性补救：

```text
normal full 6D pose IK
  -> if success: keep normal solution
  -> if fail: TCP-priority relaxed IK fallback
       primary task: TCP position
       secondary task: weak orientation cue + q_ref smoothness + limit avoidance
       hard rule: keep arm joints inside physical limits with configurable safety margin
```

设计原则：

- TCP position 是主任务，fallback 的目标是“在不超限的情况下尽量少漂 TCP”。
- orientation 只作为 nullspace soft cue，不再作为 hard 6D 约束。
- q_ref 使用上一帧关节，保证 fallback 不从一个完全无关的 seed 跳过去。
- joint limit 使用安全 margin，当前默认 `0.035 rad`；贴到 margin 边界时阻止继续往外推。
- 成功判定单独使用 `relaxed_max_pos_error_m`，当前先设 `20 mm`，实际这次远小于该阈值。

实现入口：

```text
/home/yannan/workspace/ros1_docker-main/workspaces/scripts/lfv_simulate_piper_core_gripper_pinocchio.py
/home/yannan/workspace/learning-from-video/scripts/simulate_lfv_piper_urdf_core.sh
```

新增运行参数：

```text
--relaxed-ik-fallback tcp_priority
--relaxed-max-pos-error-m 0.020
--relaxed-max-iters 250
--relaxed-limit-margin-rad 0.035
--relaxed-active-threshold-rad 0.020
--relaxed-orientation-weight 0.05
--relaxed-smooth-weight 0.02
--relaxed-limit-weight 0.02
```

CSV / metadata 新增诊断字段：

```text
normal_ik_success
normal_pos_error_m
normal_ori_error_rad
relaxed_ik_attempted
relaxed_ik_used
relaxed_ik_success
tcp_drift_m
relaxed_pos_error_m
relaxed_ori_error_rad
limit_margin_min_rad
limit_active_joints
relaxed_seed_index
```

`soap_grasp_high_4` 当前最好输入上测试：

```text
input table:
/home/yannan/workspace/ros1_docker-main/rosbag_data/lfv_demos/soap_grasp_high_4/quality/wilor_tcp_v0_state_open_regularized_index_middle_thumb_index_yz_tilt_gm1p0_button_up_trim150_590/gripper_pose_table_core_wilor_tcp_v0_state_open_regularized_index_middle_thumb_index_yz_tilt_gm1p0_button_up_trim150_590.csv

IK output:
/home/yannan/workspace/ros1_docker-main/rosbag_data/lfv_demos/soap_grasp_high_4/quality/urdf_sim/piper_pinocchio_core_absolute_scale_0p25_core_gripper_pose_table_core_wilor_tcp_v0_state_open_regularized_index_middle_thumb_index_yz_tilt_gm1p0_button_up_trim150_590_ik_pose_rot6d_relaxed_tcp_priority/

normal 6D IK success = 355 / 441
fallback attempted = 86
fallback used = 86
fallback accepted = 86
final success = 441 / 441

final TCP error:
  median = 0.071 mm
  p95 = 0.096 mm
  max = 0.100 mm

fallback TCP drift:
  median = 0.063 mm
  p95 = 0.086 mm
  max = 0.092 mm

fallback orientation error:
  median = 0.213 rad
  p95 = 0.541 rad
  max = 0.542 rad

fallback state distribution:
  open = 11 frames
  closed = 75 frames

active limit margin:
  joint4:lower = 54 frames
  joint5:lower = 8 frames
  none = 24 frames
```

fallback trigger intervals 和之前 normal 6D IK failure 一致：

```text
idx 0-10    source_frame 150-160  state open
idx 74-101  source_frame 224-251  state closed
idx 124-131 source_frame 274-281  state closed
idx 177-208 source_frame 327-358  state closed
idx 212-218 source_frame 362-368  state closed
```

MuJoCo home-ramp replay：

```text
/home/yannan/workspace/ros1_docker-main/rosbag_data/lfv_demos/soap_grasp_high_4/quality/urdf_sim/piper_pose_index_middle_thumb_index_yz_tilt_gm1p0_button_up_regularized_relaxed_tcp_priority_trim150_590/mujoco_replay_home_ramp_2s/piper_pose_ik_mujoco_replay.html
```

当前结论：

- fallback 可以把“姿态导致 wrist 贴限位”的帧转成限位内的 TCP-priority 解。
- 这版对 TCP 很强，实际 drift 小于 `0.1 mm`；joint4/joint5 没有越过物理限位。
- 代价是 fallback 帧不再保证 full 6D orientation，闭合段也会牺牲一部分姿态，所以它适合先作为 IK 可达性 fallback / QA 版本，不应被解释成 hand-to-gripper 根因已经解决。
- 真机前要重点看 fallback 的 closed 段 gripper 姿态是否仍然能保持接触关系。

`gain=-1.0` 失败段：

```text
idx 0-11    source_frame 150-161  state open->open      frames 12  max_pos 4.85 mm   max_ori 1.19 deg
idx 74-101  source_frame 224-251  state closed->closed  frames 28  max_pos 63.71 mm  max_ori 32.69 deg
idx 124-131 source_frame 274-281  state closed->closed  frames 8   max_pos 18.89 mm  max_ori 4.64 deg
idx 177-208 source_frame 327-358  state closed->closed  frames 32  max_pos 86.03 mm  max_ori 37.95 deg
idx 212-218 source_frame 362-368  state closed->closed  frames 7   max_pos 17.68 mm  max_ori 15.94 deg

open success   = 160 / 172
closed success = 194 / 269
```

结论：这个验证支持“index-middle 提供稳定横向轴，thumb-index 只补 YZ 下探角”的方向。符号非常重要，当前有效方向是 `gain=-1.0`，对应大约 `-30~-40 deg` 的下探补偿。这个版本还需要目视确认 MuJoCo 里的夹爪姿态是否真的符合 demo，但从 IK 数值上已经是目前 full 6D pose 最好的版本。

机械臂初始位姿 / home continuity：

```text
当前 rosbag 里没有机械臂 joint state，只有相机 topic，所以没有真实录制初始 q。
当前 replay/home 脚本默认 home_q7 = 0,0,0,0,0,0,0.08。
当前 Pinocchio IK 脚本里 --initial-q 为空时默认 arm q = 0,0,0,0,0,0。
```

因此，单纯把 `initial_q=0,0,0,0,0,0` 显式传给 IK 不会改变当前 pose IK 结果。home 位姿更适合放在 **replay / visualization 的启动段**：

```text
home q
  -> 2s joint-space ramp to first IK frame
  -> original IK trajectory
```

新生成的带 home ramp MuJoCo 输出：

```text
/home/yannan/workspace/ros1_docker-main/rosbag_data/lfv_demos/soap_grasp_high_4/quality/urdf_sim/piper_pose_button_up_regularized_trim150_590/mujoco_replay_home_ramp_2s/piper_position_only_mujoco_replay.html
/home/yannan/workspace/ros1_docker-main/rosbag_data/lfv_demos/soap_grasp_high_4/quality/urdf_sim/piper_pose_button_up_regularized_trim150_590/mujoco_replay_home_ramp_2s/piper_pose_ik_mujoco_replay.html
```

这个 home ramp 可以改善从机械臂初始位姿到第一帧的运动连续性，但不能解决 full 6D pose IK 第一帧和若干闭合段的不可达/错误分支问题。后续如果要继续改善 pose IK，应做 first-frame multi-seed + home-prior 分支选择，代价函数优先选离 home 近、离 joint limit 远、pose error 可接受的解，再 warm-start 后续帧。

seed sweep 诊断结果：

```text
输出目录:
/home/yannan/workspace/ros1_docker-main/rosbag_data/lfv_demos/soap_grasp_high_4/quality/urdf_sim/seed_sweep_button_up_trim150_590/

手选 initial-q 测试:
  home_zero                 success = 202 / 441, first pos error = 121.13 mm
  position_first_q           success = 202 / 441, first pos error = 121.13 mm
  pose_first_success_q47      success = 202 / 441, first pos error = 121.13 mm
  position_arm_pose_wrist     success = 202 / 441, first pos error = 121.13 mm
  mid_reach_pose_wrist        success = 202 / 441, first pos error = 121.13 mm
  mid_reach_wrist_flip        success = 148 / 441, first pos error = 135.68 mm

first-frame multi-seed:
  tested seeds = 486
  solver_success = 0
  best first-frame result still converges to:
    q = [-1.038, 1.830, -0.994, -1.658, -1.220, 0.162]
    pos error = 121.13 mm
    ori error = 29.71 deg
```

结论：当前 first-frame full 6D pose 失败不是单纯 `initial_q` seed 选择问题。大部分 seed 都收敛到同一个 joint5 下限附近的局部解，说明需要改 pose target / 放松姿态约束 / 裁掉更前面的不可达帧，而不是只换 seed。

button-up 后 local Z +30 deg 诊断：

```text
output:
/home/yannan/workspace/ros1_docker-main/rosbag_data/lfv_demos/soap_grasp_high_4/quality/urdf_sim/piper_pose_button_up_local_z_p30_regularized_trim150_590/

orientation_align_rpy:
1.570796326795,1.047197551197,3.141592653590

pose IK:
  success = 202 / 441
  max_pos_error = 121.05 mm
  p95_pos_error = 91.18 mm

MuJoCo:
/home/yannan/workspace/ros1_docker-main/rosbag_data/lfv_demos/soap_grasp_high_4/quality/urdf_sim/piper_pose_button_up_local_z_p30_regularized_trim150_590/mujoco_replay_home_ramp_2s/piper_pose_ik_mujoco_replay.html
```

结论：local Z +30 deg 基本不改变可解性，只是让 wrist/joint6 整体换一个角度；full 6D pose IK 的主要失败仍然在姿态目标和机械臂可达/限位约束之间的冲突。

姿态目标 vs 机械臂自然姿态诊断：

```text
diagnostic:
/home/yannan/workspace/ros1_docker-main/rosbag_data/lfv_demos/soap_grasp_high_4/quality/urdf_sim/piper_pose_button_up_regularized_trim150_590/orientation_vs_natural_position_ik.json

定义:
  natural orientation = position-only IK q 的 FK TCP rotation
  target orientation  = 当前 button-up rot6d full pose target
  relative angle      = angle(natural_R^T * target_R)

all frames relative angle:
  median = 59.65 deg
  p95    = 95.21 deg

pose IK success frames:
  median = 23.42 deg

pose IK failure frames:
  median = 79.06 deg

first frame:
  relative angle = 74.58 deg
  pose pos error = 121.13 mm
```

结论：full pose 解不出来的核心不是 seed，而是手部输入姿态长期带着一个相对机械臂自然姿态很大的固定夹角。这个夹角会把 wrist 推到某个固定/限位姿态，尤其容易让 `joint5` 或 `joint4` 贴限位。下一步应在 IK 前做 orientation feasibility regularizer：先用 position-only IK 得到自然机械臂姿态 `R_nat`，再把手部姿态 `R_hand` 只作为 soft cue，例如 `R_target = R_nat * Exp(clamp(beta * Log(R_nat^T R_hand), max_angle))`。open 段 beta 应更小，closed 段再保留更多 hand cue。

但是这个 regularizer 只能作为临时工程补丁，不能当作根因修复。进一步的 hand-to-gripper 诊断：

```text
diagnostic:
/home/yannan/workspace/ros1_docker-main/rosbag_data/lfv_demos/soap_grasp_high_4/quality/urdf_sim/piper_pose_button_up_regularized_trim150_590/hand_to_gripper_mapping_diagnostic.json

反推每帧需要的 A_i:
  R_nat ~= R_robot_table * R_source * A_i

如果只是固定 hand-to-gripper 映射错了，A_i 应该很集中。

A_i dispersion:
  all frames median    = 34.15 deg
  closed frames median = 29.44 deg
  open frames median   = 32.73 deg

current button-up A 到 mean needed A:
  all frames mean    = 51.76 deg
  closed frames mean = 58.57 deg
```

oracle constant mapping 测试：

```text
summary:
/home/yannan/workspace/ros1_docker-main/rosbag_data/lfv_demos/soap_grasp_high_4/quality/urdf_sim/piper_pose_oracle_mean_A_trim150_590_summary.json

current button-up:
  success = 202 / 441

oracle mean A from all frames:
  rpy = 2.062658277365,0.850803140664,3.075430818599
  success = 156 / 441

oracle mean A from closed frames:
  rpy = 2.035910241195,0.724184198728,3.015552883942
  success = 154 / 441
```

结论：只换一个固定 hand-to-gripper 矩阵没有解决问题，甚至更差。根因更可能是 hand frame 定义/hand tracking 本身随动作漂移，或者当前用人手局部轴直接定义机械夹爪完整 SO(3) 的方法不合适。后续应回到 hand tracking + hand frame construction 重新审计，而不是继续只在 IK 后端调 seed / local offset。

真机 replay：

```text
/home/yannan/workspace/learning-from-video/scripts/run_lfv_humanego_gripper_mit_replay_enter.sh
/home/yannan/workspace/ros1_docker-main/workspaces/scripts/lfv_run_right_arm_dense_replay_mit.py
```

```text
1. 切 MIT，hold 当前
2. 平滑到 dense plan 第一帧
3. Enter 开始 60Hz replay
4. 结束后回第一帧或 hold
5. 同时录制 topcam + right arm feedback/cmd/status bag
```

待补：一个 qzy baseline 专用 wrapper，固定使用 `WiLoR V2 model-only active-trim20` core CSV，避免手动传 `GRIPPER_CORE_CSV` 时混到 V0 或诊断版本。

V2.0 代码入口：

```text
/home/yannan/workspace/learning-from-video/V2.0/lfv_v2_0_gate_smooth_core.py
/home/yannan/workspace/learning-from-video/V2.0/scripts/run_v2_0_gate_smooth.sh
```

## 10. 不要混淆的点

- 不要默认使用 hand-model constrained TCP 作为最终位置。
- 当前更新：TCP 轨迹可以优先试 WiLoR V2 model-only active-trim20，因为它比 MediaPipe V0 轨迹更稳。
- 不要把 `wilor_v2_as_v0_gripper_action_active_trim20` 误认为冻结姿态；那版是 V0 在 WiLoR 21 点上重算 `mcp_3d` 姿态。
- `index_middle_mcp` 只是在手模约束姿态里替换 local X 轴定义；TCP 和 gripper state 仍然走 WiLoR TCP + V0 pinch state。
- `index_middle_mcp` 改善了 raw orientation 稳定性和 full pose IK success count，但还不能当作已解决的 Piper 6D pose IK pipeline。
- open-segment regularizer 是 IK 前的 action post-process，不是 hand tracking / pose prediction 本身。
- 夹爪 closed 段是 interaction protected segment；open 段才可以暴力平滑或直线化。
- 不要跨 demo 平均 orientation。
- 不要用 V2.0 gate 后的点数判断 raw hand detection 点数。
- V2.0 rejected frame 不等于 raw hand detection missing。
- 当前维护策略是：轨迹/TCP 优先 WiLoR，姿态/orientation 走 MediaPipe/stereo 观测 + 手模约束 SO(3) 链路。

## 11. 双手 WiLoR identity 稳定策略（2026-06-22）

双手 demo 里不能再先信 WiLoR 的 raw `left/right` 标签。当前观察到的问题包括：

- 同一个物理手在几帧内从 `right` 变成 `left` 或反过来。
- 同一个位置出现两个 MANO hand model。
- 一只手短暂漏检 1-3 帧后又回到相近位置。
- MANO palm normal 在一两帧内大角度翻转，表现为突然手心朝上/反手。

新的处理原则：

```text
WiLoR run:
  --hand best          # 保留所有候选，不提前按 raw left/right 丢掉

pre-gripper stabilizer:
  bbox continuity      # 按图像 bbox 连续性追踪物理手
  identity_hold_ms=400 # 约 12 帧@30fps；短暂 raw label flip 不允许改变身份
  missing_bridge_ms=200 # 约 6 帧@30fps；短暂漏检使用上一帧 good MANO pose hold
  palm_flip_window_ms=300
  palm_flip_angle_deg=100
  hard_palm_speed_deg_s=900
```

实现位置：

```text
/home/yannan/workspace/learning-from-video/local_pipelines/double_hand_pipeline/stabilize_wilor_predictions.py
/home/yannan/workspace/learning-from-video/local_pipelines/double_hand_pipeline/run_double_hand_pipeline.sh
```

输出位置：

```text
<demo>/quality/double_hand_pipeline/<hand>_hand_<arm>_arm/stages/wilor_temporal_stabilized_<hand>/
  wilor_predictions_stabilized_<hand>.csv
  wilor_predictions_stabilized_<hand>_events.csv
  wilor_predictions_stabilized_<hand>.json
```

`bag_20260622_1548_001` 的 595-628 帧验证：

- right lane 中 601/602/604 被标为 `hold_missing`。
- 605-613 被标为 `hold_bad_candidate / label_flip_pending`，没有把 raw `left` 当成左手输入。
- 614 被标为 `hold_bad_candidate / palm_motion_bad`。
- 626-628 被标为 `hold_bad_candidate / label_flip_pending`。
- left lane 同一段保持原来的左手 track，没有被右手坏候选污染。

注意：当前 stabilizer 对坏候选采用 hold，不做长时间插值。后续如果遮挡超过 `400ms`，需要再设计插值或重新关联策略。
