#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_ROS_ROOT="${HOST_ROS_ROOT:-/home/yannan/workspace/ros1_docker-main}"
CONTAINER_ROOT="${CONTAINER_ROOT:-/workspace/ros1_docker_jinhe}"
HOST_HUMAN_ROOT="${HOST_HUMAN_ROOT:-${HOST_ROS_ROOT}/rosbag_data/human_teaching_videos}"
HOST_TELEOP_ROOT="${HOST_TELEOP_ROOT:-${HOST_ROS_ROOT}/rosbag_data}"
DOCKER_CONTAINER="${DOCKER_CONTAINER:-ros1_noetic}"

RUN_DOUBLE_HAND="${RUN_DOUBLE_HAND:-true}"
RUN_IK="${RUN_IK:-false}"
RUN_MUJOCO="${RUN_MUJOCO:-false}"
REBUILD_WILOR="${REBUILD_WILOR:-false}"
ROBOT_SAMPLE_HZ="${ROBOT_SAMPLE_HZ:-30.0}"
AUTO_TRIM_HUMAN="${AUTO_TRIM_HUMAN:-true}"
AUTO_TRIM_ROBOT="${AUTO_TRIM_ROBOT:-true}"
SPEED_THRESHOLD_MPS="${SPEED_THRESHOLD_MPS:-0.015}"
TRIM_PAD_SEC="${TRIM_PAD_SEC:-0.20}"
LEFT_ROBOT_TOPIC="${LEFT_ROBOT_TOPIC:-/robot/arm_left/joint_states_single}"
RIGHT_ROBOT_TOPIC="${RIGHT_ROBOT_TOPIC:-/robot/arm_right/joint_states_single}"

usage() {
  cat <<EOF
Usage:
  bash local_pipelines/double_hand_pipeline/run_dual_human_vs_teleop_tcp_3d.sh <human_session_or_dir> <teleop_episode_or_bag>

Example:
  bash local_pipelines/double_hand_pipeline/run_dual_human_vs_teleop_tcp_3d.sh \\
    /home/yannan/workspace/ros1_docker-main/rosbag_data/human_teaching_videos/bag_20260622_1548_002 \\
    /home/yannan/workspace/ros1_docker-main/rosbag_data/BAG3/episode_008

Output:
  <human_session>/quality/human_vs_teleop_tcp_3d/<teleop_session>_<episode>/dual_arm_tcp_3d_table_frame.html
EOF
}

die() {
  echo "[dual_tcp_3d] ERROR: $*" >&2
  exit 1
}

log() {
  printf '\n[dual_tcp_3d] %s\n' "$*"
}

resolve_human_session() {
  local target="$1"
  if [[ "${target}" = /* ]]; then
    readlink -f "${target}"
  else
    readlink -f "${HOST_HUMAN_ROOT}/${target}"
  fi
}

resolve_teleop_bag() {
  local target="$1"
  local path=""
  if [[ "${target}" = /* ]]; then
    path="$(readlink -m "${target}")"
  else
    path="$(readlink -m "${HOST_TELEOP_ROOT}/${target}")"
  fi
  if [[ -d "${path}" ]]; then
    if [[ -f "${path}/episode.bag" ]]; then
      path="${path}/episode.bag"
    elif [[ -f "${path}/episode_0.bag" ]]; then
      path="${path}/episode_0.bag"
    else
      die "episode directory has no episode.bag or episode_0.bag: ${path}"
    fi
  fi
  readlink -f "${path}"
}

container_path() {
  local host_path="$1"
  case "${host_path}" in
    "${HOST_ROS_ROOT}"/*)
      printf '%s\n' "${host_path/#${HOST_ROS_ROOT}/${CONTAINER_ROOT}}"
      ;;
    *)
      die "cannot map host path into container: ${host_path}"
      ;;
  esac
}

latest_regularized_csv() {
  local dir="$1"
  find "${dir}" -maxdepth 1 -type f -name 'gripper_pose_table_core_*.csv' ! -name '*raw_fused.csv' -printf '%T@ %p\n' \
    | sort -n \
    | tail -1 \
    | cut -d' ' -f2-
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

human_target="${1:-}"
teleop_target="${2:-}"
if [[ -z "${human_target}" || -z "${teleop_target}" ]]; then
  usage >&2
  exit 1
fi

human_session_dir="$(resolve_human_session "${human_target}")"
teleop_bag="$(resolve_teleop_bag "${teleop_target}")"
[[ -d "${human_session_dir}" ]] || die "human session not found: ${human_session_dir}"
[[ -f "${teleop_bag}" ]] || die "teleop bag not found: ${teleop_bag}"
case "${human_session_dir}" in "${HOST_ROS_ROOT}"/*) ;; *) die "human session must be under ${HOST_ROS_ROOT}: ${human_session_dir}" ;; esac
case "${teleop_bag}" in "${HOST_ROS_ROOT}"/*) ;; *) die "teleop bag must be under ${HOST_ROS_ROOT}: ${teleop_bag}" ;; esac

teleop_episode_dir="$(dirname "${teleop_bag}")"
teleop_session="$(basename "$(dirname "${teleop_episode_dir}")")"
teleop_episode="$(basename "${teleop_episode_dir}")"
output_dir="${human_session_dir}/quality/human_vs_teleop_tcp_3d/${teleop_session}_${teleop_episode}"
mkdir -p "${output_dir}"

log "human session: ${human_session_dir}"
log "teleop bag:    ${teleop_bag}"
log "output:        ${output_dir}"
log "robot topics:  left=${LEFT_ROBOT_TOPIC} right=${RIGHT_ROBOT_TOPIC}"

if [[ "${RUN_DOUBLE_HAND}" == "true" || "${RUN_DOUBLE_HAND}" == "1" ]]; then
  log "1/3 run human left lane: HAND=left ARM=left"
  HAND=left ARM=left RUN_IK="${RUN_IK}" RUN_MUJOCO="${RUN_MUJOCO}" REBUILD_WILOR="${REBUILD_WILOR}" \
    bash "${SCRIPT_DIR}/run_double_hand_pipeline.sh" "${human_session_dir}"
  log "1/3 run human right lane: HAND=right ARM=right"
  HAND=right ARM=right RUN_IK="${RUN_IK}" RUN_MUJOCO="${RUN_MUJOCO}" REBUILD_WILOR="${REBUILD_WILOR}" \
    bash "${SCRIPT_DIR}/run_double_hand_pipeline.sh" "${human_session_dir}"
else
  log "1/3 skip human double-hand lanes by RUN_DOUBLE_HAND=${RUN_DOUBLE_HAND}"
fi

left_csv="$(latest_regularized_csv "${human_session_dir}/quality/double_hand_pipeline/left_hand_left_arm/regularized" || true)"
right_csv="$(latest_regularized_csv "${human_session_dir}/quality/double_hand_pipeline/right_hand_right_arm/regularized" || true)"
[[ -n "${left_csv}" && -f "${left_csv}" ]] || die "cannot find left human TCP CSV"
[[ -n "${right_csv}" && -f "${right_csv}" ]] || die "cannot find right human TCP CSV"

log "2/3 found human TCP CSVs"
echo "  left:  ${left_csv}"
echo "  right: ${right_csv}"

trim_human_arg=()
trim_robot_arg=()
if [[ "${AUTO_TRIM_HUMAN}" == "true" || "${AUTO_TRIM_HUMAN}" == "1" ]]; then
  trim_human_arg+=(--auto-trim-human)
fi
if [[ "${AUTO_TRIM_ROBOT}" == "true" || "${AUTO_TRIM_ROBOT}" == "1" ]]; then
  trim_robot_arg+=(--auto-trim-robot)
fi

log "3/3 draw one 3D table-frame comparison with four TCP lines"
docker exec "${DOCKER_CONTAINER}" bash -lc "
set -e
source '${CONTAINER_ROOT}/workspaces/scripts/use_robot.sh' >/dev/null
python3 '${CONTAINER_ROOT}/workspaces/scripts/compare_dual_human_teleop_tcp_table.py' \\
  --human-left-tcp-csv '$(container_path "${left_csv}")' \\
  --human-right-tcp-csv '$(container_path "${right_csv}")' \\
  --teleop-bag '$(container_path "${teleop_bag}")' \\
  --left-robot-topic '${LEFT_ROBOT_TOPIC}' \\
  --right-robot-topic '${RIGHT_ROBOT_TOPIC}' \\
  --left-robot-table-json '${CONTAINER_ROOT}/data/lfv_calibration/left_arm_table_latest.json' \\
  --right-robot-table-json '${CONTAINER_ROOT}/data/lfv_calibration/right_arm_table_latest.json' \\
  --left-tcp-json '${CONTAINER_ROOT}/data/lfv_calibration/left_arm_touch_tcp_zr2_latest.json' \\
  --right-tcp-json '${CONTAINER_ROOT}/data/lfv_calibration/right_arm_touch_tcp_xr2_latest.json' \\
  --output-dir '$(container_path "${output_dir}")' \\
  --robot-sample-hz '${ROBOT_SAMPLE_HZ}' \\
  --speed-threshold-mps '${SPEED_THRESHOLD_MPS}' \\
  --trim-pad-sec '${TRIM_PAD_SEC}' \\
  ${trim_human_arg[*]:-} \\
  ${trim_robot_arg[*]:-}
"

cat <<EOF

[dual_tcp_3d] Done.
  html:    ${output_dir}/dual_arm_tcp_3d_table_frame.html
  png:     ${output_dir}/dual_arm_tcp_3d_table_frame.png
  summary: ${output_dir}/dual_arm_tcp_3d_table_frame_summary.json
EOF
