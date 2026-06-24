#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_NAME="${CONTAINER_NAME:-ros1_noetic}"
HOST_ROS_ROOT="${HOST_ROS_ROOT:-/home/yannan/workspace/ros1_docker-main}"
CONTAINER_ROOT="${CONTAINER_ROOT:-/workspace/ros1_docker_jinhe}"
HOST_SESSION_ROOT="${HOST_SESSION_ROOT:-${HOST_ROS_ROOT}/rosbag_data/human_teaching_videos}"

OUTPUT_HZ="${OUTPUT_HZ:-60}"
REPLAY_TIME_SCALE="${REPLAY_TIME_SCALE:-2.0}"
BUILD_BAG_IF_MISSING="${BUILD_BAG_IF_MISSING:-true}"
REBUILD_BAG_BEFORE_PLAY="${REBUILD_BAG_BEFORE_PLAY:-false}"
SELECT_ROBOT_CMD_MUX="${SELECT_ROBOT_CMD_MUX:-true}"
SET_ARBITER_POLICY="${SET_ARBITER_POLICY:-true}"
WAIT_ENTER="${WAIT_ENTER:-true}"
ROSBAG_PLAY_RATE="${ROSBAG_PLAY_RATE:-1.0}"
ROSBAG_PLAY_QUIET="${ROSBAG_PLAY_QUIET:-false}"
DRY_RUN="${DRY_RUN:-false}"
LFV_ALLOW_ROBOT_MOTION="${LFV_ALLOW_ROBOT_MOTION:-0}"

usage() {
  cat <<EOF
Usage:
  DRY_RUN=true bash local_pipelines/double_hand_pipeline/play_double_arm_dense_replay_bag_enter.sh <demo_name_or_session_dir>
  LFV_ALLOW_ROBOT_MOTION=1 HAND_ARM_MAP='left:left,right:right' bash local_pipelines/double_hand_pipeline/play_double_arm_dense_replay_bag_enter.sh <demo_name_or_session_dir>

Plays the synchronized dual-arm command bag with rosbag play.
Real robot motion requires LFV_ALLOW_ROBOT_MOTION=1.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

target="${1:-}"
if [[ -z "${target}" ]]; then
  echo "[play_double_arm_replay] Missing demo name or session dir." >&2
  usage >&2
  exit 1
fi
if [[ "${DRY_RUN}" != "true" && "${LFV_ALLOW_ROBOT_MOTION}" != "1" ]]; then
  echo "[play_double_arm_replay] Refusing real robot motion." >&2
  echo "  Set LFV_ALLOW_ROBOT_MOTION=1 only when ready." >&2
  echo "  Use DRY_RUN=true first." >&2
  exit 2
fi

if [[ "${target}" = /* ]]; then
  session_dir="$(readlink -f "${target}")"
else
  session_dir="$(readlink -f "${HOST_SESSION_ROOT}/${target}")"
fi
if [[ ! -d "${session_dir}" ]]; then
  echo "[play_double_arm_replay] Session not found: ${session_dir}" >&2
  exit 1
fi
case "${session_dir}" in
  "${HOST_ROS_ROOT}"/*) ;;
  *)
    echo "[play_double_arm_replay] Session must be under ${HOST_ROS_ROOT}: ${session_dir}" >&2
    exit 1
    ;;
esac

label_float() {
  printf '%g' "$1" | sed 's/\./p/g'
}

container_path() {
  local host_path="$1"
  case "${host_path}" in
    "${HOST_ROS_ROOT}"/*)
      printf '%s\n' "${host_path/#${HOST_ROS_ROOT}/${CONTAINER_ROOT}}"
      ;;
    *)
      echo "[play_double_arm_replay] Cannot map host path into container: ${host_path}" >&2
      exit 1
      ;;
  esac
}

if [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}" 2>/dev/null || true)" != "true" ]]; then
  docker start "${CONTAINER_NAME}" >/dev/null
fi

hz_label="$(label_float "${OUTPUT_HZ}")"
time_label="$(label_float "${REPLAY_TIME_SCALE}")"
bag_dir="${session_dir}/quality/double_hand_pipeline/dual_arm_replay_${hz_label}hz_timescale_${time_label}"
bag_path="${bag_dir}/double_arm_dense_replay_cmd.bag"

if [[ "${REBUILD_BAG_BEFORE_PLAY}" == "true" || ( "${BUILD_BAG_IF_MISSING}" == "true" && ! -f "${bag_path}" ) ]]; then
  echo "[play_double_arm_replay] Building synchronized dual-arm replay bag first."
  OUTPUT_HZ="${OUTPUT_HZ}" \
  REPLAY_TIME_SCALE="${REPLAY_TIME_SCALE}" \
    bash "${SCRIPT_DIR}/build_double_arm_dense_replay_bag.sh" "${session_dir}"
fi

if [[ ! -f "${bag_path}" ]]; then
  echo "[play_double_arm_replay] Missing bag: ${bag_path}" >&2
  echo "  Build it with: HAND_ARM_MAP='left:left,right:right' bash ${SCRIPT_DIR}/build_double_arm_dense_replay_bag.sh ${session_dir}" >&2
  exit 1
fi

container_bag="$(container_path "${bag_path}")"
echo "[play_double_arm_replay] Replay bag:"
echo "  ${bag_path}"

if [[ "${DRY_RUN}" == "true" ]]; then
  docker exec "${CONTAINER_NAME}" bash -lc "
set -euo pipefail
set +u
source /opt/ros/noetic/setup.bash
set -u
rosbag info '${container_bag}'
"
  exit 0
fi

if [[ "${SELECT_ROBOT_CMD_MUX}" == "true" ]]; then
  echo "[play_double_arm_replay] Selecting left/right command route."
  docker exec "${CONTAINER_NAME}" bash -lc "
set -euo pipefail
set +u
source /opt/ros/noetic/setup.bash
if [[ -f '${CONTAINER_ROOT}/workspaces/scripts/use_robot.sh' ]]; then
  source '${CONTAINER_ROOT}/workspaces/scripts/use_robot.sh' >/dev/null || true
fi
set -u
rosservice call /robot/arm_left/joint_cmd_mux_select /robot/arm_left/robot_cmd
rosservice call /robot/arm_right/joint_cmd_mux_select /robot/arm_right/robot_cmd
"
fi

if [[ "${SET_ARBITER_POLICY}" == "true" ]]; then
  echo "[play_double_arm_replay] Requesting arbiter policy mode."
  docker exec "${CONTAINER_NAME}" bash -lc "
set -euo pipefail
set +u
source /opt/ros/noetic/setup.bash
if [[ -f '${CONTAINER_ROOT}/workspaces/scripts/use_robot.sh' ]]; then
  source '${CONTAINER_ROOT}/workspaces/scripts/use_robot.sh' >/dev/null || true
fi
set -u
rostopic pub /intervention/mode_cmd std_msgs/String \"data: 'policy'\" -1
"
fi

if [[ "${WAIT_ENTER}" == "true" ]]; then
  read -r -p "[play_double_arm_replay] Press Enter to rosbag play, Ctrl-C to cancel. " _
fi

quiet_arg=()
if [[ "${ROSBAG_PLAY_QUIET}" == "true" ]]; then
  quiet_arg+=(--quiet)
fi

docker exec -it "${CONTAINER_NAME}" bash -lc "
set -euo pipefail
set +u
source /opt/ros/noetic/setup.bash
if [[ -f '${CONTAINER_ROOT}/workspaces/scripts/use_robot.sh' ]]; then
  source '${CONTAINER_ROOT}/workspaces/scripts/use_robot.sh' >/dev/null || true
fi
set -u
rosbag play '${container_bag}' --rate '${ROSBAG_PLAY_RATE}' ${quiet_arg[*]:-}
"
