#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LFV_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONTAINER_NAME="${CONTAINER_NAME:-ros1_noetic}"
HOST_ROS_ROOT="${HOST_ROS_ROOT:-/home/yannan/workspace/ros1_docker-main}"
CONTAINER_ROOT="${CONTAINER_ROOT:-/workspace/ros1_docker_jinhe}"
HOST_SESSION_ROOT="${HOST_SESSION_ROOT:-${HOST_ROS_ROOT}/rosbag_data/human_teaching_videos}"
HAND_ARM_MAP="${HAND_ARM_MAP:-}"

OUTPUT_HZ="${OUTPUT_HZ:-60}"
REPLAY_TIME_SCALE="${REPLAY_TIME_SCALE:-2.0}"
MAX_GRIPPER_STEP="${MAX_GRIPPER_STEP:-0.002}"
MAX_JOINT_STEP_RAD="${MAX_JOINT_STEP_RAD:-0.0}"
GRIPPER_MIN="${GRIPPER_MIN:-0.0}"
GRIPPER_MAX="${GRIPPER_MAX:-0.09}"
WARN_MAX_JOINT_STEP_RAD="${WARN_MAX_JOINT_STEP_RAD:-0.015}"
HOLD_FINAL_SEC="${HOLD_FINAL_SEC:-0.5}"
PREROLL_HOLD_SEC="${PREROLL_HOLD_SEC:-0.5}"
START_TIME_SEC="${START_TIME_SEC:-1.0}"
RIGHT_TOPIC="${RIGHT_TOPIC:-/robot/arm_right/vla_joint_cmd}"
LEFT_TOPIC="${LEFT_TOPIC:-/robot/arm_left/vla_joint_cmd}"
OVERRIDE_BINARY_GRIPPER="${OVERRIDE_BINARY_GRIPPER:-true}"
GRIPPER_OPEN_COMMAND_M="${GRIPPER_OPEN_COMMAND_M:-0.09}"
GRIPPER_CLOSED_COMMAND_M="${GRIPPER_CLOSED_COMMAND_M:-0.0}"
GRIPPER_OPEN_EFFORT="${GRIPPER_OPEN_EFFORT:-1.0}"
GRIPPER_CLOSED_EFFORT="${GRIPPER_CLOSED_EFFORT:-0.8}"
REBUILD_PLAN="${REBUILD_PLAN:-false}"
OVERWRITE_BAG="${OVERWRITE_BAG:-true}"

usage() {
  cat <<EOF
Usage:
  HAND_ARM_MAP='left:left,right:right' bash local_pipelines/double_hand_pipeline/build_double_arm_dense_replay_bag.sh <demo_name_or_session_dir>

Builds one synchronized offline ROS bag from already-run per-lane IK outputs.
This does not command the robot.

Important env:
  HAND_ARM_MAP=<required comma-separated hand:arm mapping>
  OUTPUT_HZ=${OUTPUT_HZ}
  REPLAY_TIME_SCALE=${REPLAY_TIME_SCALE}
  LEFT_TOPIC=${LEFT_TOPIC}
  RIGHT_TOPIC=${RIGHT_TOPIC}
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

target="${1:-}"
if [[ -z "${target}" ]]; then
  echo "[double_arm_replay] Missing demo name or session dir." >&2
  usage >&2
  exit 1
fi
if [[ -z "${HAND_ARM_MAP}" ]]; then
  echo "[double_arm_replay] HAND_ARM_MAP is required; example: left:left,right:right" >&2
  usage >&2
  exit 1
fi

if [[ "${target}" = /* ]]; then
  session_dir="$(readlink -f "${target}")"
else
  session_dir="$(readlink -f "${HOST_SESSION_ROOT}/${target}")"
fi
if [[ ! -d "${session_dir}" ]]; then
  echo "[double_arm_replay] Session not found: ${session_dir}" >&2
  exit 1
fi
case "${session_dir}" in
  "${HOST_ROS_ROOT}"/*) ;;
  *)
    echo "[double_arm_replay] Session must be under ${HOST_ROS_ROOT}: ${session_dir}" >&2
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
      echo "[double_arm_replay] Cannot map host path into container: ${host_path}" >&2
      exit 1
      ;;
  esac
}

lane_ik_csv() {
  local hand="$1"
  local arm="$2"
  local lane_dir="${session_dir}/quality/double_hand_pipeline/${hand}_hand_${arm}_arm"
  local config_json="${lane_dir}/lane_config.json"
  if [[ ! -f "${config_json}" ]]; then
    echo "[double_arm_replay] Missing lane config: ${config_json}" >&2
    echo "  Run first: HAND=${hand} ARM=${arm} bash ${SCRIPT_DIR}/run_double_hand_pipeline.sh ${session_dir}" >&2
    exit 1
  fi
  local prefix
  prefix="$(python3 - "${config_json}" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    print(json.load(f)["pipeline_prefix"])
PY
)"
  local regularized_csv="${lane_dir}/regularized/gripper_pose_table_core_${prefix}.csv"
  local core_label
  core_label="$(basename "${regularized_csv}" .csv | sed 's/+//g;s/-/m/g;s/ /_/g;s/\./p/g;s/[^A-Za-z0-9_]/_/g')"
  local ik_csv="${session_dir}/quality/urdf_sim/piper_pinocchio_core_absolute_scale_0p25_core_${core_label}_ik_pose_rot6d_relaxed_tcp_priority/piper_pinocchio_core_ik.csv"
  if [[ ! -f "${ik_csv}" ]]; then
    ik_csv="$(find "${session_dir}/quality/urdf_sim" -path "*core_${core_label}_ik_pose_rot6d_relaxed_tcp_priority/piper_pinocchio_core_ik.csv" -print -quit 2>/dev/null || true)"
  fi
  if [[ -z "${ik_csv}" || ! -f "${ik_csv}" ]]; then
    echo "[double_arm_replay] Missing IK CSV for lane hand=${hand} arm=${arm}" >&2
    echo "  Expected near core label: ${core_label}" >&2
    exit 1
  fi
  printf '%s\n' "${ik_csv}"
}

if [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}" 2>/dev/null || true)" != "true" ]]; then
  docker start "${CONTAINER_NAME}" >/dev/null
fi

left_ik_csv=""
right_ik_csv=""
IFS=',' read -r -a lanes <<< "${HAND_ARM_MAP}"
for lane in "${lanes[@]}"; do
  lane="${lane//[[:space:]]/}"
  IFS=':' read -r hand arm extra <<< "${lane}"
  if [[ -z "${hand:-}" || -z "${arm:-}" || -n "${extra:-}" ]]; then
    echo "[double_arm_replay] Bad lane '${lane}', expected <hand>:<arm>." >&2
    exit 1
  fi
  if [[ "${hand}" != "left" && "${hand}" != "right" ]]; then
    echo "[double_arm_replay] Bad hand label '${hand}' in lane '${lane}'." >&2
    exit 1
  fi
  if [[ "${arm}" != "left" && "${arm}" != "right" ]]; then
    echo "[double_arm_replay] Bad arm label '${arm}' in lane '${lane}'." >&2
    exit 1
  fi
  ik_csv="$(lane_ik_csv "${hand}" "${arm}")"
  if [[ "${arm}" == "left" ]]; then
    if [[ -n "${left_ik_csv}" ]]; then
      echo "[double_arm_replay] Duplicate left arm lane in HAND_ARM_MAP=${HAND_ARM_MAP}" >&2
      exit 1
    fi
    left_ik_csv="${ik_csv}"
  else
    if [[ -n "${right_ik_csv}" ]]; then
      echo "[double_arm_replay] Duplicate right arm lane in HAND_ARM_MAP=${HAND_ARM_MAP}" >&2
      exit 1
    fi
    right_ik_csv="${ik_csv}"
  fi
done

if [[ -z "${left_ik_csv}" && -z "${right_ik_csv}" ]]; then
  echo "[double_arm_replay] No arm lanes found in HAND_ARM_MAP=${HAND_ARM_MAP}" >&2
  exit 1
fi

hz_label="$(label_float "${OUTPUT_HZ}")"
time_label="$(label_float "${REPLAY_TIME_SCALE}")"
out_dir="${session_dir}/quality/double_hand_pipeline/dual_arm_replay_${hz_label}hz_timescale_${time_label}"
mkdir -p "${out_dir}"

build_plan() {
  local arm="$1"
  local ik_csv="$2"
  local plan_dir="${out_dir}/${arm}_arm_dense_plan"
  local plan_csv="${plan_dir}/right_arm_dense_replay_plan.csv"
  if [[ "${REBUILD_PLAN}" == "true" || ! -f "${plan_csv}" ]]; then
    docker exec "${CONTAINER_NAME}" bash -lc "
set -euo pipefail
export MPLBACKEND=Agg
cd '${CONTAINER_ROOT}'
python3 workspaces/scripts/lfv_build_right_arm_dense_replay_plan.py \
  --ik-csv '$(container_path "${ik_csv}")' \
  --output-dir '$(container_path "${plan_dir}")' \
  --output-hz '${OUTPUT_HZ}' \
  --time-scale '${REPLAY_TIME_SCALE}' \
  --max-gripper-step '${MAX_GRIPPER_STEP}' \
  --max-joint-step-rad '${MAX_JOINT_STEP_RAD}' \
  --gripper-min '${GRIPPER_MIN}' \
  --gripper-max '${GRIPPER_MAX}' \
  --warn-max-joint-step-rad '${WARN_MAX_JOINT_STEP_RAD}' \
  --hold-final-sec '${HOLD_FINAL_SEC}'
" >&2
  fi
  if [[ ! -f "${plan_csv}" ]]; then
    echo "[double_arm_replay] Missing ${arm} plan after build: ${plan_csv}" >&2
    exit 1
  fi
  printf '%s\n' "${plan_csv}"
}

left_plan_csv=""
right_plan_csv=""
if [[ -n "${left_ik_csv}" ]]; then
  echo "[double_arm_replay] Left IK: ${left_ik_csv}"
  left_plan_csv="$(build_plan left "${left_ik_csv}")"
fi
if [[ -n "${right_ik_csv}" ]]; then
  echo "[double_arm_replay] Right IK: ${right_ik_csv}"
  right_plan_csv="$(build_plan right "${right_ik_csv}")"
fi

bag_path="${out_dir}/double_arm_dense_replay_cmd.bag"
left_arg=()
right_arg=()
if [[ -n "${left_plan_csv}" ]]; then
  left_arg+=(--left-plan-csv "$(container_path "${left_plan_csv}")")
fi
if [[ -n "${right_plan_csv}" ]]; then
  right_arg+=(--right-plan-csv "$(container_path "${right_plan_csv}")")
fi
overwrite_arg=()
if [[ "${OVERWRITE_BAG}" == "true" ]]; then
  overwrite_arg+=(--overwrite)
fi
gripper_override_arg=()
if [[ "${OVERRIDE_BINARY_GRIPPER}" == "true" || "${OVERRIDE_BINARY_GRIPPER}" == "1" || "${OVERRIDE_BINARY_GRIPPER}" == "yes" ]]; then
  gripper_override_arg+=(--override-binary-gripper)
fi

docker exec "${CONTAINER_NAME}" bash -lc "
set -euo pipefail
set +u
source /opt/ros/noetic/setup.bash
set -u
cd '${CONTAINER_ROOT}'
python3 workspaces/scripts/lfv_double_dense_plans_to_rosbag.py \
  ${left_arg[*]:-} \
  ${right_arg[*]:-} \
  --output-bag '$(container_path "${bag_path}")' \
  --left-topic '${LEFT_TOPIC}' \
  --right-topic '${RIGHT_TOPIC}' \
  --gripper-open-command-m '${GRIPPER_OPEN_COMMAND_M}' \
  --gripper-closed-command-m '${GRIPPER_CLOSED_COMMAND_M}' \
  --gripper-open-effort '${GRIPPER_OPEN_EFFORT}' \
  --gripper-closed-effort '${GRIPPER_CLOSED_EFFORT}' \
  --preroll-hold-sec '${PREROLL_HOLD_SEC}' \
  --start-time-sec '${START_TIME_SEC}' \
  ${gripper_override_arg[*]:-} \
  ${overwrite_arg[*]:-}
"

cat <<EOF

[double_arm_replay] Done.
  session_dir: ${session_dir}
  mapping: ${HAND_ARM_MAP}
  left_ik_csv: ${left_ik_csv:-}
  right_ik_csv: ${right_ik_csv:-}
  left_plan_csv: ${left_plan_csv:-}
  right_plan_csv: ${right_plan_csv:-}
  replay_bag: ${bag_path}
  replay_metadata: ${bag_path%.bag}.json
EOF
