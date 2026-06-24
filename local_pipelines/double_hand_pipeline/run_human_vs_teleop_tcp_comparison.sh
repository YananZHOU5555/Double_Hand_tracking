#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LFV_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HOST_ROS_ROOT="${HOST_ROS_ROOT:-/home/yannan/workspace/ros1_docker-main}"
CONTAINER_ROOT="${CONTAINER_ROOT:-/workspace/ros1_docker_jinhe}"
HOST_HUMAN_ROOT="${HOST_HUMAN_ROOT:-${HOST_ROS_ROOT}/rosbag_data/human_teaching_videos}"
HOST_TELEOP_ROOT="${HOST_TELEOP_ROOT:-${HOST_ROS_ROOT}/rosbag_data}"
DOCKER_CONTAINER="${DOCKER_CONTAINER:-ros1_noetic}"

HAND="${HAND:-right}"
ARM="${ARM:-right}"
RUN_DOUBLE_HAND="${RUN_DOUBLE_HAND:-true}"
RUN_FOUNDATIONSTEREO="${RUN_FOUNDATIONSTEREO:-true}"
RUN_IK="${RUN_IK:-false}"
RUN_MUJOCO="${RUN_MUJOCO:-false}"
REBUILD_WILOR="${REBUILD_WILOR:-false}"
REBUILD_DEPTH="${REBUILD_DEPTH:-false}"
ROBOT_SAMPLE_HZ="${ROBOT_SAMPLE_HZ:-30.0}"
AUTO_TRIM_HUMAN="${AUTO_TRIM_HUMAN:-true}"
AUTO_TRIM_ROBOT="${AUTO_TRIM_ROBOT:-true}"
SPEED_THRESHOLD_MPS="${SPEED_THRESHOLD_MPS:-0.015}"
TRIM_PAD_SEC="${TRIM_PAD_SEC:-0.20}"
ROBOT_TOPIC="${ROBOT_TOPIC:-/robot/arm_${ARM}/joint_states_single}"

usage() {
  cat <<EOF
Usage:
  HAND=right ARM=right bash local_pipelines/double_hand_pipeline/run_human_vs_teleop_tcp_comparison.sh <human_session_or_dir> <teleop_episode_or_bag>

Example:
  HAND=right ARM=right bash local_pipelines/double_hand_pipeline/run_human_vs_teleop_tcp_comparison.sh \\
    /home/yannan/workspace/ros1_docker-main/rosbag_data/human_teaching_videos/BAG_20260622_1804_060 \\
    /home/yannan/workspace/ros1_docker-main/rosbag_data/BAG3/episode_008

What it does:
  1. Runs the double-hand LFV pipeline for the selected HAND/ARM lane.
  2. Optionally runs FoundationStereo hand-model QA output.
  3. Converts the teleop robot joint-state bag to TCP in the same table frame.
  4. Writes phase-aligned and DTW-aligned comparison plots/CSV/HTML.

Important env:
  HAND=${HAND}
  ARM=${ARM}
  RUN_DOUBLE_HAND=${RUN_DOUBLE_HAND}
  RUN_FOUNDATIONSTEREO=${RUN_FOUNDATIONSTEREO}
  RUN_IK=${RUN_IK}
  RUN_MUJOCO=${RUN_MUJOCO}
  ROBOT_TOPIC=${ROBOT_TOPIC}
  AUTO_TRIM_HUMAN=${AUTO_TRIM_HUMAN}
  AUTO_TRIM_ROBOT=${AUTO_TRIM_ROBOT}
EOF
}

die() {
  echo "[human_vs_teleop_tcp] ERROR: $*" >&2
  exit 1
}

log() {
  printf '\n[human_vs_teleop_tcp] %s\n' "$*"
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
  find "${dir}" -maxdepth 1 -type f -name 'gripper_pose_table_core_*.csv' -printf '%T@ %p\n' \
    | sort -n \
    | tail -1 \
    | cut -d' ' -f2-
}

json_value() {
  python3 - "$1" "$2" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
key = sys.argv[2]
if not path.exists():
    raise SystemExit(0)
data = json.loads(path.read_text(encoding="utf-8"))
value = data
for part in key.split("."):
    if isinstance(value, dict):
        value = value.get(part)
    else:
        value = None
        break
if value is None:
    raise SystemExit(0)
if isinstance(value, (dict, list)):
    print(json.dumps(value, ensure_ascii=False))
else:
    print(value)
PY
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
if [[ "${HAND}" != "left" && "${HAND}" != "right" ]]; then
  die "HAND must be left or right; got ${HAND}"
fi
if [[ "${ARM}" != "left" && "${ARM}" != "right" ]]; then
  die "ARM must be left or right; got ${ARM}"
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
output_dir="${human_session_dir}/quality/human_vs_teleop_tcp/${teleop_session}_${teleop_episode}_${ARM}"
mkdir -p "${output_dir}"

if [[ "${ARM}" == "left" ]]; then
  robot_table_json="${CONTAINER_ROOT}/data/lfv_calibration/left_arm_table_latest.json"
  tcp_json="${CONTAINER_ROOT}/data/lfv_calibration/left_arm_touch_tcp_zr2_latest.json"
else
  robot_table_json="${CONTAINER_ROOT}/data/lfv_calibration/right_arm_table_latest.json"
  tcp_json="${CONTAINER_ROOT}/data/lfv_calibration/right_arm_touch_tcp_xr2_latest.json"
fi

log "human session: ${human_session_dir}"
log "teleop bag:    ${teleop_bag}"
log "HAND=${HAND} ARM=${ARM}"
log "robot topic:   ${ROBOT_TOPIC}"
log "output:        ${output_dir}"

if [[ "${RUN_DOUBLE_HAND}" == "true" || "${RUN_DOUBLE_HAND}" == "1" ]]; then
  log "1/4 run teammate double-hand pipeline to produce table-frame human TCP"
  HAND="${HAND}" \
  ARM="${ARM}" \
  RUN_IK="${RUN_IK}" \
  RUN_MUJOCO="${RUN_MUJOCO}" \
  REBUILD_WILOR="${REBUILD_WILOR}" \
  bash "${SCRIPT_DIR}/run_double_hand_pipeline.sh" "${human_session_dir}"
else
  log "1/4 skip double-hand pipeline by RUN_DOUBLE_HAND=${RUN_DOUBLE_HAND}"
fi

if [[ "${RUN_FOUNDATIONSTEREO}" == "true" || "${RUN_FOUNDATIONSTEREO}" == "1" ]]; then
  log "2/4 run FoundationStereo hand-model QA from teammate pipeline"
  REBUILD_WILOR="${REBUILD_WILOR}" \
  REBUILD_DEPTH="${REBUILD_DEPTH}" \
  bash "${SCRIPT_DIR}/run_foundationstereo_hand_model_trim20.sh" "${human_session_dir}"
else
  log "2/4 skip FoundationStereo QA by RUN_FOUNDATIONSTEREO=${RUN_FOUNDATIONSTEREO}"
fi

regularized_dir="${human_session_dir}/quality/double_hand_pipeline/${HAND}_hand_${ARM}_arm/regularized"
human_tcp_csv="$(latest_regularized_csv "${regularized_dir}" || true)"
[[ -n "${human_tcp_csv}" && -f "${human_tcp_csv}" ]] || die "cannot find human regularized TCP CSV under ${regularized_dir}"

log "3/4 compare TCP trajectories in table frame"
trim_human_arg=()
trim_robot_arg=()
if [[ "${AUTO_TRIM_HUMAN}" == "true" || "${AUTO_TRIM_HUMAN}" == "1" ]]; then
  trim_human_arg+=(--auto-trim-human)
fi
if [[ "${AUTO_TRIM_ROBOT}" == "true" || "${AUTO_TRIM_ROBOT}" == "1" ]]; then
  trim_robot_arg+=(--auto-trim-robot)
fi

container_human_tcp_csv="$(container_path "${human_tcp_csv}")"
container_teleop_bag="$(container_path "${teleop_bag}")"
container_output_dir="$(container_path "${output_dir}")"

docker exec "${DOCKER_CONTAINER}" bash -lc "
set -e
source '${CONTAINER_ROOT}/workspaces/scripts/use_robot.sh' >/dev/null
python3 '${CONTAINER_ROOT}/workspaces/scripts/compare_human_teleop_tcp_table.py' \\
  --human-tcp-csv '${container_human_tcp_csv}' \\
  --teleop-bag '${container_teleop_bag}' \\
  --arm '${ARM}' \\
  --robot-topic '${ROBOT_TOPIC}' \\
  --robot-table-json '${robot_table_json}' \\
  --tcp-json '${tcp_json}' \\
  --output-dir '${container_output_dir}' \\
  --robot-sample-hz '${ROBOT_SAMPLE_HZ}' \\
  --speed-threshold-mps '${SPEED_THRESHOLD_MPS}' \\
  --trim-pad-sec '${TRIM_PAD_SEC}' \\
  ${trim_human_arg[*]:-} \\
  ${trim_robot_arg[*]:-}
"

summary_json="${output_dir}/human_vs_teleop_tcp_table_summary.json"
html="${output_dir}/human_vs_teleop_tcp_table.html"
phase_median="$(json_value "${summary_json}" "distance_phase_m.norm.median" || true)"
dtw_median="$(json_value "${summary_json}" "distance_dtw_m.norm.median" || true)"

log "4/4 done"
cat <<EOF
  human_tcp_csv: ${human_tcp_csv}
  summary_json:  ${summary_json}
  html:          ${html}
  phase median distance m: ${phase_median:-unknown}
  DTW median distance m:   ${dtw_median:-unknown}
EOF
