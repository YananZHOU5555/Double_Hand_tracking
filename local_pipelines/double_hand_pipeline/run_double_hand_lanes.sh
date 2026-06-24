#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_ROS_ROOT="${HOST_ROS_ROOT:-/home/yannan/workspace/ros1_docker-main}"
HOST_SESSION_ROOT="${HOST_SESSION_ROOT:-${HOST_ROS_ROOT}/rosbag_data/human_teaching_videos}"
HAND_ARM_MAP="${HAND_ARM_MAP:-}"

usage() {
  cat <<EOF
Usage:
  HAND_ARM_MAP='left:left,right:right' bash local_pipelines/double_hand_pipeline/run_double_hand_lanes.sh <demo_name_or_session_dir>

Runs multiple explicit hand/arm lanes for one session. The mapping is intentionally
external: every entry is <hand_identity_label>:<robot_arm_label>.

Examples:
  HAND_ARM_MAP='left:left,right:right' bash local_pipelines/double_hand_pipeline/run_double_hand_lanes.sh BAG_20260622_1804_001
  HAND_ARM_MAP='left:right,right:left' bash local_pipelines/double_hand_pipeline/run_double_hand_lanes.sh BAG_20260622_1804_001
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

target="${1:-}"
if [[ -z "${target}" ]]; then
  echo "[double_hand_lanes] Missing demo name or session dir." >&2
  usage >&2
  exit 1
fi
if [[ -z "${HAND_ARM_MAP}" ]]; then
  echo "[double_hand_lanes] HAND_ARM_MAP is required; example: left:left,right:right" >&2
  usage >&2
  exit 1
fi

if [[ "${target}" = /* ]]; then
  session_dir="$(readlink -f "${target}")"
else
  session_dir="$(readlink -f "${HOST_SESSION_ROOT}/${target}")"
fi
if [[ ! -d "${session_dir}" ]]; then
  echo "[double_hand_lanes] Session not found: ${session_dir}" >&2
  exit 1
fi

IFS=',' read -r -a lanes <<< "${HAND_ARM_MAP}"
if [[ "${#lanes[@]}" -eq 0 ]]; then
  echo "[double_hand_lanes] HAND_ARM_MAP produced no lanes: ${HAND_ARM_MAP}" >&2
  exit 1
fi

ran_lanes=()
for lane in "${lanes[@]}"; do
  lane="${lane//[[:space:]]/}"
  IFS=':' read -r hand arm extra <<< "${lane}"
  if [[ -z "${hand:-}" || -z "${arm:-}" || -n "${extra:-}" ]]; then
    echo "[double_hand_lanes] Bad lane '${lane}', expected <hand>:<arm>." >&2
    exit 1
  fi
  if [[ "${hand}" != "left" && "${hand}" != "right" ]]; then
    echo "[double_hand_lanes] Bad hand label '${hand}' in lane '${lane}'." >&2
    exit 1
  fi
  if [[ "${arm}" != "left" && "${arm}" != "right" ]]; then
    echo "[double_hand_lanes] Bad arm label '${arm}' in lane '${lane}'." >&2
    exit 1
  fi

  printf '\n[double_hand_lanes] Running lane: hand=%s arm=%s session=%s\n' "${hand}" "${arm}" "${session_dir}"
  HAND="${hand}" ARM="${arm}" HOST_SESSION_ROOT="${HOST_SESSION_ROOT}" \
    bash "${SCRIPT_DIR}/run_double_hand_pipeline.sh" "${session_dir}"
  ran_lanes+=("${hand}_hand_${arm}_arm")
done

cat <<EOF

[double_hand_lanes] Done.
  session_dir: ${session_dir}
  mapping: ${HAND_ARM_MAP}
EOF
for lane_name in "${ran_lanes[@]}"; do
  cat <<EOF
  ${lane_name}: ${session_dir}/quality/double_hand_pipeline/${lane_name}
EOF
done
