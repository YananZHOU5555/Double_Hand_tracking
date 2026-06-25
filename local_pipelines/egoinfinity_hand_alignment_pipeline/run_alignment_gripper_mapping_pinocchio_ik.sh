#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LFV_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HOST_ROS_ROOT="${HOST_ROS_ROOT:-/home/yannan/workspace/ros1_docker-main}"
HOST_SESSION_ROOT="${HOST_SESSION_ROOT:-${HOST_ROS_ROOT}/rosbag_data/human_teaching_videos}"
CONTAINER_NAME="${CONTAINER_NAME:-ros1_noetic}"
CONTAINER_ROOT="${CONTAINER_ROOT:-/workspace/ros1_docker_jinhe}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
PYROKI_PYTHON="${PYROKI_PYTHON:-/home/yannan/workspace/.venvs/pyroki/bin/python}"

SIDE="${SIDE:-right}"
GRIPPER_MODE="${GRIPPER_MODE:-half_width}"
CORE_FPS="${CORE_FPS:-60}"
RUN_PREIK_HTML="${RUN_PREIK_HTML:-true}"
RUN_IK="${RUN_IK:-true}"  # Backward-compatible alias for RUN_PINOCCHIO_IK.
RUN_PINOCCHIO_IK="${RUN_PINOCCHIO_IK:-${RUN_IK}}"
RUN_PYROKI_IK="${RUN_PYROKI_IK:-false}"

SOURCE_STAGE_NAME="${SOURCE_STAGE_NAME:-phase_c3_mesh_visibility_anchor_locked_patch2_pose_repaired}"
SOURCE_NPZ="${SOURCE_NPZ:-}"
JOINTS_KEY="${JOINTS_KEY:-joints_cam_anchor_locked_smooth}"
UV_KEY="${UV_KEY:-joints_uv_anchor_locked_smooth}"

PHASE_D_STAGE_NAME="${PHASE_D_STAGE_NAME:-phase_d_preik_anchor_locked_patch2_pose_repaired}"
PHASE_E_STAGE_NAME="${PHASE_E_STAGE_NAME:-phase_e_piper_gripper_base_ik_input_anchor_locked_patch2_pose_repaired}"
IK_STAGE_NAME="${IK_STAGE_NAME:-}"

ROBOT_TABLE_JSON="${ROBOT_TABLE_JSON:-${CONTAINER_ROOT}/data/lfv_calibration/right_arm_table_latest.json}"
HOST_ROBOT_TABLE_JSON="${HOST_ROBOT_TABLE_JSON:-${HOST_ROS_ROOT}/data/lfv_calibration/right_arm_table_latest.json}"
PIPER_URDF="${PIPER_URDF:-${CONTAINER_ROOT}/piper_ros/src/piper_description/urdf/piper_description.urdf}"
HOST_PIPER_URDF="${HOST_PIPER_URDF:-${HOST_ROS_ROOT}/piper_ros/src/piper_description/urdf/piper_description.urdf}"
TARGET_MODE="${TARGET_MODE:-absolute}"
SCALE="${SCALE:-1.0}"
TRAJECTORY_FILTER="${TRAJECTORY_FILTER:-raw}"
HAMPEL_WINDOW_FRAMES="${HAMPEL_WINDOW_FRAMES:-11}"
HAMPEL_THRESHOLD_M="${HAMPEL_THRESHOLD_M:-0.012}"
SMOOTH_WINDOW_FRAMES="${SMOOTH_WINDOW_FRAMES:-17}"
MAX_TARGET_SPEED_MPS="${MAX_TARGET_SPEED_MPS:-0.35}"
INITIAL_Q="${INITIAL_Q:-}"
ANCHOR_XYZ="${ANCHOR_XYZ:-}"
TARGET_BIAS_XYZ="${TARGET_BIAS_XYZ:-}"
TARGET_COMPENSATION_JSON="${TARGET_COMPENSATION_JSON:-}"
TARGET_COMPENSATION_ALPHA="${TARGET_COMPENSATION_ALPHA:-1.0}"
MIN_ROBOT_Z_M="${MIN_ROBOT_Z_M:-0.06}"
DISABLE_Z_CLAMP="${DISABLE_Z_CLAMP:-false}"
TCP_OFFSET_XYZ="${TCP_OFFSET_XYZ:-0,0,0}"
TCP_OFFSET_Z_M="${TCP_OFFSET_Z_M:-0.13149316740823477}"
SIM_FPS="${SIM_FPS:-30}"
RENDER_STRIDE="${RENDER_STRIDE:-2}"
PRESERVE_GRIPPER_COMMAND="${PRESERVE_GRIPPER_COMMAND:-true}"
GRIPPER_COMMAND_MODE="${GRIPPER_COMMAND_MODE:-half_open_width}"
IK_MODE="${IK_MODE:-position}"
ORIENTATION_SOURCE="${ORIENTATION_SOURCE:-none}"
FIXED_TARGET_QUAT_XYZW="${FIXED_TARGET_QUAT_XYZW:-}"
ORIENTATION_ALIGN_RPY="${ORIENTATION_ALIGN_RPY:-0,0,0}"
ORI_TOL_RAD="${ORI_TOL_RAD:-0.25}"
ORIENTATION_WEIGHT="${ORIENTATION_WEIGHT:-0.15}"
MAX_ITERS="${MAX_ITERS:-350}"
POS_TOL_M="${POS_TOL_M:-0.0001}"
MAX_POS_ERROR_M="${MAX_POS_ERROR_M:-0.005}"
DAMPING="${DAMPING:-0.0001}"
DT="${DT:-0.60}"
SOLVER_STEP_MAX_RAD="${SOLVER_STEP_MAX_RAD:-0.35}"
NORMAL_LIMIT_MARGIN_RAD="${NORMAL_LIMIT_MARGIN_RAD:-0.035}"
NORMAL_ACTIVE_THRESHOLD_RAD="${NORMAL_ACTIVE_THRESHOLD_RAD:-0.020}"
NORMAL_SMOOTH_WEIGHT="${NORMAL_SMOOTH_WEIGHT:-0.01}"
NORMAL_LIMIT_WEIGHT="${NORMAL_LIMIT_WEIGHT:-0.02}"
FAILED_FRAME_STRATEGY="${FAILED_FRAME_STRATEGY:-interpolate}"
RELAXED_IK_FALLBACK="${RELAXED_IK_FALLBACK:-tcp_priority}"
RELAXED_MAX_POS_ERROR_M="${RELAXED_MAX_POS_ERROR_M:-0.020}"
RELAXED_MAX_ITERS="${RELAXED_MAX_ITERS:-250}"
RELAXED_LIMIT_MARGIN_RAD="${RELAXED_LIMIT_MARGIN_RAD:-0.035}"
RELAXED_ACTIVE_THRESHOLD_RAD="${RELAXED_ACTIVE_THRESHOLD_RAD:-0.020}"
RELAXED_ORIENTATION_WEIGHT="${RELAXED_ORIENTATION_WEIGHT:-0.05}"
RELAXED_SMOOTH_WEIGHT="${RELAXED_SMOOTH_WEIGHT:-0.02}"
RELAXED_LIMIT_WEIGHT="${RELAXED_LIMIT_WEIGHT:-0.02}"
PYROKI_STAGE_NAME="${PYROKI_STAGE_NAME:-}"
PYROKI_START_INDEX="${PYROKI_START_INDEX:-0}"
PYROKI_MAX_FRAMES="${PYROKI_MAX_FRAMES:-0}"
PYROKI_TARGET_LINK="${PYROKI_TARGET_LINK:-gripper_base}"
PYROKI_POS_WEIGHT="${PYROKI_POS_WEIGHT:-1.0}"
PYROKI_ORI_WEIGHT="${PYROKI_ORI_WEIGHT:-${ORIENTATION_WEIGHT}}"
PYROKI_REST_WEIGHT="${PYROKI_REST_WEIGHT:-0.01}"
PYROKI_LOCKED_JOINT_REST_WEIGHT="${PYROKI_LOCKED_JOINT_REST_WEIGHT:-10.0}"
PYROKI_LIMIT_CENTER_WEIGHT="${PYROKI_LIMIT_CENTER_WEIGHT:-0.0}"
PYROKI_LIMIT_CONSTRAINT_WEIGHT="${PYROKI_LIMIT_CONSTRAINT_WEIGHT:-1.0}"
PYROKI_MANIPULABILITY_WEIGHT="${PYROKI_MANIPULABILITY_WEIGHT:-0.0}"
PYROKI_RETRY_RANDOM_SEEDS="${PYROKI_RETRY_RANDOM_SEEDS:-0}"
PYROKI_SEED_MODE="${PYROKI_SEED_MODE:-failure_retry}"
PYROKI_SEED_RNG="${PYROKI_SEED_RNG:-17}"
PYROKI_SEED_SCORE_ORI_WEIGHT="${PYROKI_SEED_SCORE_ORI_WEIGHT:-0.02}"
PYROKI_SEED_SCORE_JOINT_WEIGHT="${PYROKI_SEED_SCORE_JOINT_WEIGHT:-0.001}"
PYROKI_SEED_SCORE_MANIP_WEIGHT="${PYROKI_SEED_SCORE_MANIP_WEIGHT:-0.001}"
PYROKI_JAX_PLATFORM="${PYROKI_JAX_PLATFORM:-}"
PYROKI_DISABLE_POSE_POSITION_CANDIDATE="${PYROKI_DISABLE_POSE_POSITION_CANDIDATE:-false}"

usage() {
  cat <<EOF
Usage:
  bash local_pipelines/egoinfinity_hand_alignment_pipeline/run_alignment_gripper_mapping_pinocchio_ik.sh <demo_name_or_session_dir>

This consumes the repaired alignment C3 hand result, exports EgoInfinity-style
gripper-base pre-IK targets, and optionally runs one or both IK backends:
Pinocchio DLS and PyRoki.

Key defaults:
  SOURCE_STAGE_NAME=${SOURCE_STAGE_NAME}
  JOINTS_KEY=${JOINTS_KEY}
  UV_KEY=${UV_KEY}
  SIDE=${SIDE}
  TCP_OFFSET_XYZ=${TCP_OFFSET_XYZ}     # gripper-base origin, not pinch/TCP offset
  IK_MODE=${IK_MODE}
  ORIENTATION_SOURCE=${ORIENTATION_SOURCE}
  RUN_PINOCCHIO_IK=${RUN_PINOCCHIO_IK}   # RUN_IK is kept as an alias
  RUN_PYROKI_IK=${RUN_PYROKI_IK}

Useful overrides:
  RUN_PINOCCHIO_IK=false RUN_PYROKI_IK=true bash ... <session>
  RUN_PINOCCHIO_IK=false RUN_PYROKI_IK=false bash ... <session>
  IK_MODE=position ORIENTATION_SOURCE=none bash ... <session>
  ORIENTATION_ALIGN_RPY=0,0,1.570796 bash ... <session>
EOF
}

timer_now_ns() {
  date +%s%N
}

timer_duration_sec() {
  python3 - "$1" "$2" <<'PY'
import sys
start = int(sys.argv[1])
end = int(sys.argv[2])
print(f"{max(0, end - start) / 1e9:.6f}")
PY
}

record_stage_timing() {
  local stage="$1"
  local status="$2"
  local start_ns="$3"
  local end_ns="$4"
  local exit_code="${5:-0}"
  local duration
  duration="$(timer_duration_sec "${start_ns}" "${end_ns}")"
  printf '{"stage":"%s","status":"%s","duration_sec":%.6f,"exit_code":%d,"started_ns":%s,"ended_ns":%s}\n' \
    "${stage}" "${status}" "${duration}" "${exit_code}" "${start_ns}" "${end_ns}" >> "${downstream_timing_jsonl}"
}

run_stage() {
  local stage="$1"
  shift
  local start_ns end_ns status rc
  start_ns="$(timer_now_ns)"
  set +e
  "$@"
  rc=$?
  set -e
  end_ns="$(timer_now_ns)"
  if [[ "${rc}" -eq 0 ]]; then
    status="ok"
  else
    status="failed"
  fi
  record_stage_timing "${stage}" "${status}" "${start_ns}" "${end_ns}" "${rc}"
  return "${rc}"
}

record_skip_stage() {
  local now
  now="$(timer_now_ns)"
  record_stage_timing "$1" "skip" "${now}" "${now}" 0
}

write_timing_summary() {
  python3 - "${downstream_timing_jsonl}" "${downstream_timing_summary}" <<'PY'
import json
import sys
from pathlib import Path

jsonl = Path(sys.argv[1])
out = Path(sys.argv[2])
records = []
if jsonl.exists():
    for line in jsonl.read_text().splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
status_counts = {}
for r in records:
    status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
out.write_text(json.dumps({
    "timing_jsonl": str(jsonl),
    "stage_count": len(records),
    "total_ok_sec": sum(float(r.get("duration_sec", 0.0)) for r in records if r.get("status") == "ok"),
    "status_counts": status_counts,
    "stages": records,
}, indent=2, ensure_ascii=False) + "\n")
PY
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

target="${1:-}"
if [[ -z "${target}" ]]; then
  echo "[alignment_gripper_ik] Missing demo name or session dir." >&2
  usage >&2
  exit 1
fi

if [[ "${target}" = /* ]]; then
  session_dir="$(readlink -f "${target}")"
else
  session_dir="$(readlink -f "${HOST_SESSION_ROOT}/${target}")"
fi

if [[ ! -d "${session_dir}" ]]; then
  echo "[alignment_gripper_ik] Session not found: ${session_dir}" >&2
  exit 1
fi

case "${session_dir}" in
  "${HOST_ROS_ROOT}"/*) ;;
  *)
    echo "[alignment_gripper_ik] Session must be under ${HOST_ROS_ROOT} so the container can see it." >&2
    echo "  got: ${session_dir}" >&2
    exit 1
    ;;
esac

if [[ "${SIDE}" != "left" && "${SIDE}" != "right" ]]; then
  echo "[alignment_gripper_ik] SIDE must be left or right; got ${SIDE}" >&2
  exit 1
fi

pipeline_dir="${session_dir}/quality/egoinfinity_hand_alignment_pipeline"
downstream_timing_jsonl="${pipeline_dir}/downstream_gripper_ik_timing.jsonl"
downstream_timing_summary="${pipeline_dir}/downstream_gripper_ik_timing_summary.json"
: > "${downstream_timing_jsonl}"
trap 'write_timing_summary >/dev/null 2>&1 || true' EXIT
source_npz="${SOURCE_NPZ}"
if [[ -z "${source_npz}" ]]; then
  source_npz="${pipeline_dir}/stages/${SOURCE_STAGE_NAME}/wilor_handresults_phase_c3_mesh_visibility.npz"
fi
source_npz="$(readlink -f "${source_npz}")"
if [[ ! -f "${source_npz}" ]]; then
  echo "[alignment_gripper_ik] Missing repaired C3 NPZ: ${source_npz}" >&2
  exit 1
fi

phase_d_dir="${pipeline_dir}/stages/${PHASE_D_STAGE_NAME}"
phase_e_dir="${pipeline_dir}/stages/${PHASE_E_STAGE_NAME}"
quality_dir="${phase_e_dir}/quality_check"
mkdir -p "${phase_d_dir}" "${phase_e_dir}" "${quality_dir}"

scale_label="$(printf '%g' "${SCALE}" | sed 's/-/m/g;s/\./p/g')"
if [[ -z "${IK_STAGE_NAME}" ]]; then
  IK_STAGE_NAME="phase_e_pinocchio_ik_${SIDE}_gripper_base_${TARGET_MODE}_s${scale_label}_${IK_MODE}_${ORIENTATION_SOURCE}"
fi
ik_output_dir="${pipeline_dir}/stages/${IK_STAGE_NAME}"

echo "[alignment_gripper_ik] Session: ${session_dir}"
echo "[alignment_gripper_ik] Source C3 NPZ: ${source_npz}"
echo "[alignment_gripper_ik] Phase-D joints key: ${JOINTS_KEY}; uv key: ${UV_KEY}"
echo "[alignment_gripper_ik] Target semantic: ${SIDE} robot gripper-base origin; TCP_OFFSET_XYZ=${TCP_OFFSET_XYZ}"

run_stage "phase_d_egoinfinity_preik" \
  "${PYTHON_BIN}" "${LFV_ROOT}/local_pipelines/egoinfinity_hand_pipeline/phase_d_egoinfinity_preik/build_egoinfinity_preik.py" \
  --session-dir "${session_dir}" \
  --input-npz "${source_npz}" \
  --output-dir "${phase_d_dir}" \
  --joints-key "${JOINTS_KEY}" \
  --uv-key "${UV_KEY}"

run_stage "phase_e_piper_gripper_base_csv" \
  "${PYTHON_BIN}" "${LFV_ROOT}/local_pipelines/egoinfinity_hand_pipeline/phase_e_piper_gripper_base_ik/export_phase_d_to_piper_core_csv.py" \
  --session-dir "${session_dir}" \
  --preik-npz "${phase_d_dir}/preik_targets.npz" \
  --output-dir "${phase_e_dir}" \
  --side "${SIDE}" \
  --core-fps "${CORE_FPS}" \
  --gripper-mode "${GRIPPER_MODE}"

gripper_csv="${phase_e_dir}/phase_d_${SIDE}_gripper_base_core.csv"
if [[ ! -f "${gripper_csv}" ]]; then
  echo "[alignment_gripper_ik] Missing Phase-E CSV: ${gripper_csv}" >&2
  exit 1
fi

if [[ "${RUN_PREIK_HTML}" == "true" || "${RUN_PREIK_HTML}" == "1" ]]; then
  for target_frame in table robot; do
    run_stage "phase_e_preik_${target_frame}_html" \
      "${PYTHON_BIN}" "${LFV_ROOT}/local_pipelines/egoinfinity_hand_pipeline/phase_e_piper_gripper_base_ik/export_preik_wrist6d_pose_html.py" \
      --session-dir "${session_dir}" \
      --input-csv "${gripper_csv}" \
      --output-html "${quality_dir}/phase_d_${SIDE}_preik_wrist6d_${target_frame}_3d.html" \
      --output-json "${quality_dir}/phase_d_${SIDE}_preik_wrist6d_${target_frame}_3d.json" \
      --side "${SIDE}" \
      --target-frame "${target_frame}" \
      --robot-table-json "${HOST_ROBOT_TABLE_JSON}" \
      --title-prefix "alignment ${SOURCE_STAGE_NAME} "
  done
else
  record_skip_stage "phase_e_preik_table_html"
  record_skip_stage "phase_e_preik_robot_html"
fi

if [[ "${RUN_PINOCCHIO_IK}" != "true" && "${RUN_PINOCCHIO_IK}" != "1" && "${RUN_PYROKI_IK}" != "true" && "${RUN_PYROKI_IK}" != "1" ]]; then
  echo "[alignment_gripper_ik] RUN_PINOCCHIO_IK=false and RUN_PYROKI_IK=false; stopped after gripper mapping."
  echo "[alignment_gripper_ik] Phase-D: ${phase_d_dir}"
  echo "[alignment_gripper_ik] Phase-E: ${phase_e_dir}"
  write_timing_summary
  echo "[alignment_gripper_ik] timing_jsonl: ${downstream_timing_jsonl}"
  echo "[alignment_gripper_ik] timing_summary_json: ${downstream_timing_summary}"
  exit 0
fi

z_clamp_arg=()
if [[ "${DISABLE_Z_CLAMP}" == "true" ]]; then
  z_clamp_arg+=(--disable-z-clamp)
fi
preserve_gripper_arg=()
if [[ "${PRESERVE_GRIPPER_COMMAND}" == "true" ]]; then
  preserve_gripper_arg+=(--preserve-gripper-command)
fi

if [[ "${RUN_PINOCCHIO_IK}" == "true" || "${RUN_PINOCCHIO_IK}" == "1" ]]; then
  if [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}" 2>/dev/null || true)" != "true" ]]; then
    docker start "${CONTAINER_NAME}" >/dev/null
  fi

  container_session_dir="${session_dir/#${HOST_ROS_ROOT}/${CONTAINER_ROOT}}"
  container_gripper_csv="${gripper_csv/#${HOST_ROS_ROOT}/${CONTAINER_ROOT}}"
  container_ik_output_dir="${ik_output_dir/#${HOST_ROS_ROOT}/${CONTAINER_ROOT}}"
  container_target_compensation_json="${TARGET_COMPENSATION_JSON}"
  if [[ -n "${TARGET_COMPENSATION_JSON}" && "${TARGET_COMPENSATION_JSON}" = "${HOST_ROS_ROOT}"/* ]]; then
    container_target_compensation_json="${TARGET_COMPENSATION_JSON/#${HOST_ROS_ROOT}/${CONTAINER_ROOT}}"
  fi

  echo "[alignment_gripper_ik] Running original Pinocchio IK in ${CONTAINER_NAME}"
  echo "[alignment_gripper_ik] Pinocchio IK output: ${ik_output_dir}"

  run_stage "phase_e_pinocchio_ik" \
    docker exec "${CONTAINER_NAME}" bash -lc "
set -euo pipefail
set +u
source /opt/ros/noetic/setup.bash
if [[ -f '${CONTAINER_ROOT}/workspaces/scripts/use_robot.sh' ]]; then
  source '${CONTAINER_ROOT}/workspaces/scripts/use_robot.sh' >/dev/null || true
fi
set -u
export PYTHONPATH='/opt/ros/noetic/lib/python3.8/site-packages:'\"\${PYTHONPATH:-}\"
export LD_LIBRARY_PATH='/opt/ros/noetic/lib:/opt/ros/noetic/lib/x86_64-linux-gnu:'\"\${LD_LIBRARY_PATH:-}\"
export MPLBACKEND=Agg
cd '${CONTAINER_ROOT}'
python3 workspaces/scripts/lfv_simulate_piper_core_gripper_pinocchio.py \
  --session-dir '${container_session_dir}' \
  --gripper-core-csv '${container_gripper_csv}' \
  --robot-table-json '${ROBOT_TABLE_JSON}' \
  --piper-urdf '${PIPER_URDF}' \
  --output-dir '${container_ik_output_dir}' \
  --target-mode '${TARGET_MODE}' \
  --scale '${SCALE}' \
  --trajectory-filter '${TRAJECTORY_FILTER}' \
  --hampel-window-frames '${HAMPEL_WINDOW_FRAMES}' \
  --hampel-threshold-m '${HAMPEL_THRESHOLD_M}' \
  --smooth-window-frames '${SMOOTH_WINDOW_FRAMES}' \
  --max-target-speed-mps '${MAX_TARGET_SPEED_MPS}' \
  --initial-q='${INITIAL_Q}' \
  --anchor-xyz='${ANCHOR_XYZ}' \
  --target-bias-xyz='${TARGET_BIAS_XYZ}' \
  --target-compensation-json '${container_target_compensation_json}' \
  --target-compensation-alpha '${TARGET_COMPENSATION_ALPHA}' \
  --min-robot-z-m '${MIN_ROBOT_Z_M}' \
  --tcp-offset-z-m '${TCP_OFFSET_Z_M}' \
  --tcp-offset-xyz '${TCP_OFFSET_XYZ}' \
  --sim-fps '${SIM_FPS}' \
  --render-stride '${RENDER_STRIDE}' \
  --gripper-command-mode '${GRIPPER_COMMAND_MODE}' \
  --ik-mode '${IK_MODE}' \
  --orientation-source '${ORIENTATION_SOURCE}' \
  --fixed-target-quat-xyzw='${FIXED_TARGET_QUAT_XYZW}' \
  --orientation-align-rpy '${ORIENTATION_ALIGN_RPY}' \
  --ori-tol-rad '${ORI_TOL_RAD}' \
  --orientation-weight '${ORIENTATION_WEIGHT}' \
  --max-iters '${MAX_ITERS}' \
  --pos-tol-m '${POS_TOL_M}' \
  --max-pos-error-m '${MAX_POS_ERROR_M}' \
  --damping '${DAMPING}' \
  --dt '${DT}' \
  --solver-step-max-rad '${SOLVER_STEP_MAX_RAD}' \
  --normal-limit-margin-rad '${NORMAL_LIMIT_MARGIN_RAD}' \
  --normal-active-threshold-rad '${NORMAL_ACTIVE_THRESHOLD_RAD}' \
  --normal-smooth-weight '${NORMAL_SMOOTH_WEIGHT}' \
  --normal-limit-weight '${NORMAL_LIMIT_WEIGHT}' \
  --failed-frame-strategy '${FAILED_FRAME_STRATEGY}' \
  --relaxed-ik-fallback '${RELAXED_IK_FALLBACK}' \
  --relaxed-max-pos-error-m '${RELAXED_MAX_POS_ERROR_M}' \
  --relaxed-max-iters '${RELAXED_MAX_ITERS}' \
  --relaxed-limit-margin-rad '${RELAXED_LIMIT_MARGIN_RAD}' \
  --relaxed-active-threshold-rad '${RELAXED_ACTIVE_THRESHOLD_RAD}' \
  --relaxed-orientation-weight '${RELAXED_ORIENTATION_WEIGHT}' \
  --relaxed-smooth-weight '${RELAXED_SMOOTH_WEIGHT}' \
  --relaxed-limit-weight '${RELAXED_LIMIT_WEIGHT}' \
  ${preserve_gripper_arg[*]:-} \
  ${z_clamp_arg[*]:-}
"
else
  echo "[alignment_gripper_ik] RUN_PINOCCHIO_IK=false; skipping Pinocchio IK."
  record_skip_stage "phase_e_pinocchio_ik"
fi

pyroki_output_dir=""
if [[ "${RUN_PYROKI_IK}" == "true" || "${RUN_PYROKI_IK}" == "1" ]]; then
  if [[ ! -x "${PYROKI_PYTHON}" ]]; then
    echo "[alignment_gripper_ik] PyRoki python not executable: ${PYROKI_PYTHON}" >&2
    echo "[alignment_gripper_ik] Create it with phase_f_pyroki_ik_backend/install_pyroki_env.sh." >&2
    exit 1
  fi
  if [[ ! -f "${HOST_ROBOT_TABLE_JSON}" ]]; then
    echo "[alignment_gripper_ik] HOST_ROBOT_TABLE_JSON not found: ${HOST_ROBOT_TABLE_JSON}" >&2
    exit 1
  fi
  if [[ ! -f "${HOST_PIPER_URDF}" ]]; then
    echo "[alignment_gripper_ik] HOST_PIPER_URDF not found: ${HOST_PIPER_URDF}" >&2
    exit 1
  fi

  if [[ -z "${PYROKI_STAGE_NAME}" ]]; then
    PYROKI_STAGE_NAME="phase_f_pyroki_ik_${SIDE}_gripper_base_${TARGET_MODE}_s${scale_label}_${IK_MODE}_${ORIENTATION_SOURCE}"
  fi
  pyroki_output_dir="${pipeline_dir}/stages/${PYROKI_STAGE_NAME}"
  pose_guard_arg=()
  if [[ "${PYROKI_DISABLE_POSE_POSITION_CANDIDATE}" == "true" || "${PYROKI_DISABLE_POSE_POSITION_CANDIDATE}" == "1" ]]; then
    pose_guard_arg+=(--disable-pose-position-candidate)
  fi

  echo "[alignment_gripper_ik] Running PyRoki IK on host"
  echo "[alignment_gripper_ik] PyRoki IK output: ${pyroki_output_dir}"
  run_stage "phase_f_pyroki_ik" \
    "${PYROKI_PYTHON}" "${LFV_ROOT}/local_pipelines/egoinfinity_hand_pipeline/phase_f_pyroki_ik_backend/solve_phase_e_pyroki_ik.py" \
    --session-dir "${session_dir}" \
    --gripper-core-csv "${gripper_csv}" \
    --start-index "${PYROKI_START_INDEX}" \
    --max-frames "${PYROKI_MAX_FRAMES}" \
    --robot-table-json "${HOST_ROBOT_TABLE_JSON}" \
    --piper-urdf "${HOST_PIPER_URDF}" \
    --output-dir "${pyroki_output_dir}" \
    --target-mode "${TARGET_MODE}" \
    --scale "${SCALE}" \
    --initial-q "${INITIAL_Q}" \
    --anchor-xyz "${ANCHOR_XYZ}" \
    --target-bias-xyz "${TARGET_BIAS_XYZ}" \
    --min-robot-z-m "${MIN_ROBOT_Z_M}" \
    --target-link "${PYROKI_TARGET_LINK}" \
    --tcp-offset-xyz "${TCP_OFFSET_XYZ}" \
    --ik-mode "${IK_MODE}" \
    --orientation-source "${ORIENTATION_SOURCE}" \
    --fixed-target-quat-xyzw "${FIXED_TARGET_QUAT_XYZW}" \
    --orientation-align-rpy "${ORIENTATION_ALIGN_RPY}" \
    --pos-weight "${PYROKI_POS_WEIGHT}" \
    --ori-weight "${PYROKI_ORI_WEIGHT}" \
    --rest-weight "${PYROKI_REST_WEIGHT}" \
    --locked-joint-rest-weight "${PYROKI_LOCKED_JOINT_REST_WEIGHT}" \
    --limit-center-weight "${PYROKI_LIMIT_CENTER_WEIGHT}" \
    --limit-constraint-weight "${PYROKI_LIMIT_CONSTRAINT_WEIGHT}" \
    --manipulability-weight "${PYROKI_MANIPULABILITY_WEIGHT}" \
    --max-pos-error-m "${MAX_POS_ERROR_M}" \
    --ori-tol-rad "${ORI_TOL_RAD}" \
    --limit-margin-rad "${NORMAL_LIMIT_MARGIN_RAD}" \
    --limit-active-threshold-rad "${NORMAL_ACTIVE_THRESHOLD_RAD}" \
    --retry-random-seeds "${PYROKI_RETRY_RANDOM_SEEDS}" \
    --seed-mode "${PYROKI_SEED_MODE}" \
    --seed-rng "${PYROKI_SEED_RNG}" \
    --seed-score-ori-weight "${PYROKI_SEED_SCORE_ORI_WEIGHT}" \
    --seed-score-joint-weight "${PYROKI_SEED_SCORE_JOINT_WEIGHT}" \
    --seed-score-manip-weight "${PYROKI_SEED_SCORE_MANIP_WEIGHT}" \
    --failed-frame-strategy "${FAILED_FRAME_STRATEGY}" \
    --gripper-command-mode "${GRIPPER_COMMAND_MODE}" \
    --jax-platform "${PYROKI_JAX_PLATFORM}" \
    "${pose_guard_arg[@]}" \
    "${z_clamp_arg[@]}"
else
  echo "[alignment_gripper_ik] RUN_PYROKI_IK=false; skipping PyRoki IK."
  record_skip_stage "phase_f_pyroki_ik"
fi

write_timing_summary

echo "[alignment_gripper_ik] Done."
echo "[alignment_gripper_ik] Phase-D: ${phase_d_dir}"
echo "[alignment_gripper_ik] Phase-E: ${phase_e_dir}"
echo "[alignment_gripper_ik] Pinocchio IK: ${ik_output_dir}"
if [[ -n "${pyroki_output_dir}" ]]; then
  echo "[alignment_gripper_ik] PyRoki IK: ${pyroki_output_dir}"
fi
echo "[alignment_gripper_ik] timing_jsonl: ${downstream_timing_jsonl}"
echo "[alignment_gripper_ik] timing_summary_json: ${downstream_timing_summary}"
