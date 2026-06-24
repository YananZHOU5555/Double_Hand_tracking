#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LFV_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HOST_ROS_ROOT="${HOST_ROS_ROOT:-/home/yannan/workspace/ros1_docker-main}"
CONTAINER_ROOT="${CONTAINER_ROOT:-/workspace/ros1_docker_jinhe}"
HOST_SESSION_ROOT="${HOST_SESSION_ROOT:-${HOST_ROS_ROOT}/rosbag_data/LFV_demos_for_new_calib}"
WILOR_ROOT="${WILOR_ROOT:-${LFV_ROOT}/WiLor}"
WILOR_PYTHON="${WILOR_PYTHON:-/home/yannan/miniforge3/envs/wilor_lfv/bin/python}"
PLOT_PYTHON="${PLOT_PYTHON:-${LFV_ROOT}/.venv_plot/bin/python}"

HAND="${HAND:-right}"
TRIM_FRAMES="${TRIM_FRAMES:-20}"
MAX_FRAMES="${MAX_FRAMES:-0}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
WILOR_CONF="${WILOR_CONF:-0.30}"
WILOR_RESCALE_FACTOR="${WILOR_RESCALE_FACTOR:-2.0}"
SAVE_WILOR_OVERLAY="${SAVE_WILOR_OVERLAY:-true}"
SAVE_WILOR_MESH_OVERLAY="${SAVE_WILOR_MESH_OVERLAY:-true}"
SAVE_HAND21_OVERLAY="${SAVE_HAND21_OVERLAY:-true}"
SAVE_CROPS="${SAVE_CROPS:-0}"
THUMB_YZ_TILT_GAIN="${THUMB_YZ_TILT_GAIN:--1.0}"
THUMB_YZ_TILT_CLAMP_DEG="${THUMB_YZ_TILT_CLAMP_DEG:-45}"
PIPELINE_PREFIX="${PIPELINE_PREFIX:-qzy_wilor_best_gm1p0_button_up}"

ROBOT_TABLE_JSON="${ROBOT_TABLE_JSON:-${CONTAINER_ROOT}/data/lfv_calibration/right_arm_table_latest.json}"
TCP_PIVOT_JSON="${TCP_PIVOT_JSON:-${HOST_ROS_ROOT}/data/lfv_calibration/right_arm_touch_tcp_xr2_latest.json}"
ORIENTATION_ALIGN_RPY="${ORIENTATION_ALIGN_RPY:-0,1.570796326795,1.570796326795}"
RUN_IK="${RUN_IK:-true}"
RUN_MUJOCO="${RUN_MUJOCO:-true}"
REBUILD_WILOR="${REBUILD_WILOR:-false}"

usage() {
  cat <<EOF
Usage:
  bash local_pipelines/qzy_wilor_fallback/run_qzy_wilor_fallback_pipeline.sh <demo_name_or_session_dir>

Runs the local QZY WiLoR + V0 state + open regularizer + relaxed fallback IK pipeline.

Important env:
  HOST_SESSION_ROOT=${HOST_SESSION_ROOT}
  WILOR_PYTHON=${WILOR_PYTHON}
  TRIM_FRAMES=${TRIM_FRAMES}
  MAX_FRAMES=${MAX_FRAMES}
  ROBOT_TABLE_JSON=${ROBOT_TABLE_JSON}
  TCP_PIVOT_JSON=${TCP_PIVOT_JSON}
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

target="${1:-}"
if [[ -z "${target}" ]]; then
  echo "[qzy_wilor_fallback] Missing demo name or session dir." >&2
  usage >&2
  exit 1
fi
if [[ "${target}" = /* ]]; then
  session_dir="$(readlink -f "${target}")"
else
  session_dir="$(readlink -f "${HOST_SESSION_ROOT}/${target}")"
fi

if [[ ! -d "${session_dir}" ]]; then
  echo "[qzy_wilor_fallback] Session not found: ${session_dir}" >&2
  exit 1
fi
case "${session_dir}" in
  "${HOST_ROS_ROOT}"/*) ;;
  *)
    echo "[qzy_wilor_fallback] Session must be under ${HOST_ROS_ROOT}: ${session_dir}" >&2
    exit 1
    ;;
esac

if [[ ! -x "${WILOR_PYTHON}" ]]; then
  echo "[qzy_wilor_fallback] WiLoR python not executable: ${WILOR_PYTHON}" >&2
  exit 1
fi
if [[ ! -f "${session_dir}/processed_topcam/left_table.mp4" ]]; then
  echo "[qzy_wilor_fallback] Missing processed_topcam; run scripts/process_lfv_demo_topcam.sh first." >&2
  exit 1
fi

pipeline_dir="${session_dir}/quality/qzy_wilor_fallback"
stage_dir="${pipeline_dir}/stages"
input_dir="${pipeline_dir}/best_inputs"
regularized_dir="${pipeline_dir}/regularized"
detection_audit_dir="${pipeline_dir}/hand_detection_audit"
mkdir -p "${stage_dir}" "${input_dir}" "${regularized_dir}"

log() {
  printf '\n[qzy_wilor_fallback] %s\n' "$*"
}

run_if_missing() {
  local marker="$1"
  shift
  if [[ "${REBUILD_WILOR}" != "true" && -f "${marker}" ]]; then
    log "reuse ${marker}"
    return 0
  fi
  "$@"
}

log "session=${session_dir}"
log "output=${pipeline_dir}"
log "robot_table=${ROBOT_TABLE_JSON}"
log "tcp_pivot=${TCP_PIVOT_JSON}"

hand21_core_overlay="${session_dir}/quality/hand21/hand21_core_segment_overlay.mp4"
mediapipe_left_overlay="${session_dir}/hand_tracking/left_${HAND}_hand_mediapipe/${HAND}_hand_tracking_overlay.mp4"
mediapipe_right_overlay="${session_dir}/hand_tracking/right_${HAND}_hand_mediapipe/${HAND}_hand_tracking_overlay.mp4"
log "1/12 MediaPipe stereo hand21 QA"
if [[ ! -f "${session_dir}/quality/hand21/${HAND}_hand_21_landmarks_table.csv" || ! -f "${session_dir}/quality/hand21/hand21_quality_report.json" || ( "${SAVE_HAND21_OVERLAY}" == "true" && ! -f "${hand21_core_overlay}" ) ]]; then
  HAND="${HAND}" SAVE_OVERLAY="${SAVE_HAND21_OVERLAY}" SAVE_CROPS="${SAVE_CROPS}" \
    bash "${LFV_ROOT}/scripts/audit_lfv_hand21_quality.sh" "${session_dir}"
else
  log "reuse hand21 outputs"
fi

log "2/12 Left/right MediaPipe detection timeline"
detection_summary_json="${detection_audit_dir}/v0_stereo_detection_summary.json"
detection_per_frame_csv="${detection_audit_dir}/v0_stereo_detection_per_frame.csv"
detection_timeline_png="${detection_audit_dir}/v0_stereo_detection_timeline.png"
detection_sbs_overlay_mp4="${detection_audit_dir}/v0_stereo_detection_sbs_overlay.mp4"
if [[ "${REBUILD_WILOR}" == "true" || ! -f "${detection_summary_json}" || ! -f "${detection_per_frame_csv}" || ! -f "${detection_timeline_png}" ]]; then
  "${WILOR_PYTHON}" "${LFV_ROOT}/scripts/lfv_v0_stereo_detection_audit.py" \
    "${session_dir}" \
    --hand "${HAND}" \
    --output-dir "${detection_audit_dir}" \
    --max-overlay-frames 0
else
  log "reuse ${detection_summary_json}"
fi

log "3/12 V0 active-trim gripper state"
v0_dir="${stage_dir}/v0_active_trim${TRIM_FRAMES}_tcp"
v0_core="${v0_dir}/gripper_pose_table_core_humanego_mit.csv"
if [[ ! -f "${v0_core}" ]]; then
  HAND="${HAND}" TRIM_FRAMES="${TRIM_FRAMES}" \
  OUTPUT_SUBDIR="quality/qzy_wilor_fallback/stages/v0_active_trim${TRIM_FRAMES}_tcp" \
  INPUT_SUBDIR="quality/qzy_wilor_fallback/stages/v0_active_trim${TRIM_FRAMES}_inputs" \
    bash "${LFV_ROOT}/scripts/build_lfv_v0_active_trim20_tcp.sh" "${session_dir}"
else
  log "reuse ${v0_core}"
fi

log "4/12 WiLoR predictions"
wilor_pred_dir="${stage_dir}/wilor_v1_left_view"
wilor_predictions="${wilor_pred_dir}/wilor_predictions.csv"
wilor_overlay="${wilor_pred_dir}/wilor_overlay.mp4"
wilor_mesh_overlay="${wilor_pred_dir}/wilor_mesh_overlay.mp4"
if [[ "${REBUILD_WILOR}" == "true" || ! -f "${wilor_predictions}" || ( "${SAVE_WILOR_OVERLAY}" == "true" && ! -f "${wilor_overlay}" ) || ( "${SAVE_WILOR_MESH_OVERLAY}" == "true" && ! -f "${wilor_mesh_overlay}" ) ]]; then
  overlay_arg=()
  if [[ "${SAVE_WILOR_OVERLAY}" == "true" ]]; then
    overlay_arg+=(--save-overlay)
  fi
  if [[ "${SAVE_WILOR_MESH_OVERLAY}" == "true" ]]; then
    overlay_arg+=(--save-mesh-overlay)
  fi
  "${WILOR_PYTHON}" "${WILOR_ROOT}/lfv_adapter/lfv_run_wilor_on_lfv_video.py" \
    --session-dir "${session_dir}" \
    --wilor-root "${WILOR_ROOT}" \
    --video "${session_dir}/processed_topcam/left_table.mp4" \
    --output-dir "${wilor_pred_dir}" \
    --hand "${HAND}" \
    --conf "${WILOR_CONF}" \
    --rescale-factor "${WILOR_RESCALE_FACTOR}" \
    --max-frames "${MAX_FRAMES}" \
    --frame-stride "${FRAME_STRIDE}" \
    "${overlay_arg[@]}"
else
  log "reuse ${wilor_predictions}"
fi

log "5/12 active trim inputs for WiLoR constrained hand"
trim_dir="${stage_dir}/active_trim${TRIM_FRAMES}_inputs"
trim_meta="${trim_dir}/active_trim_metadata.json"
if [[ "${REBUILD_WILOR}" == "true" || ! -f "${trim_meta}" ]]; then
  "${WILOR_PYTHON}" "${SCRIPT_DIR}/make_qzy_wilor_fallback_inputs.py" trim-active \
    --hand21-report "${session_dir}/quality/hand21/hand21_quality_report.json" \
    --hand21-csv "${session_dir}/quality/hand21/${HAND}_hand_21_landmarks_table.csv" \
    --wilor-predictions-csv "${wilor_predictions}" \
    --output-dir "${trim_dir}" \
    --trim-frames "${TRIM_FRAMES}"
else
  log "reuse ${trim_meta}"
fi
trimmed_hand21="$(find "${trim_dir}" -maxdepth 1 -name "*_hand_21_landmarks_table_active_trim${TRIM_FRAMES}.csv" | head -1)"
trimmed_wilor="$(find "${trim_dir}" -maxdepth 1 -name "wilor_predictions_active_trim${TRIM_FRAMES}.csv" | head -1)"
if [[ -z "${trimmed_hand21}" || -z "${trimmed_wilor}" ]]; then
  echo "[qzy_wilor_fallback] Missing trimmed inputs under ${trim_dir}" >&2
  exit 1
fi

log "6/12 WiLoR V2 stereo-constrained hand"
constrained_dir="${stage_dir}/wilor_v2_constrained_trim${TRIM_FRAMES}_scale_guard_max1p35"
constrained_csv="${constrained_dir}/wilor_v2_stereo_constrained_21_table.csv"
per_frame_csv="${constrained_dir}/wilor_v2_stereo_constrained_per_frame.csv"
if [[ "${REBUILD_WILOR}" == "true" || ! -f "${constrained_csv}" ]]; then
  "${WILOR_PYTHON}" "${WILOR_ROOT}/lfv_adapter/lfv_fit_wilor_stereo_constrained_hand.py" \
    --wilor-predictions-csv "${trimmed_wilor}" \
    --stereo-table-csv "${trimmed_hand21}" \
    --output-dir "${constrained_dir}" \
    --min-align-points 6 \
    --max-align-rms-m 0.045 \
    --max-joint-residual-m 0.055 \
    --min-scale 0.55 \
    --max-scale 1.35 \
    --smooth-alpha-t 0.28 \
    --smooth-alpha-r 0.20 \
    --smooth-alpha-scale 0.10 \
    --max-translation-speed-mps 1.20 \
    --max-translation-jump-m 0.055 \
    --max-gap-hold-frames 18 \
    --fps-fallback 60
else
  log "reuse ${constrained_csv}"
fi

log "7/12 WiLoR constrained hand table-frame viewers"
constrained_html="${constrained_dir}/qzy_wilor_constrained_hand21_table.html"
quality_html="${constrained_dir}/qzy_wilor_constrained_quality.html"
if [[ ! -f "${constrained_html}" ]]; then
  "${WILOR_PYTHON}" "${WILOR_ROOT}/lfv_adapter/lfv_export_wilor_table_skeleton_html.py" \
    --session-dir "${session_dir}" \
    --fused21-csv "${constrained_csv}" \
    --video "${session_dir}/processed_topcam/left_table.mp4" \
    --output-html "${constrained_html}" \
    --min-valid-points 21 \
    --fps 60 \
    --title "QZY WiLoR constrained hand in latest table frame: $(basename "${session_dir}")"
fi
if [[ ! -f "${quality_html}" ]]; then
  "${WILOR_PYTHON}" "${WILOR_ROOT}/lfv_adapter/lfv_export_wilor_v2_quality_html.py" \
    --session-dir "${session_dir}" \
    --v2-csv "${constrained_csv}" \
    --per-frame-csv "${per_frame_csv}" \
    --video "${session_dir}/processed_topcam/left_table.mp4" \
    --output-html "${quality_html}"
fi

log "8/12 WiLoR TCP and index-middle/thumb-YZ orientation cores"
audit_csv="${constrained_dir}/qzy_wilor_mcp_core_audit.csv"
if [[ "${REBUILD_WILOR}" == "true" || ! -f "${audit_csv}" ]]; then
  "${WILOR_PYTHON}" "${SCRIPT_DIR}/make_qzy_wilor_fallback_inputs.py" make-audit \
    --constrained-hand21-csv "${constrained_csv}" \
    --per-frame-csv "${per_frame_csv}" \
    --output-csv "${audit_csv}" \
    --fps 60
fi

tcp_core_dir="${stage_dir}/wilor_v2_model_only_active_trim${TRIM_FRAMES}"
tcp_core="${tcp_core_dir}/gripper_pose_table_core_wilor_v2_mcp_x_smooth.csv"
if [[ "${REBUILD_WILOR}" == "true" || ! -f "${tcp_core}" ]]; then
  "${WILOR_PYTHON}" "${WILOR_ROOT}/lfv_adapter/lfv_export_wilor_v2_mcp_x_smooth_core.py" \
    --audit-csv "${audit_csv}" \
    --hand21-csv "${constrained_csv}" \
    --output-dir "${tcp_core_dir}" \
    --orientation-x-axis thumb_index_mcp
fi

ori_core_dir="${stage_dir}/wilor_v2_index_middle_thumb_index_yz_tilt_gm1p0_smooth_core"
ori_core="${ori_core_dir}/gripper_pose_table_core_wilor_v2_mcp_x_smooth.csv"
ori_html="${ori_core_dir}/qzy_index_middle_thumb_yz_gm1p0_6d_pose.html"
if [[ "${REBUILD_WILOR}" == "true" || ! -f "${ori_core}" ]]; then
  "${WILOR_PYTHON}" "${WILOR_ROOT}/lfv_adapter/lfv_export_wilor_v2_mcp_x_smooth_core.py" \
    --audit-csv "${audit_csv}" \
    --hand21-csv "${constrained_csv}" \
    --output-dir "${ori_core_dir}" \
    --orientation-x-axis index_middle_thumb_index_yz_tilt \
    --thumb-yz-tilt-gain "${THUMB_YZ_TILT_GAIN}" \
    --thumb-yz-tilt-clamp-deg "${THUMB_YZ_TILT_CLAMP_DEG}"
fi
if [[ ! -f "${ori_html}" ]]; then
  "${WILOR_PYTHON}" "${WILOR_ROOT}/lfv_adapter/lfv_export_wilor_v2_6d_pose_html.py" \
    --hand21-csv "${constrained_csv}" \
    --core-csv "${ori_core}" \
    --output-html "${ori_html}" \
    --title "QZY index-middle + thumb-YZ gm1p0 6D pose: $(basename "${session_dir}")" \
    --axis-length-m 0.055
fi

log "9/12 Fuse WiLoR TCP + V0 state + best orientation"
best_summary="${input_dir}/${PIPELINE_PREFIX}_best_inputs_summary.json"
if [[ "${REBUILD_WILOR}" == "true" || ! -f "${best_summary}" ]]; then
  "${WILOR_PYTHON}" "${SCRIPT_DIR}/make_qzy_wilor_fallback_inputs.py" build-best-inputs \
    --tcp-core-csv "${tcp_core}" \
    --v0-state-core-csv "${v0_core}" \
    --orientation-core-csv "${ori_core}" \
    --output-dir "${input_dir}" \
    --prefix "${PIPELINE_PREFIX}"
fi
tcp_input="${input_dir}/gripper_pose_table_core_${PIPELINE_PREFIX}_tcp.csv"
state_ori_input="${input_dir}/gripper_pose_table_core_${PIPELINE_PREFIX}_state_orientation.csv"

log "10/12 Open-segment IK-friendly regularizer"
regularized_csv="${regularized_dir}/gripper_pose_table_core_${PIPELINE_PREFIX}.csv"
regularized_html="${regularized_dir}/${PIPELINE_PREFIX}_compare_3d.html"
if [[ "${REBUILD_WILOR}" == "true" || ! -f "${regularized_csv}" ]]; then
  "${WILOR_PYTHON}" "${LFV_ROOT}/scripts/lfv_regularize_open_segments_for_ik.py" \
    --tcp-csv "${tcp_input}" \
    --state-pose-csv "${state_ori_input}" \
    --output-dir "${regularized_dir}" \
    --prefix "${PIPELINE_PREFIX}" \
    --guard-frames 8 \
    --min-open-frames 3
else
  log "reuse ${regularized_csv}"
fi

sim_dir=""
ik_debug_dir=""
ik_debug_html=""
ik_debug_summary_json=""
if [[ "${RUN_IK}" == "true" ]]; then
  log "11/12 Piper pose IK with relaxed TCP-priority fallback"
  GRIPPER_CORE_CSV="${regularized_csv}" \
  ROBOT_TABLE_JSON="${ROBOT_TABLE_JSON}" \
  TCP_PIVOT_JSON="${TCP_PIVOT_JSON}" \
  TARGET_MODE=absolute \
  SCALE=0.25 \
  TRAJECTORY_FILTER=raw \
  IK_MODE=pose \
  ORIENTATION_SOURCE=rot6d \
  ORIENTATION_ALIGN_RPY="${ORIENTATION_ALIGN_RPY}" \
  RELAXED_IK_FALLBACK=tcp_priority \
  ENSURE_CORE=false \
  ENSURE_GRIPPER=false \
    bash "${LFV_ROOT}/scripts/simulate_lfv_piper_urdf_core.sh" "${session_dir}"

  core_label="$(basename "${regularized_csv}" .csv | sed 's/+//g;s/-/m/g;s/ /_/g;s/\./p/g;s/[^A-Za-z0-9_]/_/g')"
  sim_dir="${session_dir}/quality/urdf_sim/piper_pinocchio_core_absolute_scale_0p25_core_${core_label}_ik_pose_rot6d_relaxed_tcp_priority"
  ik_debug_dir="${sim_dir}/ik_input_output_fallback_debug"
  ik_debug_html="${ik_debug_dir}/ik_input_output_fallback_debug.html"
  ik_debug_summary_json="${ik_debug_dir}/ik_input_output_fallback_debug_summary.json"
  if [[ -f "${sim_dir}/piper_pinocchio_core_ik.csv" && ( "${REBUILD_WILOR}" == "true" || ! -f "${ik_debug_html}" ) ]]; then
    log "12/12 IK input/output fallback debugger"
    "${WILOR_PYTHON}" "${SCRIPT_DIR}/export_ik_input_output_fallback_html.py" \
      --session-dir "${session_dir}" \
      --core-csv "${regularized_csv}" \
      --ik-csv "${sim_dir}/piper_pinocchio_core_ik.csv" \
      --ik-metadata "${sim_dir}/piper_pinocchio_core_ik_metadata.json" \
      --table-json "${ROBOT_TABLE_JSON}" \
      --tcp-json "${TCP_PIVOT_JSON}" \
      --output-dir "${ik_debug_dir}" \
      --orientation-align-rpy "${ORIENTATION_ALIGN_RPY}" \
      --title "QZY IK input vs FK output with fallback: $(basename "${session_dir}")"
  elif [[ -f "${ik_debug_html}" ]]; then
    log "reuse ${ik_debug_html}"
  fi

  if [[ "${RUN_MUJOCO}" == "true" && -x "${PLOT_PYTHON}" && -f "${sim_dir}/piper_pinocchio_core_ik.csv" ]]; then
    "${PLOT_PYTHON}" "${LFV_ROOT}/scripts/export_piper_mujoco_replay_html.py" \
      --position-csv "${sim_dir}/piper_pinocchio_core_ik.csv" \
      --pose-csv "${sim_dir}/piper_pinocchio_core_ik.csv" \
      --tcp-json "${TCP_PIVOT_JSON}" \
      --output-dir "${sim_dir}/mujoco_replay_home_ramp_2s" \
      --prepend-home-sec 2 \
      --fps 60
  fi
fi

cat <<EOF

[qzy_wilor_fallback] Done.
  pipeline_dir: ${pipeline_dir}
  mediapipe_left_overlay_mp4: ${mediapipe_left_overlay}
  mediapipe_right_overlay_mp4: ${mediapipe_right_overlay}
  mediapipe_hand21_core_overlay_mp4: ${hand21_core_overlay}
  wilor_overlay_mp4: ${wilor_overlay}
  wilor_mesh_overlay_mp4: ${wilor_mesh_overlay}
  detection_summary_json: ${detection_summary_json}
  detection_per_frame_csv: ${detection_per_frame_csv}
  detection_timeline_png: ${detection_timeline_png}
  constrained_hand_html: ${constrained_html}
  constrained_quality_html: ${quality_html}
  orientation_html: ${ori_html}
  regularized_csv: ${regularized_csv}
  regularized_html: ${regularized_html}
EOF
if [[ -n "${sim_dir}" ]]; then
  cat <<EOF
  ik_dir: ${sim_dir}
  ik_csv: ${sim_dir}/piper_pinocchio_core_ik.csv
  ik_metadata: ${sim_dir}/piper_pinocchio_core_ik_metadata.json
  ik_input_output_fallback_html: ${ik_debug_html}
  ik_input_output_fallback_summary_json: ${ik_debug_summary_json}
  mujoco_html: ${sim_dir}/mujoco_replay_home_ramp_2s/piper_pose_ik_mujoco_replay.html
EOF
fi
if [[ -f "${detection_sbs_overlay_mp4}" ]]; then
  cat <<EOF
  detection_sbs_overlay_mp4: ${detection_sbs_overlay_mp4}
EOF
fi
