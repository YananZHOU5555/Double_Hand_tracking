#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LFV_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HOST_ROS_ROOT="${HOST_ROS_ROOT:-/home/yannan/workspace/ros1_docker-main}"
CONTAINER_ROOT="${CONTAINER_ROOT:-/workspace/ros1_docker_jinhe}"
HOST_SESSION_ROOT="${HOST_SESSION_ROOT:-${HOST_ROS_ROOT}/rosbag_data/human_teaching_videos}"
WILOR_ROOT="${WILOR_ROOT:-${LFV_ROOT}/WiLor}"
WILOR_PYTHON="${WILOR_PYTHON:-/home/yannan/miniforge3/envs/wilor_lfv/bin/python}"
PLOT_PYTHON="${PLOT_PYTHON:-${LFV_ROOT}/.venv_plot/bin/python}"
OCCLUSION_AUDIT_PYTHON="${OCCLUSION_AUDIT_PYTHON:-${LFV_ROOT}/.venv-dinosam/bin/python}"

HAND="${HAND:-}"
ARM="${ARM:-}"
TRIM_FRAMES="${TRIM_FRAMES:-20}"
MAX_FRAMES="${MAX_FRAMES:-0}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
WILOR_CONF="${WILOR_CONF:-0.30}"
WILOR_RESCALE_FACTOR="${WILOR_RESCALE_FACTOR:-2.0}"
WILOR_RUN_HAND="${WILOR_RUN_HAND:-best}"
SAVE_WILOR_OVERLAY="${SAVE_WILOR_OVERLAY:-true}"
SAVE_WILOR_MESH_OVERLAY="${SAVE_WILOR_MESH_OVERLAY:-true}"
RUN_WILOR_OCCLUSION_AUDIT="${RUN_WILOR_OCCLUSION_AUDIT:-true}"
OCCLUSION_AUDIT_HAND_LABEL="${OCCLUSION_AUDIT_HAND_LABEL:-}"
OCCLUSION_AUDIT_MAX_FRAMES="${OCCLUSION_AUDIT_MAX_FRAMES:-0}"
OCCLUSION_AUDIT_FRAME_START="${OCCLUSION_AUDIT_FRAME_START:-0}"
OCCLUSION_AUDIT_FRAME_END="${OCCLUSION_AUDIT_FRAME_END:--1}"
SAVE_HAND21_OVERLAY="${SAVE_HAND21_OVERLAY:-true}"
SAVE_CROPS="${SAVE_CROPS:-0}"
THUMB_YZ_TILT_GAIN="${THUMB_YZ_TILT_GAIN:--1.0}"
THUMB_YZ_TILT_CLAMP_DEG="${THUMB_YZ_TILT_CLAMP_DEG:-45}"
RUN_IK="${RUN_IK:-true}"
RUN_MUJOCO="${RUN_MUJOCO:-true}"
REBUILD_WILOR="${REBUILD_WILOR:-false}"
PROCESS_TOPCAM="${PROCESS_TOPCAM:-auto}"
TOPCAM_OUTPUT_FPS="${TOPCAM_OUTPUT_FPS:-30}"
RUN_INTERACTION_PHASES="${RUN_INTERACTION_PHASES:-true}"
STABILIZE_WILOR="${STABILIZE_WILOR:-true}"
STABILIZER_FPS="${STABILIZER_FPS:-${TOPCAM_OUTPUT_FPS}}"
IDENTITY_HOLD_MS="${IDENTITY_HOLD_MS:-400}"
MISSING_BRIDGE_MS="${MISSING_BRIDGE_MS:-200}"
PALM_FLIP_WINDOW_MS="${PALM_FLIP_WINDOW_MS:-300}"
PALM_FLIP_ANGLE_DEG="${PALM_FLIP_ANGLE_DEG:-100}"
HARD_PALM_SPEED_DEG_S="${HARD_PALM_SPEED_DEG_S:-900}"

usage() {
  cat <<EOF
Usage:
  HAND=<left|right> ARM=<left|right> bash local_pipelines/double_hand_pipeline/run_double_hand_pipeline.sh <demo_name_or_session_dir>

Runs one explicitly mapped hand/arm lane of the double-hand pipeline copy.

Important env:
  HOST_SESSION_ROOT=${HOST_SESSION_ROOT}
  HAND=<required upstream hand identity label: left or right>
  ARM=<required robot arm mapping label: left or right>
  WILOR_PYTHON=${WILOR_PYTHON}
  WILOR_RUN_HAND=${WILOR_RUN_HAND}
  TRIM_FRAMES=${TRIM_FRAMES}
  MAX_FRAMES=${MAX_FRAMES}
  STABILIZE_WILOR=${STABILIZE_WILOR}
  PROCESS_TOPCAM=${PROCESS_TOPCAM}
  ROBOT_TABLE_JSON=<optional override; otherwise selected from ARM>
  TCP_PIVOT_JSON=<optional override; otherwise selected from ARM>
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "${HAND}" != "left" && "${HAND}" != "right" ]]; then
  echo "[double_hand_pipeline] HAND must be provided explicitly as left or right; got '${HAND}'." >&2
  usage >&2
  exit 1
fi
if [[ "${ARM}" != "left" && "${ARM}" != "right" ]]; then
  echo "[double_hand_pipeline] ARM must be provided explicitly as left or right; got '${ARM}'." >&2
  usage >&2
  exit 1
fi

if [[ "${STABILIZE_WILOR}" == "true" ]]; then
  DEFAULT_PIPELINE_PREFIX="double_hand_${HAND}_hand_${ARM}_arm_qzy_wilor_stable_best_gm1p0_button_up"
  MAPPING_STAGE_SUFFIX="_stable_${HAND}"
else
  DEFAULT_PIPELINE_PREFIX="double_hand_${HAND}_hand_${ARM}_arm_qzy_wilor_best_gm1p0_button_up"
  MAPPING_STAGE_SUFFIX=""
fi
PIPELINE_PREFIX="${PIPELINE_PREFIX:-${DEFAULT_PIPELINE_PREFIX}}"

if [[ "${ARM}" == "left" ]]; then
  DEFAULT_ROBOT_TABLE_JSON="${CONTAINER_ROOT}/data/lfv_calibration/left_arm_table_latest.json"
  DEFAULT_TCP_PIVOT_JSON="${HOST_ROS_ROOT}/data/lfv_calibration/left_arm_touch_tcp_zr2_latest.json"
elif [[ "${ARM}" == "right" ]]; then
  DEFAULT_ROBOT_TABLE_JSON="${CONTAINER_ROOT}/data/lfv_calibration/right_arm_table_latest.json"
  DEFAULT_TCP_PIVOT_JSON="${HOST_ROS_ROOT}/data/lfv_calibration/right_arm_touch_tcp_xr2_latest.json"
else
  echo "[double_hand_pipeline] ARM must be left or right; got ${ARM}" >&2
  exit 1
fi
ROBOT_TABLE_JSON="${ROBOT_TABLE_JSON:-${DEFAULT_ROBOT_TABLE_JSON}}"
TCP_PIVOT_JSON="${TCP_PIVOT_JSON:-${DEFAULT_TCP_PIVOT_JSON}}"
ORIENTATION_ALIGN_RPY="${ORIENTATION_ALIGN_RPY:-0,1.570796326795,1.570796326795}"

target="${1:-}"
if [[ -z "${target}" ]]; then
  echo "[double_hand_pipeline] Missing demo name or session dir." >&2
  usage >&2
  exit 1
fi
if [[ "${target}" = /* ]]; then
  session_dir="$(readlink -f "${target}")"
else
  session_dir="$(readlink -f "${HOST_SESSION_ROOT}/${target}")"
fi

if [[ ! -d "${session_dir}" ]]; then
  echo "[double_hand_pipeline] Session not found: ${session_dir}" >&2
  exit 1
fi
case "${session_dir}" in
  "${HOST_ROS_ROOT}"/*) ;;
  *)
    echo "[double_hand_pipeline] Session must be under ${HOST_ROS_ROOT}: ${session_dir}" >&2
    exit 1
    ;;
esac

if [[ ! -x "${WILOR_PYTHON}" ]]; then
  echo "[double_hand_pipeline] WiLoR python not executable: ${WILOR_PYTHON}" >&2
  exit 1
fi
if [[ ! -f "${session_dir}/processed_topcam/left_table.mp4" ]]; then
  if [[ "${PROCESS_TOPCAM}" == "auto" || "${PROCESS_TOPCAM}" == "true" || "${PROCESS_TOPCAM}" == "1" ]]; then
    echo "[double_hand_pipeline] Missing processed_topcam; generating it from episode_0.bag."
    HOST_SESSION_ROOT="${HOST_SESSION_ROOT}" OUTPUT_FPS="${TOPCAM_OUTPUT_FPS}" \
      bash "${LFV_ROOT}/scripts/process_lfv_demo_topcam.sh" "${session_dir}"
  else
    echo "[double_hand_pipeline] Missing processed_topcam; run scripts/process_lfv_demo_topcam.sh first." >&2
    exit 1
  fi
fi
if [[ ! -f "${session_dir}/processed_topcam/left_table.mp4" ]]; then
  echo "[double_hand_pipeline] Still missing processed_topcam/left_table.mp4 after processing." >&2
  exit 1
fi

pipeline_dir="${session_dir}/quality/double_hand_pipeline/${HAND}_hand_${ARM}_arm"
stage_dir="${pipeline_dir}/stages"
input_dir="${pipeline_dir}/best_inputs"
regularized_dir="${pipeline_dir}/regularized"
detection_audit_dir="${pipeline_dir}/hand_detection_audit"
mkdir -p "${stage_dir}" "${input_dir}" "${regularized_dir}"
lane_config_json="${pipeline_dir}/lane_config.json"
LANE_CONFIG_JSON="${lane_config_json}" \
SESSION_DIR="${session_dir}" \
HAND="${HAND}" \
ARM="${ARM}" \
HOST_SESSION_ROOT="${HOST_SESSION_ROOT}" \
ROBOT_TABLE_JSON="${ROBOT_TABLE_JSON}" \
TCP_PIVOT_JSON="${TCP_PIVOT_JSON}" \
PIPELINE_PREFIX="${PIPELINE_PREFIX}" \
"${WILOR_PYTHON}" - <<'PY'
import json
import os
from datetime import datetime, timezone

payload = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "session_dir": os.environ["SESSION_DIR"],
    "host_session_root": os.environ["HOST_SESSION_ROOT"],
    "hand_identity_label": os.environ["HAND"],
    "robot_arm_label": os.environ["ARM"],
    "robot_table_json": os.environ["ROBOT_TABLE_JSON"],
    "tcp_pivot_json": os.environ["TCP_PIVOT_JSON"],
    "pipeline_prefix": os.environ["PIPELINE_PREFIX"],
    "identity_source": "upstream_explicit_HAND_env",
    "arm_mapping_source": "upstream_explicit_ARM_env",
}
with open(os.environ["LANE_CONFIG_JSON"], "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, sort_keys=True)
    f.write("\n")
PY

log() {
  printf '\n[double_hand_pipeline] %s\n' "$*"
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
log "hand=${HAND}"
log "arm=${ARM}"
log "output=${pipeline_dir}"
log "robot_table=${ROBOT_TABLE_JSON}"
log "tcp_pivot=${TCP_PIVOT_JSON}"

quality_subdir="hand21"
if [[ "${HAND}" == "left" ]]; then
  quality_subdir="left_hand21"
fi

hand21_core_overlay="${session_dir}/quality/${quality_subdir}/hand21_core_segment_overlay.mp4"
mediapipe_left_overlay="${session_dir}/hand_tracking/left_${HAND}_hand_mediapipe/${HAND}_hand_tracking_overlay.mp4"
mediapipe_right_overlay="${session_dir}/hand_tracking/right_${HAND}_hand_mediapipe/${HAND}_hand_tracking_overlay.mp4"
log "1/15 MediaPipe stereo hand21 QA"
if [[ ! -f "${session_dir}/quality/${quality_subdir}/${HAND}_hand_21_landmarks_table.csv" || ! -f "${session_dir}/quality/${quality_subdir}/hand21_quality_report.json" || ( "${SAVE_HAND21_OVERLAY}" == "true" && ! -f "${hand21_core_overlay}" ) ]]; then
  HAND="${HAND}" SAVE_OVERLAY="${SAVE_HAND21_OVERLAY}" SAVE_CROPS="${SAVE_CROPS}" \
    bash "${LFV_ROOT}/scripts/audit_lfv_hand21_quality.sh" "${session_dir}"
else
  log "reuse hand21 outputs"
fi

log "2/15 Left/right MediaPipe detection timeline"
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

log "3/15 V0 active-trim gripper state"
v0_dir="${stage_dir}/v0_active_trim${TRIM_FRAMES}_tcp"
v0_core="${v0_dir}/gripper_pose_table_core_humanego_mit.csv"
if [[ ! -f "${v0_core}" ]]; then
  HAND="${HAND}" TRIM_FRAMES="${TRIM_FRAMES}" \
  OUTPUT_SUBDIR="quality/double_hand_pipeline/${HAND}_hand_${ARM}_arm/stages/v0_active_trim${TRIM_FRAMES}_tcp" \
  INPUT_SUBDIR="quality/double_hand_pipeline/${HAND}_hand_${ARM}_arm/stages/v0_active_trim${TRIM_FRAMES}_inputs" \
    bash "${LFV_ROOT}/scripts/build_lfv_v0_active_trim20_tcp.sh" "${session_dir}"
else
  log "reuse ${v0_core}"
fi

log "4/15 WiLoR predictions"
wilor_pred_dir="${stage_dir}/wilor_v1_left_view"
wilor_predictions="${wilor_pred_dir}/wilor_predictions.csv"
wilor_geometry_npz="${wilor_pred_dir}/wilor_mesh_geometry.npz"
wilor_overlay="${wilor_pred_dir}/wilor_overlay.mp4"
wilor_mesh_overlay="${wilor_pred_dir}/wilor_mesh_overlay.mp4"
occlusion_audit_dir="${pipeline_dir}/wilor_segmentation_occlusion_confidence"
occlusion_audit_json="${occlusion_audit_dir}/wilor_segmentation_occlusion_confidence.json"
occlusion_audit_overlay="${occlusion_audit_dir}/wilor_segmentation_occlusion_confidence_overlay.mp4"
occlusion_audit_worst_sheet="${occlusion_audit_dir}/wilor_segmentation_occlusion_worst_contact_sheet.jpg"
stabilized_dir="${stage_dir}/wilor_temporal_stabilized_${HAND}"
stabilized_wilor="${stabilized_dir}/wilor_predictions_stabilized_${HAND}.csv"
stabilized_events="${stabilized_dir}/wilor_predictions_stabilized_${HAND}_events.csv"
stabilized_summary="${stabilized_dir}/wilor_predictions_stabilized_${HAND}.json"
if [[ "${REBUILD_WILOR}" == "true" || ! -f "${wilor_predictions}" || ( "${RUN_WILOR_OCCLUSION_AUDIT}" == "true" && ! -f "${wilor_geometry_npz}" ) || ( "${SAVE_WILOR_OVERLAY}" == "true" && ! -f "${wilor_overlay}" ) || ( "${SAVE_WILOR_MESH_OVERLAY}" == "true" && ! -f "${wilor_mesh_overlay}" ) ]]; then
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
    --hand "${WILOR_RUN_HAND}" \
    --conf "${WILOR_CONF}" \
    --rescale-factor "${WILOR_RESCALE_FACTOR}" \
    --max-frames "${MAX_FRAMES}" \
    --frame-stride "${FRAME_STRIDE}" \
    --save-geometry-npz \
    "${overlay_arg[@]}"
else
  log "reuse ${wilor_predictions}"
fi

log "5/15 WiLoR segmentation occlusion confidence before gripper mapping"
if [[ "${RUN_WILOR_OCCLUSION_AUDIT}" == "true" ]]; then
  if [[ ! -x "${OCCLUSION_AUDIT_PYTHON}" ]]; then
    echo "[double_hand_pipeline] Missing occlusion audit Python env: ${OCCLUSION_AUDIT_PYTHON}" >&2
    exit 2
  fi
  if [[ "${REBUILD_WILOR}" == "true" ]] || [[ ! -f "${occlusion_audit_json}" ]] || ! grep -q '"mesh_visibility_source": "mano_mesh_zbuffer"' "${occlusion_audit_json}"; then
    PYTHON_BIN="${OCCLUSION_AUDIT_PYTHON}" \
    WILOR_PREDICTIONS="${wilor_predictions}" \
    WILOR_GEOMETRY_NPZ="${wilor_geometry_npz}" \
    OUTPUT_DIR="${occlusion_audit_dir}" \
    HAND_LABEL="${OCCLUSION_AUDIT_HAND_LABEL}" \
    MAX_FRAMES="${OCCLUSION_AUDIT_MAX_FRAMES}" \
    FRAME_START="${OCCLUSION_AUDIT_FRAME_START}" \
    FRAME_END="${OCCLUSION_AUDIT_FRAME_END}" \
      bash "${LFV_ROOT}/scripts/audit_wilor_segmentation_occlusion_confidence.sh" "${session_dir}"
  else
    log "reuse ${occlusion_audit_json}"
  fi
else
  log "skip WiLoR segmentation occlusion confidence audit"
fi

log "6/15 WiLoR temporal identity stabilizer before gripper mapping"
wilor_for_mapping="${wilor_predictions}"
if [[ "${STABILIZE_WILOR}" == "true" ]]; then
  mkdir -p "${stabilized_dir}"
  if [[ "${REBUILD_WILOR}" == "true" || ! -f "${stabilized_wilor}" || ! -f "${stabilized_summary}" ]] || ! grep -q "EgoInfinity-style CSV-level WiLoR" "${stabilized_summary}" 2>/dev/null; then
    "${WILOR_PYTHON}" "${SCRIPT_DIR}/stabilize_wilor_predictions.py" \
      --input-csv "${wilor_predictions}" \
      --output-csv "${stabilized_wilor}" \
      --events-csv "${stabilized_events}" \
      --summary-json "${stabilized_summary}" \
      --target-label "${HAND}" \
      --fps "${STABILIZER_FPS}" \
      --identity-hold-ms "${IDENTITY_HOLD_MS}" \
      --missing-bridge-ms "${MISSING_BRIDGE_MS}" \
      --palm-flip-window-ms "${PALM_FLIP_WINDOW_MS}" \
      --palm-flip-angle-deg "${PALM_FLIP_ANGLE_DEG}" \
      --hard-palm-speed-deg-s "${HARD_PALM_SPEED_DEG_S}"
  else
    log "reuse ${stabilized_summary}"
  fi
  wilor_for_mapping="${stabilized_wilor}"
else
  log "skip WiLoR temporal stabilizer"
fi

log "7/15 active trim inputs for WiLoR constrained hand"
trim_dir="${stage_dir}/active_trim${TRIM_FRAMES}${MAPPING_STAGE_SUFFIX}_inputs"
trim_meta="${trim_dir}/active_trim_metadata.json"
if [[ "${REBUILD_WILOR}" == "true" || ! -f "${trim_meta}" ]]; then
  "${WILOR_PYTHON}" "${SCRIPT_DIR}/make_qzy_wilor_fallback_inputs.py" trim-active \
    --hand21-report "${session_dir}/quality/${quality_subdir}/hand21_quality_report.json" \
    --hand21-csv "${session_dir}/quality/${quality_subdir}/${HAND}_hand_21_landmarks_table.csv" \
    --wilor-predictions-csv "${wilor_for_mapping}" \
    --output-dir "${trim_dir}" \
    --trim-frames "${TRIM_FRAMES}"
else
  log "reuse ${trim_meta}"
fi
trimmed_hand21="$(find "${trim_dir}" -maxdepth 1 -name "*_hand_21_landmarks_table_active_trim${TRIM_FRAMES}.csv" | head -1)"
trimmed_wilor="$(find "${trim_dir}" -maxdepth 1 -name "wilor_predictions*_active_trim${TRIM_FRAMES}.csv" | head -1)"
if [[ -z "${trimmed_hand21}" || -z "${trimmed_wilor}" ]]; then
  echo "[double_hand_pipeline] Missing trimmed inputs under ${trim_dir}" >&2
  exit 1
fi

log "8/15 WiLoR V2 stereo-constrained hand"
constrained_dir="${stage_dir}/wilor_v2_constrained${MAPPING_STAGE_SUFFIX}_trim${TRIM_FRAMES}_scale_guard_max1p35"
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

log "9/15 WiLoR constrained hand table-frame viewers"
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

log "10/15 WiLoR TCP and index-middle/thumb-YZ orientation cores"
audit_csv="${constrained_dir}/qzy_wilor_mcp_core_audit.csv"
if [[ "${REBUILD_WILOR}" == "true" || ! -f "${audit_csv}" ]]; then
  "${WILOR_PYTHON}" "${SCRIPT_DIR}/make_qzy_wilor_fallback_inputs.py" make-audit \
    --constrained-hand21-csv "${constrained_csv}" \
    --per-frame-csv "${per_frame_csv}" \
    --output-csv "${audit_csv}" \
    --fps 60
fi

tcp_core_dir="${stage_dir}/wilor_v2_model_only${MAPPING_STAGE_SUFFIX}_active_trim${TRIM_FRAMES}"
tcp_core="${tcp_core_dir}/gripper_pose_table_core_wilor_v2_mcp_x_smooth.csv"
if [[ "${REBUILD_WILOR}" == "true" || ! -f "${tcp_core}" ]]; then
  "${WILOR_PYTHON}" "${WILOR_ROOT}/lfv_adapter/lfv_export_wilor_v2_mcp_x_smooth_core.py" \
    --audit-csv "${audit_csv}" \
    --hand21-csv "${constrained_csv}" \
    --output-dir "${tcp_core_dir}" \
    --orientation-x-axis thumb_index_mcp
fi

ori_core_dir="${stage_dir}/wilor_v2_index_middle_thumb_index_yz_tilt_gm1p0${MAPPING_STAGE_SUFFIX}_smooth_core"
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

log "11/15 Fuse WiLoR TCP + V0 state + best orientation"
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

log "12/15 Open-segment IK-friendly regularizer"
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

log "13/15 Interaction phase detection"
phase_dir="${pipeline_dir}/interaction_phases"
phase_csv="${phase_dir}/interaction_phases.csv"
phase_json="${phase_dir}/interaction_phases.json"
phase_html="${phase_dir}/interaction_phases.html"
if [[ "${RUN_INTERACTION_PHASES}" == "true" || "${RUN_INTERACTION_PHASES}" == "1" ]]; then
  mkdir -p "${phase_dir}"
  if [[ "${REBUILD_WILOR}" == "true" || ! -f "${phase_json}" || ! -f "${phase_csv}" || ! -f "${phase_html}" ]]; then
    "${PLOT_PYTHON}" "${SCRIPT_DIR}/detect_interaction_phases.py" \
      --tcp-csv "${regularized_csv}" \
      --output-dir "${phase_dir}" \
      --prefix interaction_phases \
      --fps "${TOPCAM_OUTPUT_FPS}"
  else
    log "reuse ${phase_json}"
  fi
else
  log "skip interaction phase detection"
fi

sim_dir=""
ik_debug_dir=""
ik_debug_html=""
ik_debug_summary_json=""
if [[ "${RUN_IK}" == "true" ]]; then
  log "14/15 Piper pose IK with relaxed TCP-priority fallback"
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
    log "15/15 IK input/output fallback debugger"
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

[double_hand_pipeline] Done.
  pipeline_dir: ${pipeline_dir}
  lane_config_json: ${lane_config_json}
  mediapipe_left_overlay_mp4: ${mediapipe_left_overlay}
  mediapipe_right_overlay_mp4: ${mediapipe_right_overlay}
  mediapipe_hand21_core_overlay_mp4: ${hand21_core_overlay}
  wilor_overlay_mp4: ${wilor_overlay}
  wilor_mesh_overlay_mp4: ${wilor_mesh_overlay}
  wilor_stabilized_predictions_csv: ${stabilized_wilor}
  wilor_stabilized_events_csv: ${stabilized_events}
  wilor_stabilized_summary_json: ${stabilized_summary}
  wilor_occlusion_confidence_json: ${occlusion_audit_json}
  wilor_occlusion_confidence_overlay_mp4: ${occlusion_audit_overlay}
  wilor_occlusion_confidence_worst_contact_sheet: ${occlusion_audit_worst_sheet}
  detection_summary_json: ${detection_summary_json}
  detection_per_frame_csv: ${detection_per_frame_csv}
  detection_timeline_png: ${detection_timeline_png}
  constrained_hand_html: ${constrained_html}
  constrained_quality_html: ${quality_html}
  orientation_html: ${ori_html}
  regularized_csv: ${regularized_csv}
  regularized_html: ${regularized_html}
  interaction_phase_csv: ${phase_csv}
  interaction_phase_json: ${phase_json}
  interaction_phase_html: ${phase_html}
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
