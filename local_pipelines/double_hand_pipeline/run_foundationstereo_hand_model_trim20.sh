#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LFV_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HOST_ROS_ROOT="${HOST_ROS_ROOT:-/home/yannan/workspace/ros1_docker-main}"
HOST_SESSION_ROOT="${HOST_SESSION_ROOT:-${HOST_ROS_ROOT}/rosbag_data/human_teaching_videos}"

PYTHON_BIN="${PYTHON_BIN:-${LFV_ROOT}/.venv-dinosam/bin/python}"
WILOR_PYTHON="${WILOR_PYTHON:-/home/yannan/miniforge3/envs/wilor_lfv/bin/python}"
WILOR_ROOT="${WILOR_ROOT:-${LFV_ROOT}/WiLor}"
FOUNDATIONSTEREO_ROOT="${FOUNDATIONSTEREO_ROOT:-/home/yannan/workspace/external/FoundationStereo}"
STEREO_MODEL="${STEREO_MODEL:-${FOUNDATIONSTEREO_ROOT}/pretrained_models/23-51-11/model_best_bp2.pth}"

TRIM_FRAMES="${TRIM_FRAMES:-20}"
HAND21_REPORT="${HAND21_REPORT:-}"
WILOR_CONF="${WILOR_CONF:-0.30}"
WILOR_RESCALE_FACTOR="${WILOR_RESCALE_FACTOR:-2.0}"
WILOR_RUN_HAND="${WILOR_RUN_HAND:-best}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
MAX_FRAMES="${MAX_FRAMES:-0}"
VALID_ITERS="${VALID_ITERS:-16}"
LR_CHECK="${LR_CHECK:-true}"
WRITE_PREVIEW="${WRITE_PREVIEW:-true}"
SAMPLE_RADIUS_PX="${SAMPLE_RADIUS_PX:-3}"
MIN_VISIBLE_RATIO="${MIN_VISIBLE_RATIO:-0.25}"
MAX_DEPTH_M="${MAX_DEPTH_M:-2.5}"
MIN_ALIGN_POINTS="${MIN_ALIGN_POINTS:-4}"
MAX_ALIGN_RMS_M="${MAX_ALIGN_RMS_M:-0.100}"
CONSTANT_SCALE="${CONSTANT_SCALE:-none}"
ALIGN_IDS="${ALIGN_IDS:-all}"
REBUILD_WILOR="${REBUILD_WILOR:-false}"
REBUILD_DEPTH="${REBUILD_DEPTH:-false}"

usage() {
  cat <<EOF
Usage:
  bash local_pipelines/double_hand_pipeline/run_foundationstereo_hand_model_trim20.sh <demo_name_or_session_dir>

Runs WiLoR hand-model table skeleton using FoundationStereo dense depth over
the detected active hand segment after trimming TRIM_FRAMES from both ends.

Important env:
  TRIM_FRAMES=${TRIM_FRAMES}
  CONSTANT_SCALE=${CONSTANT_SCALE}       # none|global|label
  ALIGN_IDS=${ALIGN_IDS}                 # all|default|comma ids
  REBUILD_WILOR=${REBUILD_WILOR}
  REBUILD_DEPTH=${REBUILD_DEPTH}
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

target="${1:-}"
if [[ -z "${target}" ]]; then
  echo "[foundationstereo_hand_model_trim20] Missing demo name or session dir." >&2
  usage >&2
  exit 1
fi
if [[ "${target}" = /* ]]; then
  session_dir="$(readlink -f "${target}")"
else
  session_dir="$(readlink -f "${HOST_SESSION_ROOT}/${target}")"
fi
if [[ ! -d "${session_dir}" ]]; then
  echo "[foundationstereo_hand_model_trim20] Session not found: ${session_dir}" >&2
  exit 1
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[foundationstereo_hand_model_trim20] Python not executable: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -x "${WILOR_PYTHON}" ]]; then
  echo "[foundationstereo_hand_model_trim20] WiLoR python not executable: ${WILOR_PYTHON}" >&2
  exit 1
fi
if [[ ! -f "${session_dir}/processed_topcam/left_table.mp4" || ! -f "${session_dir}/processed_topcam/right_table.mp4" ]]; then
  echo "[foundationstereo_hand_model_trim20] Missing processed_topcam table videos; run scripts/process_lfv_demo_topcam.sh first." >&2
  exit 1
fi

if [[ -z "${HAND21_REPORT}" ]]; then
  if [[ -f "${session_dir}/quality/hand21/hand21_quality_report.json" ]]; then
    HAND21_REPORT="${session_dir}/quality/hand21/hand21_quality_report.json"
  elif [[ -f "${session_dir}/quality/left_hand21/hand21_quality_report.json" ]]; then
    HAND21_REPORT="${session_dir}/quality/left_hand21/hand21_quality_report.json"
  else
    echo "[foundationstereo_hand_model_trim20] Missing hand21 quality report." >&2
    exit 1
  fi
fi

read -r frame_start frame_end <<<"$("${PYTHON_BIN}" - "${HAND21_REPORT}" "${TRIM_FRAMES}" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], "r", encoding="utf-8"))
trim = int(sys.argv[2])
active = report.get("detected_active_segment") or {}
start = int(active["start_frame"]) + trim
end = int(active["end_frame"]) - trim
if start > end:
    raise SystemExit(f"empty active trim window: {active}, trim={trim}")
print(start, end)
PY
)"
frame_tag="$(printf '%04d_%04d' "${frame_start}" "${frame_end}")"
out_root="${session_dir}/quality/foundationstereo_hand_model_active_trim${TRIM_FRAMES}_${frame_tag}"
wilor_dir="${out_root}/wilor_left_view_geometry"
visible_dir="${out_root}/wilor_visible_left_camera"
depth_dir="${out_root}/foundationstereo_depth"
foundation_points_dir="${out_root}/foundationstereo_visible_points"
table_dir="${out_root}/hand_model_table"
mkdir -p "${out_root}" "${visible_dir}" "${depth_dir}" "${foundation_points_dir}" "${table_dir}"

wilor_predictions="${wilor_dir}/wilor_predictions.csv"
wilor_geometry="${wilor_dir}/wilor_mesh_geometry.npz"
visibility_csv="${visible_dir}/wilor_visible_hand_skeleton_left_camera.csv"
depth_summary="${depth_dir}/foundationstereo_depth_summary.json"
foundation_table_csv="${foundation_points_dir}/wilor_visible_foundation_depth_table.csv"
table_html="${table_dir}/wilor_hand_model_foundationstereo_table.html"

echo "[foundationstereo_hand_model_trim20] session=${session_dir}"
echo "[foundationstereo_hand_model_trim20] active_trim=${frame_start}..${frame_end}"
echo "[foundationstereo_hand_model_trim20] output=${out_root}"

if [[ "${REBUILD_WILOR}" == "true" || ! -f "${wilor_predictions}" || ! -f "${wilor_geometry}" ]]; then
  echo "[foundationstereo_hand_model_trim20] Running WiLoR geometry."
  "${WILOR_PYTHON}" "${WILOR_ROOT}/lfv_adapter/lfv_run_wilor_on_lfv_video.py" \
    --session-dir "${session_dir}" \
    --wilor-root "${WILOR_ROOT}" \
    --video "${session_dir}/processed_topcam/left_table.mp4" \
    --output-dir "${wilor_dir}" \
    --hand "${WILOR_RUN_HAND}" \
    --conf "${WILOR_CONF}" \
    --rescale-factor "${WILOR_RESCALE_FACTOR}" \
    --frame-start "${frame_start}" \
    --frame-end "${frame_end}" \
    --frame-stride "${FRAME_STRIDE}" \
    --max-frames "${MAX_FRAMES}" \
    --save-geometry-npz
else
  echo "[foundationstereo_hand_model_trim20] Reuse WiLoR geometry: ${wilor_geometry}"
fi

if [[ ! -f "${visibility_csv}" || "${REBUILD_WILOR}" == "true" ]]; then
  echo "[foundationstereo_hand_model_trim20] Exporting MANO visible landmarks."
  "${PYTHON_BIN}" "${LFV_ROOT}/scripts/export_wilor_visible_hand_skeleton_camera_html.py" \
    --geometry-npz "${wilor_geometry}" \
    --output-html "${visible_dir}/wilor_visible_hand_skeleton_left_camera.html" \
    --video "${session_dir}/processed_topcam/left_table.mp4" \
    --frame-start "${frame_start}" \
    --frame-end "${frame_end}" \
    --fps 30
else
  echo "[foundationstereo_hand_model_trim20] Reuse visibility CSV: ${visibility_csv}"
fi

lr_args=()
if [[ "${LR_CHECK}" == "true" || "${LR_CHECK}" == "1" ]]; then
  lr_args+=(--lr-check)
fi
preview_args=()
if [[ "${WRITE_PREVIEW}" == "true" || "${WRITE_PREVIEW}" == "1" ]]; then
  preview_args+=(--write-preview)
fi

if [[ "${REBUILD_DEPTH}" == "true" || ! -f "${depth_summary}" ]]; then
  echo "[foundationstereo_hand_model_trim20] Running FoundationStereo depth."
  "${PYTHON_BIN}" "${LFV_ROOT}/scripts/run_lfv_foundationstereo_disparity.py" \
    --session-dir "${session_dir}" \
    --output-dir "${depth_dir}" \
    --backend foundationstereo \
    --foundationstereo-root "${FOUNDATIONSTEREO_ROOT}" \
    --stereo-model "${STEREO_MODEL}" \
    --valid-iters "${VALID_ITERS}" \
    --frame-start "${frame_start}" \
    --frame-end "${frame_end}" \
    --stride "${FRAME_STRIDE}" \
    --max-depth-m "${MAX_DEPTH_M}" \
    "${lr_args[@]}" \
    "${preview_args[@]}"
else
  echo "[foundationstereo_hand_model_trim20] Reuse FoundationStereo depth: ${depth_summary}"
fi

echo "[foundationstereo_hand_model_trim20] Sampling FoundationStereo depth at visible WiLoR landmarks."
"${PYTHON_BIN}" "${LFV_ROOT}/scripts/export_wilor_visible_foundation_depth_table_html.py" \
  --visibility-csv "${visibility_csv}" \
  --depth-frame-csv "${depth_dir}/foundationstereo_depth_frames.csv" \
  --depth-summary-json "${depth_summary}" \
  --output-html "${foundation_points_dir}/wilor_visible_foundation_depth_table.html" \
  --output-csv "${foundation_table_csv}" \
  --video "${session_dir}/processed_topcam/left_table.mp4" \
  --frame-start "${frame_start}" \
  --frame-end "${frame_end}" \
  --sample-radius-px "${SAMPLE_RADIUS_PX}" \
  --min-visible-ratio "${MIN_VISIBLE_RATIO}" \
  --max-depth-m "${MAX_DEPTH_M}" \
  --fps 30 \
  --title "WiLoR visible points sampled from FoundationStereo depth: active trim${TRIM_FRAMES}"

echo "[foundationstereo_hand_model_trim20] Exporting FoundationStereo-constrained hand model skeleton."
"${PYTHON_BIN}" "${LFV_ROOT}/scripts/export_wilor_visible_hand_skeleton_table_html.py" \
  --wilor-predictions-csv "${wilor_predictions}" \
  --visibility-csv "${visibility_csv}" \
  --foundation-table-csv "${foundation_table_csv}" \
  --output-html "${table_html}" \
  --video "${session_dir}/processed_topcam/left_table.mp4" \
  --frame-start "${frame_start}" \
  --frame-end "${frame_end}" \
  --constant-scale "${CONSTANT_SCALE}" \
  --align-ids "${ALIGN_IDS}" \
  --min-align-points "${MIN_ALIGN_POINTS}" \
  --max-align-rms-m "${MAX_ALIGN_RMS_M}" \
  --fps 30 \
  --title "WiLoR hand model aligned to FoundationStereo depth: active trim${TRIM_FRAMES}"

echo
echo "[foundationstereo_hand_model_trim20] Done"
echo "  active trim:       ${frame_start}..${frame_end}"
echo "  WiLoR predictions: ${wilor_predictions}"
echo "  visibility HTML:   ${visible_dir}/wilor_visible_hand_skeleton_left_camera.html"
echo "  FS points HTML:    ${foundation_points_dir}/wilor_visible_foundation_depth_table.html"
echo "  hand model HTML:   ${table_html}"
echo "  hand model CSV:    ${table_dir}/wilor_hand_model_foundationstereo_table.csv"
echo "  summary JSON:      ${table_dir}/wilor_hand_model_foundationstereo_table.json"
