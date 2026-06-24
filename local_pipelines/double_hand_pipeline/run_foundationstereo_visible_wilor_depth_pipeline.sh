#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LFV_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HOST_ROS_ROOT="${HOST_ROS_ROOT:-/home/yannan/workspace/ros1_docker-main}"
HOST_SESSION_ROOT="${HOST_SESSION_ROOT:-${HOST_ROS_ROOT}/rosbag_data/human_teaching_videos}"

PYTHON_BIN="${PYTHON_BIN:-${LFV_ROOT}/.venv-dinosam/bin/python}"
FOUNDATIONSTEREO_ROOT="${FOUNDATIONSTEREO_ROOT:-/home/yannan/workspace/external/FoundationStereo}"
STEREO_MODEL="${STEREO_MODEL:-${FOUNDATIONSTEREO_ROOT}/pretrained_models/23-51-11/model_best_bp2.pth}"
BACKEND="${BACKEND:-foundationstereo}"
VALID_ITERS="${VALID_ITERS:-16}"
FRAME_START="${FRAME_START:-580}"
FRAME_END="${FRAME_END:-640}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
MAX_FRAMES="${MAX_FRAMES:-0}"
LR_CHECK="${LR_CHECK:-true}"
WRITE_PREVIEW="${WRITE_PREVIEW:-true}"
SAMPLE_RADIUS_PX="${SAMPLE_RADIUS_PX:-3}"
MIN_VISIBLE_RATIO="${MIN_VISIBLE_RATIO:-0.25}"
MAX_DEPTH_M="${MAX_DEPTH_M:-2.5}"
FPS="${FPS:-30}"
TABLE_FRAME_JSON="${TABLE_FRAME_JSON:-${HOST_ROS_ROOT}/data/lfv_calibration/table_frame_latest.json}"
HAND_LABEL="${HAND_LABEL:-}"
OUTPUT_TAG="${OUTPUT_TAG:-}"
VISIBILITY_CSV="${VISIBILITY_CSV:-}"

usage() {
  cat <<EOF
Usage:
  bash local_pipelines/double_hand_pipeline/run_foundationstereo_visible_wilor_depth_pipeline.sh <demo_name_or_session_dir>

Runs dense FoundationStereo depth on processed_topcam/left_table.mp4 + right_table.mp4,
then samples that depth at the existing WiLoR camera-visible landmarks and exports
a table-frame 3D HTML.

Important env:
  PYTHON_BIN=${PYTHON_BIN}
  FOUNDATIONSTEREO_ROOT=${FOUNDATIONSTEREO_ROOT}
  STEREO_MODEL=${STEREO_MODEL}
  FRAME_START=${FRAME_START}
  FRAME_END=${FRAME_END}
  MAX_FRAMES=${MAX_FRAMES}   # 0 means all selected frames
  VISIBILITY_CSV=<optional explicit wilor_visible_hand_skeleton_left_camera.csv>
  HAND_LABEL=<optional left|right filter; empty means both>
  BACKEND=foundationstereo|sgbm
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

target="${1:-}"
if [[ -z "${target}" ]]; then
  echo "[foundationstereo_visible_wilor] Missing demo name or session dir." >&2
  usage >&2
  exit 1
fi
if [[ "${target}" = /* ]]; then
  session_dir="$(readlink -f "${target}")"
else
  session_dir="$(readlink -f "${HOST_SESSION_ROOT}/${target}")"
fi
if [[ ! -d "${session_dir}" ]]; then
  echo "[foundationstereo_visible_wilor] Session not found: ${session_dir}" >&2
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[foundationstereo_visible_wilor] Python not executable: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ "${BACKEND}" == "foundationstereo" ]]; then
  if [[ ! -d "${FOUNDATIONSTEREO_ROOT}" ]]; then
    echo "[foundationstereo_visible_wilor] FoundationStereo root not found: ${FOUNDATIONSTEREO_ROOT}" >&2
    exit 1
  fi
  if [[ ! -f "${STEREO_MODEL}" ]]; then
    echo "[foundationstereo_visible_wilor] Stereo checkpoint not found: ${STEREO_MODEL}" >&2
    exit 1
  fi
fi
if [[ ! -f "${session_dir}/processed_topcam/left_table.mp4" || ! -f "${session_dir}/processed_topcam/right_table.mp4" ]]; then
  echo "[foundationstereo_visible_wilor] Missing processed_topcam table videos; run scripts/process_lfv_demo_topcam.sh first." >&2
  exit 1
fi

frame_tag="$(printf '%04d_%04d' "${FRAME_START}" "${FRAME_END}")"
if [[ -z "${OUTPUT_TAG}" ]]; then
  OUTPUT_TAG="foundationstereo_visible_wilor_${frame_tag}"
fi
output_dir="${session_dir}/quality/${OUTPUT_TAG}"
depth_dir="${output_dir}/depth_prior"
mkdir -p "${output_dir}" "${depth_dir}"

if [[ -z "${VISIBILITY_CSV}" ]]; then
  expected="${session_dir}/quality/wilor_visible_hand_skeleton_left_camera_${frame_tag}/wilor_visible_hand_skeleton_left_camera.csv"
  if [[ -f "${expected}" ]]; then
    VISIBILITY_CSV="${expected}"
  else
    VISIBILITY_CSV="$(find "${session_dir}/quality" -maxdepth 3 -type f -path '*wilor_visible_hand_skeleton_left_camera*/wilor_visible_hand_skeleton_left_camera.csv' | sort | tail -n 1 || true)"
  fi
fi
if [[ -z "${VISIBILITY_CSV}" || ! -f "${VISIBILITY_CSV}" ]]; then
  echo "[foundationstereo_visible_wilor] Missing WiLoR visible landmark CSV." >&2
  echo "Expected: ${session_dir}/quality/wilor_visible_hand_skeleton_left_camera_${frame_tag}/wilor_visible_hand_skeleton_left_camera.csv" >&2
  echo "Or set VISIBILITY_CSV=/abs/path/to/wilor_visible_hand_skeleton_left_camera.csv" >&2
  exit 1
fi

lr_args=()
if [[ "${LR_CHECK}" == "true" || "${LR_CHECK}" == "1" ]]; then
  lr_args+=(--lr-check)
fi
preview_args=()
if [[ "${WRITE_PREVIEW}" == "true" || "${WRITE_PREVIEW}" == "1" ]]; then
  preview_args+=(--write-preview)
fi
hand_args=()
if [[ -n "${HAND_LABEL}" ]]; then
  hand_args+=(--hand-label "${HAND_LABEL}")
fi
max_frame_args=()
if [[ "${MAX_FRAMES}" != "0" ]]; then
  max_frame_args+=(--max-frames "${MAX_FRAMES}")
fi

echo "[foundationstereo_visible_wilor] session=${session_dir}"
echo "[foundationstereo_visible_wilor] visibility_csv=${VISIBILITY_CSV}"
echo "[foundationstereo_visible_wilor] output=${output_dir}"

"${PYTHON_BIN}" "${LFV_ROOT}/scripts/run_lfv_foundationstereo_disparity.py" \
  --session-dir "${session_dir}" \
  --output-dir "${depth_dir}" \
  --backend "${BACKEND}" \
  --foundationstereo-root "${FOUNDATIONSTEREO_ROOT}" \
  --stereo-model "${STEREO_MODEL}" \
  --valid-iters "${VALID_ITERS}" \
  --frame-start "${FRAME_START}" \
  --frame-end "${FRAME_END}" \
  --stride "${FRAME_STRIDE}" \
  --max-depth-m "${MAX_DEPTH_M}" \
  "${max_frame_args[@]}" \
  "${lr_args[@]}" \
  "${preview_args[@]}"

"${PYTHON_BIN}" "${LFV_ROOT}/scripts/export_wilor_visible_foundation_depth_table_html.py" \
  --visibility-csv "${VISIBILITY_CSV}" \
  --depth-frame-csv "${depth_dir}/foundationstereo_depth_frames.csv" \
  --depth-summary-json "${depth_dir}/foundationstereo_depth_summary.json" \
  --table-frame-json "${TABLE_FRAME_JSON}" \
  --output-html "${output_dir}/wilor_visible_foundation_depth_table.html" \
  --video "${session_dir}/processed_topcam/left_table.mp4" \
  --frame-start "${FRAME_START}" \
  --frame-end "${FRAME_END}" \
  --sample-radius-px "${SAMPLE_RADIUS_PX}" \
  --min-visible-ratio "${MIN_VISIBLE_RATIO}" \
  --max-depth-m "${MAX_DEPTH_M}" \
  --fps "${FPS}" \
  --title "WiLoR visible points sampled from FoundationStereo depth" \
  "${hand_args[@]}"

echo
echo "[foundationstereo_visible_wilor] Done"
echo "  depth summary: ${depth_dir}/foundationstereo_depth_summary.json"
echo "  depth frames:  ${depth_dir}/foundationstereo_depth_frames.csv"
echo "  preview:       ${depth_dir}/depth_preview_contact_sheet.jpg"
echo "  table html:    ${output_dir}/wilor_visible_foundation_depth_table.html"
echo "  table csv:     ${output_dir}/wilor_visible_foundation_depth_table.csv"
