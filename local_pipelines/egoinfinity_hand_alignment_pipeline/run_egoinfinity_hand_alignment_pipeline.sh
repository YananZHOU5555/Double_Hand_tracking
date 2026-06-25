#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LFV_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HOST_ROS_ROOT="${HOST_ROS_ROOT:-/home/yannan/workspace/ros1_docker-main}"
HOST_SESSION_ROOT="${HOST_SESSION_ROOT:-${HOST_ROS_ROOT}/rosbag_data/human_teaching_videos}"
WILOR_ROOT="${WILOR_ROOT:-${LFV_ROOT}/WiLor}"
WILOR_PYTHON="${WILOR_PYTHON:-/home/yannan/miniforge3/envs/wilor_lfv/bin/python}"
PYTHON_BIN="${PYTHON_BIN:-${LFV_ROOT}/.venv-dinosam/bin/python}"
FOUNDATIONSTEREO_ROOT="${FOUNDATIONSTEREO_ROOT:-/home/yannan/workspace/external/FoundationStereo}"
STEREO_MODEL="${STEREO_MODEL:-${FOUNDATIONSTEREO_ROOT}/pretrained_models/23-51-11/model_best_bp2.pth}"

HAND="${HAND:-best}"
WILOR_CONF="${WILOR_CONF:-0.30}"
WILOR_RESCALE_FACTOR="${WILOR_RESCALE_FACTOR:-2.0}"
MAX_FRAMES="${MAX_FRAMES:-0}"
FRAME_START="${FRAME_START:-0}"
FRAME_END="${FRAME_END:--1}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
PROCESS_TOPCAM="${PROCESS_TOPCAM:-auto}"
TOPCAM_OUTPUT_FPS="${TOPCAM_OUTPUT_FPS:-30}"
REBUILD_EGO_HAND="${REBUILD_EGO_HAND:-false}"
SAVE_RAW_OVERLAY="${SAVE_RAW_OVERLAY:-false}"
RUN_PHASE_C="${RUN_PHASE_C:-false}"
RUN_PHASE_C_DEPTH_STABILIZE="${RUN_PHASE_C_DEPTH_STABILIZE:-false}"
RUN_PHASE_C_DEPTH_SMOOTH="${RUN_PHASE_C_DEPTH_SMOOTH:-false}"
RUN_MOTION_INFILL="${RUN_MOTION_INFILL:-false}"
RUN_PHASE_C2="${RUN_PHASE_C2:-false}"
RUN_PHASE_C2B_REPAIR="${RUN_PHASE_C2B_REPAIR:-false}"
RUN_PHASE_C3="${RUN_PHASE_C3:-false}"
RUN_VISIBILITY_REALIGN="${RUN_VISIBILITY_REALIGN:-false}"
REBUILD_DEPTH="${REBUILD_DEPTH:-false}"
REBUILD_PHASE_C_DEPTH_STABILIZE="${REBUILD_PHASE_C_DEPTH_STABILIZE:-false}"
REBUILD_PHASE_C="${REBUILD_PHASE_C:-false}"
REBUILD_PHASE_C_DEPTH_SMOOTH="${REBUILD_PHASE_C_DEPTH_SMOOTH:-false}"
REBUILD_MOTION_INFILL="${REBUILD_MOTION_INFILL:-false}"
REBUILD_PHASE_C2="${REBUILD_PHASE_C2:-false}"
REBUILD_PHASE_C2B_REPAIR="${REBUILD_PHASE_C2B_REPAIR:-false}"
REBUILD_PHASE_C3="${REBUILD_PHASE_C3:-false}"
REBUILD_VISIBILITY_REALIGN="${REBUILD_VISIBILITY_REALIGN:-false}"
VALID_ITERS="${VALID_ITERS:-16}"
LR_CHECK="${LR_CHECK:-true}"
MAX_DEPTH_M="${MAX_DEPTH_M:-2.5}"
WRITE_DEPTH_PREVIEW="${WRITE_DEPTH_PREVIEW:-true}"
DEPTH_STABILIZE="${DEPTH_STABILIZE:-false}"
USE_FLOW_MASK="${USE_FLOW_MASK:-false}"
WRITE_STABLE_DEPTH="${WRITE_STABLE_DEPTH:-false}"
DEPTH_STABILIZE_BBOX_MARGIN="${DEPTH_STABILIZE_BBOX_MARGIN:-0.30}"
DEPTH_STABILIZE_TEMPLATE_MIN_VALID_RATIO="${DEPTH_STABILIZE_TEMPLATE_MIN_VALID_RATIO:-0.30}"
DEPTH_STABILIZE_TEMPLATE_STRIDE="${DEPTH_STABILIZE_TEMPLATE_STRIDE:-1}"
DEPTH_STABILIZE_USE_FLOW_MASK="${DEPTH_STABILIZE_USE_FLOW_MASK:-false}"
DEPTH_STABILIZE_WRITE_DYNAMIC_MASKS="${DEPTH_STABILIZE_WRITE_DYNAMIC_MASKS:-false}"
ALIGN_PATCH_SIZE="${ALIGN_PATCH_SIZE:-7}"
ALIGN_MIN_RELIABLE_JOINTS="${ALIGN_MIN_RELIABLE_JOINTS:-2}"
DEPTH_SMOOTH_SIGMA_Z="${DEPTH_SMOOTH_SIGMA_Z:-5.0}"
DEPTH_SMOOTH_MAD_FACTOR="${DEPTH_SMOOTH_MAD_FACTOR:-2.0}"
DEPTH_SMOOTH_MIN_INLIERS="${DEPTH_SMOOTH_MIN_INLIERS:-3}"
DEPTH_SMOOTH_PATCH_SIZE="${DEPTH_SMOOTH_PATCH_SIZE:-7}"
DEPTH_SMOOTH_MAX_DELTA_Z_M="${DEPTH_SMOOTH_MAX_DELTA_Z_M:-0.30}"
DEPTH_SMOOTH_VERTEX_MEAN_Z="${DEPTH_SMOOTH_VERTEX_MEAN_Z:-true}"
ALLOW_PARTIAL_QC="${ALLOW_PARTIAL_QC:-false}"
RUN_PHASE_C_BAD_VIZ="${RUN_PHASE_C_BAD_VIZ:-true}"
PHASE_C_BAD_VIZ_CONTEXT="${PHASE_C_BAD_VIZ_CONTEXT:-2}"
PHASE_C_BAD_VIZ_MAX_SHEET_FRAMES="${PHASE_C_BAD_VIZ_MAX_SHEET_FRAMES:-80}"
MOTION_INFILLER_CHECKPOINT="${MOTION_INFILLER_CHECKPOINT:-/home/yannan/workspace/EgoInfinity/pretrained_models/infiller.pt}"
MOTION_INFILL_DEVICE="${MOTION_INFILL_DEVICE:-cuda}"
SMOOTH_MANO_WINDOW="${SMOOTH_MANO_WINDOW:-7}"
SMOOTH_MANO_POLYORDER="${SMOOTH_MANO_POLYORDER:-2}"
PHASE_C2_MIN_TRACK_FRAMES="${PHASE_C2_MIN_TRACK_FRAMES:-3}"
PHASE_C2B_BAD_GLOBAL_ROT_DELTA_DEG="${PHASE_C2B_BAD_GLOBAL_ROT_DELTA_DEG:-30.0}"
PHASE_C2B_BAD_RAW_GLOBAL_ROT_JUMP_DEG="${PHASE_C2B_BAD_RAW_GLOBAL_ROT_JUMP_DEG:-90.0}"
PHASE_C2B_BAD_SMOOTH_GLOBAL_ROT_JUMP_DEG="${PHASE_C2B_BAD_SMOOTH_GLOBAL_ROT_JUMP_DEG:-90.0}"
PHASE_C2B_BAD_INFILLED_WRIST_JUMP_M="${PHASE_C2B_BAD_INFILLED_WRIST_JUMP_M:-0.080}"
PHASE_C2B_NEIGHBOR_WINDOW_FRAMES="${PHASE_C2B_NEIGHBOR_WINDOW_FRAMES:-12}"
PHASE_C2B_BRIDGE_GOOD_GAP_FRAMES="${PHASE_C2B_BRIDGE_GOOD_GAP_FRAMES:-2}"
MESH_VISIBILITY_EPSILON_M="${MESH_VISIBILITY_EPSILON_M:-0.010}"
MESH_VISIBILITY_NEAREST_VERTICES="${MESH_VISIBILITY_NEAREST_VERTICES:-18}"
MESH_VISIBILITY_RATIO_THRESHOLD="${MESH_VISIBILITY_RATIO_THRESHOLD:-0.25}"
VIS_REALIGN_ENABLE_ALL_VISIBLE_FALLBACK="${VIS_REALIGN_ENABLE_ALL_VISIBLE_FALLBACK:-false}"
VIS_REALIGN_KEEP_PREVIOUS_ON_BAD_RMS="${VIS_REALIGN_KEEP_PREVIOUS_ON_BAD_RMS:-true}"
VIS_REALIGN_PATCH_SIZE="${VIS_REALIGN_PATCH_SIZE:-7}"
VIS_REALIGN_MIN_VISIBLE_RELIABLE_JOINTS="${VIS_REALIGN_MIN_VISIBLE_RELIABLE_JOINTS:-2}"
VIS_REALIGN_MIN_VISIBLE_STABLE_JOINTS="${VIS_REALIGN_MIN_VISIBLE_STABLE_JOINTS:-3}"
VIS_REALIGN_MAX_PATCH_SPREAD_M="${VIS_REALIGN_MAX_PATCH_SPREAD_M:-0.120}"
VIS_REALIGN_MAX_PATCH_MAD_M="${VIS_REALIGN_MAX_PATCH_MAD_M:-0.040}"

usage() {
  cat <<EOF
Usage:
  bash local_pipelines/egoinfinity_hand_alignment_pipeline/run_egoinfinity_hand_alignment_pipeline.sh <demo_name_or_session_dir>

Experimental EgoInfinity-style hand pipeline.

Env:
  HAND=${HAND}                 # best, left, right
  MAX_FRAMES=${MAX_FRAMES}
  FRAME_START=${FRAME_START}
  FRAME_END=${FRAME_END}
  FRAME_STRIDE=${FRAME_STRIDE}
  REBUILD_EGO_HAND=${REBUILD_EGO_HAND}
  RUN_PHASE_C=${RUN_PHASE_C}
  RUN_PHASE_C_DEPTH_STABILIZE=${RUN_PHASE_C_DEPTH_STABILIZE}
  RUN_PHASE_C_DEPTH_SMOOTH=${RUN_PHASE_C_DEPTH_SMOOTH}
  RUN_MOTION_INFILL=${RUN_MOTION_INFILL}
  RUN_PHASE_C2=${RUN_PHASE_C2}
  RUN_PHASE_C2B_REPAIR=${RUN_PHASE_C2B_REPAIR}
  RUN_PHASE_C3=${RUN_PHASE_C3}
  RUN_VISIBILITY_REALIGN=${RUN_VISIBILITY_REALIGN}
  ALLOW_PARTIAL_QC=${ALLOW_PARTIAL_QC}
  RUN_PHASE_C_BAD_VIZ=${RUN_PHASE_C_BAD_VIZ}
  WILOR_PYTHON=${WILOR_PYTHON}
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

target="${1:-}"
if [[ -z "${target}" ]]; then
  echo "[egoinfinity_hand_alignment_pipeline] Missing demo name or session dir." >&2
  usage >&2
  exit 1
fi

if [[ "${target}" = /* ]]; then
  session_dir="$(readlink -f "${target}")"
else
  session_dir="$(readlink -f "${HOST_SESSION_ROOT}/${target}")"
fi

if [[ ! -d "${session_dir}" ]]; then
  echo "[egoinfinity_hand_alignment_pipeline] Session not found: ${session_dir}" >&2
  exit 1
fi
if [[ ! -x "${WILOR_PYTHON}" ]]; then
  echo "[egoinfinity_hand_alignment_pipeline] WiLoR python not executable: ${WILOR_PYTHON}" >&2
  exit 1
fi
if [[ "${RUN_PHASE_C}" == "true" || "${RUN_PHASE_C}" == "1" ]]; then
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "[egoinfinity_hand_alignment_pipeline] Python not executable: ${PYTHON_BIN}" >&2
    exit 1
  fi
fi

pipeline_dir="${session_dir}/quality/egoinfinity_hand_alignment_pipeline"
mkdir -p "${pipeline_dir}"
timing_jsonl="${pipeline_dir}/pipeline_timing.jsonl"
timing_summary="${pipeline_dir}/pipeline_timing_summary.json"
: > "${timing_jsonl}"

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
    "${stage}" "${status}" "${duration}" "${exit_code}" "${start_ns}" "${end_ns}" >> "${timing_jsonl}"
}

record_reuse_stage() {
  local now
  now="$(timer_now_ns)"
  record_stage_timing "$1" "reuse" "${now}" "${now}" 0
}

record_skip_stage() {
  local now
  now="$(timer_now_ns)"
  record_stage_timing "$1" "skip" "${now}" "${now}" 0
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

write_timing_summary() {
  python3 - "${timing_jsonl}" "${timing_summary}" <<'PY'
import json
import sys
from pathlib import Path

jsonl = Path(sys.argv[1])
out = Path(sys.argv[2])
records = []
if jsonl.exists():
    for line in jsonl.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
total = sum(float(r.get("duration_sec", 0.0)) for r in records if r.get("status") == "ok")
by_status = {}
for r in records:
    by_status[r["status"]] = by_status.get(r["status"], 0) + 1
out.write_text(json.dumps({
    "timing_jsonl": str(jsonl),
    "stage_count": len(records),
    "total_processed_sec": total,
    "status_counts": by_status,
    "stages": records,
}, indent=2, ensure_ascii=False) + "\n")
PY
}

trap 'write_timing_summary >/dev/null 2>&1 || true' EXIT

if [[ ! -f "${session_dir}/processed_topcam/left_table.mp4" ]]; then
  if [[ "${PROCESS_TOPCAM}" == "auto" || "${PROCESS_TOPCAM}" == "true" || "${PROCESS_TOPCAM}" == "1" ]]; then
    echo "[egoinfinity_hand_alignment_pipeline] Missing processed_topcam; generating it from episode_0.bag."
    run_stage "process_topcam" \
      env HOST_SESSION_ROOT="${HOST_SESSION_ROOT}" OUTPUT_FPS="${TOPCAM_OUTPUT_FPS}" \
      bash "${LFV_ROOT}/scripts/process_lfv_demo_topcam.sh" "${session_dir}"
  else
    echo "[egoinfinity_hand_alignment_pipeline] Missing processed_topcam/left_table.mp4" >&2
    exit 1
  fi
else
  record_reuse_stage "process_topcam"
fi

raw_dir="${pipeline_dir}/stages/raw_wilor_handresults"
phase_b_dir="${pipeline_dir}/stages/phase_b_track_postprocess"
depth_dir="${pipeline_dir}/stages/foundationstereo_depth"
depth_stabilized_dir="${pipeline_dir}/stages/foundationstereo_depth_stabilized"
phase_c_dir="${pipeline_dir}/stages/phase_c_depth_align"
phase_c1b_dir="${pipeline_dir}/stages/phase_c_depth_smooth"
phase_c1c_dir="${pipeline_dir}/stages/phase_c_motion_infiller"
phase_c2_dir="${pipeline_dir}/stages/phase_c_mano_smooth"
phase_c2b_dir="${pipeline_dir}/stages/phase_c2b_bad_pose_repair"
phase_c3_dir="${pipeline_dir}/stages/phase_c_mesh_visibility"
phase_c4_dir="${pipeline_dir}/stages/phase_c_visibility_depth_realign"
mkdir -p "${raw_dir}" "${phase_b_dir}" "${depth_dir}" "${depth_stabilized_dir}" "${phase_c_dir}" "${phase_c1b_dir}" "${phase_c1c_dir}" "${phase_c2_dir}" "${phase_c2b_dir}" "${phase_c3_dir}" "${phase_c4_dir}"

raw_npz="${raw_dir}/wilor_handresults_raw.npz"
raw_summary="${raw_dir}/wilor_handresults_raw_summary.json"
phase_b_npz="${phase_b_dir}/wilor_handresults_phase_b.npz"
phase_b_summary="${phase_b_dir}/wilor_phase_b_summary.json"
qa_dir="${pipeline_dir}/quality_check"
qa_summary="${qa_dir}/handresults_quality_summary.json"
depth_summary="${depth_dir}/foundationstereo_depth_summary.json"
depth_stabilized_summary="${depth_stabilized_dir}/foundationstereo_depth_stabilized_summary.json"
depth_stabilized_frame_csv="${depth_stabilized_dir}/foundationstereo_depth_stabilized_frames.csv"
depth_stabilize_correction_csv="${depth_stabilized_dir}/depth_stabilize_corrections.csv"
phase_c_npz="${phase_c_dir}/wilor_handresults_phase_c_depth_aligned.npz"
phase_c_summary="${phase_c_dir}/phase_c_alignment_summary.json"
phase_c_qc_csv="${phase_c_dir}/phase_c_alignment_quality.csv"
phase_c_bad_viz_dir="${qa_dir}/phase_c_bad_frames"
phase_c1b_npz="${phase_c1b_dir}/wilor_handresults_phase_c1b_depth_smooth.npz"
phase_c1b_summary="${phase_c1b_dir}/depth_smooth_summary.json"
phase_c1b_qc_csv="${phase_c1b_dir}/depth_smooth_quality.csv"
phase_c1b_track_csv="${phase_c1b_dir}/depth_smooth_track_summary.csv"
phase_c1c_npz="${phase_c1c_dir}/wilor_handresults_phase_c1c_motion_infilled.npz"
phase_c1c_summary="${phase_c1c_dir}/motion_infiller_summary.json"
phase_c1c_qc_csv="${phase_c1c_dir}/motion_infiller_quality.csv"
phase_c2_npz="${phase_c2_dir}/wilor_handresults_phase_c2_mano_smooth.npz"
phase_c2_summary="${phase_c2_dir}/mano_smoothing_summary.json"
phase_c2_qc_csv="${phase_c2_dir}/mano_smoothing_quality.csv"
phase_c2_track_csv="${phase_c2_dir}/mano_smoothing_track_summary.csv"
phase_c2b_npz="${phase_c2b_dir}/wilor_handresults_phase_c2b_bad_pose_repaired.npz"
phase_c2b_summary="${phase_c2b_dir}/bad_pose_repair_summary.json"
phase_c2b_qc_csv="${phase_c2b_dir}/bad_pose_repair_quality.csv"
phase_c3_npz="${phase_c3_dir}/wilor_handresults_phase_c3_mesh_visibility.npz"
phase_c3_summary="${phase_c3_dir}/mano_mesh_visibility_summary.json"
phase_c3_joint_csv="${phase_c3_dir}/mano_mesh_visibility_joints.csv"
phase_c3_candidate_csv="${phase_c3_dir}/mano_mesh_visibility_candidates.csv"
phase_c4_npz="${phase_c4_dir}/wilor_handresults_phase_c4_visibility_depth_realign.npz"
phase_c4_summary="${phase_c4_dir}/visibility_depth_realign_summary.json"
phase_c4_qc_csv="${phase_c4_dir}/visibility_depth_realign_quality.csv"

echo "[egoinfinity_hand_alignment_pipeline] session=${session_dir}"
echo "[egoinfinity_hand_alignment_pipeline] output=${pipeline_dir}"
echo "[egoinfinity_hand_alignment_pipeline] hand=${HAND}"

if [[ "${REBUILD_EGO_HAND}" == "true" || ! -f "${raw_npz}" || ! -f "${raw_summary}" ]]; then
  overlay_arg=()
  if [[ "${SAVE_RAW_OVERLAY}" == "true" ]]; then
    overlay_arg+=(--save-overlay)
  fi
  run_stage "raw_wilor_export" \
    "${WILOR_PYTHON}" "${SCRIPT_DIR}/export_wilor_handresults.py" \
    --session-dir "${session_dir}" \
    --wilor-root "${WILOR_ROOT}" \
    --video "${session_dir}/processed_topcam/left_table.mp4" \
    --output-dir "${raw_dir}" \
    --hand "${HAND}" \
    --conf "${WILOR_CONF}" \
    --rescale-factor "${WILOR_RESCALE_FACTOR}" \
    --max-frames "${MAX_FRAMES}" \
    --frame-start "${FRAME_START}" \
    --frame-end "${FRAME_END}" \
    --frame-stride "${FRAME_STRIDE}" \
    "${overlay_arg[@]}"
else
  echo "[egoinfinity_hand_alignment_pipeline] reuse ${raw_npz}"
  record_reuse_stage "raw_wilor_export"
fi

if [[ "${REBUILD_EGO_HAND}" == "true" || ! -f "${phase_b_npz}" || ! -f "${phase_b_summary}" ]]; then
  run_stage "phase_b_track_postprocess" \
    "${WILOR_PYTHON}" "${SCRIPT_DIR}/postprocess_handresults.py" \
    --input-npz "${raw_npz}" \
    --output-dir "${phase_b_dir}"
else
  echo "[egoinfinity_hand_alignment_pipeline] reuse ${phase_b_npz}"
  record_reuse_stage "phase_b_track_postprocess"
fi

mkdir -p "${qa_dir}"
run_stage "quality_check_handresults" \
  "${WILOR_PYTHON}" "${SCRIPT_DIR}/quality_check_handresults.py" \
  --raw-npz "${raw_npz}" \
  --phase-b-npz "${phase_b_npz}" \
  --output-dir "${qa_dir}" \
  --video "${session_dir}/processed_topcam/left_table.mp4"

if [[ "${RUN_PHASE_C}" == "true" || "${RUN_PHASE_C}" == "1" ]]; then
  lr_args=()
  if [[ "${LR_CHECK}" == "true" || "${LR_CHECK}" == "1" ]]; then
    lr_args+=(--lr-check)
  fi
  preview_args=()
  if [[ "${WRITE_DEPTH_PREVIEW}" == "true" || "${WRITE_DEPTH_PREVIEW}" == "1" ]]; then
    preview_args+=(--write-preview)
  fi
  if [[ "${REBUILD_DEPTH}" == "true" || ! -f "${depth_summary}" ]]; then
    echo "[egoinfinity_hand_alignment_pipeline] Running FoundationStereo depth."
    run_stage "foundationstereo_depth" \
      "${PYTHON_BIN}" "${LFV_ROOT}/scripts/run_lfv_foundationstereo_disparity.py" \
      --session-dir "${session_dir}" \
      --output-dir "${depth_dir}" \
      --backend foundationstereo \
      --foundationstereo-root "${FOUNDATIONSTEREO_ROOT}" \
      --stereo-model "${STEREO_MODEL}" \
      --valid-iters "${VALID_ITERS}" \
      --frame-start "${FRAME_START}" \
      --frame-end "${FRAME_END}" \
      --stride "${FRAME_STRIDE}" \
      --max-frames "${MAX_FRAMES}" \
      --max-depth-m "${MAX_DEPTH_M}" \
      "${lr_args[@]}" \
      "${preview_args[@]}"
  else
    echo "[egoinfinity_hand_alignment_pipeline] reuse ${depth_summary}"
    record_reuse_stage "foundationstereo_depth"
  fi

  phase_c_depth_summary="${depth_summary}"
  if [[ "${RUN_PHASE_C_DEPTH_STABILIZE}" == "true" || "${RUN_PHASE_C_DEPTH_STABILIZE}" == "1" ]]; then
    depth_stabilize_args=()
    if [[ "${DEPTH_STABILIZE_USE_FLOW_MASK}" == "true" || "${DEPTH_STABILIZE_USE_FLOW_MASK}" == "1" ]]; then
      depth_stabilize_args+=(--use-flow-mask)
    fi
    if [[ "${DEPTH_STABILIZE_WRITE_DYNAMIC_MASKS}" == "true" || "${DEPTH_STABILIZE_WRITE_DYNAMIC_MASKS}" == "1" ]]; then
      depth_stabilize_args+=(--write-dynamic-masks)
    fi
    if [[ "${REBUILD_PHASE_C_DEPTH_STABILIZE}" == "true" || ! -f "${depth_stabilized_summary}" || ! -f "${depth_stabilized_frame_csv}" || "${depth_summary}" -nt "${depth_stabilized_summary}" || "${phase_b_npz}" -nt "${depth_stabilized_summary}" ]]; then
      echo "[egoinfinity_hand_alignment_pipeline] Running Phase-C0b depth stabilization."
      run_stage "phase_c0b_depth_stabilize" \
        "${WILOR_PYTHON}" "${SCRIPT_DIR}/phase_c0b_depth_stabilize.py" \
        --session-dir "${session_dir}" \
        --phase-b-npz "${phase_b_npz}" \
        --depth-summary-json "${depth_summary}" \
        --output-dir "${depth_stabilized_dir}" \
        --bbox-margin "${DEPTH_STABILIZE_BBOX_MARGIN}" \
        --template-min-valid-ratio "${DEPTH_STABILIZE_TEMPLATE_MIN_VALID_RATIO}" \
        --template-stride "${DEPTH_STABILIZE_TEMPLATE_STRIDE}" \
        "${depth_stabilize_args[@]}"
    else
      echo "[egoinfinity_hand_alignment_pipeline] reuse ${depth_stabilized_summary}"
      record_reuse_stage "phase_c0b_depth_stabilize"
    fi
    phase_c_depth_summary="${depth_stabilized_summary}"
  else
    record_skip_stage "phase_c0b_depth_stabilize"
  fi

  phase_c_args=()
  if [[ "${DEPTH_STABILIZE}" == "true" || "${DEPTH_STABILIZE}" == "1" ]]; then
    phase_c_args+=(--stabilize-depth)
  fi
  if [[ "${USE_FLOW_MASK}" == "true" || "${USE_FLOW_MASK}" == "1" ]]; then
    phase_c_args+=(--use-flow-mask)
  fi
  if [[ "${WRITE_STABLE_DEPTH}" == "true" || "${WRITE_STABLE_DEPTH}" == "1" ]]; then
    phase_c_args+=(--write-stable-depth)
  fi
  phase_c_input_mismatch="false"
  if [[ -f "${phase_c_summary}" ]] && ! grep -Fq "\"depth_summary_json\": \"${phase_c_depth_summary}\"" "${phase_c_summary}"; then
    phase_c_input_mismatch="true"
  fi
  if [[ "${REBUILD_PHASE_C}" == "true" || "${phase_c_input_mismatch}" == "true" || ! -f "${phase_c_npz}" || ! -f "${phase_c_summary}" || "${phase_c_depth_summary}" -nt "${phase_c_npz}" || "${phase_b_npz}" -nt "${phase_c_npz}" ]]; then
    echo "[egoinfinity_hand_alignment_pipeline] Running Phase-C FoundationStereo depth alignment."
    run_stage "phase_c_foundation_depth_align" \
      "${WILOR_PYTHON}" "${SCRIPT_DIR}/phase_c_foundation_depth_align.py" \
      --session-dir "${session_dir}" \
      --input-npz "${phase_b_npz}" \
      --depth-summary-json "${phase_c_depth_summary}" \
      --output-dir "${phase_c_dir}" \
      --frame-start "${FRAME_START}" \
      --frame-end "${FRAME_END}" \
      --align-patch-size "${ALIGN_PATCH_SIZE}" \
      --min-reliable-joints "${ALIGN_MIN_RELIABLE_JOINTS}" \
      "${phase_c_args[@]}"
  else
    echo "[egoinfinity_hand_alignment_pipeline] reuse ${phase_c_npz}"
    record_reuse_stage "phase_c_foundation_depth_align"
  fi

  phase_c2_input_npz="${phase_c_npz}"
  if [[ "${RUN_PHASE_C_DEPTH_SMOOTH}" == "true" || "${RUN_PHASE_C_DEPTH_SMOOTH}" == "1" ]]; then
    depth_smooth_vertex_arg=(--smooth-vertex-mean-z)
    if [[ "${DEPTH_SMOOTH_VERTEX_MEAN_Z}" == "false" || "${DEPTH_SMOOTH_VERTEX_MEAN_Z}" == "0" ]]; then
      depth_smooth_vertex_arg=(--no-smooth-vertex-mean-z)
    fi
    phase_c1b_input_mismatch="false"
    if [[ -f "${phase_c1b_summary}" ]] && ! grep -Fq "\"input_npz\": \"${phase_c_npz}\"" "${phase_c1b_summary}"; then
      phase_c1b_input_mismatch="true"
    fi
    if [[ -f "${phase_c1b_summary}" ]] && ! grep -Fq "\"depth_summary_json\": \"${phase_c_depth_summary}\"" "${phase_c1b_summary}"; then
      phase_c1b_input_mismatch="true"
    fi
    if [[ "${REBUILD_PHASE_C_DEPTH_SMOOTH}" == "true" || "${phase_c1b_input_mismatch}" == "true" || ! -f "${phase_c1b_npz}" || ! -f "${phase_c1b_summary}" || "${phase_c_npz}" -nt "${phase_c1b_npz}" || "${phase_c_depth_summary}" -nt "${phase_c1b_npz}" ]]; then
      echo "[egoinfinity_hand_alignment_pipeline] Running Phase-C1b EgoInfinity depth smoothing."
      run_stage "phase_c1b_depth_smooth" \
        "${WILOR_PYTHON}" "${SCRIPT_DIR}/phase_c1b_depth_smooth.py" \
        --session-dir "${session_dir}" \
        --input-npz "${phase_c_npz}" \
        --depth-summary-json "${phase_c_depth_summary}" \
        --output-dir "${phase_c1b_dir}" \
        --sigma-z "${DEPTH_SMOOTH_SIGMA_Z}" \
        --mad-factor "${DEPTH_SMOOTH_MAD_FACTOR}" \
        --min-inliers "${DEPTH_SMOOTH_MIN_INLIERS}" \
        --patch-size "${DEPTH_SMOOTH_PATCH_SIZE}" \
        --max-delta-z-m "${DEPTH_SMOOTH_MAX_DELTA_Z_M}" \
        "${depth_smooth_vertex_arg[@]}"
    else
      echo "[egoinfinity_hand_alignment_pipeline] reuse ${phase_c1b_npz}"
      record_reuse_stage "phase_c1b_depth_smooth"
    fi
    phase_c2_input_npz="${phase_c1b_npz}"
  else
    record_skip_stage "phase_c1b_depth_smooth"
  fi

  if [[ "${RUN_MOTION_INFILL}" == "true" || "${RUN_MOTION_INFILL}" == "1" ]]; then
    motion_input_mismatch="false"
    if [[ -f "${phase_c1c_summary}" ]] && ! grep -Fq "\"input_npz\": \"${phase_c2_input_npz}\"" "${phase_c1c_summary}"; then
      motion_input_mismatch="true"
    fi
    if [[ "${REBUILD_MOTION_INFILL}" == "true" || "${motion_input_mismatch}" == "true" || ! -f "${phase_c1c_npz}" || ! -f "${phase_c1c_summary}" || "${phase_c2_input_npz}" -nt "${phase_c1c_npz}" ]]; then
      echo "[egoinfinity_hand_alignment_pipeline] Running Phase-C1c EgoInfinity MotionInfiller."
      run_stage "phase_c1c_motion_infiller" \
        "${WILOR_PYTHON}" "${SCRIPT_DIR}/phase_c1c_motion_infiller.py" \
        --session-dir "${session_dir}" \
        --input-npz "${phase_c2_input_npz}" \
        --output-dir "${phase_c1c_dir}" \
        --wilor-root "${WILOR_ROOT}" \
        --checkpoint "${MOTION_INFILLER_CHECKPOINT}" \
        --device "${MOTION_INFILL_DEVICE}"
    else
      echo "[egoinfinity_hand_alignment_pipeline] reuse ${phase_c1c_npz}"
      record_reuse_stage "phase_c1c_motion_infiller"
    fi
    phase_c2_input_npz="${phase_c1c_npz}"
  else
    record_skip_stage "phase_c1c_motion_infiller"
  fi

  if [[ "${RUN_PHASE_C2}" == "true" || "${RUN_PHASE_C2}" == "1" ]]; then
    phase_c2_input_mismatch="false"
    if [[ -f "${phase_c2_summary}" ]] && ! grep -Fq "\"input_npz\": \"${phase_c2_input_npz}\"" "${phase_c2_summary}"; then
      phase_c2_input_mismatch="true"
    fi
    if [[ "${REBUILD_PHASE_C2}" == "true" || "${phase_c2_input_mismatch}" == "true" || ! -f "${phase_c2_npz}" || ! -f "${phase_c2_summary}" || "${phase_c2_input_npz}" -nt "${phase_c2_npz}" ]]; then
      echo "[egoinfinity_hand_alignment_pipeline] Running Phase-C2 MANO temporal smoothing."
      run_stage "phase_c2_mano_temporal_smooth" \
        "${WILOR_PYTHON}" "${SCRIPT_DIR}/phase_c2_mano_temporal_smooth.py" \
        --session-dir "${session_dir}" \
        --input-npz "${phase_c2_input_npz}" \
        --output-dir "${phase_c2_dir}" \
        --wilor-root "${WILOR_ROOT}" \
        --smooth-window "${SMOOTH_MANO_WINDOW}" \
        --smooth-polyorder "${SMOOTH_MANO_POLYORDER}" \
        --min-track-frames "${PHASE_C2_MIN_TRACK_FRAMES}"
    else
      echo "[egoinfinity_hand_alignment_pipeline] reuse ${phase_c2_npz}"
      record_reuse_stage "phase_c2_mano_temporal_smooth"
    fi
  else
    record_skip_stage "phase_c2_mano_temporal_smooth"
  fi

  phase_c3_input_npz="${phase_c2_npz}"
  if [[ "${RUN_PHASE_C2B_REPAIR}" == "true" || "${RUN_PHASE_C2B_REPAIR}" == "1" ]]; then
    if [[ ! -f "${phase_c2_npz}" ]]; then
      echo "[egoinfinity_hand_alignment_pipeline] Phase-C2b requires Phase-C2 NPZ: ${phase_c2_npz}" >&2
      exit 1
    fi
    phase_c2b_input_mismatch="false"
    if [[ -f "${phase_c2b_summary}" ]] && ! grep -Fq "\"input_npz\": \"${phase_c2_npz}\"" "${phase_c2b_summary}"; then
      phase_c2b_input_mismatch="true"
    fi
    if [[ "${REBUILD_PHASE_C2B_REPAIR}" == "true" || "${phase_c2b_input_mismatch}" == "true" || ! -f "${phase_c2b_npz}" || ! -f "${phase_c2b_summary}" || "${phase_c2_npz}" -nt "${phase_c2b_npz}" ]]; then
      echo "[egoinfinity_hand_alignment_pipeline] Running Phase-C2b bad-pose repair."
      run_stage "phase_c2b_bad_pose_repair" \
        "${WILOR_PYTHON}" "${SCRIPT_DIR}/phase_c2b_bad_pose_repair.py" \
        --session-dir "${session_dir}" \
        --input-npz "${phase_c2_npz}" \
        --output-dir "${phase_c2b_dir}" \
        --bad-global-rot-delta-deg "${PHASE_C2B_BAD_GLOBAL_ROT_DELTA_DEG}" \
        --bad-raw-global-rot-jump-deg "${PHASE_C2B_BAD_RAW_GLOBAL_ROT_JUMP_DEG}" \
        --bad-smooth-global-rot-jump-deg "${PHASE_C2B_BAD_SMOOTH_GLOBAL_ROT_JUMP_DEG}" \
        --bad-infilled-wrist-jump-m "${PHASE_C2B_BAD_INFILLED_WRIST_JUMP_M}" \
        --neighbor-window-frames "${PHASE_C2B_NEIGHBOR_WINDOW_FRAMES}" \
        --bridge-good-gap-frames "${PHASE_C2B_BRIDGE_GOOD_GAP_FRAMES}"
    else
      echo "[egoinfinity_hand_alignment_pipeline] reuse ${phase_c2b_npz}"
      record_reuse_stage "phase_c2b_bad_pose_repair"
    fi
    phase_c3_input_npz="${phase_c2b_npz}"
  else
    record_skip_stage "phase_c2b_bad_pose_repair"
  fi

  if [[ "${RUN_PHASE_C3}" == "true" || "${RUN_PHASE_C3}" == "1" ]]; then
    if [[ ! -f "${phase_c3_input_npz}" ]]; then
      echo "[egoinfinity_hand_alignment_pipeline] Phase-C3 requires input NPZ: ${phase_c3_input_npz}" >&2
      exit 1
    fi
    phase_c3_input_mismatch="false"
    if [[ -f "${phase_c3_summary}" ]] && ! grep -Fq "\"input_npz\": \"${phase_c3_input_npz}\"" "${phase_c3_summary}"; then
      phase_c3_input_mismatch="true"
    fi
    if [[ "${REBUILD_PHASE_C3}" == "true" || "${phase_c3_input_mismatch}" == "true" || ! -f "${phase_c3_npz}" || ! -f "${phase_c3_summary}" || "${phase_c3_input_npz}" -nt "${phase_c3_npz}" ]]; then
      echo "[egoinfinity_hand_alignment_pipeline] Running Phase-C3 MANO mesh visibility."
      run_stage "phase_c3_mano_mesh_visibility" \
        "${WILOR_PYTHON}" "${SCRIPT_DIR}/phase_c3_mano_mesh_visibility.py" \
        --session-dir "${session_dir}" \
        --input-npz "${phase_c3_input_npz}" \
        --output-dir "${phase_c3_dir}" \
        --epsilon-m "${MESH_VISIBILITY_EPSILON_M}" \
        --nearest-vertices "${MESH_VISIBILITY_NEAREST_VERTICES}" \
        --joint-visible-ratio-threshold "${MESH_VISIBILITY_RATIO_THRESHOLD}"
    else
      echo "[egoinfinity_hand_alignment_pipeline] reuse ${phase_c3_npz}"
      record_reuse_stage "phase_c3_mano_mesh_visibility"
    fi
  else
    record_skip_stage "phase_c3_mano_mesh_visibility"
  fi

  if [[ "${RUN_VISIBILITY_REALIGN}" == "true" || "${RUN_VISIBILITY_REALIGN}" == "1" ]]; then
    if [[ ! -f "${phase_c3_npz}" ]]; then
      echo "[egoinfinity_hand_alignment_pipeline] Visibility re-align requires Phase-C3 NPZ: ${phase_c3_npz}" >&2
      exit 1
    fi
    vis_realign_args=()
    if [[ "${VIS_REALIGN_ENABLE_ALL_VISIBLE_FALLBACK}" == "true" || "${VIS_REALIGN_ENABLE_ALL_VISIBLE_FALLBACK}" == "1" ]]; then
      vis_realign_args+=(--enable-all-visible-fallback)
    fi
    if [[ "${VIS_REALIGN_KEEP_PREVIOUS_ON_BAD_RMS}" == "false" || "${VIS_REALIGN_KEEP_PREVIOUS_ON_BAD_RMS}" == "0" ]]; then
      vis_realign_args+=(--no-keep-previous-on-bad-rms)
    fi
    if [[ "${REBUILD_VISIBILITY_REALIGN}" == "true" || ! -f "${phase_c4_npz}" || ! -f "${phase_c4_summary}" || "${phase_c3_npz}" -nt "${phase_c4_npz}" ]]; then
      echo "[egoinfinity_hand_alignment_pipeline] Running Phase-C4 visibility-aware depth re-align (experimental)."
      run_stage "phase_c4_visibility_depth_realign" \
        "${WILOR_PYTHON}" "${SCRIPT_DIR}/phase_c4_visibility_depth_realign.py" \
        --session-dir "${session_dir}" \
        --input-npz "${phase_c3_npz}" \
        --depth-summary-json "${phase_c_depth_summary}" \
        --output-dir "${phase_c4_dir}" \
        --patch-size "${VIS_REALIGN_PATCH_SIZE}" \
        --min-visible-reliable-joints "${VIS_REALIGN_MIN_VISIBLE_RELIABLE_JOINTS}" \
        --min-visible-stable-joints "${VIS_REALIGN_MIN_VISIBLE_STABLE_JOINTS}" \
        --max-patch-spread-m "${VIS_REALIGN_MAX_PATCH_SPREAD_M}" \
        --max-patch-mad-m "${VIS_REALIGN_MAX_PATCH_MAD_M}" \
        "${vis_realign_args[@]}"
    else
      echo "[egoinfinity_hand_alignment_pipeline] reuse ${phase_c4_npz}"
      record_reuse_stage "phase_c4_visibility_depth_realign"
    fi
  else
    record_skip_stage "phase_c4_visibility_depth_realign"
  fi

  full_qc_args=()
  if [[ "${ALLOW_PARTIAL_QC}" == "true" || "${ALLOW_PARTIAL_QC}" == "1" ]]; then
    full_qc_args+=(--allow-partial-coverage)
  fi
  phase_c2_qc_args=()
  if [[ -f "${phase_c1b_summary}" ]]; then
    phase_c2_qc_args+=(
      --phase-c1b-npz "${phase_c1b_npz}"
      --phase-c1b-summary "${phase_c1b_summary}"
      --phase-c1b-quality-csv "${phase_c1b_qc_csv}"
      --phase-c1b-track-csv "${phase_c1b_track_csv}"
    )
  fi
  if [[ ("${RUN_PHASE_C_DEPTH_STABILIZE}" == "true" || "${RUN_PHASE_C_DEPTH_STABILIZE}" == "1") && -f "${depth_stabilized_summary}" ]]; then
    phase_c2_qc_args+=(
      --phase-c0b-summary "${depth_stabilized_summary}"
      --phase-c0b-frame-csv "${depth_stabilized_frame_csv}"
      --phase-c0b-correction-csv "${depth_stabilize_correction_csv}"
    )
  fi
  if [[ ("${RUN_MOTION_INFILL}" == "true" || "${RUN_MOTION_INFILL}" == "1") && -f "${phase_c1c_summary}" ]]; then
    phase_c2_qc_args+=(
      --phase-c1c-npz "${phase_c1c_npz}"
      --phase-c1c-summary "${phase_c1c_summary}"
      --phase-c1c-quality-csv "${phase_c1c_qc_csv}"
    )
  fi
  if [[ -f "${phase_c2_summary}" ]]; then
    phase_c2_qc_args+=(
      --phase-c2-npz "${phase_c2_npz}"
      --phase-c2-summary "${phase_c2_summary}"
      --phase-c2-quality-csv "${phase_c2_qc_csv}"
    )
  fi
  if [[ -f "${phase_c3_summary}" ]]; then
    phase_c2_qc_args+=(
      --phase-c3-npz "${phase_c3_npz}"
      --phase-c3-summary "${phase_c3_summary}"
      --phase-c3-joint-csv "${phase_c3_joint_csv}"
      --phase-c3-candidate-csv "${phase_c3_candidate_csv}"
    )
  fi
  if [[ -f "${phase_c4_summary}" ]]; then
    phase_c2_qc_args+=(
      --phase-c4-npz "${phase_c4_npz}"
      --phase-c4-summary "${phase_c4_summary}"
      --phase-c4-quality-csv "${phase_c4_qc_csv}"
    )
  fi

  run_stage "phase_c_quality_gates" \
    "${WILOR_PYTHON}" "${SCRIPT_DIR}/phase_c_quality_gates.py" \
    --session-dir "${session_dir}" \
    --phase-b-npz "${phase_b_npz}" \
    --foundation-depth-summary "${phase_c_depth_summary}" \
    --phase-c-npz "${phase_c_npz}" \
    --phase-c-summary "${phase_c_summary}" \
    --phase-c-quality-csv "${phase_c_qc_csv}" \
    "${full_qc_args[@]}" \
    "${phase_c2_qc_args[@]}" \
    --json-out "${qa_dir}/phase_c_node_quality_gates.json"

  if [[ "${RUN_PHASE_C_BAD_VIZ}" == "true" || "${RUN_PHASE_C_BAD_VIZ}" == "1" ]]; then
    echo "[egoinfinity_hand_alignment_pipeline] Rendering Phase-C bad-frame visualization."
    run_stage "phase_c_bad_frame_visualization" \
      "${WILOR_PYTHON}" "${SCRIPT_DIR}/visualize_phase_c_bad_frames.py" \
      --session-dir "${session_dir}" \
      --phase-c-npz "${phase_c_npz}" \
      --alignment-csv "${phase_c_qc_csv}" \
      --video "${session_dir}/processed_topcam/left_table.mp4" \
      --output-dir "${phase_c_bad_viz_dir}" \
      --context-frames "${PHASE_C_BAD_VIZ_CONTEXT}" \
      --max-sheet-frames "${PHASE_C_BAD_VIZ_MAX_SHEET_FRAMES}"
  else
    record_skip_stage "phase_c_bad_frame_visualization"
  fi
else
  record_skip_stage "foundationstereo_depth"
  record_skip_stage "phase_c0b_depth_stabilize"
  record_skip_stage "phase_c_foundation_depth_align"
  record_skip_stage "phase_c1b_depth_smooth"
  record_skip_stage "phase_c1c_motion_infiller"
  record_skip_stage "phase_c2_mano_temporal_smooth"
  record_skip_stage "phase_c2b_bad_pose_repair"
  record_skip_stage "phase_c3_mano_mesh_visibility"
  record_skip_stage "phase_c4_visibility_depth_realign"
  record_skip_stage "phase_c_quality_gates"
  record_skip_stage "phase_c_bad_frame_visualization"
fi

write_timing_summary

cat <<EOF

[egoinfinity_hand_alignment_pipeline] Done.
  pipeline_dir: ${pipeline_dir}
  raw_handresults_npz: ${raw_npz}
  raw_predictions_csv: ${raw_dir}/wilor_predictions_raw.csv
  raw_detections_csv: ${raw_dir}/wilor_detections_raw.csv
  raw_summary_json: ${raw_summary}
  phase_b_handresults_npz: ${phase_b_npz}
  phase_b_predictions_csv: ${phase_b_dir}/wilor_predictions_phase_b.csv
  phase_b_events_csv: ${phase_b_dir}/wilor_phase_b_events.csv
  phase_b_summary_json: ${phase_b_summary}
  quality_summary_json: ${qa_summary}
  quality_per_candidate_csv: ${qa_dir}/handresults_quality_per_candidate.csv
  quality_track_timeline_png: ${qa_dir}/handresults_track_timeline.png
  quality_contact_sheet_jpg: ${qa_dir}/handresults_phase_b_contact_sheet.jpg
  foundation_depth_summary_json: ${depth_summary}
  foundation_depth_stabilized_summary_json: ${depth_stabilized_summary}
  foundation_depth_stabilized_frame_csv: ${depth_stabilized_frame_csv}
  depth_stabilize_correction_csv: ${depth_stabilize_correction_csv}
  phase_c_depth_aligned_npz: ${phase_c_npz}
  phase_c_alignment_quality_csv: ${phase_c_qc_csv}
  phase_c_alignment_summary_json: ${phase_c_summary}
  phase_c1b_depth_smooth_npz: ${phase_c1b_npz}
  phase_c1b_depth_smooth_summary_json: ${phase_c1b_summary}
  phase_c1b_depth_smooth_quality_csv: ${phase_c1b_qc_csv}
  phase_c1b_depth_smooth_track_summary_csv: ${phase_c1b_track_csv}
  phase_c1c_motion_infiller_npz: ${phase_c1c_npz}
  phase_c1c_motion_infiller_summary_json: ${phase_c1c_summary}
  phase_c1c_motion_infiller_quality_csv: ${phase_c1c_qc_csv}
  phase_c2_mano_smooth_npz: ${phase_c2_npz}
  phase_c2_mano_smooth_summary_json: ${phase_c2_summary}
  phase_c2_mano_smooth_quality_csv: ${phase_c2_qc_csv}
  phase_c2_mano_smooth_track_summary_csv: ${phase_c2_track_csv}
  phase_c2b_bad_pose_repair_npz: ${phase_c2b_npz}
  phase_c2b_bad_pose_repair_summary_json: ${phase_c2b_summary}
  phase_c2b_bad_pose_repair_quality_csv: ${phase_c2b_qc_csv}
  phase_c3_mesh_visibility_npz: ${phase_c3_npz}
  phase_c3_mesh_visibility_summary_json: ${phase_c3_summary}
  phase_c3_mesh_visibility_joint_csv: ${phase_c3_joint_csv}
  phase_c3_mesh_visibility_candidate_csv: ${phase_c3_candidate_csv}
  phase_c4_visibility_realign_npz: ${phase_c4_npz}
  phase_c4_visibility_realign_summary_json: ${phase_c4_summary}
  phase_c4_visibility_realign_quality_csv: ${phase_c4_qc_csv}
  phase_c_bad_alignment_overlay_mp4: ${phase_c_bad_viz_dir}/phase_c_bad_alignment_overlay.mp4
  phase_c_bad_alignment_contact_sheet_jpg: ${phase_c_bad_viz_dir}/phase_c_bad_alignment_contact_sheet.jpg
  phase_c_bad_alignment_summary_json: ${phase_c_bad_viz_dir}/phase_c_bad_alignment_viz_summary.json
  pipeline_timing_jsonl: ${timing_jsonl}
  pipeline_timing_summary_json: ${timing_summary}
EOF
