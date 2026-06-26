#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LFV_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HOST_ROS_ROOT="${HOST_ROS_ROOT:-/home/yannan/workspace/ros1_docker-main}"
HOST_SESSION_ROOT="${HOST_SESSION_ROOT:-${HOST_ROS_ROOT}/rosbag_data/human_teaching_videos}"
CONTAINER_ROOT="${CONTAINER_ROOT:-/workspace/ros1_docker_jinhe}"
WILOR_PYTHON="${WILOR_PYTHON:-/home/yannan/miniforge3/envs/wilor_lfv/bin/python}"
PYTHON_BIN="${PYTHON_BIN:-${LFV_ROOT}/.venv-dinosam/bin/python}"

SIDE="${SIDE:-both}"
SMOKE="${SMOKE:-false}"
SMOKE_FRAME_START="${SMOKE_FRAME_START:-200}"
SMOKE_FRAME_END="${SMOKE_FRAME_END:-230}"
SMOKE_FRAME_STRIDE="${SMOKE_FRAME_STRIDE:-1}"
SMOKE_MAX_FRAMES="${SMOKE_MAX_FRAMES:-0}"

if [[ "${SMOKE}" == "true" || "${SMOKE}" == "1" ]]; then
  DEFAULT_REBUILD="true"
  DEFAULT_VALID_ITERS="4"
  DEFAULT_ALLOW_PARTIAL_QC="true"
  DEFAULT_RUN_BAD_VIZ="false"
  DEFAULT_WRITE_DEPTH_PREVIEW="false"
  DEFAULT_RENDER_ANCHOR_OVERLAY="false"
  DEFAULT_RUN_PREIK_HTML="false"
  DEFAULT_RENDER_STRIDE="999999"
else
  DEFAULT_REBUILD="false"
  DEFAULT_VALID_ITERS="16"
  DEFAULT_ALLOW_PARTIAL_QC="false"
  DEFAULT_RUN_BAD_VIZ="false"
  DEFAULT_WRITE_DEPTH_PREVIEW="true"
  DEFAULT_RENDER_ANCHOR_OVERLAY="true"
  DEFAULT_RUN_PREIK_HTML="true"
  DEFAULT_RENDER_STRIDE="2"
fi

BASELINE_REBUILD="${BASELINE_REBUILD:-${DEFAULT_REBUILD}}"
BASELINE_VALID_ITERS="${BASELINE_VALID_ITERS:-${DEFAULT_VALID_ITERS}}"
BASELINE_ALLOW_PARTIAL_QC="${BASELINE_ALLOW_PARTIAL_QC:-${DEFAULT_ALLOW_PARTIAL_QC}}"
BASELINE_RUN_BAD_VIZ="${BASELINE_RUN_BAD_VIZ:-${DEFAULT_RUN_BAD_VIZ}}"
BASELINE_WRITE_DEPTH_PREVIEW="${BASELINE_WRITE_DEPTH_PREVIEW:-${DEFAULT_WRITE_DEPTH_PREVIEW}}"
BASELINE_SEGMENTER="${BASELINE_SEGMENTER:-sam}"
BASELINE_RENDER_ANCHOR_OVERLAY="${BASELINE_RENDER_ANCHOR_OVERLAY:-${DEFAULT_RENDER_ANCHOR_OVERLAY}}"
BASELINE_RUN_PREIK_HTML="${BASELINE_RUN_PREIK_HTML:-${DEFAULT_RUN_PREIK_HTML}}"
BASELINE_RENDER_STRIDE="${BASELINE_RENDER_STRIDE:-${DEFAULT_RENDER_STRIDE}}"
BASELINE_RUN_PINOCCHIO_IK="${BASELINE_RUN_PINOCCHIO_IK:-true}"
BASELINE_RUN_PYROKI_IK="${BASELINE_RUN_PYROKI_IK:-false}"
BASELINE_SKIP_NO_HAND="${BASELINE_SKIP_NO_HAND:-true}"
BASELINE_SIDE_SPEC="${BASELINE_SIDES:-${SIDE}}"

ANCHOR_STAGE_NAME="${ANCHOR_STAGE_NAME:-phase_c2_anchor_locked_depth_patch2_pose_repaired}"
C3_STAGE_NAME="${C3_STAGE_NAME:-phase_c3_mesh_visibility_anchor_locked_patch2_pose_repaired}"
PHASE_D_STAGE_NAME="${PHASE_D_STAGE_NAME:-phase_d_preik_anchor_locked_patch2_pose_repaired}"
PHASE_E_STAGE_NAME="${PHASE_E_STAGE_NAME:-phase_e_piper_gripper_base_ik_input_anchor_locked_patch2_pose_repaired}"
IK_STAGE_NAME_OVERRIDE="${IK_STAGE_NAME:-}"

ORIENTATION_ALIGN_RPY="${ORIENTATION_ALIGN_RPY:-0,-1.57079632679,-1.57079632679}"
IK_MODE="${IK_MODE:-pose}"
ORIENTATION_SOURCE="${ORIENTATION_SOURCE:-rot6d}"
TARGET_MODE="${TARGET_MODE:-absolute}"
SCALE="${SCALE:-1.0}"
TCP_OFFSET_XYZ="${TCP_OFFSET_XYZ:-0,0,0}"

usage() {
  cat <<EOF
Usage:
  bash local_pipelines/egoinfinity_hand_alignment_pipeline/run_axisperm_pose_baseline_pipeline.sh <demo_name_or_session_dir>

Runs the current confirmed baseline:
  WiLoR/EgoInfinity hand pipeline -> anchor-locked patch2 C3 repaired output
  -> gripper-base mapping -> Pinocchio pose IK with axis permutation.

Smoke example:
  SMOKE=true SMOKE_FRAME_START=200 SMOKE_FRAME_END=230 \\
    bash local_pipelines/egoinfinity_hand_alignment_pipeline/run_axisperm_pose_baseline_pipeline.sh bag_20260622_1548_001

Useful env:
  SIDE=${SIDE}                         # both, right, left, or "right,left"
  BASELINE_SIDES=${BASELINE_SIDE_SPEC} # overrides SIDE when set
  SMOKE=${SMOKE}
  BASELINE_SEGMENTER=${BASELINE_SEGMENTER}       # sam, grabcut, bbox
  BASELINE_VALID_ITERS=${BASELINE_VALID_ITERS}
  BASELINE_REBUILD=${BASELINE_REBUILD}
  BASELINE_RUN_PINOCCHIO_IK=${BASELINE_RUN_PINOCCHIO_IK}
  BASELINE_RUN_PYROKI_IK=${BASELINE_RUN_PYROKI_IK}
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
    "${stage}" "${status}" "${duration}" "${exit_code}" "${start_ns}" "${end_ns}" >> "${baseline_timing_jsonl}"
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

record_reuse_stage() {
  local now
  now="$(timer_now_ns)"
  record_stage_timing "$1" "reuse" "${now}" "${now}" 0
}

write_timing_summary() {
  python3 - "${baseline_timing_jsonl}" "${baseline_timing_summary}" <<'PY'
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
  echo "[axisperm_baseline] Missing demo name or session dir." >&2
  usage >&2
  exit 1
fi

if [[ "${target}" = /* ]]; then
  source_session_dir="$(readlink -f "${target}")"
else
  source_session_dir="$(readlink -f "${HOST_SESSION_ROOT}/${target}")"
fi

if [[ ! -d "${source_session_dir}" ]]; then
  echo "[axisperm_baseline] Session not found: ${source_session_dir}" >&2
  exit 1
fi

case "${source_session_dir}" in
  "${HOST_ROS_ROOT}"/*) ;;
  *)
    echo "[axisperm_baseline] Session must be under ${HOST_ROS_ROOT} for Docker visibility." >&2
    echo "  got: ${source_session_dir}" >&2
    exit 1
    ;;
esac

declare -a run_sides=()
seen_right=false
seen_left=false
for side_token in ${BASELINE_SIDE_SPEC//,/ }; do
  case "${side_token}" in
    both)
      if [[ "${seen_right}" == "false" ]]; then run_sides+=(right); seen_right=true; fi
      if [[ "${seen_left}" == "false" ]]; then run_sides+=(left); seen_left=true; fi
      ;;
    right)
      if [[ "${seen_right}" == "false" ]]; then run_sides+=(right); seen_right=true; fi
      ;;
    left)
      if [[ "${seen_left}" == "false" ]]; then run_sides+=(left); seen_left=true; fi
      ;;
    "")
      ;;
    *)
      echo "[axisperm_baseline] SIDE/BASELINE_SIDES must be both, right, left, or a comma-separated list; got ${BASELINE_SIDE_SPEC}" >&2
      exit 1
      ;;
  esac
done
if [[ "${#run_sides[@]}" -eq 0 ]]; then
  echo "[axisperm_baseline] No sides selected from SIDE/BASELINE_SIDES=${BASELINE_SIDE_SPEC}" >&2
  exit 1
fi

session_dir="${source_session_dir}"
frame_start="${FRAME_START:-0}"
frame_end="${FRAME_END:--1}"
frame_stride="${FRAME_STRIDE:-1}"
max_frames="${MAX_FRAMES:-0}"

if [[ "${SMOKE}" == "true" || "${SMOKE}" == "1" ]]; then
  source_name="$(basename "${source_session_dir}")"
  smoke_name="_smoke_axisperm_${source_name}_${SMOKE_FRAME_START}_${SMOKE_FRAME_END}_$(date +%Y%m%d_%H%M%S)"
  session_dir="${HOST_SESSION_ROOT}/${smoke_name}"
  mkdir -p "${session_dir}"
  if [[ ! -d "${source_session_dir}/processed_topcam" ]]; then
    echo "[axisperm_baseline] Smoke source missing processed_topcam: ${source_session_dir}" >&2
    exit 1
  fi
  ln -s "${source_session_dir}/processed_topcam" "${session_dir}/processed_topcam"
  if [[ -d "${source_session_dir}/config" ]]; then
    ln -s "${source_session_dir}/config" "${session_dir}/config"
  fi
  if [[ -f "${source_session_dir}/episode_0.bag" ]]; then
    ln -s "${source_session_dir}/episode_0.bag" "${session_dir}/episode_0.bag"
  fi
  printf 'source_session=%s\nsmoke_frame_start=%s\nsmoke_frame_end=%s\n' \
    "${source_session_dir}" "${SMOKE_FRAME_START}" "${SMOKE_FRAME_END}" > "${session_dir}/baseline_smoke_source.txt"
  frame_start="${SMOKE_FRAME_START}"
  frame_end="${SMOKE_FRAME_END}"
  frame_stride="${SMOKE_FRAME_STRIDE}"
  max_frames="${SMOKE_MAX_FRAMES}"
fi

pipeline_dir="${session_dir}/quality/egoinfinity_hand_alignment_pipeline"
mkdir -p "${pipeline_dir}"
baseline_timing_jsonl="${pipeline_dir}/axisperm_baseline_timing.jsonl"
baseline_timing_summary="${pipeline_dir}/axisperm_baseline_timing_summary.json"
baseline_summary="${pipeline_dir}/axisperm_pose_baseline_summary.json"
: > "${baseline_timing_jsonl}"
trap 'write_timing_summary >/dev/null 2>&1 || true' EXIT

echo "[axisperm_baseline] source_session=${source_session_dir}"
echo "[axisperm_baseline] session=${session_dir}"
echo "[axisperm_baseline] sides=${run_sides[*]} smoke=${SMOKE} frames=${frame_start}..${frame_end} stride=${frame_stride} max=${max_frames}"
echo "[axisperm_baseline] baseline C3 stage=${C3_STAGE_NAME}"

stages_root="${pipeline_dir}/stages"
phase_b_summary="${stages_root}/phase_b_track_postprocess/wilor_phase_b_summary.json"
c2b_npz="${stages_root}/phase_c2b_bad_pose_repair/wilor_handresults_phase_c2b_bad_pose_repaired.npz"
depth_summary="${stages_root}/foundationstereo_depth_stabilized/foundationstereo_depth_stabilized_summary.json"
anchor_dir="${stages_root}/${ANCHOR_STAGE_NAME}"
anchor_npz="${anchor_dir}/wilor_handresults_phase_c2_anchor_locked_depth_patch2_smooth.npz"
c3_dir="${stages_root}/${C3_STAGE_NAME}"
c3_npz="${c3_dir}/wilor_handresults_phase_c3_mesh_visibility.npz"

run_stage "hand_detection_phase_b_preflight" \
  env \
    HAND="${HAND:-best}" \
    MAX_FRAMES="${max_frames}" \
    FRAME_START="${frame_start}" \
    FRAME_END="${frame_end}" \
    FRAME_STRIDE="${frame_stride}" \
    REBUILD_EGO_HAND="${BASELINE_REBUILD}" \
    RUN_PHASE_C=false \
    RUN_PHASE_C_BAD_VIZ=false \
    PROCESS_TOPCAM=auto \
    bash "${SCRIPT_DIR}/run_egoinfinity_hand_alignment_pipeline.sh" "${session_dir}"

phase_b_candidates="$("${PYTHON_BIN}" - "${phase_b_summary}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print(-1)
    raise SystemExit
data = json.loads(path.read_text())
print(int(data.get("candidates_out", data.get("candidates", -1))))
PY
)"

if [[ "${BASELINE_SKIP_NO_HAND}" == "true" || "${BASELINE_SKIP_NO_HAND}" == "1" ]]; then
  if [[ "${phase_b_candidates}" -eq 0 ]]; then
    python3 - "${baseline_summary}" <<PY
import json
from pathlib import Path

summary = {
    "status": "skipped_no_hand_candidates",
    "source_session": "${source_session_dir}",
    "session": "${session_dir}",
    "smoke": "${SMOKE}",
    "sides": "${run_sides[*]}".split(),
    "frame_start": int("${frame_start}"),
    "frame_end": int("${frame_end}"),
    "frame_stride": int("${frame_stride}"),
    "max_frames": int("${max_frames}"),
    "phase_b_summary": "${phase_b_summary}",
    "phase_b_candidates": int("${phase_b_candidates}"),
    "timing_jsonl": "${baseline_timing_jsonl}",
    "timing_summary_json": "${baseline_timing_summary}",
}
Path("${baseline_summary}").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\\n", encoding="utf-8")
PY
    write_timing_summary
    cat <<EOF

[axisperm_baseline] Skipped: no Phase-B hand candidates.
  session: ${session_dir}
  phase_b_summary: ${phase_b_summary}
  summary: ${baseline_summary}
EOF
    exit 0
  fi
fi

run_stage "hand_alignment_to_c2b_repair" \
  env \
    HAND="${HAND:-best}" \
    MAX_FRAMES="${max_frames}" \
    FRAME_START="${frame_start}" \
    FRAME_END="${frame_end}" \
    FRAME_STRIDE="${frame_stride}" \
    REBUILD_EGO_HAND=false \
    REBUILD_DEPTH="${BASELINE_REBUILD}" \
    REBUILD_PHASE_C_DEPTH_STABILIZE="${BASELINE_REBUILD}" \
    REBUILD_PHASE_C="${BASELINE_REBUILD}" \
    REBUILD_PHASE_C_DEPTH_SMOOTH="${BASELINE_REBUILD}" \
    REBUILD_MOTION_INFILL="${BASELINE_REBUILD}" \
    REBUILD_PHASE_C2="${BASELINE_REBUILD}" \
    REBUILD_PHASE_C2B_REPAIR="${BASELINE_REBUILD}" \
    REBUILD_PHASE_C3="${BASELINE_REBUILD}" \
    RUN_PHASE_C=true \
    RUN_PHASE_C_DEPTH_STABILIZE=true \
    RUN_PHASE_C_DEPTH_SMOOTH=true \
    RUN_MOTION_INFILL=true \
    RUN_PHASE_C2=true \
    RUN_PHASE_C2B_REPAIR=true \
    RUN_PHASE_C3=true \
    RUN_VISIBILITY_REALIGN=false \
    ALLOW_PARTIAL_QC="${BASELINE_ALLOW_PARTIAL_QC}" \
    RUN_PHASE_C_BAD_VIZ="${BASELINE_RUN_BAD_VIZ}" \
    VALID_ITERS="${BASELINE_VALID_ITERS}" \
    WRITE_DEPTH_PREVIEW="${BASELINE_WRITE_DEPTH_PREVIEW}" \
    PROCESS_TOPCAM=auto \
    bash "${SCRIPT_DIR}/run_egoinfinity_hand_alignment_pipeline.sh" "${session_dir}"

if [[ ! -f "${depth_summary}" ]]; then
  depth_summary="${stages_root}/foundationstereo_depth/foundationstereo_depth_summary.json"
fi

if [[ ! -f "${c2b_npz}" ]]; then
  echo "[axisperm_baseline] Missing C2b input: ${c2b_npz}" >&2
  exit 1
fi
if [[ ! -f "${depth_summary}" ]]; then
  echo "[axisperm_baseline] Missing depth summary: ${depth_summary}" >&2
  exit 1
fi

anchor_overlay_arg=()
if [[ "${BASELINE_RENDER_ANCHOR_OVERLAY}" == "false" || "${BASELINE_RENDER_ANCHOR_OVERLAY}" == "0" ]]; then
  anchor_overlay_arg+=(--no-render-overlay)
fi

if [[ "${BASELINE_REBUILD}" == "true" || ! -f "${anchor_npz}" || "${c2b_npz}" -nt "${anchor_npz}" || "${depth_summary}" -nt "${anchor_npz}" ]]; then
  run_stage "phase_c2_anchor_locked_patch2_pose_repaired" \
    "${PYTHON_BIN}" "${SCRIPT_DIR}/phase_c2_anchor_locked_depth_patch2.py" \
      --session-dir "${session_dir}" \
      --input-npz "${c2b_npz}" \
      --depth-summary-json "${depth_summary}" \
      --output-dir "${anchor_dir}" \
      --video "${session_dir}/processed_topcam/left_table.mp4" \
      --segmenter "${BASELINE_SEGMENTER}" \
      "${anchor_overlay_arg[@]}"
else
  echo "[axisperm_baseline] reuse ${anchor_npz}"
  record_reuse_stage "phase_c2_anchor_locked_patch2_pose_repaired"
fi

if [[ "${BASELINE_REBUILD}" == "true" || ! -f "${c3_npz}" || "${anchor_npz}" -nt "${c3_npz}" ]]; then
  run_stage "phase_c3_mesh_visibility_anchor_locked_patch2_pose_repaired" \
    "${PYTHON_BIN}" "${SCRIPT_DIR}/phase_c3_mano_mesh_visibility.py" \
      --session-dir "${session_dir}" \
      --input-npz "${anchor_npz}" \
      --output-dir "${c3_dir}" \
      --vertices-cam-field vertices_cam_anchor_locked_smooth \
      --joints-cam-field joints_cam_anchor_locked_smooth \
      --joints-uv-field joints_uv_anchor_locked_smooth
else
  echo "[axisperm_baseline] reuse ${c3_npz}"
  record_reuse_stage "phase_c3_mesh_visibility_anchor_locked_patch2_pose_repaired"
fi

for run_side in "${run_sides[@]}"; do
  host_robot_table_json="${HOST_ROBOT_TABLE_JSON:-${HOST_ROS_ROOT}/data/lfv_calibration/${run_side}_arm_table_latest.json}"
  robot_table_json="${ROBOT_TABLE_JSON:-${CONTAINER_ROOT}/data/lfv_calibration/${run_side}_arm_table_latest.json}"
  if [[ ! -f "${host_robot_table_json}" ]]; then
    echo "[axisperm_baseline] Missing ${run_side} robot-table calibration: ${host_robot_table_json}" >&2
    exit 1
  fi
  if [[ -n "${IK_STAGE_NAME_OVERRIDE}" && "${#run_sides[@]}" -eq 1 ]]; then
    side_ik_stage_name="${IK_STAGE_NAME_OVERRIDE}"
  elif [[ -n "${IK_STAGE_NAME_OVERRIDE}" ]]; then
    side_ik_stage_name="${IK_STAGE_NAME_OVERRIDE}_${run_side}"
  else
    side_ik_stage_name="phase_e_pinocchio_ik_${run_side}_gripper_base_absolute_s1_pose_rot6d_axisperm_hz_hx_hy"
  fi

  run_stage "gripper_mapping_axisperm_pose_ik_${run_side}" \
    env \
      SIDE="${run_side}" \
      SOURCE_STAGE_NAME="${C3_STAGE_NAME}" \
      SOURCE_NPZ="${c3_npz}" \
      JOINTS_KEY=joints_cam_anchor_locked_smooth \
      UV_KEY=joints_uv_anchor_locked_smooth \
      PHASE_D_STAGE_NAME="${PHASE_D_STAGE_NAME}" \
      PHASE_E_STAGE_NAME="${PHASE_E_STAGE_NAME}" \
      IK_STAGE_NAME="${side_ik_stage_name}" \
      ROBOT_TABLE_JSON="${robot_table_json}" \
      HOST_ROBOT_TABLE_JSON="${host_robot_table_json}" \
      TARGET_MODE="${TARGET_MODE}" \
      SCALE="${SCALE}" \
      TCP_OFFSET_XYZ="${TCP_OFFSET_XYZ}" \
      IK_MODE="${IK_MODE}" \
      ORIENTATION_SOURCE="${ORIENTATION_SOURCE}" \
      ORIENTATION_ALIGN_RPY="${ORIENTATION_ALIGN_RPY}" \
      RUN_PREIK_HTML="${BASELINE_RUN_PREIK_HTML}" \
      RUN_PINOCCHIO_IK="${BASELINE_RUN_PINOCCHIO_IK}" \
      RUN_PYROKI_IK="${BASELINE_RUN_PYROKI_IK}" \
      RENDER_STRIDE="${BASELINE_RENDER_STRIDE}" \
      bash "${SCRIPT_DIR}/run_alignment_gripper_mapping_ik_backends.sh" "${session_dir}"
done

phase_d_dir="${stages_root}/${PHASE_D_STAGE_NAME}"
phase_e_dir="${stages_root}/${PHASE_E_STAGE_NAME}"

python3 - "${baseline_summary}" <<PY
import json
from pathlib import Path

sides = "${run_sides[*]}".split()
ik_override = "${IK_STAGE_NAME_OVERRIDE}"
side_outputs = {}
for side in sides:
    if ik_override and len(sides) == 1:
        ik_stage = ik_override
    elif ik_override:
        ik_stage = f"{ik_override}_{side}"
    else:
        ik_stage = f"phase_e_pinocchio_ik_{side}_gripper_base_absolute_s1_pose_rot6d_axisperm_hz_hx_hy"
    ik_dir = Path("${stages_root}") / ik_stage
    phase_e_dir = Path("${phase_e_dir}")
    side_outputs[side] = {
        "phase_e_csv": str(phase_e_dir / f"phase_d_{side}_gripper_base_core.csv"),
        "pinocchio_ik_dir": str(ik_dir),
        "pinocchio_ik_csv": str(ik_dir / "piper_pinocchio_core_ik.csv"),
        "pinocchio_ik_metadata_json": str(ik_dir / "piper_pinocchio_core_ik_metadata.json"),
    }

summary = {
    "status": "done",
    "source_session": "${source_session_dir}",
    "session": "${session_dir}",
    "smoke": "${SMOKE}",
    "sides": sides,
    "frame_start": int("${frame_start}"),
    "frame_end": int("${frame_end}"),
    "frame_stride": int("${frame_stride}"),
    "max_frames": int("${max_frames}"),
    "baseline_stage": "${C3_STAGE_NAME}",
    "c2b_npz": "${c2b_npz}",
    "depth_summary_json": "${depth_summary}",
    "anchor_npz": "${anchor_npz}",
    "c3_npz": "${c3_npz}",
    "phase_d_dir": "${phase_d_dir}",
    "phase_e_dir": "${phase_e_dir}",
    "side_outputs": side_outputs,
    "timing_jsonl": "${baseline_timing_jsonl}",
    "timing_summary_json": "${baseline_timing_summary}",
}
exists = {k: Path(v).exists() for k, v in summary.items() if isinstance(v, str) and (k.endswith(("_npz", "_json", "_csv")) or k.endswith("_dir"))}
for side, outputs in side_outputs.items():
    exists[side] = {k: Path(v).exists() for k, v in outputs.items()}
summary["exists"] = exists
Path("${baseline_summary}").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\\n", encoding="utf-8")
PY

write_timing_summary

cat <<EOF

[axisperm_baseline] Done.
  session: ${session_dir}
  summary: ${baseline_summary}
  C3 NPZ: ${c3_npz}
  timing: ${baseline_timing_summary}
EOF

for run_side in "${run_sides[@]}"; do
  if [[ -n "${IK_STAGE_NAME_OVERRIDE}" && "${#run_sides[@]}" -eq 1 ]]; then
    side_ik_stage_name="${IK_STAGE_NAME_OVERRIDE}"
  elif [[ -n "${IK_STAGE_NAME_OVERRIDE}" ]]; then
    side_ik_stage_name="${IK_STAGE_NAME_OVERRIDE}_${run_side}"
  else
    side_ik_stage_name="phase_e_pinocchio_ik_${run_side}_gripper_base_absolute_s1_pose_rot6d_axisperm_hz_hx_hy"
  fi
  echo "  ${run_side} Phase-E CSV: ${phase_e_dir}/phase_d_${run_side}_gripper_base_core.csv"
  echo "  ${run_side} Pinocchio IK CSV: ${stages_root}/${side_ik_stage_name}/piper_pinocchio_core_ik.csv"
done
