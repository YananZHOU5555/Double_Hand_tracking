#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LFV_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${LFV_ROOT}/.." && pwd)"
EGOINFINITY_ROOT="${EGOINFINITY_ROOT:-${WORKSPACE_ROOT}/EgoInfinity}"
CKPT_DIR="${EGOINFINITY_CKPT_DIR:-${EGOINFINITY_ROOT}/pretrained_models}"
WILOR_PYTHON="${WILOR_PYTHON:-/home/yannan/miniforge3/envs/wilor_lfv/bin/python}"
INSTALL_MEMFOF="${INSTALL_MEMFOF:-true}"
PREFETCH_MEMFOF="${PREFETCH_MEMFOF:-false}"

if [[ ! -d "${EGOINFINITY_ROOT}" ]]; then
  echo "[setup_egoinfinity_strict] Missing EgoInfinity repo: ${EGOINFINITY_ROOT}" >&2
  exit 1
fi
if [[ ! -x "${WILOR_PYTHON}" ]]; then
  echo "[setup_egoinfinity_strict] Missing python: ${WILOR_PYTHON}" >&2
  exit 1
fi

mkdir -p "${CKPT_DIR}"

echo "[setup_egoinfinity_strict] EgoInfinity root: ${EGOINFINITY_ROOT}"
echo "[setup_egoinfinity_strict] Checkpoint dir:   ${CKPT_DIR}"
echo "[setup_egoinfinity_strict] Python:           ${WILOR_PYTHON}"

echo "[setup_egoinfinity_strict] Ensuring EgoInfinity WiLoR source snapshot."
(cd "${EGOINFINITY_ROOT}" && bash scripts/setup_wilor.sh)

echo "[setup_egoinfinity_strict] Copying MANO files from LFV WiLoR when available."
mkdir -p "${EGOINFINITY_ROOT}/third_party/wilor/mano_data"
if [[ -f "${LFV_ROOT}/WiLor/mano_data/MANO_RIGHT.pkl" ]]; then
  cp -n "${LFV_ROOT}/WiLor/mano_data/MANO_RIGHT.pkl" "${EGOINFINITY_ROOT}/third_party/wilor/mano_data/MANO_RIGHT.pkl"
fi
if [[ -f "${LFV_ROOT}/WiLor/mano_data/mano_mean_params.npz" ]]; then
  cp -n "${LFV_ROOT}/WiLor/mano_data/mano_mean_params.npz" "${EGOINFINITY_ROOT}/third_party/wilor/mano_data/mano_mean_params.npz"
fi

echo "[setup_egoinfinity_strict] Downloading strict checkpoints."
"${WILOR_PYTHON}" - "${CKPT_DIR}" <<'PY'
import os
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download

ckpt_dir = Path(sys.argv[1]).expanduser().resolve()
ckpt_dir.mkdir(parents=True, exist_ok=True)

def valid(path: Path, min_size: int) -> bool:
    return path.exists() and path.stat().st_size >= min_size

def atomic_copy(src: str, dst: Path) -> None:
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copyfile(src, tmp)
    if tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"downloaded empty file for {dst}")
    tmp.replace(dst)

def hf_file(repo_id: str, filename: str, dst_name: str, min_size: int, repo_type=None):
    dst = ckpt_dir / dst_name
    if valid(dst, min_size):
        print(f"  ok {dst.name} ({dst.stat().st_size} bytes)")
        return
    if dst.exists() and dst.stat().st_size < min_size:
        dst.unlink()
    print(f"  download {dst.name} from hf://{repo_id}/{filename}")
    src = hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)
    atomic_copy(src, dst)
    if not valid(dst, min_size):
        raise RuntimeError(f"{dst} too small after download: {dst.stat().st_size}")

def url_file(url: str, dst_name: str, min_size: int):
    dst = ckpt_dir / dst_name
    if valid(dst, min_size):
        print(f"  ok {dst.name} ({dst.stat().st_size} bytes)")
        return
    if dst.exists() and dst.stat().st_size < min_size:
        dst.unlink()
    print(f"  download {dst.name} from {url}")
    with tempfile.NamedTemporaryFile(delete=False) as f:
        tmp_name = f.name
    try:
        urllib.request.urlretrieve(url, tmp_name)
        atomic_copy(tmp_name, dst)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
    if not valid(dst, min_size):
        raise RuntimeError(f"{dst} too small after download: {dst.stat().st_size}")

hf_file("rolpotamias/WiLoR", "pretrained_models/detector.pt", "detector.pt", 1_000_000, repo_type="space")
hf_file("rolpotamias/WiLoR", "pretrained_models/wilor_final.ckpt", "wilor_final.ckpt", 10_000_000, repo_type="space")
hf_file("ThunderVVV/HaWoR", "hawor/checkpoints/infiller.pt", "infiller.pt", 1_000_000)
try:
    url_file("https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt", "sam2.1_hiera_small.pt", 10_000_000)
except Exception as exc:
    print(f"  WARN optional SAM2 small download failed: {exc}")
PY

if [[ "${INSTALL_MEMFOF}" == "true" || "${INSTALL_MEMFOF}" == "1" ]]; then
  echo "[setup_egoinfinity_strict] Installing MEMFOF into ${WILOR_PYTHON} environment."
  "${WILOR_PYTHON}" -m pip install "git+https://github.com/msu-video-group/memfof"
else
  echo "[setup_egoinfinity_strict] Skipping MEMFOF install (INSTALL_MEMFOF=${INSTALL_MEMFOF})."
fi

if [[ "${PREFETCH_MEMFOF}" == "true" || "${PREFETCH_MEMFOF}" == "1" ]]; then
  echo "[setup_egoinfinity_strict] Prefetching MEMFOF HuggingFace weights."
  "${WILOR_PYTHON}" - <<'PY'
from memfof import MEMFOF
MEMFOF.from_pretrained("egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH")
print("MEMFOF checkpoint cached")
PY
fi

echo "[setup_egoinfinity_strict] Checking components."
"${WILOR_PYTHON}" "${SCRIPT_DIR}/check_egoinfinity_strict_components.py" \
  --egoinfinity-root "${EGOINFINITY_ROOT}" \
  --checkpoint-dir "${CKPT_DIR}" \
  --json-out "${SCRIPT_DIR}/strict_components_status.json"

echo "[setup_egoinfinity_strict] Done."
