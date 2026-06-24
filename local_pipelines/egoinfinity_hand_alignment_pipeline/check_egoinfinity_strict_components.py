#!/usr/bin/env python3
"""Check strict EgoInfinity dependencies for the LFV hand pipeline."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def file_status(path: Path, min_size: int = 1) -> Dict[str, Any]:
    path = path.expanduser().resolve()
    exists = path.exists()
    size = path.stat().st_size if exists else 0
    return {
        "path": str(path),
        "exists": bool(exists),
        "size_bytes": int(size),
        "min_size_bytes": int(min_size),
        "ok": bool(exists and size >= min_size),
    }


def import_status(module: str) -> Dict[str, Any]:
    try:
        mod = importlib.import_module(module)
        return {
            "module": module,
            "ok": True,
            "version": str(getattr(mod, "__version__", "")),
            "error": "",
        }
    except Exception as exc:
        return {"module": module, "ok": False, "version": "", "error": repr(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--egoinfinity-root", default="/home/yannan/workspace/EgoInfinity")
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    ego = Path(args.egoinfinity_root).expanduser().resolve()
    ckpt = Path(args.checkpoint_dir).expanduser().resolve() if args.checkpoint_dir else ego / "pretrained_models"
    lfv_pipeline = Path(__file__).resolve().parent

    files = {
        "detector_pt": file_status(ckpt / "detector.pt", 1_000_000),
        "wilor_final_ckpt": file_status(ckpt / "wilor_final.ckpt", 10_000_000),
        "infiller_pt": file_status(ckpt / "infiller.pt", 1_000_000),
        "sam2_small": file_status(ckpt / "sam2.1_hiera_small.pt", 10_000_000),
        "mano_right": file_status(ego / "third_party" / "wilor" / "mano_data" / "MANO_RIGHT.pkl", 1_000_000),
        "mano_mean_params": file_status(ego / "third_party" / "wilor" / "mano_data" / "mano_mean_params.npz", 100),
        "egoinfinity_wilor_source": file_status(ego / "third_party" / "wilor" / "models" / "wilor.py", 1_000),
    }

    imports = {
        "torch": import_status("torch"),
        "cv2": import_status("cv2"),
        "scipy": import_status("scipy"),
        "huggingface_hub": import_status("huggingface_hub"),
        "memfof": import_status("memfof"),
        "local_depth_align": import_status("egoinfinity_strict.depth_align"),
        "local_depth_stabilize": import_status("egoinfinity_strict.depth_stabilize"),
        "local_biomech": import_status("egoinfinity_strict.biomech_constraints"),
        "local_mano_smoothing": import_status("egoinfinity_strict.mano_smoothing"),
        "local_motion_infiller": import_status("egoinfinity_strict.motion_infiller"),
    }

    hard_errors: List[str] = []
    warnings: List[str] = []
    optional_files = {"sam2_small"}
    for name, st in files.items():
        if not st["ok"]:
            if name in optional_files:
                warnings.append(f"optional_file_missing_or_too_small:{name}:{st['path']}")
            else:
                hard_errors.append(f"missing_or_too_small_file:{name}:{st['path']}")
    for name, st in imports.items():
        if name == "memfof":
            if not st["ok"]:
                warnings.append(f"memfof_unavailable:{st['error']}")
            continue
        if not st["ok"]:
            hard_errors.append(f"import_failed:{name}:{st['error']}")

    # Local imports require this script directory on sys.path.  If the caller
    # did not run from this folder, retry with the local pipeline root inserted.
    if any(k.startswith("local_") and not v["ok"] for k, v in imports.items()):
        if str(lfv_pipeline) not in sys.path:
            sys.path.insert(0, str(lfv_pipeline))
        for key, module in [
            ("local_depth_align", "egoinfinity_strict.depth_align"),
            ("local_depth_stabilize", "egoinfinity_strict.depth_stabilize"),
            ("local_biomech", "egoinfinity_strict.biomech_constraints"),
            ("local_mano_smoothing", "egoinfinity_strict.mano_smoothing"),
            ("local_motion_infiller", "egoinfinity_strict.motion_infiller"),
        ]:
            imports[key] = import_status(module)
        hard_errors = [e for e in hard_errors if not e.startswith("import_failed:local_")]
        for key, st in imports.items():
            if key.startswith("local_") and not st["ok"]:
                hard_errors.append(f"import_failed:{key}:{st['error']}")

    summary = {
        "semantic": "Strict EgoInfinity component check for LFV hand Phase-C",
        "egoinfinity_root": str(ego),
        "checkpoint_dir": str(ckpt),
        "files": files,
        "imports": imports,
        "strict_ready": len(hard_errors) == 0 and imports["memfof"]["ok"],
        "phase_c_ready_without_flow": len(hard_errors) == 0,
        "sam2_object_tracking_ready": files["sam2_small"]["ok"],
        "hard_errors": hard_errors,
        "warnings": warnings,
    }

    text = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.json_out:
        out = Path(args.json_out).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if len(hard_errors) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
