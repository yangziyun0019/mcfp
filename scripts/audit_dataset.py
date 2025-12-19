from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from omegaconf import OmegaConf

from mcfp.data.io import read_jsonl
from mcfp.data.io import load_capability_map, load_morph_spec
from mcfp.utils.logging import setup_logger


def _check_array_1d(name: str, arr: np.ndarray, n: int) -> List[str]:
    errs: List[str] = []
    if arr.ndim != 1:
        errs.append(f"{name}: expected 1D, got shape={arr.shape}")
        return errs
    if arr.shape[0] != n:
        errs.append(f"{name}: expected length={n}, got length={arr.shape[0]}")
    if not np.all(np.isfinite(arr.astype(np.float32))):
        errs.append(f"{name}: contains non-finite values")
    return errs


def main() -> None:
    cfg = OmegaConf.load("configs/audit_dataset.yaml")
    logger = setup_logger(name="mcfp.scripts.audit_dataset", log_dir=cfg.logging.log_dir)

    repo_root = Path(cfg.data.repo_root).resolve()
    manifest_path = Path(cfg.data.manifest_path).resolve()
    report_path = Path(cfg.output.report_path).resolve()

    required_cap_keys = list(cfg.data.required_cap_keys)

    records = read_jsonl(manifest_path)
    logger.info(f"[audit] Loaded {len(records)} manifest records from {manifest_path}")

    report: Dict[str, Any] = {
        "manifest_path": str(manifest_path),
        "num_records": len(records),
        "errors": [],
        "warnings": [],
        "summary": {},
    }

    per_record: List[Dict[str, Any]] = []
    total_errors = 0
    total_warnings = 0

    for rec in records:
        family = rec["family"]
        variant_id = rec["variant_id"]
        spec_path = (repo_root / rec["spec_path"]).resolve()
        cap_path = (repo_root / rec["cap_path"]).resolve()

        entry: Dict[str, Any] = {
            "family": family,
            "variant_id": variant_id,
            "spec_path": str(spec_path),
            "cap_path": str(cap_path),
            "errors": [],
            "warnings": [],
            "valid_ratio": rec.get("valid_ratio", None),
            "num_cells": rec.get("num_cells", None),
        }

        if not spec_path.exists():
            entry["errors"].append("missing spec_path")
        if not cap_path.exists():
            entry["errors"].append("missing cap_path")

        if entry["errors"]:
            total_errors += len(entry["errors"])
            per_record.append(entry)
            continue

        spec = load_morph_spec(spec_path)
        meta = spec.get("meta", {})
        dof = meta.get("dof", None)
        if dof is None:
            entry["warnings"].append("spec.meta.dof missing")

        cap = load_capability_map(cap_path)
        missing = [k for k in required_cap_keys if k not in cap]
        if missing:
            entry["errors"].append(f"cap missing keys: {missing}")

        if "cell_centers" in cap:
            centers = np.asarray(cap["cell_centers"])
            if centers.ndim != 2 or centers.shape[1] != 3:
                entry["errors"].append(f"cell_centers bad shape: {centers.shape}")
            n = int(centers.shape[0]) if centers.ndim == 2 else -1
        else:
            n = -1

        # g_ws sanity
        if "g_ws" in cap and n > 0:
            gws = np.asarray(cap["g_ws"]).astype(np.float32).reshape(-1)
            entry["errors"].extend(_check_array_1d("g_ws", gws, n))
            uniq = np.unique(np.round(gws, 6))
            if not np.all(np.isin(uniq, np.array([0.0, 1.0], dtype=np.float32))):
                entry["warnings"].append(f"g_ws not binary; unique={uniq.tolist()}")

            valid_ratio = float(np.mean(gws))
            entry["valid_ratio_recomputed"] = valid_ratio
        else:
            entry["warnings"].append("missing g_ws or invalid n")

        # Other g_* arrays
        if n > 0:
            for k in required_cap_keys:
                if k in ("cell_centers",):
                    continue
                if k not in cap:
                    continue
                arr = np.asarray(cap[k])
                if k.startswith("g_") or k in ("sample_counts",):
                    if k == "sample_counts":
                        if arr.ndim != 1 or arr.shape[0] != n:
                            entry["errors"].append(f"sample_counts bad shape: {arr.shape}")
                    else:
                        entry["errors"].extend(_check_array_1d(k, arr.astype(np.float32), n))

        if entry["errors"]:
            total_errors += len(entry["errors"])
        if entry["warnings"]:
            total_warnings += len(entry["warnings"])

        per_record.append(entry)

    report["per_record"] = per_record
    report["summary"] = {
        "total_errors": total_errors,
        "total_warnings": total_warnings,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(f"[audit] Report written to {report_path}")
    logger.info(f"[audit] total_errors={total_errors}, total_warnings={total_warnings}")


if __name__ == "__main__":
    main()
