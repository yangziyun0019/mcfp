from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from omegaconf import OmegaConf

from mcfp.data.io import read_jsonl
from mcfp.data.io import load_morph_spec, load_pose_samples
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

    required_keys = list(cfg.data.required_keys)

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
        pose_path = (repo_root / rec["pose_path"]).resolve()

        entry: Dict[str, Any] = {
            "family": family,
            "variant_id": variant_id,
            "spec_path": str(spec_path),
            "pose_path": str(pose_path),
            "errors": [],
            "warnings": [],
            "valid_ratio": rec.get("valid_ratio", None),
            "num_cells": rec.get("num_cells", None),
        }

        if not spec_path.exists():
            entry["errors"].append("missing spec_path")
        if not pose_path.exists():
            entry["errors"].append("missing pose_path")

        if entry["errors"]:
            total_errors += len(entry["errors"])
            per_record.append(entry)
            continue

        spec = load_morph_spec(spec_path)
        meta = spec.get("meta", {})
        dof = meta.get("dof", None)
        if dof is None:
            entry["warnings"].append("spec.meta.dof missing")

        data = load_pose_samples(pose_path)
        missing = [k for k in required_keys if k not in data]
        if missing:
            entry["errors"].append(f"pose samples missing keys: {missing}")

        poses = np.asarray(data.get("poses", np.empty((0, 7))), dtype=np.float32)
        labels = np.asarray(data.get("labels", np.empty((0,))), dtype=np.float32).reshape(-1)
        delta_pos = np.asarray(data.get("delta_pos", np.empty((0, 3))), dtype=np.float32)
        delta_rot = np.asarray(data.get("delta_rot", np.empty((0, 3))), dtype=np.float32)
        delta_mask = np.asarray(data.get("delta_mask", np.empty((0,))), dtype=np.float32).reshape(-1)

        n = int(poses.shape[0]) if poses.ndim == 2 else -1

        if poses.ndim != 2 or poses.shape[1] != 7:
            entry["errors"].append(f"poses bad shape: {poses.shape}")
        if labels.ndim != 1 or labels.shape[0] != n:
            entry["errors"].append(f"labels bad shape: {labels.shape}")
        if delta_pos.ndim != 2 or delta_pos.shape[1] != 3 or delta_pos.shape[0] != n:
            entry["errors"].append(f"delta_pos bad shape: {delta_pos.shape}")
        if delta_rot.ndim != 2 or delta_rot.shape[1] != 3 or delta_rot.shape[0] != n:
            entry["errors"].append(f"delta_rot bad shape: {delta_rot.shape}")
        if delta_mask.ndim != 1 or delta_mask.shape[0] != n:
            entry["errors"].append(f"delta_mask bad shape: {delta_mask.shape}")

        if n > 0:
            entry["errors"].extend(_check_array_1d("labels", labels, n))
            entry["errors"].extend(_check_array_1d("delta_mask", delta_mask, n))
            uniq = np.unique(np.round(labels, 6))
            if not np.all(np.isin(uniq, np.array([0.0, 1.0], dtype=np.float32))):
                entry["warnings"].append(f"labels not binary; unique={uniq.tolist()}")
            uniq_m = np.unique(np.round(delta_mask, 6))
            if not np.all(np.isin(uniq_m, np.array([0.0, 1.0], dtype=np.float32))):
                entry["warnings"].append(f"delta_mask not binary; unique={uniq_m.tolist()}")

            valid_ratio = float(np.mean(labels)) if labels.size > 0 else float("nan")
            entry["valid_ratio_recomputed"] = valid_ratio

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
