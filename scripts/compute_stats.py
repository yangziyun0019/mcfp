from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from omegaconf import OmegaConf

from mcfp.data.io import read_jsonl, load_pose_samples
from mcfp.utils.logging import setup_logger


def _percentiles(x: np.ndarray, ps: List[float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for p in ps:
        out[f"p{int(p)}"] = float(np.percentile(x, p))
    return out


def main() -> None:
    cfg = OmegaConf.load("configs/compute_stats.yaml")
    logger = setup_logger(name="mcfp.scripts.compute_stats", log_dir=cfg.logging.log_dir)

    repo_root = Path(cfg.data.repo_root).resolve()
    manifest_path = Path(cfg.data.manifest_path).resolve()
    split_path = Path(cfg.data.train_split_path).resolve()
    out_path = Path(cfg.output.stats_path).resolve()

    percentiles = list(cfg.data.percentiles)
    label_keys = list(getattr(cfg.data, "label_keys", [])) or [
        "g_ws",
        "delta_pos_x",
        "delta_pos_y",
        "delta_pos_z",
        "delta_rot_x",
        "delta_rot_y",
        "delta_rot_z",
    ]

    records = read_jsonl(manifest_path)
    rec_by_id = {r["variant_id"]: r for r in records}

    train_ids = [s.strip() for s in split_path.read_text(encoding="utf-8").splitlines() if s.strip()]
    logger.info(f"[stats] train morphologies={len(train_ids)}")

    agg: Dict[str, List[np.ndarray]] = {k: [] for k in label_keys}

    for vid in train_ids:
        if vid not in rec_by_id:
            logger.warning(f"[stats] Missing vid in manifest: {vid}")
            continue
        pose_path = (repo_root / rec_by_id[vid]["pose_path"]).resolve()
        data = load_pose_samples(pose_path)

        labels = np.asarray(data.get("labels"), dtype=np.float32).reshape(-1)
        delta_pos = np.asarray(data.get("delta_pos"), dtype=np.float32)
        delta_rot = np.asarray(data.get("delta_rot"), dtype=np.float32)
        delta_mask = np.asarray(data.get("delta_mask"), dtype=np.float32).reshape(-1) > 0.5

        if labels.ndim != 1:
            logger.warning(f"[stats] Bad labels shape: {pose_path} {labels.shape}")
            continue
        if delta_pos.ndim != 2 or delta_pos.shape[1] != 3:
            logger.warning(f"[stats] Bad delta_pos shape: {pose_path} {delta_pos.shape}")
            continue
        if delta_rot.ndim != 2 or delta_rot.shape[1] != 3:
            logger.warning(f"[stats] Bad delta_rot shape: {pose_path} {delta_rot.shape}")
            continue

        for k in label_keys:
            if k == "g_ws":
                agg[k].append(labels)
                continue
            if not delta_mask.any():
                continue
            if k.startswith("delta_pos_"):
                axis = {"x": 0, "y": 1, "z": 2}.get(k.split("_")[-1], None)
                if axis is None:
                    raise ValueError(f"[stats] Unknown label key: {k}")
                agg[k].append(delta_pos[delta_mask][:, axis])
            elif k.startswith("delta_rot_"):
                axis = {"x": 0, "y": 1, "z": 2}.get(k.split("_")[-1], None)
                if axis is None:
                    raise ValueError(f"[stats] Unknown label key: {k}")
                agg[k].append(delta_rot[delta_mask][:, axis])
            else:
                raise ValueError(f"[stats] Unknown label key: {k}")

    stats: Dict[str, Any] = {
        "label_keys": list(agg.keys()),
        "percentiles": percentiles,
        "per_key": {},
    }

    for k in label_keys:
        if len(agg[k]) == 0:
            stats["per_key"][k] = {"count": 0}
            continue
        vec = np.concatenate(agg[k], axis=0)
        vec = vec[np.isfinite(vec)]
        if vec.size == 0:
            stats["per_key"][k] = {"count": 0}
            continue

        stats["per_key"][k] = {
            "count": int(vec.size),
            "mean": float(np.mean(vec)),
            "std": float(np.std(vec)),
            "min": float(np.min(vec)),
            "max": float(np.max(vec)),
            **_percentiles(vec, percentiles),
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"[stats] Wrote stats to {out_path}")


if __name__ == "__main__":
    main()
