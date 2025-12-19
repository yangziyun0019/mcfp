from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from omegaconf import OmegaConf

from mcfp.data.io import read_jsonl, load_capability_map
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

    label_keys = list(cfg.data.label_keys)
    percentiles = list(cfg.data.percentiles)

    records = read_jsonl(manifest_path)
    rec_by_id = {r["variant_id"]: r for r in records}

    train_ids = [s.strip() for s in split_path.read_text(encoding="utf-8").splitlines() if s.strip()]
    logger.info(f"[stats] train morphologies={len(train_ids)}")

    agg: Dict[str, List[np.ndarray]] = {k: [] for k in label_keys}

    for vid in train_ids:
        if vid not in rec_by_id:
            logger.warning(f"[stats] Missing vid in manifest: {vid}")
            continue
        cap_path = (repo_root / rec_by_id[vid]["cap_path"]).resolve()
        cap = load_capability_map(cap_path)

        gws = np.asarray(cap["g_ws"], dtype=np.float32).reshape(-1)
        ws_mask = gws > 0.5

        for k in label_keys:
            if k not in cap:
                logger.warning(f"[stats] Missing key={k} in {cap_path}")
                continue
            x = np.asarray(cap[k], dtype=np.float32).reshape(-1)
            if x.shape[0] != gws.shape[0]:
                logger.warning(f"[stats] Shape mismatch key={k} in {cap_path}: {x.shape} vs {gws.shape}")
                continue

            if k == "g_ws":
                agg[k].append(x)
            else:
                # Only statistics over valid workspace cells.
                agg[k].append(x[ws_mask])

    stats: Dict[str, Any] = {
        "label_keys": label_keys,
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
