#!/usr/bin/env python3
"""Sample one Position-SDF training batch and print sampler statistics.

This script is a lightweight debugging tool for checking batch balance and distance bucket coverage.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from mcfp.data.pos_dataset import PositionDataset
from mcfp.utils.config import load_config


def _get_path(cfg: Any, key: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in key.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            if part not in cur:
                return default
            cur = cur[part]
        else:
            if not hasattr(cur, part):
                return default
            cur = getattr(cur, part)
    return cur


def _resolve_path(path: str | Path, repo_root: Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        return (repo_root / p).resolve()
    return p.resolve()


def _stats(name: str, arr: np.ndarray) -> None:
    if arr.size == 0:
        print(f"{name}: empty")
        return
    print(
        f"{name}: n={arr.size} min={float(np.min(arr)):.6f} max={float(np.max(arr)):.6f} "
        f"mean={float(np.mean(arr)):.6f} std={float(np.std(arr)):.6f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample one batch from dataset_pos and print stats.")
    parser.add_argument("--config", type=str, default="configs/train_pos.yaml")
    parser.add_argument("--split", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    repo_root = Path(_get_path(cfg, "paths.repo_root", ".")).resolve()

    pos_path = _resolve_path(_get_path(cfg, "paths.position_h5"), repo_root)
    split_voxel_cfg = _get_path(cfg, "paths.splits_voxel", None)
    split_voxel = _resolve_path(split_voxel_cfg, repo_root) if split_voxel_cfg else None

    dataset = PositionDataset(pos_path, split_voxel, cache_grid=bool(_get_path(cfg, "data.cache_grid", True)))

    batch_size = int(_get_path(cfg, "data.batch_size", 8192))
    ratios = tuple(_get_path(cfg, "data.ratios", [0.30, 0.70]))
    s_bucket_weights_cfg = _get_path(cfg, "data.s_bucket_weights", None)
    s_bucket_weights = tuple(s_bucket_weights_cfg) if s_bucket_weights_cfg is not None else None
    if s_bucket_weights is None:
        legacy = _get_path(cfg, "data.inner_tier_ratios", None)
        if legacy is not None:
            s_bucket_weights = tuple(legacy)
    s_bucket_tau = _get_path(cfg, "data.s_bucket_tau", None)
    if s_bucket_tau is not None:
        s_bucket_tau = float(s_bucket_tau)
    sign_balance = float(_get_path(cfg, "data.sign_balance", 0.5))

    rng = np.random.default_rng(int(args.seed))
    batch = dataset.sample_batch(
        rng,
        batch_size,
        ratios,
        split_id=int(args.split),
        s_bucket_weights=s_bucket_weights,
        s_bucket_tau=s_bucket_tau,
        sign_balance=sign_balance,
    )

    s = batch["s"].astype(np.float64)
    source = batch["source"].astype(np.int64)
    tier = batch.get("tier", np.zeros_like(source, dtype=np.uint8)).astype(np.int64)

    print(f"batch_size={s.size} split={args.split}")
    print(f"source counts: boundary={int(np.sum(source==0))} nonboundary={int(np.sum(source==1))}")

    _stats("s_all", s)
    mask_bnd = source == 0
    mask_nb = source == 1
    _stats("s_boundary", s[mask_bnd])
    _stats("s_nonboundary", s[mask_nb])

    if np.any(mask_nb):
        mask_in = s[mask_nb] >= 0
        mask_out = ~mask_in
        _stats("s_inside(nb)", s[mask_nb][mask_in])
        _stats("s_outside(nb)", s[mask_nb][mask_out])
        print(
            f"sign ratio (nb): pos={float(np.mean(mask_in)):.4f} "
            f"neg={float(np.mean(mask_out)):.4f}"
        )

    if tier.size > 0:
        counts = np.bincount(tier, minlength=int(np.max(tier)) + 1)
        print("tier counts:")
        for i, c in enumerate(counts):
            if c == 0:
                continue
            print(f"  tier {i}: {int(c)}")

        print("tier s stats:")
        for i, c in enumerate(counts):
            if c == 0:
                continue
            mask = tier == i
            _stats(f"  tier {i} s", s[mask])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
