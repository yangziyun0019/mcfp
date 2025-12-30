from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterator, List, Tuple

import numpy as np
from torch.utils.data import Sampler

from mcfp.data.io import load_pose_samples


@dataclass(frozen=True)
class BalancedSamplingConfig:
    """Sampling configuration for mixing reachable/non-reachable poses."""

    ws_ratio: float = 0.8
    seed: int = 42
    batch_size: int = 256


class GroupedBalancedPoseBatchSampler(Sampler[List[int]]):
    """Grouped batch sampler with target g_ws ratio.

    Each batch contains samples from a single variant_id (morphology).
    It balances reachable (g_ws=1) and non-reachable (g_ws=0) samples.
    """

    def __init__(
        self,
        dataset,
        cfg: BalancedSamplingConfig,
        repo_root,
        manifest_by_id: Dict[str, Dict],
    ) -> None:
        self.dataset = dataset
        self.cfg = cfg
        self.repo_root = repo_root
        self.manifest_by_id = manifest_by_id

        if self.cfg.batch_size <= 0:
            raise ValueError("[sampling] cfg.batch_size must be positive.")

        self.variant_ids: List[str] = list(getattr(self.dataset, "variant_ids", []))
        if len(self.variant_ids) == 0:
            self.variant_ids = sorted(list({vid for vid, _ in self.dataset._index}))
        if len(self.variant_ids) == 0:
            raise ValueError("[sampling] Empty variant_ids. Check your split file and manifest.")

        sample_counts: Dict[str, int] = {vid: 0 for vid in self.variant_ids}
        for vid, _ in getattr(self.dataset, "_index", []):
            if vid in sample_counts:
                sample_counts[vid] += 1
        self._variant_weights = []
        for vid in self.variant_ids:
            w = float(sample_counts.get(vid, 0))
            if not np.isfinite(w) or w <= 0.0:
                w = 1.0
            self._variant_weights.append(w)

        self._pools: Dict[str, Tuple[List[int], List[int]]] = {}
        self._ptr_ws: Dict[str, int] = {vid: 0 for vid in self.variant_ids}
        self._ptr_nw: Dict[str, int] = {vid: 0 for vid in self.variant_ids}
        self._build_pools()

    def _build_pools(self) -> None:
        label_cache: Dict[str, np.ndarray] = {}
        ws_pool: Dict[str, List[int]] = {vid: [] for vid in self.variant_ids}
        nw_pool: Dict[str, List[int]] = {vid: [] for vid in self.variant_ids}

        for global_idx, (vid, sample_idx) in enumerate(self.dataset._index):
            if vid not in label_cache:
                pose_path = (self.repo_root / self.manifest_by_id[vid]["pose_path"]).resolve()
                data = load_pose_samples(pose_path)
                label_cache[vid] = np.asarray(data["labels"], dtype=np.float32).reshape(-1)

            g_ws = float(label_cache[vid][sample_idx])
            if g_ws > 0.5:
                ws_pool[vid].append(global_idx)
            else:
                nw_pool[vid].append(global_idx)

        rng = random.Random(self.cfg.seed)
        for vid in self.variant_ids:
            a = ws_pool.get(vid, [])
            b = nw_pool.get(vid, [])
            rng.shuffle(a)
            rng.shuffle(b)
            self._pools[vid] = (a, b)

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.cfg.seed)

        while True:
            if len(self.variant_ids) == 1:
                vid = self.variant_ids[0]
            else:
                vid = rng.choices(self.variant_ids, weights=self._variant_weights, k=1)[0]
            ws, nw = self._pools[vid]

            batch: List[int] = []
            for _ in range(self.cfg.batch_size):
                pick_ws = rng.random() < self.cfg.ws_ratio
                if pick_ws and len(ws) > 0:
                    i = self._ptr_ws[vid] % len(ws)
                    batch.append(ws[i])
                    self._ptr_ws[vid] += 1
                elif len(nw) > 0:
                    i = self._ptr_nw[vid] % len(nw)
                    batch.append(nw[i])
                    self._ptr_nw[vid] += 1
                else:
                    if len(ws) > 0:
                        i = self._ptr_ws[vid] % len(ws)
                        batch.append(ws[i])
                        self._ptr_ws[vid] += 1

            yield batch

    def __len__(self) -> int:
        return max(1, len(self.dataset) // max(1, self.cfg.batch_size))
