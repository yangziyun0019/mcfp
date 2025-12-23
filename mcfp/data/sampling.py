# mcfp/data/sampling.py

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterator, List, Tuple

import numpy as np
import torch
from torch.utils.data import Sampler

from mcfp.data.io import load_capability_map


@dataclass(frozen=True)
class BalancedSamplingConfig:
    """Sampling configuration for mixing ws/non-ws cells."""
    ws_ratio: float = 0.8
    seed: int = 42
    batch_size: int = 256


class BalancedCellSampler(Sampler[int]):
    """Index sampler that enforces a target ws/non-ws ratio.

    This sampler operates on Stage1Dataset's flattened index space.
    It precomputes ws_mask per (variant_id, cell_idx) by reading g_ws from maps.

    Notes
    -----
    - This sampler is designed for Stage I: regression losses are masked by ws_mask,
      but g_ws head benefits from a stable stream of negative samples.
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

        self._ws_indices: List[int] = []
        self._non_ws_indices: List[int] = []
        self._build_pools()

    def _build_pools(self) -> None:
        gws_cache: Dict[str, np.ndarray] = {}

        for global_idx, (vid, cell_idx) in enumerate(self.dataset._index):
            if vid not in gws_cache:
                cap_path = (self.repo_root / self.manifest_by_id[vid]["cap_path"]).resolve()
                cap = load_capability_map(cap_path)
                gws_cache[vid] = np.asarray(cap["g_ws"], dtype=np.float32).reshape(-1)

            ws = float(gws_cache[vid][cell_idx]) > 0.5
            if ws:
                self._ws_indices.append(global_idx)
            else:
                self._non_ws_indices.append(global_idx)

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.cfg.seed)

        ws = list(self._ws_indices)
        non_ws = list(self._non_ws_indices)

        rng.shuffle(ws)
        rng.shuffle(non_ws)

        i_ws = 0
        i_nw = 0

        while True:
            pick_ws = rng.random() < self.cfg.ws_ratio
            if pick_ws and len(ws) > 0:
                yield ws[i_ws]
                i_ws = (i_ws + 1) % len(ws)
            elif len(non_ws) > 0:
                yield non_ws[i_nw]
                i_nw = (i_nw + 1) % len(non_ws)
            else:
                if len(ws) > 0:
                    yield ws[i_ws]
                    i_ws = (i_ws + 1) % len(ws)

    def __len__(self) -> int:
        return len(self.dataset)


class GroupedBalancedBatchSampler(Sampler[List[int]]):
    """Batch sampler that yields same-morphology (grouped) batches with ws/non-ws mixing.

    Each yielded batch is a list of global indices, all belonging to the same variant_id.

    Why this matters
    ----------------
    - Enables caching morphology encoding once per batch (single morph_spec).
    - Stabilizes cross-morph generalization by learning a conditional mapping under fixed morph.
    - Still provides negative g_ws examples via ws/non-ws mixture.
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
            # Fallback: infer from dataset index.
            self.variant_ids = sorted(list({vid for vid, _ in self.dataset._index}))

        # Pools per variant: vid -> (ws_indices, non_ws_indices)
        self._pools: Dict[str, Tuple[List[int], List[int]]] = {}
        self._build_pools()

        # Weights for variant sampling (proportional to num_cells when available).
        self._variant_weights: List[float] = []
        for vid in self.variant_ids:
            rec = self.manifest_by_id.get(vid, {})
            w = float(rec.get("num_cells", 1.0))
            if not np.isfinite(w) or w <= 0.0:
                w = 1.0
            self._variant_weights.append(w)

        # Pointers per variant to cycle through pools.
        self._ptr_ws: Dict[str, int] = {vid: 0 for vid in self.variant_ids}
        self._ptr_nw: Dict[str, int] = {vid: 0 for vid in self.variant_ids}

    def _build_pools(self) -> None:
        gws_cache: Dict[str, np.ndarray] = {}

        ws_pool: Dict[str, List[int]] = {vid: [] for vid in self.variant_ids}
        nw_pool: Dict[str, List[int]] = {vid: [] for vid in self.variant_ids}

        for global_idx, (vid, cell_idx) in enumerate(self.dataset._index):
            if vid not in gws_cache:
                cap_path = (self.repo_root / self.manifest_by_id[vid]["cap_path"]).resolve()
                cap = load_capability_map(cap_path)
                gws_cache[vid] = np.asarray(cap["g_ws"], dtype=np.float32).reshape(-1)

            ws = float(gws_cache[vid][cell_idx]) > 0.5
            if ws:
                ws_pool[vid].append(global_idx)
            else:
                nw_pool[vid].append(global_idx)

        # Shuffle pools deterministically.
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
            # Weighted choice by num_cells to approximate global cell distribution.
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
                    # Degenerate fallback: if one pool is empty, draw from the other.
                    if len(ws) > 0:
                        i = self._ptr_ws[vid] % len(ws)
                        batch.append(ws[i])
                        self._ptr_ws[vid] += 1

            yield batch

    def __len__(self) -> int:
        # Define one "epoch" as iterating over the flattened dataset once (approx).
        return max(1, len(self.dataset) // max(1, self.cfg.batch_size))
