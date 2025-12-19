from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np
import torch
from torch.utils.data import Sampler

from mcfp.data.io import load_capability_map


@dataclass(frozen=True)
class BalancedSamplingConfig:
    """Sampling configuration for mixing ws/non-ws cells."""
    ws_ratio: float = 0.8
    seed: int = 42


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
        # Build a cache of g_ws per morphology to avoid reloading per cell.
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

        # Infinite-like sampling: cycle through pools.
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
                # Fallback if one pool is empty.
                if len(ws) > 0:
                    yield ws[i_ws]
                    i_ws = (i_ws + 1) % len(ws)

    def __len__(self) -> int:
        # PyTorch requires a finite length; define as dataset size.
        return len(self.dataset)
