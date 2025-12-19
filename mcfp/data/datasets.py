from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from mcfp.data.io import GridMeta, infer_grid_meta_from_centers, load_capability_map, load_morph_spec


@dataclass(frozen=True)
class PoseFeatureConfig:
    """Configuration for task-space point feature construction."""
    use_aabb_centered: bool = True
    use_morph_scale: bool = True
    grid_round_decimals: int = 6


class Stage1Dataset(Dataset):
    """Stage I dataset: (morphology, cell) -> point_feats, labels, masks, meta.

    Notes
    -----
    - Regression labels are only defined on ws_mask (g_ws==1). The training loop
      must mask regression losses accordingly.
    - This dataset is manifest-driven: it does not require pre-flattened samples.
    """

    def __init__(
        self,
        repo_root: Path,
        manifest_records: List[Dict[str, Any]],
        variant_ids: List[str],
        label_keys: List[str],
        pose_cfg: PoseFeatureConfig,
        cache_maps: bool = True,
        cache_specs: bool = True,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.records_by_id = {r["variant_id"]: r for r in manifest_records}
        self.variant_ids = list(variant_ids)
        self.label_keys = list(label_keys)
        self.pose_cfg = pose_cfg
        self.cache_maps = bool(cache_maps)
        self.cache_specs = bool(cache_specs)

        # Cache: variant_id -> arrays/spec/grid_meta/morph_scale
        self._cap_cache: Dict[str, Dict[str, np.ndarray]] = {}
        self._spec_cache: Dict[str, Dict[str, Any]] = {}
        self._grid_cache: Dict[str, GridMeta] = {}
        self._scale_cache: Dict[str, float] = {}

        # Global index: (variant_id, cell_idx)
        self._index: List[Tuple[str, int]] = []
        self._build_index()

    def _build_index(self) -> None:
        self._index.clear()
        for vid in self.variant_ids:
            rec = self.records_by_id[vid]
            n = int(rec["num_cells"])
            for i in range(n):
                self._index.append((vid, i))

    def __len__(self) -> int:
        return len(self._index)

    def _load_cap(self, vid: str) -> Dict[str, np.ndarray]:
        if self.cache_maps and vid in self._cap_cache:
            return self._cap_cache[vid]
        cap_path = (self.repo_root / self.records_by_id[vid]["cap_path"]).resolve()
        cap = load_capability_map(cap_path)
        if self.cache_maps:
            self._cap_cache[vid] = cap
        return cap

    def _load_spec(self, vid: str) -> Dict[str, Any]:
        if self.cache_specs and vid in self._spec_cache:
            return self._spec_cache[vid]
        spec_path = (self.repo_root / self.records_by_id[vid]["spec_path"]).resolve()
        spec = load_morph_spec(spec_path)
        if self.cache_specs:
            self._spec_cache[vid] = spec
        return spec

    def _get_grid_meta(self, vid: str, centers: np.ndarray) -> GridMeta:
        if vid in self._grid_cache:
            return self._grid_cache[vid]
        # Prefer manifest stored meta if present; otherwise infer.
        rec = self.records_by_id[vid]
        gm = rec.get("grid_meta", None)
        if isinstance(gm, dict) and "bounds_min" in gm and "bounds_max" in gm and "resolution" in gm and "cell_shape" in gm:
            grid = GridMeta(
                bounds_min=np.asarray(gm["bounds_min"], dtype=np.float32),
                bounds_max=np.asarray(gm["bounds_max"], dtype=np.float32),
                resolution=np.asarray(gm["resolution"], dtype=np.float32),
                cell_shape=tuple(int(x) for x in gm["cell_shape"]),
            )
        else:
            grid = infer_grid_meta_from_centers(
                centers, decimals=self.pose_cfg.grid_round_decimals
            )
        self._grid_cache[vid] = grid
        return grid

    def _get_scale(self, vid: str, spec: Dict[str, Any]) -> float:
        if vid in self._scale_cache:
            return self._scale_cache[vid]
        rec = self.records_by_id[vid]
        Lr = float(rec.get("morph_scale", 1.0))
        if not np.isfinite(Lr) or Lr <= 0.0:
            # Fallback: compute from spec if manifest missing.
            links = spec.get("chain", {}).get("links", [])
            Lr = float(np.sum([float(l.get("length_estimate", 0.0)) for l in links])) if links else 1.0
            if not np.isfinite(Lr) or Lr <= 0.0:
                Lr = 1.0
        self._scale_cache[vid] = Lr
        return Lr

    def _build_point_features(self, p: np.ndarray, grid: GridMeta, Lr: float) -> np.ndarray:
        feats: List[np.ndarray] = []

        if self.pose_cfg.use_aabb_centered:
            bmin = grid.bounds_min.astype(np.float32)
            bmax = grid.bounds_max.astype(np.float32)
            denom = np.maximum(bmax - bmin, 1e-8).astype(np.float32)
            phat = (p.astype(np.float32) - bmin) / denom
            # Center to [-1, 1]
            phat_c = (2.0 * phat - 1.0).astype(np.float32)
            feats.append(phat_c)

        if self.pose_cfg.use_morph_scale:
            Lr = float(Lr)
            ps = (p.astype(np.float32) / max(Lr, 1e-8)).astype(np.float32)
            feats.append(ps)

        if len(feats) == 0:
            feats.append(p.astype(np.float32))

        return np.concatenate(feats, axis=0).astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        vid, cell_idx = self._index[idx]
        rec = self.records_by_id[vid]

        cap = self._load_cap(vid)
        spec = self._load_spec(vid)

        centers = np.asarray(cap["cell_centers"], dtype=np.float32)
        p = centers[cell_idx]  # (3,)

        grid = self._get_grid_meta(vid, centers)
        Lr = self._get_scale(vid, spec)

        point_feats = self._build_point_features(p, grid, Lr)  # (F,)

        # Labels
        labels = []
        for k in self.label_keys:
            arr = np.asarray(cap[k], dtype=np.float32).reshape(-1)
            labels.append(arr[cell_idx])
        y = np.asarray(labels, dtype=np.float32)

        gws = float(np.asarray(cap["g_ws"], dtype=np.float32).reshape(-1)[cell_idx])
        ws_mask = 1.0 if gws > 0.5 else 0.0

        sample = {
            "variant_id": vid,
            "family": rec["family"],
            "dof": int(rec["dof"]),
            "cell_idx": int(cell_idx),
            "point_feats": torch.from_numpy(point_feats),  # (F,)
            "labels": torch.from_numpy(y),                 # (K,)
            "ws_mask": torch.tensor(ws_mask, dtype=torch.float32),
        }

        # Morphology graph encoding is model-side; here we only return spec
        # as a structured dict for downstream graph builder.
        sample["morph_spec"] = spec
        return sample
