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
    """Configuration for task-space point feature construction.

    Contract v1 default expects 9D:
        aabb_ratio(3) + aabb_centered(3) + p_over_Lr(3)
    """
    use_aabb_ratio: bool = False
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

        if len(self.label_keys) == 0:
            raise ValueError("[Stage1Dataset] label_keys is empty. This violates the Stage-1 contract.")

        self.ws_name = "g_ws"
        if self.ws_name not in self.label_keys:
            raise ValueError(
                f"[Stage1Dataset] ws_name='{self.ws_name}' must be included in label_keys. "
                f"Got label_keys={self.label_keys}"
            )
        self.ws_idx = int(self.label_keys.index(self.ws_name))

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
        """Construct pose features in a fixed order.

        Contract v1 default (F=9):
            [aabb_ratio(3), aabb_centered(3), p_over_Lr(3)]
        """
        feats: List[np.ndarray] = []

        # AABB-derived features (ratio / centered)
        if self.pose_cfg.use_aabb_ratio or self.pose_cfg.use_aabb_centered:
            bmin = grid.bounds_min.astype(np.float32)
            bmax = grid.bounds_max.astype(np.float32)
            denom = np.maximum(bmax - bmin, 1e-8).astype(np.float32)
            phat = ((p.astype(np.float32) - bmin) / denom).astype(np.float32)  # [0,1]

            # IMPORTANT: fixed concat order
            if self.pose_cfg.use_aabb_ratio:
                feats.append(phat)
            if self.pose_cfg.use_aabb_centered:
                feats.append((2.0 * phat - 1.0).astype(np.float32))  # [-1,1]

        # Morph-scale normalized feature
        if self.pose_cfg.use_morph_scale:
            Lr = float(Lr)
            feats.append((p.astype(np.float32) / max(Lr, 1e-8)).astype(np.float32))

        if len(feats) == 0:
            feats.append(p.astype(np.float32))
                                                                                                                                                                                
        out = np.concatenate(feats, axis=0).astype(np.float32)

        # Contract assertions (shape)
        expected = 0
        expected += 3 if self.pose_cfg.use_aabb_ratio else 0
        expected += 3 if self.pose_cfg.use_aabb_centered else 0
        expected += 3 if self.pose_cfg.use_morph_scale else 0
        expected = expected if expected > 0 else 3
        if out.shape != (expected,):
            raise ValueError(f"[Stage1Dataset] pose_feats shape mismatch: got {out.shape}, expected {(expected,)}")

        return out


    def __getitem__(self, idx: int) -> Dict[str, Any]:
        vid, cell_idx = self._index[idx]
        rec = self.records_by_id[vid]

        cap = self._load_cap(vid)
        spec = self._load_spec(vid)

        centers = np.asarray(cap["cell_centers"], dtype=np.float32)
        p = centers[cell_idx]  # (3,)

        grid = self._get_grid_meta(vid, centers)
        Lr = self._get_scale(vid, spec)

        pose_feats = self._build_point_features(p, grid, Lr)  # (F,)

        # Labels
        labels = []
        for k in self.label_keys:
            arr = np.asarray(cap[k], dtype=np.float32).reshape(-1)
            labels.append(arr[cell_idx])
        y = np.asarray(labels, dtype=np.float32)


        # Contract assertion: labels shape
        if y.shape != (len(self.label_keys),):
            raise ValueError(f"[Stage1Dataset] labels shape mismatch: got {y.shape}, expected {(len(self.label_keys),)}")

        # Contract v1: ws_mask is derived strictly from labels[g_ws].
        ws_val = float(y[self.ws_idx])
        ws_mask = bool(ws_val > 0.5)

        # Optional audit: check cap['g_ws'] consistency if it exists.
        if "g_ws" in cap:
            gws_cap = float(np.asarray(cap["g_ws"], dtype=np.float32).reshape(-1)[cell_idx])
            if abs(gws_cap - ws_val) > 1e-5:
                raise ValueError(
                    f"[Stage1Dataset] Inconsistent g_ws between labels and cap['g_ws']: "
                    f"labels={ws_val} cap={gws_cap} (variant_id={vid}, cell_idx={cell_idx})"
                )

        sample = {
            "variant_id": vid,
            "family": rec["family"],
            "dof": int(rec["dof"]),
            "cell_idx": int(cell_idx),
            "pose_feats": torch.from_numpy(pose_feats),          # (F,)
            "labels": torch.from_numpy(y),                       # (K,)
            "ws_mask": torch.tensor(ws_mask, dtype=torch.bool),  # ()
        }
        sample["morph_spec"] = spec
        return sample
    
    @staticmethod
    def collate_stage1_grouped(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate function for grouped-by-variant Stage I batches.

        Requirements
        ------------
        - All samples in `batch` must share the same variant_id (grouped sampling).
        - Keep only one morph_spec to avoid redundant graph building / caching.
        """
        if len(batch) == 0:
            raise ValueError("Empty batch is not allowed.")

        vid0 = batch[0]["variant_id"]
        for s in batch:
            if s["variant_id"] != vid0:
                raise ValueError(
                    f"Grouped collate expects a single variant_id per batch, "
                    f"but got {vid0} and {s['variant_id']}."
                )

        point_feats = torch.stack([s["point_feats"] for s in batch], dim=0)  # (B, F)
        labels = torch.stack([s["labels"] for s in batch], dim=0)            # (B, K)
        ws_mask = torch.stack([s["ws_mask"] for s in batch], dim=0).view(-1) # (B,)

        out: Dict[str, Any] = {
            "variant_id": vid0,
            "family": batch[0].get("family", None),
            "dof": batch[0].get("dof", None),
            "cell_idx": torch.tensor([s["cell_idx"] for s in batch], dtype=torch.long),
            "point_feats": point_feats,
            "labels": labels,
            "ws_mask": ws_mask,
            # Keep a single spec for the entire batch.
            "morph_spec": batch[0]["morph_spec"],
        }
        return out
