# mcfp/data/collate.py

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from mcfp.models.morph_graph import GraphData, build_link_graph


def _to_tensor_1d(x: Any, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert a scalar or 1D array-like to a 1D tensor."""
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype).view(-1)
    t = torch.as_tensor(x, dtype=dtype)
    return t.view(-1)


def _to_tensor_2d(x: Any, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert array-like to a 2D tensor."""
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype)
    return torch.as_tensor(x, dtype=dtype)


def _read_json(path: Path) -> Dict[str, Any]:
    """Read a JSON file into a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class MorphGraphCache:
    """In-memory cache of morphology GraphData keyed by morph_id.

    This cache is intentionally simple (dict-based) since morph_id count is small (~204).
    """
    morph_specs_root: Path
    device: Optional[torch.device] = None
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        self.morph_specs_root = Path(self.morph_specs_root)
        self._cache: Dict[str, GraphData] = {}

    def get(self, morph_id: str, morph_spec_relpath: Optional[str] = None) -> GraphData:
        """Get (or build) GraphData for a morph_id.

        Args:
            morph_id: Morphology identifier (e.g., 'wx200_v0001').
            morph_spec_relpath: Optional relative path to json under morph_specs_root.

        Returns:
            GraphData on CPU by default (device can be specified).
        """
        morph_id = str(morph_id)
        if morph_id in self._cache:
            return self._cache[morph_id]

        if morph_spec_relpath is None:
            # Default convention: <root>/<family>/<morph_id>.json OR <root>/<morph_id>.json
            # We try a few reasonable candidates to avoid hard-coding a single layout.
            candidates = [
                self.morph_specs_root / f"{morph_id}.json",
            ]
            # If morph_id has a family prefix like "wx200_v0001", also try root/wx200/morph_id.json
            family = morph_id.split("_")[0]
            candidates.append(self.morph_specs_root / family / f"{morph_id}.json")

            spec_path = None
            for p in candidates:
                if p.exists():
                    spec_path = p
                    break
            if spec_path is None:
                raise FileNotFoundError(
                    f"[MorphGraphCache] Cannot locate morph spec json for morph_id={morph_id}. "
                    f"Tried: {[str(p) for p in candidates]}"
                )
        else:
            spec_path = self.morph_specs_root / morph_spec_relpath
            if not spec_path.exists():
                raise FileNotFoundError(f"[MorphGraphCache] morph_spec path not found: {spec_path}")

        spec = _read_json(spec_path)
        graph = build_link_graph(spec, device=self.device, dtype=self.dtype)
        self._cache[morph_id] = graph
        return graph


class GroupedBatchCollator:
    """Contract v1 collator: one morphology (variant_id) per batch.

    Input item (from Stage1Dataset.__getitem__) MUST contain:
      - 'pose_feats': Tensor[9]
      - 'labels': Tensor[K]
      - 'ws_mask': bool or Tensor scalar
      - 'variant_id': str
      - 'morph_spec': dict

    Output batch MUST contain (contract v1):
      - variant_id: str
      - pose_feats: Tensor[B,9]
      - labels: Tensor[B,K]
      - ws_mask: BoolTensor[B]
      - label_keys: List[str]
      - morph_graph: GraphData
    """

    def __init__(
        self,
        label_keys: Sequence[str],
        pose_key: str = "pose_feats",
        variant_id_key: str = "variant_id",
        labels_key: str = "labels",
        ws_mask_key: str = "ws_mask",
        morph_spec_key: str = "morph_spec",
        strict_one_morph_per_batch: bool = True,
        cache_graph: bool = True,
    ) -> None:
        self.label_keys = [str(k) for k in label_keys]
        self.pose_key = str(pose_key)
        self.variant_id_key = str(variant_id_key)
        self.labels_key = str(labels_key)
        self.ws_mask_key = str(ws_mask_key)
        self.morph_spec_key = str(morph_spec_key)
        self.strict_one_morph_per_batch = bool(strict_one_morph_per_batch)
        self.cache_graph = bool(cache_graph)
        self._graph_cache: Dict[str, Any] = {}

        # Contract invariant
        if "g_ws" not in self.label_keys:
            raise ValueError(
                f"[GroupedBatchCollator] label_keys must include 'g_ws'. Got {self.label_keys}"
            )

    def __call__(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(items) == 0:
            raise ValueError("[GroupedBatchCollator] Empty batch.")

        # ---- enforce single-morph batch ----
        vids = [str(it[self.variant_id_key]) for it in items]
        vid0 = vids[0]
        if self.strict_one_morph_per_batch and any(v != vid0 for v in vids):
            raise ValueError(
                f"[GroupedBatchCollator] Batch contains multiple variant_id: {sorted(set(vids))}. "
                f"Sampler must enforce grouped batches."
            )

        # ---- build / cache morph graph once per batch ----
        morph_spec = items[0][self.morph_spec_key]
        if self.cache_graph and vid0 in self._graph_cache:
            morph_graph = self._graph_cache[vid0]
        else:
            morph_graph = build_link_graph(morph_spec)
            if self.cache_graph:
                self._graph_cache[vid0] = morph_graph

        # ---- stack pose feats ----
        pose_list = []
        for it in items:
            x = it[self.pose_key]
            if not torch.is_tensor(x):
                x = torch.as_tensor(x, dtype=torch.float32)
            x = x.to(dtype=torch.float32)
            if x.ndim != 1:
                raise ValueError(f"[GroupedBatchCollator] pose_feats must be 1D, got shape={tuple(x.shape)}")
            if x.shape[0] != 9:
                raise ValueError(f"[GroupedBatchCollator] pose_feats must be 9D, got shape={tuple(x.shape)}")
            pose_list.append(x)
        pose_feats = torch.stack(pose_list, dim=0)  # [B,9]

        # ---- stack labels ----
        y_list = []
        K = len(self.label_keys)
        for it in items:
            y = it[self.labels_key]
            if not torch.is_tensor(y):
                y = torch.as_tensor(y, dtype=torch.float32)
            y = y.to(dtype=torch.float32)
            if y.ndim != 1 or y.shape[0] != K:
                raise ValueError(
                    f"[GroupedBatchCollator] labels must be 1D with K={K}, got shape={tuple(y.shape)}"
                )
            y_list.append(y)
        labels = torch.stack(y_list, dim=0)  # [B,K]

        # ---- ws_mask ----
        ws_vals = []
        for it in items:
            w = it[self.ws_mask_key]
            if torch.is_tensor(w):
                w = bool(w.item())
            else:
                w = bool(w)
            ws_vals.append(w)
        ws_mask = torch.as_tensor(ws_vals, dtype=torch.bool)  # [B]

        return {
            "variant_id": vid0,
            "pose_feats": pose_feats,
            "labels": labels,
            "ws_mask": ws_mask,
            "label_keys": self.label_keys,
            "morph_graph": morph_graph,
        }