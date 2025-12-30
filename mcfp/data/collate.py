from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import torch

from mcfp.models.morph_graph import GraphData, build_link_graph


@dataclass
class PoseBatchCollator:
    """Grouped collator for pose samples (one morphology per batch).

    Expected input item fields:
      - pose_feats: Tensor[F] or array-like
      - labels: Tensor[K] or array-like
      - delta_mask: float or Tensor scalar
      - ws_mask: float or Tensor scalar
      - variant_id: str
      - morph_spec: dict

    Output batch fields:
      - pose_feats: Tensor[B, F]
      - labels: Tensor[B, K]
      - delta_mask: Tensor[B]
      - ws_mask: Tensor[B]
      - label_keys: List[str]
      - morph_graph: GraphData
    """

    label_keys: Sequence[str]
    graph_bidirectional: bool = True
    graph_use_link_index: bool = True
    strict_one_morph_per_batch: bool = True
    cache_graph: bool = True

    def __post_init__(self) -> None:
        self.label_keys = [str(k) for k in self.label_keys]
        if "g_ws" not in self.label_keys:
            raise ValueError("[PoseBatchCollator] label_keys must include 'g_ws'.")
        self._graph_cache: Dict[str, GraphData] = {}

    def __call__(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(items) == 0:
            raise ValueError("[PoseBatchCollator] Empty batch.")

        vids = [str(it["variant_id"]) for it in items]
        vid0 = vids[0]
        if self.strict_one_morph_per_batch and any(v != vid0 for v in vids):
            raise ValueError("[PoseBatchCollator] Batch contains multiple variant_id values.")

        if self.cache_graph and vid0 in self._graph_cache:
            morph_graph = self._graph_cache[vid0]
        else:
            spec = items[0]["morph_spec"]
            g = build_link_graph(
                spec,
                device=None,
                dtype=torch.float32,
                bidirectional=self.graph_bidirectional,
                use_link_index_feature=self.graph_use_link_index,
            )
            morph_graph = g if g.batch is not None else GraphData(
                x=g.x,
                edge_index=g.edge_index,
                edge_attr=g.edge_attr,
                batch=torch.zeros((g.x.shape[0],), dtype=torch.long),
                node_names=g.node_names,
                meta=g.meta,
            )
            if self.cache_graph:
                self._graph_cache[vid0] = morph_graph

        pose_list = []
        for it in items:
            x = it["pose_feats"]
            if not torch.is_tensor(x):
                x = torch.as_tensor(x, dtype=torch.float32)
            x = x.to(dtype=torch.float32).view(-1)
            pose_list.append(x)
        pose_feats = torch.stack(pose_list, dim=0)

        label_list = []
        K = len(self.label_keys)
        for it in items:
            y = it["labels"]
            if not torch.is_tensor(y):
                y = torch.as_tensor(y, dtype=torch.float32)
            y = y.to(dtype=torch.float32).view(-1)
            if y.shape[0] != K:
                raise ValueError(f"[PoseBatchCollator] labels must be K={K}, got {y.shape}")
            label_list.append(y)
        labels = torch.stack(label_list, dim=0)

        delta_mask_list = []
        ws_mask_list = []
        for it in items:
            dm = it.get("delta_mask", 0.0)
            wm = it.get("ws_mask", 0.0)
            if torch.is_tensor(dm):
                dm = float(dm.view(-1)[0].item())
            if torch.is_tensor(wm):
                wm = float(wm.view(-1)[0].item())
            delta_mask_list.append(1.0 if dm > 0.5 else 0.0)
            ws_mask_list.append(1.0 if wm > 0.5 else 0.0)

        delta_mask = torch.as_tensor(delta_mask_list, dtype=torch.float32)
        ws_mask = torch.as_tensor(ws_mask_list, dtype=torch.float32)

        return {
            "variant_id": vid0,
            "pose_feats": pose_feats,
            "labels": labels,
            "delta_mask": delta_mask,
            "ws_mask": ws_mask,
            "label_keys": list(self.label_keys),
            "morph_graph": morph_graph,
        }
