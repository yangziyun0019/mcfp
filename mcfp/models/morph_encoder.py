"""Encode robot morphology specifications into conditioning embeddings.

The morphology encoder combines link geometry and graph structure for downstream MCFP models.
"""

from __future__ import annotations

from typing import Tuple

import torch
import numpy as np
from torch import nn

from mcfp.data.morph_spec_io import MorphologySpec
from mcfp.models.pointnet import PointNetEncoder
from mcfp.models.graph_transformer import GraphData, GraphTransformer


class MorphologyEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        depth_emb_dim: int = 16,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.depth_emb = nn.Embedding(64, depth_emb_dim)
        self.pointnet = PointNetEncoder(in_dim=6, out_dim=128)
        # mass(1) + inertia(6) + bbox(3) + depth_emb + pointnet
        in_dim = depth_emb_dim + 128 + 1 + 6 + 3
        self.proj = nn.Linear(in_dim, d_model)
        self.graph = GraphTransformer(dim=d_model, num_layers=num_layers, num_heads=num_heads)
        self.dropout = nn.Dropout(dropout)

    def _build_node_features(self, spec: MorphologySpec, device: torch.device) -> torch.Tensor:
        points = torch.from_numpy(spec.points).to(device)
        depth = torch.from_numpy(spec.depth.astype("int64")).to(device)
        bbox_min = torch.from_numpy(spec.bbox_min).to(device)
        bbox_max = torch.from_numpy(spec.bbox_max).to(device)
        mass = torch.from_numpy(spec.mass).to(device)
        inertia = torch.from_numpy(spec.inertia).to(device)

        l_ref = float(spec.l_ref) if spec.l_ref > 0 else 1.0
        # normalize
        bbox = (bbox_max - bbox_min) / l_ref
        mass_norm = mass / torch.clamp(torch.nan_to_num(mass).max(), min=1e-6)
        inertia_norm = inertia / torch.clamp(torch.nan_to_num(mass).max() * (l_ref ** 2), min=1e-6)

        g_i = self.pointnet(points)
        d_i = self.depth_emb(torch.clamp(depth, max=63))

        feats = torch.cat([
            d_i,
            g_i,
            mass_norm.unsqueeze(-1),
            inertia_norm,
            bbox,
        ], dim=1)
        return feats

    def forward(self, spec: MorphologySpec, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._build_node_features(spec, device)
        x = self.dropout(self.proj(x))

        kin_mask = torch.from_numpy(spec.kin_mask).to(device)
        col_mask = torch.from_numpy(spec.col_mask).to(device)
        tree_dist = torch.from_numpy(spec.tree_dist).to(device)
        acm_mask = torch.from_numpy(spec.acm_mask).to(device)


        n = x.shape[0]
        # build kinematic edge features (N,N,F)
        edge_dim = 19
        kin_edge_feat = torch.zeros((n, n, edge_dim), device=device)
        # joint type mapping
        type_map = {"revolute": 0, "prismatic": 1, "fixed": 2, "continuous": 3}
        for j_idx, (p, c) in enumerate(zip(spec.joint_parent, spec.joint_child)):
            if p < 0 or c < 0:
                continue
            jtype = spec.joint_type[j_idx] if j_idx < len(spec.joint_type) else "fixed"
            t_idx = type_map.get(str(jtype), 2)
            onehot = torch.zeros((4,), device=device)
            onehot[t_idx] = 1.0
            axis = torch.from_numpy(spec.joint_axis[j_idx]).to(device)
            origin_xyz = torch.from_numpy(spec.joint_origin_xyz[j_idx]).to(device)
            origin_rpy = torch.from_numpy(spec.joint_origin_rpy[j_idx]).to(device)
            lo = float(spec.joint_limit_lower[j_idx]) if j_idx < len(spec.joint_limit_lower) else 0.0
            hi = float(spec.joint_limit_upper[j_idx]) if j_idx < len(spec.joint_limit_upper) else 0.0
            if np.isnan(lo) or np.isnan(hi):
                lo, hi = 0.0, 0.0
            limit_mid = 0.5 * (lo + hi)
            limit_range = hi - lo
            is_act = 1.0 if str(jtype) in ("revolute", "prismatic", "continuous") else 0.0
            feat = torch.cat([
                onehot,
                axis,
                origin_xyz,
                origin_rpy,
                torch.tensor([lo, hi, limit_range, limit_mid, is_act], device=device),
            ], dim=0)
            # parent -> child (edge_dir=+1)
            kin_edge_feat[p, c, :-1] = feat
            kin_edge_feat[p, c, -1] = 1.0
            # child -> parent (edge_dir=-1)
            kin_edge_feat[c, p, :-1] = feat
            kin_edge_feat[c, p, -1] = -1.0

        graph = GraphData(x=x, kin_mask=kin_mask, col_mask=col_mask, tree_dist=tree_dist, acm_mask=acm_mask, kin_edge_feat=kin_edge_feat)
        node_emb = self.graph(graph)
        global_emb = torch.mean(node_emb, dim=0, keepdim=True)
        return global_emb, node_emb
