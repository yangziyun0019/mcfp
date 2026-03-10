"""Graph transformer blocks used by the morphology encoder.

This module implements attention-based message passing over robot link graphs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class GraphData:
    x: torch.Tensor  # (N, D)
    kin_mask: torch.Tensor  # (N, N) bool
    col_mask: torch.Tensor  # (N, N) bool
    tree_dist: torch.Tensor  # (N, N) int
    acm_mask: torch.Tensor  # (N, N) bool
    kin_edge_feat: Optional[torch.Tensor] = None  # (N, N, F)


class GraphTransformerLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        # x: (N,D), bias: (N,N,H)
        n = x.shape[0]
        qkv = self.qkv(x).reshape(n, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # (N,H,dh)

        # scores: (H,N,N)
        scores = torch.einsum("nhd,mhd->hnm", q, k) / (self.head_dim ** 0.5)
        scores = scores + bias.permute(2, 0, 1)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum("hnm,mhd->nhd", attn, v).reshape(n, self.dim)
        out = self.out_proj(out)
        x = self.norm1(x + out)
        x = self.norm2(x + self.ffn(x))
        return x


class GraphTransformer(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        max_tree_dist: int = 16,
        edge_dim: int = 19,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [GraphTransformerLayer(dim, num_heads) for _ in range(num_layers)]
        )
        self.num_heads = num_heads
        self.type_embed = nn.Embedding(3, num_heads)  # 0 none, 1 kin, 2 col
        self.dist_embed = nn.Embedding(max_tree_dist + 1, num_heads)
        self.acm_embed = nn.Embedding(2, num_heads)
        self.edge_proj = nn.Linear(edge_dim, num_heads)

    def _build_bias(self, graph: GraphData) -> torch.Tensor:
        # bias: (N,N,H)
        n = graph.x.shape[0]
        kin = graph.kin_mask
        col = graph.col_mask
        dist = graph.tree_dist.clamp(min=0)
        max_d = self.dist_embed.num_embeddings - 1
        dist = torch.clamp(dist, max=max_d)
        acm = graph.acm_mask.long()

        bias = torch.zeros((n, n, self.num_heads), device=graph.x.device)
        kin_bias = self.type_embed.weight[1]  # (H,)
        col_bias = self.type_embed.weight[2]
        bias = bias + kin.unsqueeze(-1) * kin_bias
        bias = bias + col.unsqueeze(-1) * col_bias
        bias = bias + self.dist_embed(dist)
        bias = bias + self.acm_embed(acm)
        if graph.kin_edge_feat is not None:
            edge_bias = self.edge_proj(graph.kin_edge_feat)
            bias = bias + edge_bias
        return bias

    def forward(self, graph: GraphData) -> torch.Tensor:
        x = graph.x
        bias = self._build_bias(graph)
        for layer in self.layers:
            x = layer(x, bias)
        return x
