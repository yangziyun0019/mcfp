from __future__ import annotations

from pathlib import Path
from typing import Optional

import math

import torch
from torch import nn

from mcfp.data.io import load_morph_spec
from mcfp.models.morph_graph import GraphData, build_joint_feature_graph


def build_morph_graph_from_json(
    spec_path: Path | str,
    *,
    l_ref: float,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    bidirectional: bool = True,
) -> GraphData:
    """Load a morph spec JSON and build a graph."""
    spec = load_morph_spec(Path(spec_path))
    return build_joint_feature_graph(
        spec,
        l_ref=l_ref,
        device=device,
        dtype=dtype,
        bidirectional=bidirectional,
    )


class GraphTransformerLayer(nn.Module):
    """Graph Transformer layer with dense neighbor attention."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.d_head = int(d_model // n_heads)

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor, adj_mask: torch.Tensor) -> torch.Tensor:
        """Apply one transformer layer.

        Args:
            x: Node features (N, d_model).
            adj_mask: Boolean adjacency mask (N, N), True for valid neighbors.
        """
        n = x.shape[0]
        qkv = self.qkv(x).view(n, 3, self.n_heads, self.d_head)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        q = q.permute(1, 0, 2)
        k = k.permute(1, 0, 2)
        v = v.permute(1, 0, 2)

        scores = torch.einsum("hnd,hmd->hnm", q, k) / math.sqrt(self.d_head)
        mask_value = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~adj_mask.unsqueeze(0), mask_value)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum("hnm,hmd->hnd", attn, v).permute(1, 0, 2)
        out = out.reshape(n, self.d_model)
        out = self.out_proj(out)

        x = self.norm1(x + self.dropout(out))
        ff = self.ffn(x)
        x = self.norm2(x + self.dropout(ff))
        return x


class GraphTransformerEncoder(nn.Module):
    """Graph Transformer encoder for morphology graphs."""

    def __init__(
        self,
        in_dim: int,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 8,
        d_ff: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_dim, d_model)
        self.layers = nn.ModuleList(
            [
                GraphTransformerLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                )
                for _ in range(int(n_layers))
            ]
        )

    def forward(self, graph: GraphData) -> torch.Tensor:
        """Encode a single morphology graph."""
        x = self.input_proj(graph.x)
        n = x.shape[0]
        adj_mask = torch.zeros((n, n), dtype=torch.bool, device=x.device)
        if graph.edge_index.numel() > 0:
            src = graph.edge_index[0].long()
            dst = graph.edge_index[1].long()
            adj_mask[src, dst] = True
        adj_mask.fill_diagonal_(True)

        for layer in self.layers:
            x = layer(x, adj_mask=adj_mask)
        return x


class MorphologyEncoder(nn.Module):
    """Morphology encoder producing a global embedding."""

    def __init__(
        self,
        node_feat_dim: int,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 8,
        d_ff: int = 512,
        dropout: float = 0.1,
        out_dim: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = GraphTransformerEncoder(
            in_dim=node_feat_dim,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.proj = nn.Linear(d_model, out_dim)

    @staticmethod
    def _mean_pool(x: torch.Tensor, batch: Optional[torch.Tensor]) -> torch.Tensor:
        """Mean-pool node features into graph features."""
        if batch is None:
            return x.mean(dim=0, keepdim=True)
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        out = torch.zeros((num_graphs, x.shape[1]), device=x.device, dtype=x.dtype)
        counts = torch.zeros((num_graphs,), device=x.device, dtype=x.dtype)
        out.index_add_(0, batch, x)
        counts.index_add_(0, batch, torch.ones_like(batch, dtype=x.dtype))
        out = out / counts.clamp_min(1.0).unsqueeze(1)
        return out

    def forward(self, graph: GraphData) -> torch.Tensor:
        """Compute morphology embedding."""
        h = self.encoder(graph)
        pooled = self._mean_pool(h, graph.batch)
        return self.proj(pooled)
