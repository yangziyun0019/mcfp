# mcfp/models/morph_encoder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mcfp.models.morph_graph import GraphData


def _segment_mean(x: torch.Tensor, batch: torch.Tensor, num_segments: int) -> torch.Tensor:
    """Compute segment-wise mean without external scatter dependencies.

    Args:
        x: Tensor of shape (N, D).
        batch: Long tensor of shape (N,) with values in [0, num_segments-1].
        num_segments: Number of segments.

    Returns:
        Tensor of shape (num_segments, D).
    """
    if x.numel() == 0:
        return torch.zeros((num_segments, x.shape[-1]), device=x.device, dtype=x.dtype)

    out = torch.zeros((num_segments, x.shape[-1]), device=x.device, dtype=x.dtype)
    out = out.index_add(0, batch, x)

    ones = torch.ones((x.shape[0],), device=x.device, dtype=x.dtype)
    cnt = torch.zeros((num_segments,), device=x.device, dtype=x.dtype)
    cnt = cnt.index_add(0, batch, ones).clamp_min(1.0)

    return out / cnt.unsqueeze(-1)


def _aggregate_mean(
    h: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Mean aggregation of neighbor features.

    Args:
        h: Node features (N, D).
        edge_index: (2, E), where edges are src -> dst.
        num_nodes: Number of nodes N.

    Returns:
        Aggregated neighbor features (N, D).
    """
    if edge_index.numel() == 0:
        return torch.zeros((num_nodes, h.shape[-1]), device=h.device, dtype=h.dtype)

    src = edge_index[0]
    dst = edge_index[1]

    out = torch.zeros((num_nodes, h.shape[-1]), device=h.device, dtype=h.dtype)
    out = out.index_add(0, dst, h[src])

    ones = torch.ones((dst.shape[0],), device=h.device, dtype=h.dtype)
    deg = torch.zeros((num_nodes,), device=h.device, dtype=h.dtype)
    deg = deg.index_add(0, dst, ones).clamp_min(1.0)

    return out / deg.unsqueeze(-1)


def _aggregate_mean_edge(
    h: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    msg_mlp: nn.Module,
    num_nodes: int,
) -> torch.Tensor:
    """Mean aggregation with edge features.

    Args:
        h: Node features (N, D).
        edge_index: (2, E), where edges are src -> dst.
        edge_attr: (E, F) edge features.
        msg_mlp: MLP mapping [h_src, edge_attr] -> (Dout).
        num_nodes: Number of nodes N.
    """
    if edge_index.numel() == 0:
        return torch.zeros((num_nodes, msg_mlp[-1].out_features), device=h.device, dtype=h.dtype)

    src = edge_index[0]
    dst = edge_index[1]

    h_src = h[src]
    m_in = torch.cat([h_src, edge_attr], dim=-1)
    msg = msg_mlp(m_in)

    out = torch.zeros((num_nodes, msg.shape[-1]), device=h.device, dtype=h.dtype)
    out = out.index_add(0, dst, msg)

    ones = torch.ones((dst.shape[0],), device=h.device, dtype=h.dtype)
    deg = torch.zeros((num_nodes,), device=h.device, dtype=h.dtype)
    deg = deg.index_add(0, dst, ones).clamp_min(1.0)

    return out / deg.unsqueeze(-1)


@dataclass(frozen=True)
class MorphologyEncoderConfig:
    """Configuration for MorphologyEncoderGNN."""
    input_dim: int
    hidden_dim: int
    num_layers: int
    edge_dim: int = 0
    dropout: float = 0.0
    use_layernorm: bool = True


class GraphSAGELayer(nn.Module):
    """A lightweight GraphSAGE-like layer using mean aggregation."""
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float,
        use_layernorm: bool,
        edge_dim: int = 0,
    ) -> None:
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim, bias=True)
        self.lin_neigh = nn.Linear(in_dim, out_dim, bias=True)
        self.dropout = float(dropout)
        self.norm = nn.LayerNorm(out_dim) if use_layernorm else None
        self.edge_dim = int(edge_dim)
        if self.edge_dim > 0:
            self.msg_mlp = nn.Sequential(
                nn.Linear(in_dim + self.edge_dim, out_dim),
                nn.GELU(),
                nn.Linear(out_dim, out_dim),
            )
            # When edge features are used, neighbor aggregation already outputs out_dim.
            self.lin_neigh = None
        else:
            self.msg_mlp = None

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            h: Node features (N, Din).
            edge_index: COO edges (2, E).
            edge_attr: Optional edge features (E, F).

        Returns:
            Updated node features (N, Dout).
        """
        n = h.shape[0]
        if self.edge_dim > 0 and edge_attr is not None:
            neigh = _aggregate_mean_edge(
                h, edge_index=edge_index, edge_attr=edge_attr, msg_mlp=self.msg_mlp, num_nodes=n
            )
            out = self.lin_self(h) + neigh
        else:
            neigh = _aggregate_mean(h, edge_index=edge_index, num_nodes=n)
            out = self.lin_self(h) + self.lin_neigh(neigh)
        out = F.silu(out)
        if self.norm is not None:
            out = self.norm(out)
        if self.dropout > 0.0:
            out = F.dropout(out, p=self.dropout, training=self.training)
        return out


class MorphologyEncoderGNN(nn.Module):
    """Encode a morphology link-graph into link-level and graph-level embeddings."""
    def __init__(self, cfg: MorphologyEncoderConfig) -> None:
        super().__init__()
        if cfg.num_layers < 1:
            raise ValueError("MorphologyEncoderGNN requires num_layers >= 1")

        layers = []
        in_dim = cfg.input_dim
        for _ in range(cfg.num_layers):
            layers.append(
                GraphSAGELayer(
                    in_dim=in_dim,
                    out_dim=cfg.hidden_dim,
                    dropout=cfg.dropout,
                    use_layernorm=cfg.use_layernorm,
                    edge_dim=cfg.edge_dim,
                )
            )
            in_dim = cfg.hidden_dim
        self.layers = nn.ModuleList(layers)

    def forward(self, graph: GraphData) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            graph: GraphData with fields:
                - x: (N, Din)
                - edge_index: (2, E)
                - edge_attr: Optional (E, F)
                - batch: Optional (N,), graph ids for batching

        Returns:
            H_links: (N, D) node embeddings
            z_morph: (B, D) graph embeddings, where B=1 if graph.batch is None
        """
        h = graph.x
        edge_attr = graph.edge_attr
        for layer in self.layers:
            h = layer(h, graph.edge_index, edge_attr=edge_attr)

        if graph.batch is None:
            z = h.mean(dim=0, keepdim=True)
        else:
            num_graphs = int(graph.batch.max().item()) + 1 if graph.batch.numel() > 0 else 0
            z = _segment_mean(h, graph.batch, num_segments=num_graphs)

        return h, z
