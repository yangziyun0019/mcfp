from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from mcfp.models.backbone import FiLMSirenBackbone
from mcfp.models.heads import ScalarHead
from mcfp.models.morph_encoder import MorphologyEncoder, build_morph_graph_from_json
from mcfp.models.morph_graph import GraphData
from mcfp.models.pose_encoder import PoseEncoder


class MCFPStage1(nn.Module):
    """Stage-1 implicit field model."""

    def __init__(
        self,
        node_feat_dim: int,
        morph_dim: int = 128,
        pose_dim: int = 128,
        pose_hidden_dims: Optional[list[int]] = None,
        pose_fourier_dim: int = 0,
        pose_fourier_scale: float = 10.0,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 8,
        d_ff: int = 512,
        dropout: float = 0.1,
        backbone_hidden_dim: int = 128,
        backbone_num_layers: int = 4,
        backbone_out_dim: int = 128,
        w0_first: float = 30.0,
        w0: float = 1.0,
    ) -> None:
        super().__init__()
        self.morph_encoder = MorphologyEncoder(
            node_feat_dim=node_feat_dim,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            out_dim=morph_dim,
        )
        self.pose_encoder = PoseEncoder(
            in_dim=6,
            hidden_dims=pose_hidden_dims,
            out_dim=pose_dim,
            fourier_dim=pose_fourier_dim,
            fourier_scale=pose_fourier_scale,
        )
        self.backbone = FiLMSirenBackbone(
            in_dim=pose_dim,
            morph_dim=morph_dim,
            hidden_dim=backbone_hidden_dim,
            num_layers=backbone_num_layers,
            out_dim=backbone_out_dim,
            w0_first=w0_first,
            w0=w0,
        )
        self.head = ScalarHead(backbone_out_dim)

    def encode_morph(self, spec_path: str, l_ref: float, device: Optional[torch.device] = None) -> torch.Tensor:
        """Encode morphology from spec JSON."""
        graph = build_morph_graph_from_json(spec_path, l_ref=l_ref, device=device)
        return self.morph_encoder(graph)

    def forward(
        self,
        w: torch.Tensor,
        *,
        morph_graph: Optional[GraphData] = None,
        morph_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for s(x)."""
        if morph_emb is None:
            if morph_graph is None:
                raise ValueError("morph_graph or morph_emb must be provided.")
            morph_emb = self.morph_encoder(morph_graph)
        h = self.pose_encoder(w)
        feat = self.backbone(h, morph_emb)
        return self.head(feat)
