# mcfp/models/backbone.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    """Small helper to read config from dict-like or OmegaConf-like objects."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _pack_padded_nodes(
    node_emb: torch.Tensor,
    node_batch: torch.Tensor,
    max_nodes: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pack variable-length node embeddings into a padded tensor.

    Args:
        node_emb: [N, D] node embeddings.
        node_batch: [N] graph index for each node, in [0..B-1].
        max_nodes: Optional cap for max nodes per graph (truncate if exceeded).

    Returns:
        padded: [B, M, D]
        key_padding_mask: [B, M] bool mask, True for PAD positions.
    """
    if node_emb.ndim != 2:
        raise ValueError(f"[backbone] node_emb must be [N,D], got {tuple(node_emb.shape)}")
    if node_batch.ndim != 1 or node_batch.shape[0] != node_emb.shape[0]:
        raise ValueError(
            f"[backbone] node_batch must be [N], got {tuple(node_batch.shape)}"
        )

    device = node_emb.device
    B = int(node_batch.max().item()) + 1 if node_batch.numel() > 0 else 0
    D = node_emb.shape[1]

    if B == 0:
        raise ValueError("[backbone] Empty batch is not supported.")

    counts = torch.bincount(node_batch, minlength=B)
    M = int(counts.max().item())
    if max_nodes is not None:
        M = min(M, int(max_nodes))

    padded = torch.zeros((B, M, D), dtype=node_emb.dtype, device=device)
    mask = torch.ones((B, M), dtype=torch.bool, device=device)

    # Stable per-graph fill
    for b in range(B):
        idx = (node_batch == b).nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            continue
        if idx.numel() > M:
            idx = idx[:M]
        nb = idx.numel()
        padded[b, :nb, :] = node_emb.index_select(0, idx)
        mask[b, :nb] = False  # not padded
    return padded, mask


@dataclass
class BackboneOutput:
    """Container for fusion backbone outputs."""
    z: torch.Tensor                 # [B, D]
    fused_pose_token: torch.Tensor  # [B, D]
    fused_node_tokens: torch.Tensor # [B, M, D]
    node_key_padding_mask: torch.Tensor  # [B, M]
    fused_morph_token: Optional[torch.Tensor] = None  # [B, D] if provided


class TokenFusionBackbone(nn.Module):
    """Fuse morphology node tokens and a pose token using a Transformer encoder.

    Sequence layout per graph:
        [POSE] + [NODE_1 ... NODE_M]

    The output z is taken as the transformed [POSE] token.

    Args:
        d_model: Token dimension.
        nhead: Number of attention heads.
        num_layers: Transformer encoder layers.
        dim_feedforward: FFN dimension.
        dropout: Dropout rate.
        max_nodes: Optional truncation for nodes per graph.
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_nodes: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.max_nodes = max_nodes

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(nhead),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))
        self.pose_ln = nn.LayerNorm(self.d_model)

    @classmethod
    def from_cfg(cls, cfg: Any) -> "TokenFusionBackbone":
        """Build TokenFusionBackbone from config."""
        return cls(
            d_model=int(_get(cfg, "d_model", 256)),
            nhead=int(_get(cfg, "nhead", 8)),
            num_layers=int(_get(cfg, "num_layers", 4)),
            dim_feedforward=int(_get(cfg, "dim_feedforward", 1024)),
            dropout=float(_get(cfg, "dropout", 0.1)),
            max_nodes=_get(cfg, "max_nodes", None),
        )

    def forward(
        self,
        pose_token: torch.Tensor,
        node_emb: torch.Tensor,
        node_batch: torch.Tensor,
        morph_token: Optional[torch.Tensor] = None,
    ) -> BackboneOutput:
        """
        Args:
            pose_token: [B, D] pose embedding.
            node_emb: [N, D] node embeddings.
            node_batch: [N] node -> graph index.

        Returns:
            BackboneOutput with z: [B, D].
        """
        if pose_token.ndim != 2 or pose_token.shape[-1] != self.d_model:
            raise ValueError(f"[backbone] pose_token must be [B,{self.d_model}]")
        if node_emb.ndim != 2 or node_emb.shape[-1] != self.d_model:
            raise ValueError(f"[backbone] node_emb must be [N,{self.d_model}]")

        nodes_padded, node_pad_mask = _pack_padded_nodes(
            node_emb=node_emb,
            node_batch=node_batch,
            max_nodes=self.max_nodes,
        )  # [B,M,D], [B,M]

        B = pose_token.shape[0]
        if nodes_padded.shape[0] != B:
            raise ValueError(
                f"[backbone] Batch mismatch: pose B={B}, nodes B={nodes_padded.shape[0]}"
            )

        pose_token = self.pose_ln(pose_token)
        pose_token = pose_token.unsqueeze(1)  # [B,1,D]

        tokens = [pose_token]
        if morph_token is not None:
            if morph_token.ndim != 2 or morph_token.shape[-1] != self.d_model:
                raise ValueError(f"[backbone] morph_token must be [B,{self.d_model}]")
            morph_token = morph_token.unsqueeze(1)  # [B,1,D]
            tokens.append(morph_token)
        tokens.append(nodes_padded)
        tokens = torch.cat(tokens, dim=1)  # [B,1(+1)+M,D]

        # key_padding_mask: True means "ignore"
        pose_mask = torch.zeros((B, 1), dtype=torch.bool, device=tokens.device)
        if morph_token is not None:
            morph_mask = torch.zeros((B, 1), dtype=torch.bool, device=tokens.device)
            key_padding_mask = torch.cat([pose_mask, morph_mask, node_pad_mask], dim=1)
        else:
            key_padding_mask = torch.cat([pose_mask, node_pad_mask], dim=1)

        fused = self.encoder(tokens, src_key_padding_mask=key_padding_mask)  # [B,1+M,D]

        fused_pose = fused[:, 0, :]
        if morph_token is not None:
            fused_morph = fused[:, 1, :]
            fused_nodes = fused[:, 2:, :]
        else:
            fused_morph = None
            fused_nodes = fused[:, 1:, :]

        return BackboneOutput(
            z=fused_pose,
            fused_pose_token=fused_pose,
            fused_node_tokens=fused_nodes,
            node_key_padding_mask=node_pad_mask,
            fused_morph_token=fused_morph,
        )
