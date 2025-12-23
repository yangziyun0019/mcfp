# mcfp/models/stage1.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from mcfp.models.pose_encoder import PoseEncoder
from mcfp.models.backbone import TokenFusionBackbone
from mcfp.models.heads import MultiIndicatorHeads


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    """Small helper to read config from dict-like or OmegaConf-like objects."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


@dataclass
class Stage1Output:
    """Container for Stage-1 outputs."""
    preds: Dict[str, torch.Tensor]  # name -> [B]
    z: torch.Tensor                # [B, D]


def _infer_pose_key(batch: Dict[str, Any]) -> str:
    """Infer the pose feature key from a batch dict."""
    for k in ["pose_feats", "pose", "x", "query_feats", "query"]:
        if k in batch:
            return k
    raise KeyError("[MCFPStage1] Cannot find pose features in batch.")


def _infer_graph_key(batch: Dict[str, Any]) -> str:
    """Infer the morphology graph key from a batch dict."""
    for k in ["morph_graph", "graph", "morph"]:
        if k in batch:
            return k
    raise KeyError("[MCFPStage1] Cannot find morphology graph in batch.")


def _ensure_node_batch_for_grouped(
    h_links: torch.Tensor,
    pose_batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Expand link embeddings for grouped batches (one morph per batch).

    Given:
      - h_links: (N, D) link embeddings for a single morphology
      - pose_batch_size: B

    Return:
      - node_emb: (B*N, D) expanded embeddings
      - node_batch: (B*N,) each node assigned to sample index [0..B-1]

    This makes TokenFusionBackbone treat each sample as an independent "graph"
    with the same node token set (valid for grouped sampling).
    """
    if h_links.ndim != 2:
        raise ValueError("[MCFPStage1] h_links must be (N, D).")
    N, D = h_links.shape
    B = int(pose_batch_size)

    if B <= 0:
        raise ValueError("[MCFPStage1] pose_batch_size must be positive.")

    if B == 1:
        node_emb = h_links
        node_batch = torch.zeros((N,), dtype=torch.long, device=h_links.device)
        return node_emb, node_batch

    # Expand: (N,D) -> (B,N,D) -> (B*N,D)
    node_emb = h_links.unsqueeze(0).expand(B, N, D).reshape(B * N, D)

    # Build node_batch: [0..0, 1..1, ..., B-1..B-1], each repeated N times
    node_batch = torch.arange(B, device=h_links.device, dtype=torch.long).repeat_interleave(N)

    return node_emb, node_batch


class MCFPStage1(nn.Module):
    """Stage-1 MCFP model: (morphology graph, query pose) -> multi-indicator predictions.

    Assumptions for Stage-1 (grouped sampling):
      - A batch contains query points from the SAME morphology.
      - The morphology graph is provided once per batch.
      - Morph encoder returns (h_links, z_morph), where h_links is (N_links, D).
    """

    def __init__(
        self,
        morph_encoder: nn.Module,
        pose_encoder: PoseEncoder,
        backbone: TokenFusionBackbone,
        heads: MultiIndicatorHeads,
    ) -> None:
        super().__init__()
        self.morph_encoder = morph_encoder
        self.pose_encoder = pose_encoder
        self.backbone = backbone
        self.heads = heads

    @classmethod
    def from_cfg(cls, cfg: Any, morph_encoder: nn.Module) -> "MCFPStage1":
        """Build Stage-1 model from config and an already-constructed morph encoder."""
        pose_enc = PoseEncoder.from_cfg(_get(cfg, "pose_encoder", None))
        backbone = TokenFusionBackbone.from_cfg(_get(cfg, "backbone", None))

        heads_cfg = _get(cfg, "heads", None)
        if heads_cfg is None:
            raise ValueError("[MCFPStage1] Missing cfg.heads for MultiIndicatorHeads.")
        heads = MultiIndicatorHeads.from_cfg(in_dim=backbone.d_model, cfg=heads_cfg)

        return cls(
            morph_encoder=morph_encoder,
            pose_encoder=pose_enc,
            backbone=backbone,
            heads=heads,
        )

    def forward(self, batch: Dict[str, Any]) -> Stage1Output:
        """
        Args:
            batch: Dict containing at least:
              - pose_feats: [B, pose_dim]
              - morph_graph: graph object expected by morph_encoder

        Returns:
            Stage1Output with preds dict and fused latent z: [B, D].
        """
        pose_key = _infer_pose_key(batch)
        graph_key = _infer_graph_key(batch)

        pose_feats = batch[pose_key]
        graph = batch[graph_key]

        if not isinstance(pose_feats, torch.Tensor):
            raise TypeError("[MCFPStage1] pose_feats must be a torch.Tensor.")

        B = int(pose_feats.shape[0])

        pose_out = self.pose_encoder(pose_feats)

        # Morph encoder must return (h_links, z_morph) at minimum.
        morph_out = self.morph_encoder(graph)
        if not isinstance(morph_out, (tuple, list)) or len(morph_out) < 1:
            raise TypeError("[MCFPStage1] morph_encoder must return a tuple/list (h_links, z_morph).")

        h_links = morph_out[0]  # (N_links, D)
        if h_links.ndim != 2:
            raise ValueError("[MCFPStage1] morph_encoder first output must be (N_links, D).")

        z_morph = morph_out[1] if len(morph_out) > 1 else None
        if z_morph is not None:
            if z_morph.ndim != 2:
                raise ValueError("[MCFPStage1] z_morph must be (B, D) or (1, D).")
            if z_morph.shape[0] == 1 and B > 1:
                z_morph = z_morph.expand(B, -1)
            elif z_morph.shape[0] not in (1, B):
                raise ValueError("[MCFPStage1] z_morph batch dim must be 1 or B.")

        # Grouped sampling: expand link tokens for each sample in batch.
        node_emb, node_batch = _ensure_node_batch_for_grouped(h_links, pose_batch_size=B)

        fused = self.backbone(
            pose_token=pose_out.pose_emb,
            node_emb=node_emb,
            node_batch=node_batch,
            morph_token=z_morph,
        )

        head_out = self.heads(fused.z)
        return Stage1Output(preds=head_out.preds, z=fused.z)
