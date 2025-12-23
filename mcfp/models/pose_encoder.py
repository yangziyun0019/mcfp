# mcfp/models/pose_encoder.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import math
import torch
import torch.nn as nn


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    """Small helper to read config from dict-like or OmegaConf-like objects."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class FourierFeatures(nn.Module):
    """Fourier feature mapping for continuous coordinates.

    This module maps x in R^D to [sin(2*pi*B*x), cos(2*pi*B*x)].
    Frequencies are geometric by default: 2^k for k=0..(K-1).

    Args:
        in_dim: Input dimension (e.g., 3 for xyz).
        num_bands: Number of frequency bands K.
        include_input: Whether to concatenate raw input x.
        base: Frequency base for geometric progression.
        scale: Multiplicative scale applied before sin/cos (commonly 1.0).
    """

    def __init__(
        self,
        in_dim: int,
        num_bands: int,
        include_input: bool = True,
        base: float = 2.0,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.num_bands = int(num_bands)
        self.include_input = bool(include_input)
        self.base = float(base)
        self.scale = float(scale)

        # Precompute frequency bands as a buffer to keep it on the right device.
        freqs = torch.tensor([self.base**k for k in range(self.num_bands)], dtype=torch.float32)
        self.register_buffer("freqs", freqs, persistent=False)

    @property
    def out_dim(self) -> int:
        d = 2 * self.in_dim * self.num_bands
        if self.include_input:
            d += self.in_dim
        return d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, in_dim].

        Returns:
            Tensor of shape [B, out_dim].
        """
        if x.ndim != 2 or x.shape[-1] != self.in_dim:
            raise ValueError(f"[FourierFeatures] Expected [B,{self.in_dim}], got {tuple(x.shape)}")

        # [B, in_dim, 1] * [K] -> [B, in_dim, K]
        xb = (x * self.scale).unsqueeze(-1) * self.freqs.view(1, 1, -1)
        xb = 2.0 * math.pi * xb

        sin = torch.sin(xb)
        cos = torch.cos(xb)

        # Flatten bands: [B, in_dim*K]
        sin = sin.reshape(x.shape[0], -1)
        cos = cos.reshape(x.shape[0], -1)

        out = torch.cat([sin, cos], dim=-1)
        if self.include_input:
            out = torch.cat([x, out], dim=-1)
        return out


class MLP(nn.Module):
    """Standard MLP with LayerNorm and GELU."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("[MLP] num_layers must be >= 1")

        layers = []
        d = in_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class PoseEncoderOutput:
    """Container for pose encoding outputs."""
    pose_emb: torch.Tensor  # [B, D]


class PoseEncoder(nn.Module):
    """Encode a task-space query point feature vector into a latent embedding.

    Expected input convention (Stage-1 default):
      - pose_feats: [B, 9]
        first 3 dims: xyz normalized to [0,1] (or [-1,1], both workable)
        remaining dims: auxiliary scalars (e.g., radial ratio, anisotropic scale ratios)

    The encoder uses Fourier features on xyz and concatenates auxiliary scalars.

    Args:
        pose_dim: Input feature dimension (default 9).
        emb_dim: Output embedding dimension.
        num_bands: Fourier bands for xyz.
        mlp_hidden: MLP hidden size.
        mlp_layers: MLP depth.
        dropout: Dropout in MLP.
        include_xyz_raw: Whether to include raw xyz in FourierFeatures.
    """

    def __init__(
        self,
        pose_dim: int = 9,
        emb_dim: int = 256,
        num_bands: int = 10,
        mlp_hidden: int = 256,
        mlp_layers: int = 3,
        dropout: float = 0.0,
        include_xyz_raw: bool = True,
    ) -> None:
        super().__init__()
        self.pose_dim = int(pose_dim)
        self.emb_dim = int(emb_dim)

        self.ff = FourierFeatures(
            in_dim=3,
            num_bands=int(num_bands),
            include_input=bool(include_xyz_raw),
            base=2.0,
            scale=1.0,
        )

        aux_dim = max(0, self.pose_dim - 3)
        self.aux_dim = aux_dim

        in_dim = self.ff.out_dim + aux_dim
        self.mlp = MLP(
            in_dim=in_dim,
            hidden_dim=int(mlp_hidden),
            out_dim=self.emb_dim,
            num_layers=int(mlp_layers),
            dropout=float(dropout),
        )

    @classmethod
    def from_cfg(cls, cfg: Any) -> "PoseEncoder":
        """Build PoseEncoder from config."""
        return cls(
            pose_dim=int(_get(cfg, "pose_dim", 9)),
            emb_dim=int(_get(cfg, "emb_dim", 256)),
            num_bands=int(_get(cfg, "num_bands", 10)),
            mlp_hidden=int(_get(cfg, "mlp_hidden", 256)),
            mlp_layers=int(_get(cfg, "mlp_layers", 3)),
            dropout=float(_get(cfg, "dropout", 0.0)),
            include_xyz_raw=bool(_get(cfg, "include_xyz_raw", True)),
        )

    def forward(self, pose_feats: torch.Tensor) -> PoseEncoderOutput:
        """
        Args:
            pose_feats: [B, pose_dim].

        Returns:
            PoseEncoderOutput with pose_emb: [B, emb_dim].
        """
        if pose_feats.ndim != 2 or pose_feats.shape[-1] != self.pose_dim:
            raise ValueError(
                f"[PoseEncoder] Expected [B,{self.pose_dim}], got {tuple(pose_feats.shape)}"
            )

        xyz = pose_feats[:, :3]
        aux = pose_feats[:, 3:] if self.aux_dim > 0 else None

        xyz_ff = self.ff(xyz)
        if aux is None:
            x = xyz_ff
        else:
            x = torch.cat([xyz_ff, aux], dim=-1)

        emb = self.mlp(x)
        return PoseEncoderOutput(pose_emb=emb)
