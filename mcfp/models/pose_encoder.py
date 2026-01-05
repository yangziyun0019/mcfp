from __future__ import annotations

from typing import Optional, Sequence

import math

import torch
from torch import nn


class PoseEncoder(nn.Module):
    """Pose encoder for w=[p/L_ref, xi/lambda]."""

    def __init__(
        self,
        in_dim: int = 6,
        hidden_dims: Optional[Sequence[int]] = None,
        out_dim: int = 128,
        fourier_dim: int = 0,
        fourier_scale: float = 10.0,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.fourier_dim = int(fourier_dim)

        if hidden_dims is None:
            hidden_dims = [128, 128]

        if self.fourier_dim > 0:
            b = torch.randn(self.in_dim, self.fourier_dim) * float(fourier_scale)
            self.register_buffer("fourier_b", b)
            feat_dim = self.in_dim + 2 * self.fourier_dim
        else:
            self.fourier_b = None
            feat_dim = self.in_dim

        layers = []
        prev = feat_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, int(h)))
            layers.append(nn.SiLU())
            prev = int(h)
        layers.append(nn.Linear(prev, self.out_dim))
        self.mlp = nn.Sequential(*layers)

    def _apply_fourier(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Fourier features if enabled."""
        if self.fourier_dim <= 0 or self.fourier_b is None:
            return x
        proj = x @ self.fourier_b
        proj = 2.0 * math.pi * proj
        return torch.cat([x, torch.sin(proj), torch.cos(proj)], dim=-1)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """Encode pose in w coordinates."""
        x = self._apply_fourier(w)
        return self.mlp(x)
