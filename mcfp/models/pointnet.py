"""PointNet encoder modules for per-link collision point clouds.

These layers turn sampled link geometry into feature vectors for morphology encoding.
"""

from __future__ import annotations

import torch
from torch import nn


class PointNetEncoder(nn.Module):
    """PointNet encoder for per-link collision point clouds."""

    def __init__(self, in_dim: int = 6, out_dim: int = 128) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(256, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Encode points.

        Args:
            points: (N_links, N_pts, 6)
        Returns:
            (N_links, out_dim)
        """
        if points.ndim != 3 or points.shape[-1] != 6:
            raise ValueError(f"points must be (N_links,N_pts,6). Got {points.shape}.")
        n_links = points.shape[0]
        x = points.reshape(-1, points.shape[-1])
        x = self.mlp(x)
        x = x.reshape(n_links, -1, x.shape[-1])
        x = torch.max(x, dim=1).values
        return self.fc(x)
