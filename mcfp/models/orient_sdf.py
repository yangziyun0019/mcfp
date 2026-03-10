"""Orientation-SDF decoder modules for anchor-conditioned attitude prediction.

This module defines the mixture-of-experts decoder used for orientation reachability learning.
"""

from __future__ import annotations

import math
from typing import List

import torch
from torch import nn

from mcfp.models.pos_sdf import FiLMSiren


class OrientationSDFModel(nn.Module):
    def __init__(
        self,
        orient_in_dim: int,
        context_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 4,
        experts: int = 4,
        cond_in_dim: int = 256 + 63 + 1,
    ) -> None:
        super().__init__()
        self.experts = experts
        self.context = nn.Sequential(
            nn.Linear(cond_in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, context_dim),
            nn.ReLU(inplace=True),
        )
        self.decoders = nn.ModuleList([
            FiLMSiren(
                in_dim=orient_in_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                out_dim=1,
                w0_first=30.0,
                w0=1.0,
                cond_dim=context_dim,
            )
            for _ in range(experts)
        ])

    def forward(self, e_r: torch.Tensor, cond: torch.Tensor, tau: float = 0.05) -> torch.Tensor:
        """Return raw u_k and aggregated u.

        Args:
            e_r: (B, orient_in_dim)
            cond: (B, cond_in_dim)
        Returns:
            u_raw: (B, K)
            u: (B,)
        """
        c = self.context(cond)
        u_list = [decoder(e_r, c).squeeze(-1) for decoder in self.decoders]
        u_raw = torch.stack(u_list, dim=1)
        u = tau * torch.logsumexp(u_raw / tau, dim=1)
        return u_raw, u
