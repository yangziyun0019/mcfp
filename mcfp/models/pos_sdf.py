"""Position-SDF decoder modules for workspace reachability prediction.

This module defines the FiLM-SIREN network used for position field learning.
"""

from __future__ import annotations

import math
from typing import List

import torch
from torch import nn


class FiLMSiren(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 5,
        out_dim: int = 1,
        w0_first: float = 30.0,
        w0: float = 1.0,
        cond_dim: int = 256,
    ) -> None:
        super().__init__()
        self.w0_first = w0_first
        self.w0 = w0

        self.layers = nn.ModuleList()
        self.film = nn.ModuleList()
        dims = [in_dim] + [hidden_dim] * num_layers
        for i in range(num_layers):
            self.layers.append(nn.Linear(dims[i], hidden_dim))
            self.film.append(nn.Linear(cond_dim, hidden_dim * 2))

        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # cond: (B,cond_dim) or (1,cond_dim)
        if cond.shape[0] == 1 and x.shape[0] > 1:
            cond = cond.expand(x.shape[0], -1)
        h = x
        for i, (layer, film) in enumerate(zip(self.layers, self.film)):
            gamma_beta = film(cond)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            h = layer(h)
            h = gamma * h + beta
            w0 = self.w0_first if i == 0 else self.w0
            h = torch.sin(w0 * h)
        return self.out(h)


class PositionSDFModel(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 5,
        w0_first: float = 30.0,
        w0: float = 1.0,
        cond_dim: int = 256,
    ) -> None:
        super().__init__()
        self.decoder = FiLMSiren(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            out_dim=1,
            w0_first=w0_first,
            w0=w0,
            cond_dim=cond_dim,
        )

    def forward(self, e_p: torch.Tensor, morph_emb: torch.Tensor) -> torch.Tensor:
        return self.decoder(e_p, morph_emb).squeeze(-1)
