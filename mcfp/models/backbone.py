from __future__ import annotations

import math

import torch
from torch import nn


def _init_siren_linear(linear: nn.Linear, in_dim: int, w0: float, is_first: bool) -> None:
    """Initialize SIREN linear layer."""
    if is_first:
        bound = 1.0 / float(in_dim)
    else:
        bound = math.sqrt(6.0 / float(in_dim)) / float(w0)
    nn.init.uniform_(linear.weight, -bound, bound)
    nn.init.uniform_(linear.bias, -bound, bound)


class FiLMSirenBackbone(nn.Module):
    """FiLM-modulated SIREN backbone."""

    def __init__(
        self,
        in_dim: int,
        morph_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        out_dim: int = 128,
        w0_first: float = 30.0,
        w0: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.morph_dim = int(morph_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.out_dim = int(out_dim)
        self.w0_first = float(w0_first)
        self.w0 = float(w0)

        self.layers = nn.ModuleList()
        self.film = nn.ModuleList()

        prev = self.in_dim
        for i in range(self.num_layers):
            linear = nn.Linear(prev, self.hidden_dim)
            _init_siren_linear(linear, prev, self.w0_first if i == 0 else self.w0, is_first=(i == 0))
            self.layers.append(linear)
            self.film.append(nn.Linear(self.morph_dim, 2 * self.hidden_dim))
            prev = self.hidden_dim

        self.out = nn.Linear(self.hidden_dim, self.out_dim)
        _init_siren_linear(self.out, self.hidden_dim, self.w0, is_first=False)

    def _expand_morph(self, x: torch.Tensor, morph: torch.Tensor) -> torch.Tensor:
        """Broadcast morph embedding to match batch size."""
        if morph.dim() == 1:
            morph = morph.unsqueeze(0)
        if morph.shape[0] == 1 and x.shape[0] > 1:
            morph = morph.expand(x.shape[0], -1)
        if morph.shape[0] != x.shape[0]:
            raise ValueError("morph embedding batch size must match input.")
        return morph

    def forward(self, x: torch.Tensor, morph: torch.Tensor) -> torch.Tensor:
        """Forward pass with FiLM modulation."""
        morph = self._expand_morph(x, morph)
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            gamma_beta = self.film[i](morph)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            h = (1.0 + gamma) * h + beta
            w0 = self.w0_first if i == 0 else self.w0
            h = torch.sin(w0 * h)
        return self.out(h)
