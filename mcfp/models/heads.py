from __future__ import annotations

import torch
from torch import nn


class ScalarHead(nn.Module):
    """Scalar output head."""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(int(in_dim), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map features to scalar output."""
        return self.linear(x).squeeze(-1)
