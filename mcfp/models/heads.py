# mcfp/models/heads.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    """Small helper to read config from dict-like or OmegaConf-like objects."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class HeadMLP(nn.Module):
    """A small MLP head for scalar prediction."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.0,
        out_activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("[HeadMLP] num_layers must be >= 1")

        layers: List[nn.Module] = []
        d = int(in_dim)
        h = int(hidden_dim)

        for i in range(num_layers - 1):
            layers.append(nn.Linear(d, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.GELU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            d = h

        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

        act = str(out_activation).lower()
        if act == "sigmoid":
            self.act = nn.Sigmoid()
        elif act == "identity" or act == "none":
            self.act = nn.Identity()
        else:
            raise ValueError(f"[HeadMLP] Unsupported out_activation: {out_activation}")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return a flat scalar tensor [B]."""
        y = self.net(z)           # [B,1]
        y = self.act(y)           # [B,1]
        return y.squeeze(-1)      # [B]


@dataclass
class HeadsOutput:
    """Container for heads outputs."""
    preds: Dict[str, torch.Tensor]  # each [B]


class MultiIndicatorHeads(nn.Module):
    """Multi-head predictor for per-indicator scalar outputs.

    The heads are created from a config list:
        heads:
          - name: g_self
            hidden_dim: 256
            num_layers: 2
            dropout: 0.0
            out_activation: sigmoid
          - name: g_man
            ...

    Args:
        in_dim: Input latent dimension.
        head_cfgs: List of per-head configs.
    """

    def __init__(self, in_dim: int, head_cfgs: Sequence[Any]) -> None:
        super().__init__()
        self.in_dim = int(in_dim)

        heads = nn.ModuleDict()
        names: List[str] = []
        for hc in head_cfgs:
            name = _get(hc, "name", None)
            if not name:
                raise ValueError("[MultiIndicatorHeads] Each head must have a 'name'.")
            names.append(str(name))

            hidden_dim = int(_get(hc, "hidden_dim", 256))
            num_layers = int(_get(hc, "num_layers", 2))
            dropout = float(_get(hc, "dropout", 0.0))
            out_activation = str(_get(hc, "out_activation", "sigmoid"))

            heads[str(name)] = HeadMLP(
                in_dim=self.in_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                out_activation=out_activation,
            )

        self.heads = heads
        self.names = names

    @classmethod
    def from_cfg(cls, in_dim: int, cfg: Any) -> "MultiIndicatorHeads":
        """Build heads from cfg.heads (list)."""
        head_cfgs = _get(cfg, "heads", None)
        if head_cfgs is None:
            raise ValueError("[MultiIndicatorHeads] Missing cfg.heads list.")
        return cls(in_dim=in_dim, head_cfgs=head_cfgs)

    def forward(self, z: torch.Tensor) -> HeadsOutput:
        """
        Args:
            z: [B, D].

        Returns:
            HeadsOutput with preds dict: name -> [B].
        """
        preds: Dict[str, torch.Tensor] = {}
        for name, head in self.heads.items():
            preds[name] = head(z)
        return HeadsOutput(preds=preds)
