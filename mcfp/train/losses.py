# mcfp/train/losses.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    """Read config from dict-like or OmegaConf-like objects."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _as_float_tensor(x: Any, device: torch.device) -> torch.Tensor:
    """Convert input to float tensor on device."""
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.float32)
    return torch.as_tensor(x, device=device, dtype=torch.float32)


def _labels_to_dict(
    labels: Union[Dict[str, torch.Tensor], torch.Tensor],
    label_keys: Optional[List[str]],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Normalize labels into a dict[name -> Tensor[B]] on device.

    Supported formats
    -----------------
    1) dict: {name: Tensor[B] or Tensor[B,1]}
    2) tensor: Tensor[B,K] with an accompanying label_keys list of length K
    """
    if isinstance(labels, dict):
        out: Dict[str, torch.Tensor] = {}
        for k, v in labels.items():
            t = _as_float_tensor(v, device)
            if t.ndim == 2 and t.shape[1] == 1:
                t = t[:, 0]
            if t.ndim != 1:
                raise ValueError(f"[losses] labels[{k}] must be 1D (got shape={tuple(t.shape)}).")
            out[str(k)] = t
        return out

    if not isinstance(labels, torch.Tensor):
        raise TypeError(f"[losses] labels must be dict or torch.Tensor, got {type(labels)}.")

    if label_keys is None:
        raise ValueError("[losses] labels is a Tensor but batch['label_keys'] is missing.")

    y = _as_float_tensor(labels, device)
    if y.ndim != 2:
        raise ValueError(f"[losses] tensor labels must be 2D [B,K], got shape={tuple(y.shape)}.")

    if y.shape[1] != len(label_keys):
        raise ValueError(
            f"[losses] labels second dim K={y.shape[1]} != len(label_keys)={len(label_keys)}."
        )

    out = {}
    for j, name in enumerate(label_keys):
        out[str(name)] = y[:, j]
    return out


@dataclass
class LossOutput:
    """Container for loss outputs."""
    total: torch.Tensor
    per_head: Dict[str, torch.Tensor]
    stats: Dict[str, float]


class MultiTaskLoss(nn.Module):
    """Stage-1 multi-task loss for MCFP.

    Conventions
    -----------
    - g_ws is a binary reachability target in {0,1}.
    - Other heads are continuous in [0,1] and are meaningful only when ws_mask == 1.

    Expected batch keys
    -------------------
    - batch["labels"]: Dict[str, Tensor[B]] OR Tensor[B,K]
    - batch["label_keys"]: List[str] if labels is Tensor
    - batch["ws_mask"]: Tensor[B] (preferred for masking regression)
    - batch["sample_weight"]: Tensor[B] (optional)

    Args:
        head_weights: Dict[str, float] weighting each head loss.
        ws_name: Name of reachability head (default "g_ws").
        ws_with_logits: If True, treat prediction for g_ws as logits and use BCEWithLogits.
        mask_by_ws: If True, only supervise non-ws heads where ws_mask==1.
        huber_delta: Delta for SmoothL1/Huber loss.
        ws_pos_weight: Optional positive-class weight for g_ws (for BCEWithLogits).
        eps: Numerical epsilon.
    """

    def __init__(
        self,
        head_weights: Dict[str, float],
        ws_name: str = "g_ws",
        ws_with_logits: bool = True,
        mask_by_ws: bool = True,
        huber_delta: float = 0.05,
        ws_pos_weight: Optional[float] = None,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.head_weights = {str(k): float(v) for k, v in head_weights.items()}
        self.ws_name = str(ws_name)
        self.ws_with_logits = bool(ws_with_logits)
        self.mask_by_ws = bool(mask_by_ws)
        self.huber_delta = float(huber_delta)
        self.ws_pos_weight = None if ws_pos_weight is None else float(ws_pos_weight)
        self.eps = float(eps)

        # Uncertainty weighting: learnable log-sigma per head.
        log_sigma = nn.ParameterDict()
        for name, w in self.head_weights.items():
            if w is None or w <= 0.0:
                init = 0.0
            else:
                init = -0.5 * math.log(2.0 * float(w))
            log_sigma[str(name)] = nn.Parameter(torch.tensor(init, dtype=torch.float32))
        self.log_sigma = log_sigma

    @classmethod
    def from_cfg(cls, cfg: Any) -> "MultiTaskLoss":
        """Build MultiTaskLoss from config."""
        loss_cfg = _get(cfg, "loss", cfg)
        head_weights = _get(loss_cfg, "head_weights", None)
        if head_weights is None:
            raise ValueError("[losses] cfg.loss.head_weights is required.")
        return cls(
            head_weights=dict(head_weights),
            ws_name=str(_get(loss_cfg, "ws_name", "g_ws")),
            ws_with_logits=bool(_get(loss_cfg, "ws_with_logits", True)),
            mask_by_ws=bool(_get(loss_cfg, "mask_by_ws", True)),
            huber_delta=float(_get(loss_cfg, "huber_delta", 0.05)),
            ws_pos_weight=_get(loss_cfg, "ws_pos_weight", None),
            eps=float(_get(loss_cfg, "eps", 1e-8)),
        )

    def forward(self, preds: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> LossOutput:
        """Compute total and per-head losses."""
        if len(preds) == 0:
            raise ValueError("[losses] preds is empty.")

        device = next(iter(preds.values())).device

        labels_raw = batch.get("labels", None)
        if labels_raw is None:
            raise ValueError("[losses] batch['labels'] is required.")

        label_keys = batch.get("label_keys", None)
        if label_keys is not None and not isinstance(label_keys, list):
            label_keys = list(label_keys)

        labels = _labels_to_dict(labels_raw, label_keys, device)

        sample_weight = batch.get("sample_weight", None)
        if sample_weight is None:
            w = torch.ones((next(iter(labels.values())).shape[0],), device=device, dtype=torch.float32)
        else:
            w = _as_float_tensor(sample_weight, device).reshape(-1)

        # Prefer ws_mask for regression masking (contract-level mask).
        ws_mask = batch.get("ws_mask", None)
        if ws_mask is not None:
            ws_mask_t = _as_float_tensor(ws_mask, device).reshape(-1)
            ws_valid = (ws_mask_t > 0.5).float()
        else:
            if self.ws_name not in labels:
                raise ValueError("[losses] ws_mask missing and ws label not found; cannot build mask.")
            ws_valid = (labels[self.ws_name] > 0.5).float()

        per_head: Dict[str, torch.Tensor] = {}
        stats: Dict[str, float] = {}
        total = torch.zeros((), device=device, dtype=torch.float32)

        # Compute g_ws loss (classification).
        if self.ws_name in preds:
            if self.ws_name not in labels:
                raise ValueError(f"[losses] ws head '{self.ws_name}' in preds but missing in labels.")
            y_ws = labels[self.ws_name].clamp(0.0, 1.0)
            p_ws = preds[self.ws_name].reshape(-1)

            if self.ws_with_logits:
                pos_weight = None
                if self.ws_pos_weight is not None:
                    pos_weight = torch.tensor([self.ws_pos_weight], device=device, dtype=torch.float32)
                bce = F.binary_cross_entropy_with_logits(p_ws, y_ws, reduction="none", pos_weight=pos_weight)
            else:
                p_prob = p_ws.clamp(self.eps, 1.0 - self.eps)
                bce = F.binary_cross_entropy(p_prob, y_ws, reduction="none")

            loss_ws = (bce * w).mean()
            per_head[self.ws_name] = loss_ws
            stats[f"loss/{self.ws_name}"] = float(loss_ws.detach().cpu())

        # Regression heads: masked by ws_valid if enabled.
        for name, p in preds.items():
            if name == self.ws_name:
                continue
            if name not in labels:
                # Allow model to output extra heads without supervision in stage1.
                continue

            y = labels[name].clamp(0.0, 1.0)
            p = p.reshape(-1)

            if self.mask_by_ws:
                mask = ws_valid
            else:
                mask = torch.ones_like(ws_valid)

            # SmoothL1 == Huber variant.
            reg = F.smooth_l1_loss(p, y, reduction="none", beta=self.huber_delta)

            num = (reg * w * mask).sum()
            den = (w * mask).sum().clamp_min(self.eps)
            loss_reg = num / den

            per_head[name] = loss_reg
            stats[f"loss/{name}"] = float(loss_reg.detach().cpu())
            stats[f"mask/{name}_den"] = float(den.detach().cpu())

        # Uncertainty weighting aggregation
        raw_sum = torch.zeros((), device=device, dtype=torch.float32)
        weighted_no_reg = torch.zeros((), device=device, dtype=torch.float32)
        for name, loss_val in per_head.items():
            raw_sum = raw_sum + loss_val
            if name not in self.log_sigma:
                # Fallback to unweighted if no parameter exists.
                total = total + loss_val
                weighted_no_reg = weighted_no_reg + loss_val
                continue
            ls = self.log_sigma[name]
            precision = torch.exp(-2.0 * ls)
            weighted_no_reg = weighted_no_reg + 0.5 * precision * loss_val
            total = total + 0.5 * precision * loss_val + ls

        stats["loss/raw_sum"] = float(raw_sum.detach().cpu())
        stats["loss/weighted_no_reg"] = float(weighted_no_reg.detach().cpu())
        stats["loss/total"] = float(total.detach().cpu())
        return LossOutput(total=total, per_head=per_head, stats=stats)
