# mcfp/train/metrics.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


def _as_float_tensor(x: Any, device: torch.device) -> torch.Tensor:
    """Convert to float tensor on device."""
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.float32)
    return torch.as_tensor(x, device=device, dtype=torch.float32)


@dataclass
class MetricsOutput:
    """Container for computed metrics."""
    scalars: Dict[str, float]


class Stage1Metrics:
    """Streaming metrics for Stage-1.

    Metrics:
      - For g_ws: accuracy, precision, recall, f1 (threshold 0.5).
      - For continuous heads: MAE, RMSE (masked by g_ws=1).

    Notes:
      - For g_ws, predictions can be logits or probabilities. If logits are provided,
        pass ws_is_logit=True to update().
    """

    def __init__(
        self,
        ws_name: str = "g_ws",
        ws_is_logit: bool = True,
        mask_by_ws: bool = True,
        eps: float = 1e-8,
    ) -> None:
        self.ws_name = str(ws_name)
        self.ws_is_logit = bool(ws_is_logit)
        self.mask_by_ws = bool(mask_by_ws)
        self.eps = float(eps)
        self.reset()

    def reset(self) -> None:
        """Reset all accumulators."""
        self._n = 0

        # ws classification accumulators
        self._ws_tp = 0.0
        self._ws_tn = 0.0
        self._ws_fp = 0.0
        self._ws_fn = 0.0

        # regression accumulators per head
        self._sum_abs: Dict[str, float] = {}
        self._sum_sq: Dict[str, float] = {}
        self._sum_w: Dict[str, float] = {}

    @torch.no_grad()
    def update(self, preds: Dict[str, torch.Tensor], labels: Dict[str, Any]) -> None:
        """Update accumulators from a batch."""
        if self.ws_name not in labels:
            raise KeyError(f"[metrics] labels must include '{self.ws_name}'.")

        # Infer device from any prediction
        any_pred = next(iter(preds.values()))
        device = any_pred.device

        ws_tgt = _as_float_tensor(labels[self.ws_name], device=device).view(-1)  # [B]
        ws_pred = preds[self.ws_name].view(-1)  # [B]

        if self.ws_is_logit:
            ws_prob = torch.sigmoid(ws_pred)
        else:
            ws_prob = ws_pred

        ws_hat = (ws_prob > 0.5).to(dtype=torch.float32)
        ws_true = (ws_tgt > 0.5).to(dtype=torch.float32)

        tp = float(((ws_hat == 1) & (ws_true == 1)).float().sum().item())
        tn = float(((ws_hat == 0) & (ws_true == 0)).float().sum().item())
        fp = float(((ws_hat == 1) & (ws_true == 0)).float().sum().item())
        fn = float(((ws_hat == 0) & (ws_true == 1)).float().sum().item())

        self._ws_tp += tp
        self._ws_tn += tn
        self._ws_fp += fp
        self._ws_fn += fn

        # Mask for regression heads
        if self.mask_by_ws:
            mask = (ws_true > 0.5).to(dtype=torch.float32)  # [B]
        else:
            mask = torch.ones_like(ws_true, dtype=torch.float32)

        # Regression heads
        for name, y_pred in preds.items():
            if name == self.ws_name:
                continue
            if name not in labels:
                continue

            y_true = _as_float_tensor(labels[name], device=device).view(-1)
            y_pred = y_pred.view(-1)

            err = y_pred - y_true
            abs_err = torch.abs(err) * mask
            sq_err = (err * err) * mask
            w = mask.sum().item()

            self._sum_abs[name] = self._sum_abs.get(name, 0.0) + float(abs_err.sum().item())
            self._sum_sq[name] = self._sum_sq.get(name, 0.0) + float(sq_err.sum().item())
            self._sum_w[name] = self._sum_w.get(name, 0.0) + float(w)

        self._n += int(ws_true.numel())

    def compute(self) -> MetricsOutput:
        """Compute scalar metrics."""
        eps = self.eps
        tp, tn, fp, fn = self._ws_tp, self._ws_tn, self._ws_fp, self._ws_fn
        total = tp + tn + fp + fn

        acc = (tp + tn) / max(total, eps)
        prec = tp / max(tp + fp, eps)
        rec = tp / max(tp + fn, eps)
        f1 = 2.0 * prec * rec / max(prec + rec, eps)

        scalars: Dict[str, float] = {
            "ws_acc": float(acc),
            "ws_precision": float(prec),
            "ws_recall": float(rec),
            "ws_f1": float(f1),
        }

        # Regression metrics
        for name in sorted(self._sum_w.keys()):
            w = self._sum_w[name]
            mae = self._sum_abs[name] / max(w, eps)
            rmse = (self._sum_sq[name] / max(w, eps)) ** 0.5
            scalars[f"{name}_mae"] = float(mae)
            scalars[f"{name}_rmse"] = float(rmse)

        return MetricsOutput(scalars=scalars)
