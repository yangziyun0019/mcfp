# mcfp/train/runner.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from torch.cuda.amp import GradScaler, autocast

from mcfp.train.losses import MultiTaskLoss
from mcfp.train.metrics import Stage1Metrics


@dataclass
class TrainStepOutput:
    """Container for a single train step output."""
    loss: float
    loss_stats: Dict[str, float]


@dataclass
class EvalOutput:
    """Container for evaluation output."""
    loss: float
    loss_stats: Dict[str, float]
    metrics: Dict[str, float]


def _move_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Recursively move tensors in a batch dict to device."""
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = _move_to_device(v, device)
        else:
            out[k] = v
    return out


class Stage1Runner:
    """Training/evaluation runner for MCFP Stage-1.

    This runner is intentionally model-agnostic regarding graph batching.
    It assumes `batch` is already collated into the expected model input format.

    Key requirements:
      - model(batch) returns an object with `.preds` dict: name -> [B]
      - batch["labels"] is a dict: name -> [B]
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: MultiTaskLoss,
        device: torch.device,
        logger: Any,
        scheduler: Optional[Any] = None,
        use_amp: bool = True,
        grad_clip_norm: Optional[float] = 1.0,
        grad_accum_steps: int = 1,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.logger = logger
        self.scheduler = scheduler
        self.use_amp = bool(use_amp and torch.cuda.is_available())
        self.grad_clip_norm = grad_clip_norm
        self.grad_accum_steps = int(grad_accum_steps)

        self.scaler = GradScaler(enabled=self.use_amp)

    def train_one_epoch(
        self,
        loader: Any,
        epoch: int,
        log_interval: int = 50,
        max_steps: Optional[int] = None,
    ) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        # Streaming meters
        last_stats: Dict[str, float] = {}
        step = 0

        it = iter(loader)
        while True:
            if max_steps is not None and step >= int(max_steps):
                break
            try:
                batch = next(it)
            except StopIteration:
                break

            batch = _move_to_device(batch, self.device)

            with autocast(enabled=self.use_amp):
                out = self.model(batch)
                loss_out = self.loss_fn(out.preds, batch)
                loss = loss_out.total / float(self.grad_accum_steps)

            self.scaler.scale(loss).backward()

            if (step + 1) % self.grad_accum_steps == 0:
                if self.grad_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.grad_clip_norm))

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                if self.scheduler is not None:
                    self.scheduler.step()

            last_stats = dict(loss_out.stats)

            if log_interval > 0 and (step % log_interval == 0):
                self.logger.info(
                    f"[train][epoch={epoch}][step={step}] total_loss={last_stats.get('total_loss', -1.0):.6f}"
                )

            step += 1

        # Return last seen loss stats (trainer will log more in eval)
        return last_stats

    @torch.no_grad()
    def evaluate(
        self,
        loader: Any,
        epoch: int,
        ws_name: str = "g_ws",
        ws_is_logit: bool = True,
        mask_by_ws: bool = True,
        max_steps: Optional[int] = None,
    ) -> EvalOutput:
        """Evaluate model on a loader."""
        self.model.eval()

        # Accumulate loss
        total_loss = 0.0
        n_batches = 0
        agg_stats: Dict[str, float] = {}

        metrics = Stage1Metrics(
            ws_name=ws_name,
            ws_is_logit=ws_is_logit,
            mask_by_ws=mask_by_ws,
        )

        it = iter(loader)
        step = 0
        while True:
            if max_steps is not None and step >= int(max_steps):
                break
            try:
                batch = next(it)
            except StopIteration:
                break

            batch = _move_to_device(batch, self.device)

            out = self.model(batch)
            loss_out = self.loss_fn(out.preds, batch)

            total_loss += float(loss_out.total.item())
            n_batches += 1

            # Update metrics
            labels = batch["labels"]
            metrics.update(out.preds, labels)

            # Aggregate stats by averaging over batches
            for k, v in loss_out.stats.items():
                agg_stats[k] = agg_stats.get(k, 0.0) + float(v)

            step += 1

        if n_batches == 0:
            raise RuntimeError("[runner] Empty evaluation loader.")

        for k in list(agg_stats.keys()):
            agg_stats[k] /= float(n_batches)

        m = metrics.compute().scalars
        avg_loss = total_loss / float(n_batches)

        self.logger.info(
            f"[eval][epoch={epoch}] loss={avg_loss:.6f} ws_acc={m.get('ws_acc', -1.0):.4f} ws_f1={m.get('ws_f1', -1.0):.4f}"
        )

        return EvalOutput(
            loss=avg_loss,
            loss_stats=agg_stats,
            metrics=m,
        )
