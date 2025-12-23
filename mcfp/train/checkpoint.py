# mcfp/train/checkpoint.py

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


def _ensure_dir(p: Path) -> None:
    """Create directory if it does not exist."""
    p.mkdir(parents=True, exist_ok=True)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a training checkpoint."""
    path = Path(path)
    _ensure_dir(path.parent)

    payload: Dict[str, Any] = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "torch_version": torch.__version__,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "meta": meta or {},
    }
    torch.save(payload, str(path))


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    """Load a training checkpoint and restore states."""
    path = Path(path)
    payload = torch.load(str(path), map_location=map_location)

    model.load_state_dict(payload["model"], strict=strict)
    if optimizer is not None and payload.get("optimizer", None) is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler", None) is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler", None) is not None:
        scaler.load_state_dict(payload["scaler"])

    return payload.get("meta", {})


@dataclass
class CheckpointState:
    """Track best checkpoint state."""
    best_metric: float
    best_path: Optional[Path]


class CheckpointManager:
    """Manage 'last' and 'best' checkpoints.

    Args:
        ckpt_dir: Directory to write checkpoints.
        monitor_key: The metric name to monitor.
        mode: "min" or "max".
        keep_last: If True, always write last.pt.
        keep_best: If True, write best.pt when improved.
    """

    def __init__(
        self,
        ckpt_dir: Path,
        monitor_key: str,
        mode: str = "min",
        keep_last: bool = True,
        keep_best: bool = True,
    ) -> None:
        self.ckpt_dir = Path(ckpt_dir)
        self.monitor_key = str(monitor_key)
        self.mode = str(mode).lower()
        if self.mode not in ("min", "max"):
            raise ValueError("[checkpoint] mode must be 'min' or 'max'.")
        self.keep_last = bool(keep_last)
        self.keep_best = bool(keep_best)

        _ensure_dir(self.ckpt_dir)

        init_best = float("inf") if self.mode == "min" else -float("inf")
        self.state = CheckpointState(best_metric=init_best, best_path=None)

    def _is_improved(self, value: float) -> bool:
        if self.mode == "min":
            return value < self.state.best_metric
        return value > self.state.best_metric

    def save_last(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[Any],
        scaler: Optional[Any],
        meta: Dict[str, Any],
    ) -> Path:
        """Save last checkpoint."""
        path = self.ckpt_dir / "last.pt"
        save_checkpoint(path, model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, meta=meta)
        return path

    def maybe_save_best(
        self,
        metrics: Dict[str, float],
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[Any],
        scaler: Optional[Any],
        meta: Dict[str, Any],
    ) -> Optional[Path]:
        """Save best checkpoint if improved."""
        if not self.keep_best:
            return None
        if self.monitor_key not in metrics:
            raise KeyError(f"[checkpoint] monitor_key '{self.monitor_key}' not found in metrics.")

        value = float(metrics[self.monitor_key])
        if self._is_improved(value):
            self.state.best_metric = value
            path = self.ckpt_dir / "best.pt"
            save_checkpoint(path, model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, meta=meta)
            self.state.best_path = path
            return path
        return None
