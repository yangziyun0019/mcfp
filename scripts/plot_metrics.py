#!/usr/bin/env python
"""Plot loss curves from metrics.csv."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

# -----------------------------
# User config (edit here)
# -----------------------------
CSV_PATH = Path("runs/stage1/exp_aligned_v1/metrics.csv")
OUT_PATH = None  # e.g. Path("runs/stage1/exp_aligned_v1/loss_plot.png")
MA_WINDOW = 200  # moving average window (steps)
SHOW = False  # set True to open a window
WATCH = True  # set True to refresh when CSV updates
WATCH_INTERVAL_S = 5  # seconds between checks

# Plot config
TRAIN_LOSS_KEY = "loss/weighted_no_reg"
VAL_PLOT_MODE = "loss"  # "loss" or "metric"
VAL_LOSS_KEY = "loss/weighted_no_reg"
VAL_METRIC_KEY = "ws_f1"


def _to_float(s: str) -> float | None:
    try:
        return float(s)
    except Exception:
        return None


def _read_metrics(path: Path, phase: str) -> Tuple[List[int], Dict[str, List[float]]]:
    steps: List[int] = []
    series: Dict[str, List[float]] = {}

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if phase and row.get("phase", "") != phase:
                continue
            step = _to_float(row.get("step", ""))
            if step is None:
                continue
            step_i = int(step)

            loss_cols = [k for k in row.keys() if k == "loss/total" or k.startswith("loss/")]
            loss_cols = sorted(set(loss_cols))
            if not loss_cols:
                continue

            steps.append(step_i)
            for k in loss_cols:
                v = _to_float(row.get(k, ""))
                if v is None:
                    continue
                series.setdefault(k, []).append(v)

    return steps, series


def _moving_average(values: List[float], window: int) -> List[float]:
    if window <= 1 or len(values) == 0:
        return list(values)
    window = min(window, len(values))
    out: List[float] = []
    csum = [0.0]
    for v in values:
        csum.append(csum[-1] + v)
    for i in range(len(values)):
        j0 = max(0, i - window + 1)
        j1 = i + 1
        out.append((csum[j1] - csum[j0]) / float(j1 - j0))
    return out


def _plot_once(csv_path: Path) -> None:
    if not csv_path.exists():
        raise SystemExit(f"metrics.csv not found: {csv_path}")

    train_steps, train_series = _read_metrics(csv_path, phase="train")
    if TRAIN_LOSS_KEY not in train_series:
        raise SystemExit(f"{TRAIN_LOSS_KEY} not found in metrics.csv.")

    val_steps, val_series = _read_metrics(csv_path, phase="val")
    if VAL_PLOT_MODE == "loss":
        if VAL_LOSS_KEY not in val_series:
            raise SystemExit(f"{VAL_LOSS_KEY} not found in metrics.csv.")
        val_key = VAL_LOSS_KEY
    else:
        if VAL_METRIC_KEY not in val_series:
            raise SystemExit(f"{VAL_METRIC_KEY} not found in metrics.csv.")
        val_key = VAL_METRIC_KEY

    # Sort by step for stable curves.
    train_pairs = sorted(zip(train_steps, train_series[TRAIN_LOSS_KEY]), key=lambda p: p[0])
    train_steps = [p[0] for p in train_pairs]
    train_vals = [p[1] for p in train_pairs]

    val_pairs = sorted(zip(val_steps, val_series[val_key]), key=lambda p: p[0])
    val_steps = [p[0] for p in val_pairs]
    val_vals = [p[1] for p in val_pairs]

    plt.clf()
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(train_steps, train_vals, label=f"{TRAIN_LOSS_KEY} (raw)", linewidth=1.0, alpha=0.35)
    train_ma = _moving_average(train_vals, int(MA_WINDOW))
    axes[0].plot(train_steps, train_ma, label=f"{TRAIN_LOSS_KEY} (ma{int(MA_WINDOW)})", linewidth=2.0)
    axes[0].set_title("train loss")
    axes[0].set_ylabel("loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, ncol=1)

    axes[1].plot(val_steps, val_vals, label=f"{val_key} (raw)", linewidth=1.0, alpha=0.35)
    val_ma = _moving_average(val_vals, int(MA_WINDOW))
    axes[1].plot(val_steps, val_ma, label=f"{val_key} (ma{int(MA_WINDOW)})", linewidth=2.0)
    axes[1].set_title(f"val {val_key}")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("value")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8, ncol=1)

    if OUT_PATH:
        out_path = Path(OUT_PATH)
    else:
        out_path = csv_path.with_name("loss_plot_train_val.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)

    if SHOW:
        plt.pause(0.1)


def main() -> None:
    csv_path = Path(CSV_PATH)
    if SHOW:
        plt.ion()
    if not WATCH:
        _plot_once(csv_path)
        if SHOW:
            plt.show()
        return

    last_mtime = None
    while True:
        if csv_path.exists():
            mtime = csv_path.stat().st_mtime
            if last_mtime is None or mtime > last_mtime:
                _plot_once(csv_path)
                last_mtime = mtime
        if SHOW:
            plt.pause(0.1)
        import time
        time.sleep(float(WATCH_INTERVAL_S))


if __name__ == "__main__":
    main()
