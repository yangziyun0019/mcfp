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
CSV_PATH = Path("runs/stage1/exp_pose6d_v1/metrics.csv")
OUT_PATH = None  # e.g. Path("runs/stage1/exp_pose6d_v1/loss_plot.png")
MA_WINDOW = 200  # moving average window (steps)
SHOW = False  # set True to open a window
WATCH = True  # set True to refresh when CSV updates
WATCH_INTERVAL_S = 5  # seconds between checks

# Plot config
GWS_LOSS_KEY = "loss/g_ws"
DELTA_POS_KEYS = ["loss/delta_pos_x", "loss/delta_pos_y", "loss/delta_pos_z"]
DELTA_ROT_KEYS = ["loss/delta_rot_x", "loss/delta_rot_y", "loss/delta_rot_z"]


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
            row_series: Dict[str, float] = {}
            for k, v in row.items():
                if k in ("phase", "step", "max_steps"):
                    continue
                fv = _to_float(v)
                if fv is None:
                    continue
                row_series[k] = fv

            if len(row_series) == 0:
                continue

            steps.append(step_i)
            for k, v in row_series.items():
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


def _sum_series(series: Dict[str, List[float]], keys: List[str], label: str) -> List[float]:
    vals: List[List[float]] = []
    for k in keys:
        if k not in series:
            raise SystemExit(f"{label} missing in metrics.csv: {k}")
        vals.append(series[k])
    n = len(vals[0])
    if any(len(v) != n for v in vals):
        raise SystemExit(f"{label} length mismatch across keys: {keys}")
    out = []
    for i in range(n):
        out.append(float(sum(v[i] for v in vals)))
    return out


def _plot_once(csv_path: Path) -> None:
    if not csv_path.exists():
        raise SystemExit(f"metrics.csv not found: {csv_path}")

    train_steps, train_series = _read_metrics(csv_path, phase="train")
    val_steps, val_series = _read_metrics(csv_path, phase="val")

    if GWS_LOSS_KEY not in train_series:
        raise SystemExit(f"{GWS_LOSS_KEY} not found in training rows of metrics.csv.")

    train_gws = train_series[GWS_LOSS_KEY]
    train_pos = _sum_series(train_series, DELTA_POS_KEYS, "delta_pos")
    train_rot = _sum_series(train_series, DELTA_ROT_KEYS, "delta_rot")

    has_val = True
    if len(val_steps) == 0:
        has_val = False
    else:
        required_val = [GWS_LOSS_KEY] + DELTA_POS_KEYS + DELTA_ROT_KEYS
        missing_val = [k for k in required_val if k not in val_series]
        if missing_val:
            has_val = False

    if has_val:
        val_gws = val_series[GWS_LOSS_KEY]
        val_pos = _sum_series(val_series, DELTA_POS_KEYS, "delta_pos")
        val_rot = _sum_series(val_series, DELTA_ROT_KEYS, "delta_rot")
    else:
        val_gws = []
        val_pos = []
        val_rot = []

    # Sort by step for stable curves.
    train_pairs = sorted(zip(train_steps, train_gws, train_pos, train_rot), key=lambda p: p[0])
    train_steps = [p[0] for p in train_pairs]
    train_gws = [p[1] for p in train_pairs]
    train_pos = [p[2] for p in train_pairs]
    train_rot = [p[3] for p in train_pairs]

    if has_val:
        val_pairs = sorted(zip(val_steps, val_gws, val_pos, val_rot), key=lambda p: p[0])
        val_steps = [p[0] for p in val_pairs]
        val_gws = [p[1] for p in val_pairs]
        val_pos = [p[2] for p in val_pairs]
        val_rot = [p[3] for p in val_pairs]

    plt.clf()
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(train_steps, train_gws, label="g_ws (raw)", linewidth=1.0, alpha=0.35)
    axes[0].plot(train_steps, _moving_average(train_gws, int(MA_WINDOW)), label=f"g_ws (ma{int(MA_WINDOW)})", linewidth=2.0)
    axes[0].plot(train_steps, train_pos, label="delta_pos_sum (raw)", linewidth=1.0, alpha=0.35)
    axes[0].plot(train_steps, _moving_average(train_pos, int(MA_WINDOW)), label=f"delta_pos_sum (ma{int(MA_WINDOW)})", linewidth=2.0)
    axes[0].plot(train_steps, train_rot, label="delta_rot_sum (raw)", linewidth=1.0, alpha=0.35)
    axes[0].plot(train_steps, _moving_average(train_rot, int(MA_WINDOW)), label=f"delta_rot_sum (ma{int(MA_WINDOW)})", linewidth=2.0)
    axes[0].set_title("train losses")
    axes[0].set_ylabel("loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, ncol=1)

    if has_val:
        axes[1].plot(val_steps, val_gws, label="g_ws (raw)", linewidth=1.0, alpha=0.35)
        axes[1].plot(val_steps, _moving_average(val_gws, int(MA_WINDOW)), label=f"g_ws (ma{int(MA_WINDOW)})", linewidth=2.0)
        axes[1].plot(val_steps, val_pos, label="delta_pos_sum (raw)", linewidth=1.0, alpha=0.35)
        axes[1].plot(val_steps, _moving_average(val_pos, int(MA_WINDOW)), label=f"delta_pos_sum (ma{int(MA_WINDOW)})", linewidth=2.0)
        axes[1].plot(val_steps, val_rot, label="delta_rot_sum (raw)", linewidth=1.0, alpha=0.35)
        axes[1].plot(val_steps, _moving_average(val_rot, int(MA_WINDOW)), label=f"delta_rot_sum (ma{int(MA_WINDOW)})", linewidth=2.0)
        axes[1].set_title("val losses")
    else:
        axes[1].set_title("val losses (no data yet)")
        axes[1].text(0.5, 0.5, "no val rows found", ha="center", va="center", transform=axes[1].transAxes)
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
