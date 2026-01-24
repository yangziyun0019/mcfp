from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List, Tuple


def _read_metrics(path: Path) -> Dict[str, List[Tuple[int, float]]]:
    data = {"train": [], "val": []}
    if not path.is_file():
        return data
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = (row.get("split") or "train").strip()
            if split not in data:
                continue
            try:
                step = int(float(row["step"]))
                loss = float(row["loss"])
            except (KeyError, ValueError):
                continue
            data[split].append((step, loss))
    return data


def _tail_series(series: List[Tuple[int, float]], tail: int) -> List[Tuple[int, float]]:
    if tail <= 0:
        return series
    return series[-tail:]


def main() -> int:
    parser = argparse.ArgumentParser(description="Live plot of SDF Stage1 metrics.csv")
    parser.add_argument(
        "--metrics",
        type=str,
        default="runs/sdf_stage1/exp001/metrics.csv",
        help="Path to metrics.csv",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Output PNG path (default: metrics.png next to metrics.csv).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Refresh interval in seconds",
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=0,
        help="Number of latest points to display (0 for all)",
    )
    args = parser.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for plotting. Install it with conda or pip.")
        return 1

    metrics_path = Path(args.metrics).expanduser()
    if args.out:
        out_path = Path(args.out).expanduser()
    else:
        out_path = metrics_path.with_suffix(".png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    (train_line,) = ax.plot([], [], label="train", color="#1f77b4")
    (val_line,) = ax.plot([], [], label="val", color="#ff7f0e")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(f"Live metrics: {metrics_path}")
    ax.grid(True, alpha=0.3)
    ax.legend()

    try:
        last_mtime = None
        while True:
            if metrics_path.is_file():
                mtime = metrics_path.stat().st_mtime
            else:
                mtime = None

            if mtime is not None and mtime == last_mtime:
                time.sleep(args.interval)
                continue
            last_mtime = mtime

            data = _read_metrics(metrics_path)
            train = _tail_series(data["train"], args.tail)
            val = _tail_series(data["val"], args.tail)

            if train:
                xs, ys = zip(*train)
                train_line.set_data(xs, ys)
            else:
                train_line.set_data([], [])

            if val:
                xs, ys = zip(*val)
                val_line.set_data(xs, ys)
            else:
                val_line.set_data([], [])

            ax.relim()
            ax.autoscale_view()
            fig.tight_layout()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
