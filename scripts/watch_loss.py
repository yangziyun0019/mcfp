"""Watch a metrics CSV file and refresh a loss plot on disk.

This script is intended for long-running training jobs that need a lightweight plotting monitor.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def plot_metrics(csv_path: Path, out_path: Path) -> None:
    if not csv_path.is_file():
        return
    df = pd.read_csv(csv_path)
    if df.empty or "step" not in df.columns:
        return
    step = df["step"].values
    cols = [c for c in df.columns if c not in ("step", "elapsed_sec")]
    if not cols:
        return

    plt.figure(figsize=(8, 5))
    for c in cols:
        plt.plot(step, df[c].values, label=c)
    plt.xlabel("step")
    plt.ylabel("value")
    plt.title(csv_path.stem)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch metrics.csv and update a loss PNG.")
    parser.add_argument("--metrics", type=str, required=True, help="Path to metrics.csv")
    parser.add_argument("--out", type=str, required=True, help="Output png path")
    parser.add_argument("--interval", type=float, default=5.0, help="Refresh interval (sec)")
    parser.add_argument("--once", action="store_true", help="Plot once and exit")
    args = parser.parse_args()

    csv_path = Path(args.metrics)
    out_path = Path(args.out)

    if args.once:
        plot_metrics(csv_path, out_path)
        return

    last_mtime = None
    while True:
        if csv_path.is_file():
            mtime = csv_path.stat().st_mtime
            if last_mtime is None or mtime > last_mtime:
                plot_metrics(csv_path, out_path)
                last_mtime = mtime
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
