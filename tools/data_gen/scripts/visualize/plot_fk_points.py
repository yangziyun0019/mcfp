#!/usr/bin/env python3
"""Script: plot_fk_points.py
Purpose: Plot sampled FK end-effector positions from a generated position dataset.
Usage: python3 tools/data_gen/scripts/visualize/plot_fk_points.py --input-dir <output_dir>
"""

import argparse
import pathlib
import sys

import numpy as np

# RealMan RM65:
DEFAULT_INPUT_DIR = "tools/data_gen/outputs/realman/rm65/voxel_3mm"
# Aubo i5:
# DEFAULT_INPUT_DIR = "tools/data_gen/outputs/aubo/aubo_i5/voxel_3mm"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visualize FK sample positions from dataset_generator_cli"
    )
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help="Output directory containing fk_positions.npy",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=100000,
        help="Number of points to plot (random subset)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--save",
        default="",
        help="Optional output image path (png). If empty, show window.",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.3, help="Point transparency"
    )
    args = parser.parse_args()

    input_dir = pathlib.Path(args.input_dir)
    fk_path = input_dir / "fk_positions.npy"
    if not fk_path.exists():
        print(f"Missing fk_positions.npy in {input_dir}", file=sys.stderr)
        print("Hint: set output.write_fk_samples=true and rerun generator.")
        return 1

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not found. Install with: pip install matplotlib", file=sys.stderr)
        return 1

    points = np.load(fk_path)
    if points.ndim != 2 or points.shape[1] != 3:
        print(f"Unexpected fk_positions shape: {points.shape}", file=sys.stderr)
        return 1

    n = points.shape[0]
    if n == 0:
        print("No FK points found.")
        return 1

    sample_n = min(args.sample, n)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(n, size=sample_n, replace=False) if sample_n < n else np.arange(n)
    pts = points[idx]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, c=pts[:, 2], cmap="viridis", alpha=args.alpha)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"FK samples: {sample_n}/{n}")

    if args.save:
        plt.savefig(args.save, dpi=200, bbox_inches="tight")
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
