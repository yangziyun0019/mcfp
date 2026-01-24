from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def _pick_pose_array(data: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    preferred = ["poses", "pose", "positions", "points", "anchors", "samples"]
    for key in preferred:
        arr = data.get(key)
        if arr is not None and arr.ndim == 2 and arr.shape[1] >= 3:
            return arr

    for arr in data.values():
        if arr.ndim == 2 and arr.shape[1] >= 3:
            return arr

    return None


def _resolve_positions(anchors_path: Path, pool_path: Optional[Path]) -> np.ndarray:
    anchor_data = _load_npz(anchors_path)
    poses = _pick_pose_array(anchor_data)
    if poses is not None:
        return np.asarray(poses)[:, :3]

    if "indices" not in anchor_data:
        raise ValueError(
            "No pose-like array in anchors npz and missing 'indices' to map into a pool."
        )

    indices = np.asarray(anchor_data["indices"]).astype(np.int64).ravel()
    if indices.size == 0:
        raise ValueError("Anchor indices are empty.")

    required_size = int(indices.max()) + 1
    candidates = []
    if pool_path is not None:
        candidates.append(pool_path)
    else:
        default_pool = anchors_path.with_name("positive_pool.npz")
        candidates.append(default_pool)
        for path in sorted(anchors_path.parent.glob("*.npz")):
            if path == anchors_path or path == default_pool:
                continue
            candidates.append(path)

    checked = []
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            pool_data = _load_npz(candidate)
        except Exception as exc:
            checked.append(f"{candidate} (load failed: {exc})")
            continue
        pool_poses = _pick_pose_array(pool_data)
        if pool_poses is None:
            checked.append(f"{candidate} (no pose-like array)")
            continue
        pool_poses = np.asarray(pool_poses)
        if pool_poses.ndim != 2:
            checked.append(f"{candidate} (pose array not 2D)")
            continue
        if pool_poses.shape[0] < required_size:
            checked.append(f"{candidate} (poses={pool_poses.shape[0]})")
            continue
        return pool_poses[indices][:, :3]

    checked_msg = ", ".join(checked) if checked else "none"
    raise ValueError(
        "No pose pool with enough entries for the anchor indices. "
        f"Need >= {required_size}, checked: {checked_msg}. "
        "Pass --pool to a matching pose pool npz."
    )


def _positions_from_file(path: Path, pool_path: Optional[Path]) -> np.ndarray:
    data = _load_npz(path)
    poses = _pick_pose_array(data)
    if poses is not None:
        return np.asarray(poses)[:, :3]

    if "indices" in data:
        return _resolve_positions(path, pool_path)

    raise ValueError(f"No pose-like array found in {path}.")


def _plot_positions(
    positions: np.ndarray,
    title: str,
    out_path: Optional[Path],
    show: bool,
) -> None:
    fig = plt.figure(figsize=(7.0, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        s=1.0,
        c=positions[:, 2],
        cmap="viridis",
        alpha=0.8,
    )
    fig.colorbar(sc, ax=ax, shrink=0.7, label="Z (m)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title)
    _set_axes_equal(ax, positions)
    fig.tight_layout()

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300)
        print(f"Saved plot to: {out_path}")

    if show:
        plt.show()


def _downsample(points: np.ndarray, max_points: Optional[int], seed: int) -> np.ndarray:
    if max_points is None or points.shape[0] <= max_points:
        return points

    rng = np.random.default_rng(seed)
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx]


def _set_axes_equal(ax, points: np.ndarray) -> None:
    min_xyz = points.min(axis=0)
    max_xyz = points.max(axis=0)
    center = (min_xyz + max_xyz) * 0.5
    span = max_xyz - min_xyz
    max_range = float(span.max())
    if max_range == 0.0:
        max_range = 1.0

    half = 0.5 * max_range
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize anchor boundary points (position only) from an NPZ file."
    )
    parser.add_argument(
        "--npz",
        type=Path,
        default=Path("data/sdf/ur5/anchors.npz"),
        help="Path to anchors npz (or a pose npz).",
    )
    parser.add_argument(
        "--pool",
        type=Path,
        default=None,
        help="Optional pose pool npz (used when anchors contain indices).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("tests/anchors_positions.png"),
        help="Output image path. Use --show to display instead.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot interactively.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Visualize all standard SDF NPZ files in a directory.",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("data/sdf/ur5"),
        help="Directory containing the 5 SDF NPZ files for --all.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tests/anchor_plots"),
        help="Output directory for batch images when using --all.",
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Skip files that cannot be visualized in --all mode.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=200000,
        help="Optional cap on number of points to plot.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for downsampling.",
    )
    args = parser.parse_args()

    if args.all:
        default_files: Sequence[str] = (
            "anchors.npz",
            "boundary_samples.npz",
            "positive_pool.npz",
            "pos_far_samples.npz",
            "rot_far_samples.npz",
        )
        errors: list[str] = []
        pool_path = args.pool or (args.dir / "positive_pool.npz")
        for name in default_files:
            path = args.dir / name
            if not path.exists():
                msg = f"Missing npz: {path}"
                if args.skip_errors:
                    print(msg)
                    continue
                errors.append(msg)
                continue
            try:
                positions = _positions_from_file(path, pool_path)
                positions = _downsample(positions, args.max_points, args.seed)
                out_path = args.out_dir / f"{path.stem}_positions.png"
                title = f"{path.stem} positions (N={positions.shape[0]})"
                _plot_positions(positions, title, out_path, show=False)
            except Exception as exc:
                msg = f"{path}: {exc}"
                if args.skip_errors:
                    print(msg)
                    continue
                errors.append(msg)
        if errors:
            raise ValueError("Batch visualization failed:\n" + "\n".join(errors))
        return

    if not args.npz.exists():
        raise FileNotFoundError(f"Anchors npz not found: {args.npz}")

    positions = _positions_from_file(args.npz, args.pool)
    positions = _downsample(positions, args.max_points, args.seed)
    title = f"{args.npz.stem} positions (N={positions.shape[0]})"
    _plot_positions(positions, title, args.out, show=args.show)


if __name__ == "__main__":
    main()
