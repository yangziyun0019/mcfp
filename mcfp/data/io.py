from __future__ import annotations

from pathlib import Path
from typing import Dict
from typing import Optional

import numpy as np


def save_capability_map(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    """Save capability map arrays to a compressed NPZ file.

    Parameters
    ----------
    path:
        Output file path.
    arrays:
        Dictionary of named arrays, e.g. {
            "cell_centers": (N, 3),
            "g_ws": (N,),
            ...
        }.
    """
    path = Path(path)
    np.savez_compressed(path, **arrays)


def load_capability_map(path: Path) -> Dict[str, np.ndarray]:
    """Load capability map arrays from a NPZ file.

    Parameters
    ----------
    path:
        Input file path.

    Returns
    -------
    arrays:
        Dictionary of named arrays.
    """
    path = Path(path)
    with np.load(path) as data:
        return {k: data[k] for k in data.files}

def save_workspace_samples(
    output_path: Path,
    positions: np.ndarray,
    joint_samples: Optional[np.ndarray] = None,
    logger=None,
) -> None:
    """Save workspace sampling points for visualization.

    Parameters
    ----------
    output_path:
        Path to the main capability map file. The workspace samples will be
        stored next to it, with suffix '_workspace_samples.npz'.
    positions:
        Array of shape (N, 3) with end-effector positions.
    joint_samples:
        Optional array of shape (N, num_joints) with joint configurations
        used for sampling.
    logger:
        Optional logger instance.
    """
    output_path = Path(output_path)
    out_dir = output_path.parent
    out_name = output_path.stem + "_workspace_samples.npz"
    save_path = out_dir / out_name

    data = {"positions": np.asarray(positions, dtype=np.float32)}
    if joint_samples is not None:
        data["joint_samples"] = np.asarray(joint_samples, dtype=np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(save_path, **data)

    if logger is not None:
        logger.info(
            f"[data.io] Saved workspace samples to: {save_path}"
        )