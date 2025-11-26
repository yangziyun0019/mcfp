from __future__ import annotations

from pathlib import Path
from typing import Dict

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
