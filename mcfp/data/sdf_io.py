from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np


def save_npz(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    """Save arrays to a compressed NPZ file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    """Save a JSON file with UTF-8 encoding."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
