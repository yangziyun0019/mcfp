from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import yaml


def _dict_to_namespace(obj: Any) -> Any:
    """Recursively convert dictionaries into SimpleNamespace objects."""
    if isinstance(obj, Mapping):
        return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_dict_to_namespace(v) for v in obj]
    return obj


def load_config(path: str | Path):
    """Load a YAML config file and return an attribute-accessible object.

    Parameters
    ----------
    path:
        Path to the YAML configuration file.

    Returns
    -------
    cfg:
        A SimpleNamespace-like object with nested attributes matching
        the YAML structure.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return _dict_to_namespace(data)
