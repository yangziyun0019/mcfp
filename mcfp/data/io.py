from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
        logger.info(f"[data.io] Saved workspace samples to: {save_path}")


# =============================================================================
# New utilities for Stage I data preparation (manifest/splits/stats).
# =============================================================================

_REQUIRED_CAP_KEYS_DEFAULT = [
    "cell_centers",
    "g_ws",
    "g_red",
    "g_margin",
    "g_self",
    "g_selfpass",
    "g_lim",
    "g_rot",
    "g_man",
    "g_iso",
    "g_sigma",
]


@dataclass(frozen=True)
class GridMeta:
    """Structured grid metadata inferred from voxel centers.

    Attributes
    ----------
    bounds_min:
        Axis-aligned min bound (inclusive) in meters.
    bounds_max:
        Axis-aligned max bound (inclusive) in meters.
    resolution:
        Grid resolution per axis (dx, dy, dz).
    cell_shape:
        Grid shape (Nx, Ny, Nz).
    """

    bounds_min: np.ndarray  # (3,)
    bounds_max: np.ndarray  # (3,)
    resolution: np.ndarray  # (3,)
    cell_shape: Tuple[int, int, int]


def load_morph_spec(path: Path) -> Dict[str, Any]:
    """Load a morphology spec JSON.

    Parameters
    ----------
    path:
        Path to a spec JSON file.

    Returns
    -------
    spec:
        Parsed JSON as dict.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_morph_scale(spec: Dict[str, Any]) -> float:
    """Compute a morphology scale scalar L_r from spec.

    This is used for morphology-scale normalization of task-space positions.

    Current default:
        L_r = sum(link.length_estimate) over chain.links

    Parameters
    ----------
    spec:
        Morphology spec dict.

    Returns
    -------
    L_r:
        Positive scale scalar. Falls back to 1.0 if missing/invalid.
    """
    links = (
        spec.get("chain", {}).get("links", [])
        if isinstance(spec, dict)
        else []
    )
    lengths: List[float] = []
    for lk in links:
        try:
            lengths.append(float(lk.get("length_estimate", 0.0)))
        except Exception:
            lengths.append(0.0)

    L_r = float(np.sum(lengths)) if len(lengths) > 0 else 0.0
    if not np.isfinite(L_r) or L_r <= 0.0:
        # Conservative fallback to avoid division by zero in downstream code.
        return 1.0
    return L_r


def get_spec_identity(spec: Dict[str, Any], fallback_stem: str) -> Tuple[str, str, int]:
    """Extract (family, variant_id, dof) from spec.

    Parameters
    ----------
    spec:
        Morphology spec dict.
    fallback_stem:
        File stem used as fallback variant_id.

    Returns
    -------
    family:
        Spec family string.
    variant_id:
        Spec variant id string.
    dof:
        Degrees of freedom (int).
    """
    meta = spec.get("meta", {}) if isinstance(spec, dict) else {}
    family = str(meta.get("family", "")).strip() or "unknown"
    variant_id = str(meta.get("variant_id", "")).strip() or fallback_stem
    dof_raw = meta.get("dof", None)
    dof = int(dof_raw) if dof_raw is not None else -1
    return family, variant_id, dof


def _unique_sorted_with_rounding(values: np.ndarray, decimals: int = 6) -> np.ndarray:
    """Get sorted unique values with rounding to stabilize float noise."""
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    v = np.round(v, decimals=decimals)
    u = np.unique(v)
    u.sort()
    return u


def infer_grid_meta_from_centers(
    cell_centers: np.ndarray,
    decimals: int = 6,
    logger=None,
) -> GridMeta:
    """Infer grid metadata (bounds/resolution/shape) from cell centers.

    Assumptions
    ----------
    - cell_centers are produced by a Cartesian product grid (meshgrid),
      i.e., centers lie on a regular lattice.
    - Resolution is approximately constant along each axis.

    Parameters
    ----------
    cell_centers:
        Array of shape (N, 3).
    decimals:
        Rounding decimals for stable unique coordinate detection.
    logger:
        Optional logger.

    Returns
    -------
    grid_meta:
        GridMeta object.

    Raises
    ------
    ValueError:
        If grid structure cannot be inferred reliably.
    """
    c = np.asarray(cell_centers, dtype=np.float64)
    if c.ndim != 2 or c.shape[1] != 3:
        raise ValueError(f"[data.io] cell_centers must be (N,3). Got {c.shape}.")

    ux = _unique_sorted_with_rounding(c[:, 0], decimals=decimals)
    uy = _unique_sorted_with_rounding(c[:, 1], decimals=decimals)
    uz = _unique_sorted_with_rounding(c[:, 2], decimals=decimals)

    Nx, Ny, Nz = int(len(ux)), int(len(uy)), int(len(uz))
    N = int(c.shape[0])
    if Nx * Ny * Nz != N:
        msg = (
            f"[data.io] Cannot infer grid shape: Nx*Ny*Nz={Nx*Ny*Nz} != N={N}. "
            "This suggests centers are not a full Cartesian grid."
        )
        if logger is not None:
            logger.error(msg)
        raise ValueError(msg)

    def _infer_res(u: np.ndarray) -> float:
        if len(u) <= 1:
            return 0.0
        d = np.diff(u)
        d = d[np.isfinite(d)]
        if d.size == 0:
            return 0.0
        # Median is robust to a few outliers.
        return float(np.median(d))

    dx, dy, dz = _infer_res(ux), _infer_res(uy), _infer_res(uz)
    res = np.asarray([dx, dy, dz], dtype=np.float64)
    if not np.all(np.isfinite(res)) or np.any(res <= 0.0):
        msg = f"[data.io] Invalid inferred resolution: {res.tolist()}."
        if logger is not None:
            logger.error(msg)
        raise ValueError(msg)

    # AABB bounds for voxel grid are assumed to be center-extents +/- res/2.
    bounds_min = np.asarray([ux[0] - dx / 2.0, uy[0] - dy / 2.0, uz[0] - dz / 2.0], dtype=np.float64)
    bounds_max = np.asarray([ux[-1] + dx / 2.0, uy[-1] + dy / 2.0, uz[-1] + dz / 2.0], dtype=np.float64)

    return GridMeta(
        bounds_min=bounds_min.astype(np.float32),
        bounds_max=bounds_max.astype(np.float32),
        resolution=res.astype(np.float32),
        cell_shape=(Nx, Ny, Nz),
    )


def resolve_capability_map_path(
    capability_root: Path,
    family: str,
    variant_id: str,
) -> Path:
    """Resolve capability_map.npz path from naming convention."""
    return Path(capability_root) / family / variant_id / "capability_map.npz"


def iter_spec_files(morph_specs_root: Path) -> Iterable[Path]:
    """Iterate all spec JSON files under morph_specs_root."""
    root = Path(morph_specs_root)
    yield from root.rglob("*.json")


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    """Write JSONL records."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read JSONL records."""
    path = Path(path)
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            records.append(json.loads(s))
    return records


def to_posix_relpath(path: Path, repo_root: Path) -> str:
    """Convert an absolute path into a POSIX-style relative path to repo root."""
    p = Path(path).resolve()
    root = Path(repo_root).resolve()
    rel = p.relative_to(root)
    return rel.as_posix()


def build_manifest_records(
    repo_root: Path,
    morph_specs_root: Path,
    capability_root: Path,
    required_cap_keys: Optional[List[str]] = None,
    grid_round_decimals: int = 6,
    logger=None,
) -> List[Dict[str, Any]]:
    """Build manifest records for all morphologies.

    Parameters
    ----------
    repo_root:
        Repository root path. All stored paths are relative to this root.
    morph_specs_root:
        Root directory containing morphology spec JSONs.
    capability_root:
        Root directory containing capability maps, organized as:
            capability_root / family / variant_id / capability_map.npz
    required_cap_keys:
        Keys required to exist in capability_map.npz.
    grid_round_decimals:
        Rounding decimals for grid inference from cell_centers.
    logger:
        Optional logger.

    Returns
    -------
    records:
        List of manifest dict records.
    """
    req_keys = required_cap_keys if required_cap_keys is not None else list(_REQUIRED_CAP_KEYS_DEFAULT)

    records: List[Dict[str, Any]] = []
    for spec_path in iter_spec_files(morph_specs_root):
        spec = load_morph_spec(spec_path)
        family, variant_id, dof = get_spec_identity(spec, fallback_stem=spec_path.stem)

        cap_path = resolve_capability_map_path(capability_root, family=family, variant_id=variant_id)
        if not cap_path.exists():
            if logger is not None:
                logger.warning(f"[data.io] Missing capability map: {cap_path} (spec={spec_path})")
            continue

        cap = load_capability_map(cap_path)
        missing = [k for k in req_keys if k not in cap]
        if missing:
            if logger is not None:
                logger.warning(
                    f"[data.io] Capability map missing keys {missing}: {cap_path}"
                )
            continue

        centers = cap["cell_centers"]
        grid_meta = infer_grid_meta_from_centers(
            cell_centers=centers,
            decimals=grid_round_decimals,
            logger=logger,
        )

        g_ws = cap.get("g_ws", None)
        if g_ws is not None:
            ws_coverage = float(np.mean(np.asarray(g_ws, dtype=np.float32)))
        else:
            ws_coverage = float("nan")

        num_cells = int(np.asarray(centers).shape[0])
        cap_keys = sorted(list(cap.keys()))

        L_r = compute_morph_scale(spec)

        rec = {
            "family": family,
            "variant_id": variant_id,
            "dof": int(dof),
            "spec_path": to_posix_relpath(spec_path, repo_root),
            "cap_path": to_posix_relpath(cap_path, repo_root),
            "cap_keys": cap_keys,
            "num_cells": num_cells,
            "valid_ratio": ws_coverage,
            "grid_meta": {
                "bounds_min": grid_meta.bounds_min.tolist(),
                "bounds_max": grid_meta.bounds_max.tolist(),
                "resolution": grid_meta.resolution.tolist(),
                "cell_shape": list(grid_meta.cell_shape),
                "round_decimals": int(grid_round_decimals),
            },
            "morph_scale": float(L_r),
        }
        records.append(rec)

    # Deterministic ordering for reproducibility.
    records.sort(key=lambda r: (r["family"], r["variant_id"]))
    return records
