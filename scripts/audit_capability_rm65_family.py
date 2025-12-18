# scripts/audit_capability_rm65_family.py
"""Audit capability maps across a morphology family using fixed probes from a baseline map.

This script is intentionally config-free for rapid debugging and dataset QA.
All paths and rules are defined in the constants section below.
"""

from __future__ import annotations

import csv
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from mcfp.utils.logging import setup_logger


# =============================================================================
# Constants (hard-coded as requested)
# =============================================================================

# Baseline capability map path (Windows).
BASELINE_NPZ = Path("/home/user/mcfp/data/capability/morph_specs/rm65/rm65_base/capability_map.npz")

# Family root directory. All spec subfolders under this directory will be audited.
FAMILY_ROOT = BASELINE_NPZ.parent.parent  # ...\rm65\

# Capability filename inside each spec folder.
CAP_FILENAME = "capability_map.npz"

# Output directory (default: family root).
OUTPUT_DIR = FAMILY_ROOT

# Spec folder name filter. Adjust if you want to include/exclude certain variants.
# - None: include any folder containing capability_map.npz
# - Regex: only include matching folder names
SPEC_DIR_REGEX = re.compile(r"^rm65_(base|v\d{4,})$")

# Probe selection parameters (from baseline only).
EXTREME_QUANTILE = 0.995         # near-boundary robust extreme
CENTRAL_Q_LOW = 0.40             # central region lower quantile
CENTRAL_Q_HIGH = 0.60            # central region upper quantile
ADD_LOW_MARGIN_PROBES = True
LOW_MARGIN_K = 2                 # number of lowest g_margin probes to add from baseline

# Nearest-cell match tolerance in meters.
MAX_MATCH_DIST_M = 0.05

# If True, only choose probes among baseline reachable cells (g_ws==1).
PROBES_REQUIRE_REACHABLE = True

# If sample_counts exists, prefer cells with larger counts when selecting extreme probes.
PREFER_HIGH_SAMPLE_COUNTS_FOR_PROBES = True

# =============================================================================
# Data structures
# =============================================================================


@dataclass(frozen=True)
class Probe:
    """A fixed probe defined by a target position in task space."""
    name: str
    xyz: np.ndarray  # shape (3,), float32


# =============================================================================
# IO utilities
# =============================================================================


def load_capability_npz(path: Path) -> Dict[str, Any]:
    """Load a capability_map.npz file.

    Args:
        path: Path to capability_map.npz.

    Returns:
        A dict mapping keys to numpy arrays.
    """
    data = np.load(str(path), allow_pickle=False)
    return {k: data[k] for k in data.files}


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write a list of dict rows into a CSV file with stable column ordering.

    Args:
        path: Output CSV path.
        rows: Rows to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    keys: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
    """Write JSON to disk.

    Args:
        path: Output JSON path.
        obj: JSON-serializable object.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# =============================================================================
# Math / selection utilities
# =============================================================================


def safe_float(x: Any) -> float:
    """Convert numpy scalar or python numeric to float."""
    return float(np.asarray(x).item())


def quantile_threshold(values: np.ndarray, q: float) -> float:
    """Compute a quantile threshold with float output."""
    return float(np.quantile(values, q))


def nearest_cell_index(cell_centers: np.ndarray, target_xyz: np.ndarray) -> Tuple[int, float]:
    """Find nearest cell center.

    Args:
        cell_centers: (N, 3) float array.
        target_xyz: (3,) float array.

    Returns:
        (idx, dist_m): nearest index and Euclidean distance in meters.
    """
    diff = cell_centers - target_xyz[None, :]
    d2 = np.sum(diff * diff, axis=1)
    idx = int(np.argmin(d2))
    dist = float(math.sqrt(float(d2[idx])))
    return idx, dist


def compute_aabb(centers: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute AABB stats from cell centers.

    Args:
        centers: (N, 3)

    Returns:
        xyz_min, xyz_max, span, volume
    """
    xyz_min = np.min(centers, axis=0)
    xyz_max = np.max(centers, axis=0)
    span = xyz_max - xyz_min
    volume = float(span[0] * span[1] * span[2])
    return xyz_min, xyz_max, span, volume


def summarize_reachable(cap: Dict[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
    """Summarize metric distributions over reachable cells.

    Args:
        cap: capability map dict.
        keys: metric keys to summarize.

    Returns:
        A dict with mean/p10/p50/p90 fields per key.
    """
    g_ws = np.asarray(cap["g_ws"], dtype=np.float32)
    reachable = g_ws > 0.5
    out: Dict[str, Any] = {}

    n = int(g_ws.shape[0])
    n_reach = int(np.sum(reachable))
    out["n_cells"] = n
    out["n_reachable"] = n_reach
    out["coverage"] = (n_reach / max(n, 1))

    if n_reach == 0:
        return out

    for k in keys:
        if k not in cap:
            continue
        v = np.asarray(cap[k], dtype=np.float32)[reachable]
        out[f"{k}_mean"] = safe_float(np.mean(v))
        out[f"{k}_p10"] = safe_float(np.quantile(v, 0.10))
        out[f"{k}_p50"] = safe_float(np.quantile(v, 0.50))
        out[f"{k}_p90"] = safe_float(np.quantile(v, 0.90))

    return out


# =============================================================================
# Probe selection from baseline
# =============================================================================


def select_probes_from_baseline(
    baseline: Dict[str, Any],
    extreme_quantile: float,
    central_q_low: float,
    central_q_high: float,
    add_low_margin: bool,
    low_margin_k: int,
    require_reachable: bool,
    prefer_high_counts: bool,
) -> List[Probe]:
    """Select probe points from baseline capability map.

    Strategy:
      - 6 robust extremes (+/-X, +/-Y, +/-Z) among reachable cells.
      - 1 central best-manipulability cell inside a central quantile box.
      - Optional: K lowest g_margin reachable cells as boundary probes.

    Args:
        baseline: Baseline capability map dict.
        extreme_quantile: Quantile for extreme selection (e.g., 0.995).
        central_q_low: Lower quantile for central region.
        central_q_high: Upper quantile for central region.
        add_low_margin: Whether to add lowest-margin probes.
        low_margin_k: Number of low-margin probes.
        require_reachable: Restrict selection to reachable cells.
        prefer_high_counts: Prefer larger sample_counts when selecting extremes.

    Returns:
        List of Probe.
    """
    centers = np.asarray(baseline["cell_centers"], dtype=np.float32)
    g_ws = np.asarray(baseline["g_ws"], dtype=np.float32)

    if require_reachable:
        mask = g_ws > 0.5
    else:
        mask = np.ones((centers.shape[0],), dtype=bool)

    if not np.any(mask):
        raise ValueError("Baseline has no selectable cells (mask is empty).")

    centers_m = centers[mask]

    counts_m: Optional[np.ndarray] = None
    if prefer_high_counts and "sample_counts" in baseline:
        counts_m = np.asarray(baseline["sample_counts"], dtype=np.int32)[mask]

    probes: List[Probe] = []

    # Extreme probes per axis
    for axis_name, axis_idx in (("x", 0), ("y", 1), ("z", 2)):
        vals = centers_m[:, axis_idx]
        hi_th = quantile_threshold(vals, extreme_quantile)
        lo_th = quantile_threshold(vals, 1.0 - extreme_quantile)

        hi_candidates = np.where(vals >= hi_th)[0]
        lo_candidates = np.where(vals <= lo_th)[0]

        if hi_candidates.size == 0 or lo_candidates.size == 0:
            # Fallback to strict argmax/argmin if quantile yields empty sets.
            hi_candidates = np.array([int(np.argmax(vals))], dtype=int)
            lo_candidates = np.array([int(np.argmin(vals))], dtype=int)

        if counts_m is not None:
            hi_pick = int(hi_candidates[np.argmax(counts_m[hi_candidates])])
            lo_pick = int(lo_candidates[np.argmax(counts_m[lo_candidates])])
        else:
            hi_pick = int(hi_candidates[0])
            lo_pick = int(lo_candidates[0])

        probes.append(Probe(name=f"{axis_name}_pos_extreme", xyz=centers_m[hi_pick].astype(np.float32)))
        probes.append(Probe(name=f"{axis_name}_neg_extreme", xyz=centers_m[lo_pick].astype(np.float32)))

    # Central best manipulability (as a stability control point)
    if "g_man" in baseline:
        x = centers_m[:, 0]
        y = centers_m[:, 1]
        z = centers_m[:, 2]

        cx_lo = quantile_threshold(x, central_q_low)
        cx_hi = quantile_threshold(x, central_q_high)
        cy_lo = quantile_threshold(y, central_q_low)
        cy_hi = quantile_threshold(y, central_q_high)
        cz_lo = quantile_threshold(z, central_q_low)
        cz_hi = quantile_threshold(z, central_q_high)

        central_mask = (
            (x >= cx_lo) & (x <= cx_hi) &
            (y >= cy_lo) & (y <= cy_hi) &
            (z >= cz_lo) & (z <= cz_hi)
        )

        if np.any(central_mask):
            g_man_m = np.asarray(baseline["g_man"], dtype=np.float32)[mask]
            central_idx = int(np.argmax(g_man_m[central_mask]))
            central_xyz = centers_m[central_mask][central_idx]
            probes.append(Probe(name="central_best_man", xyz=central_xyz.astype(np.float32)))

    # Optional: low margin probes near reachable boundary
    if add_low_margin and ("g_margin" in baseline):
        g_margin_m = np.asarray(baseline["g_margin"], dtype=np.float32)[mask]
        if g_margin_m.size > 0:
            k = min(int(low_margin_k), int(g_margin_m.size))
            low_idx = np.argsort(g_margin_m)[:k]
            for i, idx in enumerate(low_idx, start=1):
                probes.append(Probe(name=f"low_margin_{i}", xyz=centers_m[int(idx)].astype(np.float32)))

    return probes


# =============================================================================
# Audit logic
# =============================================================================


def list_spec_cap_files(family_root: Path, cap_filename: str) -> List[Tuple[str, Path]]:
    """List all (spec_name, capability_npz_path) in the family root.

    Args:
        family_root: Family directory containing spec subfolders.
        cap_filename: capability_map.npz filename.

    Returns:
        Sorted list of (spec_name, path).
    """
    items: List[Tuple[str, Path]] = []
    for p in family_root.iterdir():
        if not p.is_dir():
            continue
        if SPEC_DIR_REGEX is not None and not SPEC_DIR_REGEX.match(p.name):
            continue
        cap_path = p / cap_filename
        if cap_path.exists():
            items.append((p.name, cap_path))
    items.sort(key=lambda x: x[0])
    return items


def audit_global(spec_name: str, cap: Dict[str, Any]) -> Dict[str, Any]:
    """Compute global summary row for one spec.

    Args:
        spec_name: Spec folder name.
        cap: Capability map dict.

    Returns:
        A dict row for CSV.
    """
    centers = np.asarray(cap["cell_centers"], dtype=np.float32)
    xyz_min, xyz_max, span, volume = compute_aabb(centers)

    row: Dict[str, Any] = {
        "spec": spec_name,
        "n_cells": int(centers.shape[0]),
        "aabb_min_x": safe_float(xyz_min[0]),
        "aabb_min_y": safe_float(xyz_min[1]),
        "aabb_min_z": safe_float(xyz_min[2]),
        "aabb_max_x": safe_float(xyz_max[0]),
        "aabb_max_y": safe_float(xyz_max[1]),
        "aabb_max_z": safe_float(xyz_max[2]),
        "span_x": safe_float(span[0]),
        "span_y": safe_float(span[1]),
        "span_z": safe_float(span[2]),
        "aabb_volume": float(volume),
    }

    summ = summarize_reachable(
        cap,
        keys=["g_lim", "g_selfpass", "g_man", "g_iso", "g_sigma", "g_rot", "g_red", "g_margin"],
    )
    row.update(summ)
    return row


def audit_probes(
    spec_name: str,
    cap: Dict[str, Any],
    probes: List[Probe],
    max_match_dist_m: float,
) -> List[Dict[str, Any]]:
    """Audit a list of probes against one capability map.

    Args:
        spec_name: Spec folder name.
        cap: Capability map dict.
        probes: Fixed probe list from baseline.
        max_match_dist_m: Maximum allowed nearest-cell distance to consider a match.

    Returns:
        List of rows (one row per probe).
    """
    centers = np.asarray(cap["cell_centers"], dtype=np.float32)
    rows: List[Dict[str, Any]] = []

    for probe in probes:
        idx, dist = nearest_cell_index(centers, probe.xyz)
        matched = dist <= max_match_dist_m

        row: Dict[str, Any] = {
            "spec": spec_name,
            "probe": probe.name,
            "probe_x": safe_float(probe.xyz[0]),
            "probe_y": safe_float(probe.xyz[1]),
            "probe_z": safe_float(probe.xyz[2]),
            "nearest_idx": int(idx),
            "nearest_dist_m": float(dist),
            "matched": bool(matched),
        }

        if matched:
            row["cell_x"] = safe_float(centers[idx, 0])
            row["cell_y"] = safe_float(centers[idx, 1])
            row["cell_z"] = safe_float(centers[idx, 2])

            # Always include g_ws
            row["g_ws"] = safe_float(np.asarray(cap["g_ws"], dtype=np.float32)[idx])

            # Core metrics if present
            for k in ["g_lim", "g_selfpass", "g_man", "g_iso", "g_sigma", "g_rot", "g_red", "g_margin"]:
                if k in cap:
                    row[k] = safe_float(np.asarray(cap[k], dtype=np.float32)[idx])

            if "sample_counts" in cap:
                row["sample_counts"] = int(np.asarray(cap["sample_counts"], dtype=np.int32)[idx])

        rows.append(row)

    return rows


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Run the audit for a family rooted at FAMILY_ROOT."""
    logger = setup_logger(name="mcfp.audit.capability_family", log_dir=str(OUTPUT_DIR))

    if not BASELINE_NPZ.exists():
        raise FileNotFoundError(f"Baseline capability map not found: {BASELINE_NPZ}")

    logger.info(f"Baseline: {BASELINE_NPZ}")
    logger.info(f"Family root: {FAMILY_ROOT}")
    logger.info(f"Output dir: {OUTPUT_DIR}")

    t0 = time.time()

    # Load baseline and select probes
    baseline_cap = load_capability_npz(BASELINE_NPZ)
    probes = select_probes_from_baseline(
        baseline=baseline_cap,
        extreme_quantile=EXTREME_QUANTILE,
        central_q_low=CENTRAL_Q_LOW,
        central_q_high=CENTRAL_Q_HIGH,
        add_low_margin=ADD_LOW_MARGIN_PROBES,
        low_margin_k=LOW_MARGIN_K,
        require_reachable=PROBES_REQUIRE_REACHABLE,
        prefer_high_counts=PREFER_HIGH_SAMPLE_COUNTS_FOR_PROBES,
    )
    logger.info(f"Selected probes: {len(probes)}")
    for p in probes:
        logger.info(f"  - {p.name}: [{p.xyz[0]:.4f}, {p.xyz[1]:.4f}, {p.xyz[2]:.4f}]")

    # Enumerate all specs under family root
    spec_caps = list_spec_cap_files(FAMILY_ROOT, CAP_FILENAME)
    if not spec_caps:
        logger.warning(f"No capability files found under: {FAMILY_ROOT}")
        return

    logger.info(f"Found {len(spec_caps)} specs with capability maps.")

    global_rows: List[Dict[str, Any]] = []
    probe_rows: List[Dict[str, Any]] = []
    summary_json: Dict[str, Any] = {
        "baseline": str(BASELINE_NPZ),
        "family_root": str(FAMILY_ROOT),
        "spec_count": len(spec_caps),
        "probe_count": len(probes),
        "max_match_dist_m": MAX_MATCH_DIST_M,
        "specs": {},
    }

    for i, (spec_name, cap_path) in enumerate(spec_caps, start=1):
        logger.info(f"[{i}/{len(spec_caps)}] Auditing {spec_name} -> {cap_path}")
        cap = load_capability_npz(cap_path)

        g_row = audit_global(spec_name, cap)
        global_rows.append(g_row)

        p_rows = audit_probes(spec_name, cap, probes, MAX_MATCH_DIST_M)
        probe_rows.extend(p_rows)

        # Minimal per-spec JSON summary
        summary_json["specs"][spec_name] = {
            "cap_path": str(cap_path),
            "coverage": g_row.get("coverage", None),
            "n_cells": g_row.get("n_cells", None),
            "n_reachable": g_row.get("n_reachable", None),
            "span": [g_row.get("span_x", None), g_row.get("span_y", None), g_row.get("span_z", None)],
            "aabb_volume": g_row.get("aabb_volume", None),
        }

    # Write outputs
    out_global = OUTPUT_DIR / "audit_global_summary.csv"
    out_probe = OUTPUT_DIR / "audit_probe_table.csv"
    out_json = OUTPUT_DIR / "audit_summary.json"

    write_csv(out_global, global_rows)
    write_csv(out_probe, probe_rows)
    write_json(out_json, summary_json)

    elapsed = time.time() - t0
    logger.info(f"Wrote: {out_global}")
    logger.info(f"Wrote: {out_probe}")
    logger.info(f"Wrote: {out_json}")
    logger.info(f"Done in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
