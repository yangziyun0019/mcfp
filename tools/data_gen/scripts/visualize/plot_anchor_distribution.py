#!/usr/bin/env python3
"""Script: plot_anchor_distribution.py
Purpose: Visualize orientation anchors together with the position-dataset boundary.
Usage: python3 tools/data_gen/scripts/visualize/plot_anchor_distribution.py
"""

import sys
from pathlib import Path

import h5py
import numpy as np
# Prefer interactive 3D (pyvista)
USE_PYVISTA = True

# ---- Config (edit here) ----
BASE_H5 = "tools/data_gen/outputs/aubo/aubo_i5/voxel_3mm/dataset.h5"
ORIENT_CFG = "tools/data_gen/configs/robots/aubo/aubo_i5/orientation_3mm.yaml"
ORIENT_H5 = None  # if None -> read from ORIENT_CFG output.path
USE_ORIENT_H5_IF_AVAILABLE = True
BOUNDARY_BAND = None  # meters; if None -> 2 * voxel_size
BOUNDARY_USE_INSIDE_ONLY = False  # True: only label==1
MAX_BOUNDARY_POINTS = 80000
ANCHOR_MAX_POINTS = 2000  # downsample anchors if too many
ANCHOR_SIZE = 6
ANCHOR_RADIUS = 0.004  # meters; used when USE_SPHERE_ANCHORS=True
USE_SPHERE_ANCHORS = True
BOUNDARY_SIZE = 1
BOUNDARY_ALPHA = 0.03
ANCHOR_ALPHA = 0.9
ANCHOR_COLOR_BY = "none"  # none | s_v | c_v | g_v | n_seed
SHOW_ANCHOR_INDEX = False
RANDOM_SEED = 0
# ----------------------------


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError:
        print("Missing dependency: pyyaml (python3 -m pip install pyyaml)")
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        print(f"Failed to read config: {path} ({exc})")
        return {}


def _get_orient_h5_path() -> Path | None:
    if ORIENT_H5:
        return Path(ORIENT_H5)
    cfg_path = Path(ORIENT_CFG)
    if not cfg_path.exists():
        return None
    cfg = _load_yaml(cfg_path)
    out = cfg.get("output", {}) if isinstance(cfg, dict) else {}
    path = out.get("path")
    if path:
        return Path(path)
    return None


def load_grid_meta(path):
    with h5py.File(path, "r") as f:
        origin = f["/grid/origin"][:]
        dims = f["/grid/dims"][:]
        voxel = float(f["/grid/voxel_size"][0])
        label = f["/grid/label"][:] if "/grid/label" in f else None
        sdf = f["/grid/sdf"][:] if "/grid/sdf" in f else None
    return origin, dims, voxel, label, sdf


def load_anchors(path: Path):
    with h5py.File(path, "r") as f:
        if "/anchors/pos" not in f:
            raise RuntimeError("No /anchors/pos in orientation dataset.")
        pos = f["/anchors/pos"][:]
        meta = {}
        for key in ("s_v", "c_v", "g_v", "n_seed"):
            if f"/anchors/{key}" in f:
                meta[key] = f[f"/anchors/{key}"][:]
    return pos, meta


def _sample_boundary_points(label, sdf, origin, dims, voxel, max_points, boundary_band, inside_only, seed):
    rng = np.random.default_rng(seed)
    nx, ny, nz = int(dims[0]), int(dims[1]), int(dims[2])
    total = nx * ny * nz
    keep = []
    seen = 0

    if label is not None:
        data = label.reshape(-1)
        if np.any(data == 0):
            boundary_mask = (data == 0)
            idxs = np.flatnonzero(boundary_mask)
        else:
            data = None
            idxs = None
    elif sdf is not None:
        data = sdf.reshape(-1)
        boundary_mask = np.abs(data) <= boundary_band
        if inside_only:
            boundary_mask &= (data > 0)
        idxs = np.flatnonzero(boundary_mask)
    else:
        return np.zeros((0, 3), dtype=np.float32)

    if idxs is None:
        return np.zeros((0, 3), dtype=np.float32)

    for idx in idxs:
        seen += 1
        if len(keep) < max_points:
            keep.append(int(idx))
        else:
            j = rng.integers(0, seen)
            if j < max_points:
                keep[j] = int(idx)

    if not keep:
        return np.zeros((0, 3), dtype=np.float32)
    keep = np.asarray(keep, dtype=np.int64)
    x = keep // (ny * nz)
    rem = keep % (ny * nz)
    y = rem // nz
    z = rem % nz
    pts = np.stack([x, y, z], axis=1).astype(np.float64)
    return (origin + voxel * (pts + 0.5)).astype(np.float32)


def main():
    origin, dims, voxel, label, sdf = load_grid_meta(BASE_H5)
    boundary_band = BOUNDARY_BAND if BOUNDARY_BAND is not None else 2.0 * voxel
    pts = _sample_boundary_points(label, sdf, origin, dims, voxel, MAX_BOUNDARY_POINTS,
                                  boundary_band, BOUNDARY_USE_INSIDE_ONLY, RANDOM_SEED)
    if pts.size == 0:
        print("No boundary voxels found with current settings.")
        return 1

    orient_path = _get_orient_h5_path()
    anchors = None
    meta = {}
    if USE_ORIENT_H5_IF_AVAILABLE and orient_path and orient_path.exists():
        anchors, meta = load_anchors(orient_path)
    else:
        raise RuntimeError(
            "Orientation anchors not found. Run orientation_dataset_cli to generate /anchors/pos "
            "or set USE_ORIENT_H5_IF_AVAILABLE=True with a valid output path."
        )
    rng = np.random.default_rng(RANDOM_SEED)
    if ANCHOR_MAX_POINTS > 0 and anchors.shape[0] > ANCHOR_MAX_POINTS:
        pick = rng.choice(anchors.shape[0], size=ANCHOR_MAX_POINTS, replace=False)
        anchors = anchors[pick]
        for key in meta:
            meta[key] = meta[key][pick]

    if USE_PYVISTA:
        try:
            import pyvista as pv
        except ImportError:
            print("Missing dependency: pyvista (python3 -m pip install pyvista)")
            return 1

        plotter = pv.Plotter()
        boundary_cloud = pv.PolyData(pts)
        plotter.add_points(boundary_cloud, color="gray", opacity=BOUNDARY_ALPHA, point_size=BOUNDARY_SIZE)

        anchor_cloud = pv.PolyData(anchors)
        if ANCHOR_COLOR_BY != "none" and ANCHOR_COLOR_BY in meta:
            values = meta[ANCHOR_COLOR_BY].astype(np.float32)
            anchor_cloud[ANCHOR_COLOR_BY] = values
            if USE_SPHERE_ANCHORS:
                spheres = anchor_cloud.glyph(scale=False, geom=pv.Sphere(radius=ANCHOR_RADIUS))
                plotter.add_mesh(
                    spheres,
                    scalars=ANCHOR_COLOR_BY,
                    cmap="viridis",
                    opacity=ANCHOR_ALPHA,
                )
            else:
                plotter.add_points(
                    anchor_cloud,
                    scalars=ANCHOR_COLOR_BY,
                    cmap="viridis",
                    opacity=ANCHOR_ALPHA,
                    point_size=ANCHOR_SIZE,
                )
        else:
            if USE_SPHERE_ANCHORS:
                spheres = anchor_cloud.glyph(scale=False, geom=pv.Sphere(radius=ANCHOR_RADIUS))
                plotter.add_mesh(spheres, color="red", opacity=ANCHOR_ALPHA)
            else:
                plotter.add_points(anchor_cloud, color="red", opacity=ANCHOR_ALPHA, point_size=ANCHOR_SIZE)

        if SHOW_ANCHOR_INDEX:
            labels = [str(i) for i in range(anchors.shape[0])]
            plotter.add_point_labels(anchors, labels, font_size=10, point_size=ANCHOR_SIZE, text_color="black")

        plotter.add_axes()
        plotter.add_title("Position boundary + selected anchors")
        plotter.show()
    else:
        print("USE_PYVISTA is False. Enable it for interactive 3D.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
