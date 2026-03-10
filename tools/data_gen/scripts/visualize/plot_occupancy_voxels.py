#!/usr/bin/env python3
"""Script: plot_occupancy_voxels.py
Purpose: Inspect voxel labels, occupancy statistics, and related fields from a generated dataset.
Usage: python3 tools/data_gen/scripts/visualize/plot_occupancy_voxels.py
"""

import os
import sys

import numpy as np


# ======== CONFIG (edit here) ========
INPUT_DIR = "tools/data_gen/outputs/aubo/aubo_i5/voxel_3mm"
HDF5_PATH = "tools/data_gen/outputs/aubo/aubo_i5/voxel_3mm/dataset.h5"
USE_HDF5 = True
RANDOM_SEED = 42

SHOW_OCCUPANCY = False
SHOW_COUNTS = False
SHOW_BOUNDARY = False
SHOW_UNIFORM = False
SHOW_ORIENTATION_COVERAGE = False
SHOW_FARFIELD_BUCKETS = False
SHOW_OUTER_SHELL = False
SHOW_LABELS = True

MAX_DIM = 200

OCCUPANCY_OPACITY = 0.2
COUNTS_OPACITY = 0.35
COUNTS_LOG_SCALE = True

ORIENT_COVERAGE_OPACITY = 0.45
ORIENT_COVERAGE_MIN = 0.001
ORIENT_COVERAGE_CMAP = "viridis"

BOUNDARY_POINT_SIZE = 4.0
BOUNDARY_COLOR = (1.0, 0.6, 0.2)
BOUNDARY_OPACITY = 0.6
BOUNDARY_MAX_POINTS = 200000

UNIFORM_POINT_SIZE = 3.0
UNIFORM_POS_COLOR = (0.2, 0.6, 1.0)
UNIFORM_NEG_COLOR = (1.0, 0.3, 0.3)
UNIFORM_MIN_ALPHA = 0.05
UNIFORM_MAX_ALPHA = 0.9
UNIFORM_MAX_POINTS = 300000

SHOW_AXES = True

# Farfield bucket/tier visualization (requires HDF5 + /grid/farfield_*)
FARFIELD_MODE = "tier"  # "tier" or "bucket"
FARFIELD_TIERS = [1, 2, 3, 4, 5]  # near-field tiers; far-far uses /outer_grid
FARFIELD_BUCKETS = [(0, 0, 0), (2, 2, 2), (4, 4, 4)]
FARFIELD_POINTS_PER_GROUP = 5000
FARFIELD_OCCUPANCY_FILTER = "all"  # "outside" | "inside" | "all"
FARFIELD_POINT_SIZE = 4.0
FARFIELD_OPACITY = 0.9
FARFIELD_COLORS = [
    (0.91, 0.29, 0.24),  # red
    (0.95, 0.61, 0.07),  # orange
    (0.18, 0.80, 0.44),  # green
    (0.20, 0.55, 0.90),  # blue
    (0.65, 0.41, 0.85),  # purple
    (0.60, 0.60, 0.60),  # gray
]

OUTER_POINT_SIZE = 4.0
OUTER_OPACITY = 0.6
OUTER_COLOR = (0.9, 0.3, 0.9)
OUTER_MAX_POINTS = 200000

LABEL_MAX_POINTS = 1000000
LABEL0_COLOR = (0.85, 0.85, 0.85)
LABEL0_OPACITY = 0.6
LABEL0_POINT_SIZE = 3.0
LABEL_INSIDE_CMAP = "viridis"
LABEL_OUTSIDE_CMAP = "magma"
LABEL_MIN_OPACITY = 0.2
LABEL_MAX_OPACITY = 0.9
LABEL_SLICE_COORD = 0.0  # world coordinate for x/y/z slices
LABEL_SLICE_HALF_WIDTH = 0.01 # meters; if None -> 0.5 * voxel_size
LABEL_USE_MPL = False  # label 1/2: matplotlib 2x2 figure
LABEL_USE_PV_GRID = True  # label 1/2: pyvista 2x2 interactive figure
LABEL_SHOW_PV_FOR_1_2 = True # set True to also show pyvista 3D for label 1/2
# ======== CONFIG END ========


def load_optional(path, default, mmap=True):
    if os.path.exists(path):
        arr = np.load(path, mmap_mode="r" if mmap else None)
        if arr.size == 1:
            return float(arr.reshape(-1)[0])
        return np.asarray(arr)
    return default


def load_from_hdf5(path):
    try:
        import h5py
    except ImportError:
        print("Missing dependency: h5py", file=sys.stderr)
        print("Install with: pip install h5py", file=sys.stderr)
        return None

    if not os.path.exists(path):
        return None

    with h5py.File(path, "r") as f:
        if "/grid/label" in f:
            occupancy = f["/grid/label"][...]
        else:
            occupancy = f["/grid/occupancy"][...]
        counts = f["/grid/voxel_counts"][...] if "/grid/voxel_counts" in f else None
        boundary = f["/samples/boundary"][...] if "/samples/boundary" in f else None
        uniform = f["/samples/uniform"][...] if "/samples/uniform" in f else None
        coverage = (
            f["/grid/orientation_coverage"][...] if "/grid/orientation_coverage" in f else None
        )
        farfield_tier = f["/grid/farfield_tier"][...] if "/grid/farfield_tier" in f else None
        farfield_bucket = (
            f["/grid/farfield_bucket"][...] if "/grid/farfield_bucket" in f else None
        )
        outer_mask = f["/outer_grid/mask"][...] if "/outer_grid/mask" in f else None
        outer_origin = f["/outer_grid/origin"][...] if "/outer_grid/origin" in f else None
        outer_voxel = f["/outer_grid/voxel_size"][...] if "/outer_grid/voxel_size" in f else None
        sdf = f["/grid/sdf"][...] if "/grid/sdf" in f else None
        origin = f["/grid/origin"][...] if "/grid/origin" in f else np.zeros(3)
        voxel_size = f["/grid/voxel_size"][...]
        if np.asarray(voxel_size).size == 1:
            voxel_size = float(np.asarray(voxel_size).reshape(-1)[0])
        else:
            voxel_size = float(np.asarray(voxel_size).ravel()[0])
        if outer_voxel is not None:
            if np.asarray(outer_voxel).size == 1:
                outer_voxel = float(np.asarray(outer_voxel).reshape(-1)[0])
            else:
                outer_voxel = float(np.asarray(outer_voxel).ravel()[0])
    return {
        "occupancy": occupancy,
        "counts": counts,
        "boundary": boundary,
        "uniform": uniform,
        "coverage": coverage,
        "farfield_tier": farfield_tier,
        "farfield_bucket": farfield_bucket,
        "sdf": sdf,
        "outer_mask": outer_mask,
        "outer_origin": None if outer_origin is None else np.asarray(outer_origin, dtype=np.float64),
        "outer_voxel": None if outer_voxel is None else float(outer_voxel),
        "origin": np.asarray(origin, dtype=np.float64),
        "voxel_size": float(voxel_size),
    }


def compute_stride(shape, max_dim):
    stride = int(np.ceil(max(shape) / float(max_dim)))
    return max(1, stride)


def downsample_volume(vol, stride):
    if stride <= 1:
        return np.asarray(vol)
    return np.asarray(vol[::stride, ::stride, ::stride])


def make_grid(shape, origin, spacing):
    try:
        import pyvista as pv
    except ImportError:
        print("Missing dependency: pyvista", file=sys.stderr)
        print("Install with: pip install pyvista", file=sys.stderr)
        return None

    nx, ny, nz = shape
    grid = pv.ImageData(
        dimensions=(nx + 1, ny + 1, nz + 1),
        spacing=(spacing, spacing, spacing),
        origin=origin,
    )
    return grid


def show_occupancy(occupancy, origin, spacing, stride):
    import pyvista as pv

    inside = occupancy == 1
    grid = make_grid(inside.shape, origin, spacing)
    if grid is None:
        return

    grid.cell_data["inside"] = inside.ravel(order="F")
    inside_cells = grid.threshold(value=0.5, scalars="inside")

    plotter = pv.Plotter()
    plotter.add_mesh(
        inside_cells,
        color=(0.12, 0.47, 0.71),
        opacity=OCCUPANCY_OPACITY,
        show_edges=False,
    )
    plotter.add_title(
        f"Occupancy (inside voxels), stride={stride} (voxel={spacing:.4f} m)"
    )
    if SHOW_AXES:
        plotter.show_axes()
    plotter.show()


def show_counts(counts, origin, spacing, stride):
    import pyvista as pv

    grid = make_grid(counts.shape, origin, spacing)
    if grid is None:
        return

    grid.cell_data["counts"] = counts.ravel(order="F")
    nonzero = grid.threshold(value=1, scalars="counts")
    if nonzero.n_cells == 0:
        print("No nonzero counts to visualize.")
        return

    plotter = pv.Plotter()
    plotter.add_mesh(
        nonzero,
        scalars="counts",
        cmap="viridis",
        opacity=COUNTS_OPACITY,
        log_scale=COUNTS_LOG_SCALE,
        show_edges=False,
    )
    plotter.add_title(
        f"Voxel counts (stride={stride}, voxel={spacing:.4f} m)"
    )
    if SHOW_AXES:
        plotter.show_axes()
    plotter.show()


def show_orientation_coverage(coverage, origin, spacing, stride):
    import pyvista as pv

    grid = make_grid(coverage.shape, origin, spacing)
    if grid is None:
        return

    grid.cell_data["coverage"] = coverage.ravel(order="F")
    visible = grid.threshold(value=ORIENT_COVERAGE_MIN, scalars="coverage")
    if visible.n_cells == 0:
        print("No orientation coverage values above threshold.")
        return

    plotter = pv.Plotter()
    plotter.add_mesh(
        visible,
        scalars="coverage",
        cmap=ORIENT_COVERAGE_CMAP,
        opacity=ORIENT_COVERAGE_OPACITY,
        clim=(0.0, 1.0),
        show_edges=False,
        show_scalar_bar=True,
    )
    plotter.add_title(
        f"Orientation coverage (stride={stride}, voxel={spacing:.4f} m)"
    )
    if SHOW_AXES:
        plotter.show_axes()
    plotter.show()


def show_boundary(boundary_samples, rng):
    import pyvista as pv

    if boundary_samples.size == 0:
        print("No boundary samples to visualize.")
        return

    points = boundary_samples[:, 6:9]
    if points.shape[0] > BOUNDARY_MAX_POINTS:
        choice = rng.choice(points.shape[0], size=BOUNDARY_MAX_POINTS, replace=False)
        points = points[choice]

    poly = pv.PolyData(points)
    plotter = pv.Plotter()
    plotter.add_mesh(
        poly,
        color=BOUNDARY_COLOR,
        opacity=BOUNDARY_OPACITY,
        point_size=BOUNDARY_POINT_SIZE,
        render_points_as_spheres=True,
    )
    plotter.add_title(f"Boundary samples (n={points.shape[0]})")
    if SHOW_AXES:
        plotter.show_axes()
    plotter.show()


def show_uniform(uniform_samples, rng):
    import pyvista as pv

    if uniform_samples.size == 0:
        print("No uniform samples to visualize.")
        return

    if uniform_samples.shape[0] > UNIFORM_MAX_POINTS:
        choice = rng.choice(
            uniform_samples.shape[0], size=UNIFORM_MAX_POINTS, replace=False
        )
        uniform_samples = uniform_samples[choice]

    pts = uniform_samples[:, :3]
    s = uniform_samples[:, 3]
    s_abs = np.abs(s)
    max_abs = np.max(s_abs) if s_abs.size > 0 else 1.0
    alpha = s_abs / max_abs
    alpha = np.clip(alpha, UNIFORM_MIN_ALPHA, UNIFORM_MAX_ALPHA)

    pos_mask = s >= 0
    neg_mask = ~pos_mask

    def add_cloud(mask, color):
        if not np.any(mask):
            return
        cloud = pv.PolyData(pts[mask])
        rgba = np.zeros((cloud.n_points, 4), dtype=np.uint8)
        rgb = (np.array(color) * 255.0).astype(np.uint8)
        rgba[:, 0] = rgb[0]
        rgba[:, 1] = rgb[1]
        rgba[:, 2] = rgb[2]
        rgba[:, 3] = (alpha[mask] * 255.0).astype(np.uint8)
        cloud.point_data["rgba"] = rgba
        plotter.add_mesh(
            cloud,
            scalars="rgba",
            rgba=True,
            point_size=UNIFORM_POINT_SIZE,
            render_points_as_spheres=True,
        )

    plotter = pv.Plotter()
    add_cloud(pos_mask, UNIFORM_POS_COLOR)
    add_cloud(neg_mask, UNIFORM_NEG_COLOR)
    plotter.add_title(
        f"Uniform samples (+s blue / -s red, n={uniform_samples.shape[0]})"
    )
    if SHOW_AXES:
        plotter.show_axes()
    plotter.show()


def voxel_centers(indices, origin, spacing):
    if indices.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    centers = (indices.astype(np.float64) + 0.5) * float(spacing) + origin.reshape(1, 3)
    return centers.astype(np.float32)


def sample_indices(mask, rng, max_points):
    flat = np.flatnonzero(mask)
    if flat.size == 0:
        return np.zeros((0, 3), dtype=np.int64)
    if flat.size > max_points:
        chosen = rng.choice(flat, size=max_points, replace=False)
    else:
        chosen = flat
    idx = np.array(np.unravel_index(chosen, mask.shape)).T
    return idx


def show_farfield_buckets(tier, bucket, occupancy, origin, spacing, rng):
    import pyvista as pv

    if tier is None and bucket is None:
        print("No farfield tiers/buckets available.")
        return

    outside_value = 0
    if occupancy.max() >= 2:
        outside_value = 2

    if FARFIELD_OCCUPANCY_FILTER == "outside":
        occ_mask = occupancy == outside_value
    elif FARFIELD_OCCUPANCY_FILTER == "inside":
        occ_mask = occupancy == 1
    else:
        occ_mask = None

    plotter = pv.Plotter()
    legend = []
    color_idx = 0

    if FARFIELD_MODE == "tier":
        for t in FARFIELD_TIERS:
            mask = tier == t
            if occ_mask is not None:
                mask = mask & occ_mask
            idx = sample_indices(mask, rng, FARFIELD_POINTS_PER_GROUP)
            if idx.size == 0:
                continue
            pts = voxel_centers(idx, origin, spacing)
            color = FARFIELD_COLORS[color_idx % len(FARFIELD_COLORS)]
            color_idx += 1
            poly = pv.PolyData(pts)
            plotter.add_mesh(
                poly,
                color=color,
                opacity=FARFIELD_OPACITY,
                point_size=FARFIELD_POINT_SIZE,
                render_points_as_spheres=True,
            )
            legend.append([f"tier={t}", color])
    else:
        for b in FARFIELD_BUCKETS:
            bx, by, bz = b
            mask = (
                (bucket[..., 0] == bx)
                & (bucket[..., 1] == by)
                & (bucket[..., 2] == bz)
            )
            if occ_mask is not None:
                mask = mask & occ_mask
            idx = sample_indices(mask, rng, FARFIELD_POINTS_PER_GROUP)
            if idx.size == 0:
                continue
            pts = voxel_centers(idx, origin, spacing)
            color = FARFIELD_COLORS[color_idx % len(FARFIELD_COLORS)]
            color_idx += 1
            poly = pv.PolyData(pts)
            plotter.add_mesh(
                poly,
                color=color,
                opacity=FARFIELD_OPACITY,
                point_size=FARFIELD_POINT_SIZE,
                render_points_as_spheres=True,
            )
            legend.append([f"bucket={b}", color])

    if not legend:
        print("No farfield points matched the filter.")
        return

    plotter.add_title(
        f"Farfield {FARFIELD_MODE} samples (n≈{FARFIELD_POINTS_PER_GROUP} each)"
    )
    plotter.add_legend(legend)
    if SHOW_AXES:
        plotter.show_axes()
    plotter.show()


def show_outer_shell(mask, origin, spacing, rng):
    import pyvista as pv

    if mask is None:
        print("No outer_grid/mask found.")
        return
    idx = sample_indices(mask == 1, rng, OUTER_MAX_POINTS)
    if idx.size == 0:
        print("No outer shell voxels to visualize.")
        return

    pts = voxel_centers(idx, origin, spacing)
    poly = pv.PolyData(pts)
    plotter = pv.Plotter()
    plotter.add_mesh(
        poly,
        color=OUTER_COLOR,
        opacity=OUTER_OPACITY,
        point_size=OUTER_POINT_SIZE,
        render_points_as_spheres=True,
    )
    plotter.add_title(f"Outer shell (far-far field) n={pts.shape[0]}")
    if SHOW_AXES:
        plotter.show_axes()
    plotter.show()


def show_label_points(labels, sdf, origin, spacing, rng):
    import pyvista as pv
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None

    def sample_label(value):
        mask = labels == value
        idx = sample_indices(mask, rng, LABEL_MAX_POINTS)
        if idx.size == 0:
            return None, None
        pts = voxel_centers(idx, origin, spacing)
        lin = np.ravel_multi_index(
            (idx[:, 0], idx[:, 1], idx[:, 2]),
            labels.shape,
            mode="clip",
        )
        return pts, lin

    # label 0 (boundary)
    pts0, _ = sample_label(0)
    if pts0 is not None:
        plotter = pv.Plotter()
        poly = pv.PolyData(pts0)
        plotter.add_mesh(
            poly,
            color=LABEL0_COLOR,
            opacity=LABEL0_OPACITY,
            point_size=LABEL0_POINT_SIZE,
            render_points_as_spheres=True,
        )
        plotter.add_title(f"Label 0 (boundary), n={pts0.shape[0]}")
        if SHOW_AXES:
            plotter.show_axes()
        plotter.show()

    def show_mpl(label_value, pts, s_abs, cmap, title):
        if plt is None:
            print("matplotlib not available; skip slice plots.")
            return
        slice_half = (
            LABEL_SLICE_HALF_WIDTH
            if LABEL_SLICE_HALF_WIDTH is not None
            else 0.5 * spacing
        )
        x0 = LABEL_SLICE_COORD
        y0 = LABEL_SLICE_COORD
        z0 = LABEL_SLICE_COORD

        fig = plt.figure(figsize=(12, 9))
        ax0 = fig.add_subplot(2, 2, 1, projection="3d")
        ax0.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=s_abs, cmap=cmap, s=2, alpha=0.6)
        ax0.set_title(f"{title} (overall)")
        ax0.set_xlabel("x")
        ax0.set_ylabel("y")
        ax0.set_zlabel("z")

        # x slice (x ~ x0): plot y vs z
        mask_x = np.abs(pts[:, 0] - x0) <= slice_half
        ax1 = fig.add_subplot(2, 2, 2)
        ax1.scatter(pts[mask_x, 1], pts[mask_x, 2], c=s_abs[mask_x], cmap=cmap, s=3, alpha=0.7)
        ax1.set_title(f"x={x0:.3f} ± {slice_half:.4f} (n={mask_x.sum()})")
        ax1.set_xlabel("y")
        ax1.set_ylabel("z")

        # y slice (y ~ y0): plot x vs z
        mask_y = np.abs(pts[:, 1] - y0) <= slice_half
        ax2 = fig.add_subplot(2, 2, 3)
        ax2.scatter(pts[mask_y, 0], pts[mask_y, 2], c=s_abs[mask_y], cmap=cmap, s=3, alpha=0.7)
        ax2.set_title(f"y={y0:.3f} ± {slice_half:.4f} (n={mask_y.sum()})")
        ax2.set_xlabel("x")
        ax2.set_ylabel("z")

        # z slice (z ~ z0): plot x vs y
        mask_z = np.abs(pts[:, 2] - z0) <= slice_half
        ax3 = fig.add_subplot(2, 2, 4)
        ax3.scatter(pts[mask_z, 0], pts[mask_z, 1], c=s_abs[mask_z], cmap=cmap, s=3, alpha=0.7)
        ax3.set_title(f"z={z0:.3f} ± {slice_half:.4f} (n={mask_z.sum()})")
        ax3.set_xlabel("x")
        ax3.set_ylabel("y")

        fig.suptitle(title)
        fig.tight_layout()
        plt.show()

    def show_pv_grid(pts, s_abs, cmap, title):
        slice_half = (
            LABEL_SLICE_HALF_WIDTH
            if LABEL_SLICE_HALF_WIDTH is not None
            else 0.5 * spacing
        )
        x0 = LABEL_SLICE_COORD
        y0 = LABEL_SLICE_COORD
        z0 = LABEL_SLICE_COORD

        plotter = pv.Plotter(shape=(2, 2))

        # Overall 3D
        plotter.subplot(0, 0)
        poly = pv.PolyData(pts)
        poly.point_data["s_abs"] = s_abs
        plotter.add_mesh(
            poly,
            scalars="s_abs",
            cmap=cmap,
            opacity=LABEL_MAX_OPACITY,
            point_size=LABEL0_POINT_SIZE,
            render_points_as_spheres=True,
        )
        plotter.add_title(f"{title} (overall)")
        if SHOW_AXES:
            plotter.show_axes()

        # x slice (x ~ x0): y-z
        plotter.subplot(0, 1)
        mask_x = np.abs(pts[:, 0] - x0) <= slice_half
        poly_x = pv.PolyData(pts[mask_x])
        if poly_x.n_points > 0:
            poly_x.point_data["s_abs"] = s_abs[mask_x]
            plotter.add_mesh(
                poly_x,
                scalars="s_abs",
                cmap=cmap,
                opacity=LABEL_MAX_OPACITY,
                point_size=LABEL0_POINT_SIZE,
                render_points_as_spheres=True,
            )
        plotter.add_title(f"x={x0:.3f} ± {slice_half:.4f} (n={mask_x.sum()})")
        if SHOW_AXES:
            plotter.show_axes()

        # y slice (y ~ y0): x-z
        plotter.subplot(1, 0)
        mask_y = np.abs(pts[:, 1] - y0) <= slice_half
        poly_y = pv.PolyData(pts[mask_y])
        if poly_y.n_points > 0:
            poly_y.point_data["s_abs"] = s_abs[mask_y]
            plotter.add_mesh(
                poly_y,
                scalars="s_abs",
                cmap=cmap,
                opacity=LABEL_MAX_OPACITY,
                point_size=LABEL0_POINT_SIZE,
                render_points_as_spheres=True,
            )
        plotter.add_title(f"y={y0:.3f} ± {slice_half:.4f} (n={mask_y.sum()})")
        if SHOW_AXES:
            plotter.show_axes()

        # z slice (z ~ z0): x-y
        plotter.subplot(1, 1)
        mask_z = np.abs(pts[:, 2] - z0) <= slice_half
        poly_z = pv.PolyData(pts[mask_z])
        if poly_z.n_points > 0:
            poly_z.point_data["s_abs"] = s_abs[mask_z]
            plotter.add_mesh(
                poly_z,
                scalars="s_abs",
                cmap=cmap,
                opacity=LABEL_MAX_OPACITY,
                point_size=LABEL0_POINT_SIZE,
                render_points_as_spheres=True,
            )
        plotter.add_title(f"z={z0:.3f} ± {slice_half:.4f} (n={mask_z.sum()})")
        if SHOW_AXES:
            plotter.show_axes()

        plotter.show()

    # label 1 (inside) colored by |s|
    pts1, lin1 = sample_label(1)
    if pts1 is not None:
        s1 = np.abs(sdf.ravel()[lin1]) if sdf is not None else np.zeros(pts1.shape[0], dtype=np.float32)
        if LABEL_USE_MPL:
            show_mpl(1, pts1, s1, LABEL_INSIDE_CMAP, f"Label 1 (inside) |s|")
        if LABEL_USE_PV_GRID:
            show_pv_grid(pts1, s1, LABEL_INSIDE_CMAP, "Label 1 (inside) |s|")
        if LABEL_SHOW_PV_FOR_1_2:
            plotter = pv.Plotter()
            poly = pv.PolyData(pts1)
            poly.point_data["s_abs"] = s1
            plotter.add_mesh(
                poly,
                scalars="s_abs",
                cmap=LABEL_INSIDE_CMAP,
                opacity=LABEL_MAX_OPACITY,
                point_size=LABEL0_POINT_SIZE,
                render_points_as_spheres=True,
            )
            plotter.add_title(f"Label 1 (inside) |s| colored, n={pts1.shape[0]}")
            if SHOW_AXES:
                plotter.show_axes()
            plotter.show()

    # label 2 (outside) colored by |s|
    pts2, lin2 = sample_label(2)
    if pts2 is not None:
        s2 = np.abs(sdf.ravel()[lin2]) if sdf is not None else np.zeros(pts2.shape[0], dtype=np.float32)
        if LABEL_USE_MPL:
            show_mpl(2, pts2, s2, LABEL_OUTSIDE_CMAP, f"Label 2 (outside) |s|")
        if LABEL_USE_PV_GRID:
            show_pv_grid(pts2, s2, LABEL_OUTSIDE_CMAP, "Label 2 (outside) |s|")
        if LABEL_SHOW_PV_FOR_1_2:
            plotter = pv.Plotter()
            poly = pv.PolyData(pts2)
            poly.point_data["s_abs"] = s2
            plotter.add_mesh(
                poly,
                scalars="s_abs",
                cmap=LABEL_OUTSIDE_CMAP,
                opacity=LABEL_MAX_OPACITY,
                point_size=LABEL0_POINT_SIZE,
                render_points_as_spheres=True,
            )
            plotter.add_title(f"Label 2 (outside) |s| colored, n={pts2.shape[0]}")
            if SHOW_AXES:
                plotter.show_axes()
            plotter.show()


def main():
    try:
        import pyvista  # noqa: F401
    except ImportError:
        print("Missing dependency: pyvista", file=sys.stderr)
        print("Install with: pip install pyvista", file=sys.stderr)
        return 1

    use_hdf5 = USE_HDF5 and os.path.exists(HDF5_PATH)
    if use_hdf5:
        data = load_from_hdf5(HDF5_PATH)
        if data is None:
            return 1
        occupancy = data["occupancy"]
        counts = data["counts"]
        boundary_samples = data["boundary"]
        uniform_samples = data["uniform"]
        coverage = data["coverage"]
        farfield_tier = data["farfield_tier"]
        farfield_bucket = data["farfield_bucket"]
        outer_mask = data["outer_mask"]
        outer_origin = data["outer_origin"]
        outer_voxel = data["outer_voxel"]
        sdf = data["sdf"]
        origin = data["origin"]
        voxel_size = data["voxel_size"]
    else:
        occ_path = os.path.join(INPUT_DIR, "label.npy")
        fallback_occ = os.path.join(INPUT_DIR, "occupancy.npy")
        counts_path = os.path.join(INPUT_DIR, "voxel_counts.npy")
        boundary_path = os.path.join(INPUT_DIR, "boundary_samples.npy")
        uniform_path = os.path.join(INPUT_DIR, "uniform_samples.npy")
        orient_cov_path = os.path.join(INPUT_DIR, "orientation_coverage.npy")
        origin_path = os.path.join(INPUT_DIR, "origin.npy")
        voxel_path = os.path.join(INPUT_DIR, "voxel_size.npy")

        if not os.path.exists(occ_path):
            if os.path.exists(fallback_occ):
                occ_path = fallback_occ
            else:
                print("Missing label.npy at:", occ_path, file=sys.stderr)
                return 1

        origin = load_optional(origin_path, np.zeros(3), mmap=False).astype(np.float64)
        voxel_size = float(load_optional(voxel_path, 1.0, mmap=False))

        occupancy = load_optional(occ_path, None, mmap=True)
        if occupancy is None:
            print("Failed to load label.npy", file=sys.stderr)
            return 1
        counts = load_optional(counts_path, None, mmap=True) if os.path.exists(counts_path) else None
        boundary_samples = load_optional(boundary_path, None, mmap=False) if os.path.exists(boundary_path) else None
        uniform_samples = load_optional(uniform_path, None, mmap=False) if os.path.exists(uniform_path) else None
        coverage = load_optional(orient_cov_path, None, mmap=True) if os.path.exists(orient_cov_path) else None
        farfield_tier = None
        farfield_bucket = None
        outer_mask = None
        outer_origin = None
        outer_voxel = None
        sdf = None

    stride = compute_stride(occupancy.shape, MAX_DIM)
    spacing = voxel_size * stride

    rng = np.random.default_rng(RANDOM_SEED)

    if SHOW_OCCUPANCY:
        occ_ds = downsample_volume(occupancy, stride)
        show_occupancy(occ_ds, origin, spacing, stride)

    if SHOW_COUNTS and counts is not None:
        counts_ds = downsample_volume(counts, stride).astype(np.float32)
        show_counts(counts_ds, origin, spacing, stride)

    if SHOW_ORIENTATION_COVERAGE and coverage is not None:
        coverage_ds = downsample_volume(coverage, stride).astype(np.float32)
        show_orientation_coverage(coverage_ds, origin, spacing, stride)

    if SHOW_BOUNDARY and boundary_samples is not None and boundary_samples.ndim == 2:
        show_boundary(boundary_samples, rng)

    if SHOW_UNIFORM and uniform_samples is not None and uniform_samples.ndim == 2:
        show_uniform(uniform_samples, rng)

    if SHOW_LABELS:
        show_label_points(occupancy, sdf, origin, voxel_size, rng)

    if SHOW_FARFIELD_BUCKETS and farfield_tier is not None:
        if FARFIELD_MODE == "bucket" and farfield_bucket is None:
            print("Farfield bucket data not found in HDF5.")
        else:
            show_farfield_buckets(
                farfield_tier,
                farfield_bucket,
                occupancy,
                origin,
                voxel_size,
                rng,
            )

    if SHOW_OUTER_SHELL and outer_mask is not None:
        show_outer_shell(
            outer_mask,
            outer_origin if outer_origin is not None else origin,
            outer_voxel if outer_voxel is not None else voxel_size,
            rng,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
