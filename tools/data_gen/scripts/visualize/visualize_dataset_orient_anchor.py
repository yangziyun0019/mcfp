#!/usr/bin/env python3
"""Script: visualize_dataset_orient_anchor.py
Purpose: Render a detailed anchor-level view of orientation samples and nearby position boundaries.
Usage: python3 tools/data_gen/scripts/visualize/visualize_dataset_orient_anchor.py
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple
from types import SimpleNamespace

import numpy as np

# =========================
# CONFIG (edit here)
# =========================
CONFIG = {
    # anchor to inspect
    "anchor_id": 2000,
    # datasets
    "orient_h5": "tools/data_gen/outputs/aubo/aubo_i5/voxel_3mm/dataset_orient.h5",
    "pos_h5": "tools/data_gen/outputs/aubo/aubo_i5/voxel_3mm/dataset.h5",
    # output
    "out_dir": "tools/data_gen/outputs/aubo/aubo_i5/voxel_3mm/orient_vis",
    "seed": 42,
    # visualization params
    "tool_axis": "z",  # x/y/z
    "B_dir": 1024,
    "N_psi": 36,
    "N_slice": 8,
    "min_count_per_dir_bin": 5,
    "max_boundary_points": 50000,
    "boundary_alpha": 0.08,
    "boundary_stride": 2,
    "show_aabb": True,
    "max_rotvec_points": 30000,
    "rotvec_color": "phi",  # phi or label
    # matplotlib backend (None = default). Example: "Qt5Agg"
    "backend": None,
    # show interactive windows
    "show": False,
    # pyvista 3D interactive window (V0 + V3 overlay)
    "show_pyvista": True,
    "pv_sphere_radius": 0.12,  # meters
    "pv_point_size": 6.0,
    "pv_boundary_point_size": 2.0,
    "pv_sphere_alpha": 0.15,
    "pv_min_count": 5,
    # global mode: random anchors + slider to toggle V3
    "global_mode": False,
    "global_num_anchors": 100,
    "global_seed": 42,
    "global_out_name": "V0V3_global_slider.png",
    "global_html_name": "V0V3_global_slider.html",
    "global_export_html": False,
    "global_html_backend": "plotly",  # plotly | pyvista
    # global mode overrides (lighter sampling)
    "global_B_dir": 512,
    "global_N_psi": 18,
    "global_min_count": 3,
    "global_max_boundary_points": 30000,
    "global_boundary_stride": 3,
    "global_pv_point_size": 5.0,
    "global_pv_sphere_radius": 0.08,
    "global_anchor_point_size": 14.0,
}


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    n = np.linalg.norm(q, axis=1, keepdims=True)
    return q / np.clip(n, 1e-9, None)


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    q = _normalize_quat(q)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    r00 = 1.0 - 2.0 * (yy + zz)
    r01 = 2.0 * (xy - wz)
    r02 = 2.0 * (xz + wy)
    r10 = 2.0 * (xy + wz)
    r11 = 1.0 - 2.0 * (xx + zz)
    r12 = 2.0 * (yz - wx)
    r20 = 2.0 * (xz - wy)
    r21 = 2.0 * (yz + wx)
    r22 = 1.0 - 2.0 * (xx + yy)
    rot = np.stack([
        np.stack([r00, r01, r02], axis=1),
        np.stack([r10, r11, r12], axis=1),
        np.stack([r20, r21, r22], axis=1),
    ], axis=1)
    return rot


def _compute_tool_dir(q: np.ndarray, tool_axis: str) -> Tuple[np.ndarray, np.ndarray]:
    rot = _quat_to_rotmat(q)
    if tool_axis == "x":
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    elif tool_axis == "y":
        axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    else:
        axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    n = rot @ axis
    x_axis = rot @ np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return n.astype(np.float32), x_axis.astype(np.float32)


def _compute_twist(n: np.ndarray, x_axis: np.ndarray) -> np.ndarray:
    ex = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    ey = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    dot_ex = np.abs(np.sum(n * ex, axis=1))
    use_ey = dot_ex > 0.95
    a = np.where(use_ey[:, None], ey[None, :], ex[None, :])
    proj = a - np.sum(a * n, axis=1, keepdims=True) * n
    proj_norm = np.linalg.norm(proj, axis=1, keepdims=True)
    proj_norm = np.clip(proj_norm, 1e-9, None)
    x0 = proj / proj_norm
    y0 = np.cross(n, x0)
    x_dot_x0 = np.sum(x_axis * x0, axis=1)
    x_dot_y0 = np.sum(x_axis * y0, axis=1)
    psi = np.arctan2(x_dot_y0, x_dot_x0)
    return psi.astype(np.float32)


def _rotvec_from_quat(q: np.ndarray) -> np.ndarray:
    q = _normalize_quat(q)
    w = q[:, 3:4]
    flip = w < 0.0
    q = np.where(flip, -q, q)
    w = np.clip(q[:, 3], -1.0, 1.0)
    theta = 2.0 * np.arccos(w)
    sin_half = np.sin(theta / 2.0)
    axis = q[:, :3] / np.clip(sin_half[:, None], 1e-9, None)
    axis = np.where(theta[:, None] < 1e-8, 0.0, axis)
    return axis * theta[:, None]


def _fibonacci_sphere(n: int) -> np.ndarray:
    n = int(n)
    idx = np.arange(n, dtype=np.float32)
    phi = (1.0 + 5.0 ** 0.5) / 2.0
    theta = 2.0 * math.pi * idx / phi
    z = 1.0 - (2.0 * idx + 1.0) / n
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def _dir_bins(n: np.ndarray, centers: np.ndarray) -> np.ndarray:
    dots = n @ centers.T
    return np.argmax(dots, axis=1).astype(np.int64)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_anchor_samples(orient_h5: Path, anchor_id: int) -> Dict[str, np.ndarray]:
    import h5py
    with h5py.File(orient_h5, "r") as h5:
        anchor_pos = np.asarray(h5["/anchors/pos"][anchor_id], dtype=np.float32)
        start = int(h5["/csr/anchor_start"][anchor_id])
        end = int(h5["/csr/anchor_start"][anchor_id + 1])
        if end <= start:
            raise RuntimeError("Empty anchor samples")
        if "/csr/sample_index" in h5:
            ids = np.asarray(h5["/csr/sample_index"][start:end], dtype=np.int64)
        else:
            ids = np.arange(start, end, dtype=np.int64)

        quat = np.asarray(h5["/samples/quat"][ids], dtype=np.float32)
        phi = np.asarray(h5["/samples/phi"][ids], dtype=np.float32)
        label = np.asarray(h5["/samples/label"][ids], dtype=np.int8)
        method = np.asarray(h5["/samples/method"][ids], dtype=np.uint8)
        bucket_r = np.asarray(h5["/samples/bucket_r"][ids], dtype=np.int64) if "/samples/bucket_r" in h5 else None
        q_ref = np.asarray(h5["/meta/q_ref"], dtype=np.float32) if "/meta/q_ref" in h5 else None
    # normalize labels to +/-1
    uniq = set(np.unique(label).tolist())
    if uniq <= {0, 1, 2}:
        label = np.where(label == 0, -1, 1).astype(np.int8)
    elif uniq <= {-1, 1}:
        label = label.astype(np.int8)
    else:
        label = np.where(label >= 0, 1, -1).astype(np.int8)
    return {
        "anchor_pos": anchor_pos,
        "quat": quat,
        "phi": phi,
        "label": label,
        "method": method,
        "bucket_r": bucket_r,
        "q_ref": q_ref,
    }


def _collect_boundary_points(
    pos_h5: Path,
    max_points: int,
    stride: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    import h5py
    rng = np.random.default_rng(int(seed))
    with h5py.File(pos_h5, "r") as h5:
        origin = np.asarray(h5["/grid/origin"], dtype=np.float64).reshape(3)
        dims = np.asarray(h5["/grid/dims"], dtype=np.int64).reshape(3)
        voxel = float(np.asarray(h5["/grid/voxel_size"], dtype=np.float64).reshape(-1)[0])
        label_ds = h5["/grid/label"] if "/grid/label" in h5 else None
        sdf_ds = h5["/grid/sdf"] if "/grid/sdf" in h5 else None

        points = []
        nx, ny, nz = [int(x) for x in dims]
        for i in range(0, nx, max(1, stride)):
            if label_ds is not None:
                if label_ds.ndim == 3:
                    slab = label_ds[i, ::stride, ::stride]
                else:
                    start = i * ny * nz
                    end = (i + 1) * ny * nz
                    slab = label_ds[start:end].reshape(ny, nz)[::stride, ::stride]
                idx = np.argwhere(slab == 0)
                if idx.size == 0:
                    continue
                jj = idx[:, 0] * stride
                kk = idx[:, 1] * stride
            elif sdf_ds is not None:
                if sdf_ds.ndim == 3:
                    slab = sdf_ds[i, ::stride, ::stride]
                else:
                    start = i * ny * nz
                    end = (i + 1) * ny * nz
                    slab = sdf_ds[start:end].reshape(ny, nz)[::stride, ::stride]
                idx = np.argwhere(np.abs(slab) <= 0.5 * voxel)
                if idx.size == 0:
                    continue
                jj = idx[:, 0] * stride
                kk = idx[:, 1] * stride
            else:
                break
            ii = np.full_like(jj, i)
            pts = np.stack([ii, jj, kk], axis=1).astype(np.float64)
            points.append(pts)

        if not points:
            return origin, dims, np.zeros((0, 3), dtype=np.float32), voxel
        ijk = np.vstack(points)
        if ijk.shape[0] > max_points:
            idx = rng.choice(ijk.shape[0], size=max_points, replace=False)
            ijk = ijk[idx]
        pts = origin[None, :] + (ijk + 0.5) * voxel
        return origin, dims, pts.astype(np.float32), voxel


def _sanity_check(phi: np.ndarray, label: np.ndarray, method: np.ndarray) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    methods = [0, 1, 2, 3]
    for m in methods:
        mask = method == m
        stats[f"count_m{m}"] = int(np.sum(mask))
        if np.any(mask):
            stats[f"reach_ratio_m{m}"] = float(np.mean(label[mask] > 0))
            p = phi[mask]
            stats[f"phi_min_m{m}"] = float(np.min(p))
            stats[f"phi_max_m{m}"] = float(np.max(p))
            stats[f"phi_mean_m{m}"] = float(np.mean(p))
            stats[f"phi_p10_m{m}"] = float(np.percentile(p, 10))
            stats[f"phi_p50_m{m}"] = float(np.percentile(p, 50))
            stats[f"phi_p90_m{m}"] = float(np.percentile(p, 90))

    mask_shell_in = method == 0
    mask_shell_out = method == 1
    stats["shell_in_bad"] = float(np.mean(label[mask_shell_in] < 0)) if np.any(mask_shell_in) else 0.0
    stats["shell_out_bad"] = float(np.mean(label[mask_shell_out] > 0)) if np.any(mask_shell_out) else 0.0

    # consistency checks
    stats["method2_phi_zero"] = float(np.mean(np.abs(phi[method == 2]) <= 1e-6)) if np.any(method == 2) else 1.0
    stats["method0_phi_pos"] = float(np.mean(phi[method == 0] > 0)) if np.any(method == 0) else 1.0
    stats["method1_phi_neg"] = float(np.mean(phi[method == 1] < 0)) if np.any(method == 1) else 1.0
    if np.any(method == 3):
        stats["method3_sign_match"] = float(np.mean(np.sign(phi[method == 3]) == np.sign(label[method == 3])))
    else:
        stats["method3_sign_match"] = 1.0
    return stats


def _save_sanity(stats: Dict[str, float], out_path: Path) -> None:
    lines = ["key,value"]
    for k in sorted(stats.keys()):
        lines.append(f"{k},{stats[k]}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _render_pyvista_v0_v2_v3(
    out_path: Path,
    anchor_pos: np.ndarray,
    boundary_pts: np.ndarray,
    centers: np.ndarray,
    p_reach: np.ndarray,
    cov_ratio: np.ndarray,
    total: np.ndarray,
    *,
    sphere_radius: float,
    point_size: float,
    boundary_point_size: float,
    sphere_alpha: float,
    min_count: int,
    show: bool,
) -> None:
    try:
        import pyvista as pv
    except ImportError:
        print("Missing dependency: pyvista (skip 3D interactive)")
        return

    plotter = pv.Plotter(shape=(1, 2))
    plotter.set_background("white")

    def _draw_base(cell_idx: int) -> None:
        plotter.subplot(0, cell_idx)
        if boundary_pts.size > 0:
            plotter.add_points(
                boundary_pts,
                color=(0.7, 0.7, 0.7),
                point_size=boundary_point_size,
                render_points_as_spheres=True,
            )
        plotter.add_points(
            anchor_pos.reshape(1, 3),
            color=(1.0, 0.0, 0.0),
            point_size=12.0,
            render_points_as_spheres=True,
        )
        sphere = pv.Sphere(radius=float(sphere_radius), center=anchor_pos.tolist(), theta_resolution=32, phi_resolution=32)
        plotter.add_mesh(sphere, color=(0.9, 0.9, 0.9), opacity=float(sphere_alpha), smooth_shading=True)
        plotter.add_axes(line_width=1.0)

    mask = total >= int(min_count)
    pts = anchor_pos[None, :] + sphere_radius * centers[mask] if np.any(mask) else None

    # left: V0 + V2 (reachability)
    _draw_base(0)
    if pts is not None:
        scalars = p_reach[mask]
        plotter.add_points(
            pts,
            scalars=scalars,
            cmap="viridis",
            point_size=float(point_size),
            render_points_as_spheres=True,
            clim=[0.0, 1.0],
        )
    plotter.add_text("V0 + V2", font_size=12)

    # right: V0 + V3 (twist coverage)
    _draw_base(1)
    if pts is not None:
        scalars = cov_ratio[mask]
        plotter.add_points(
            pts,
            scalars=scalars,
            cmap="plasma",
            point_size=float(point_size),
            render_points_as_spheres=True,
            clim=[0.0, 1.0],
        )
    plotter.add_text("V0 + V3", font_size=12)

    plotter.show(screenshot=str(out_path), auto_close=not show)
    if show:
        plotter.show()


def _load_anchor_samples_from_h5(h5, anchor_id: int) -> Dict[str, np.ndarray]:
    anchor_pos = np.asarray(h5["/anchors/pos"][anchor_id], dtype=np.float32)
    start = int(h5["/csr/anchor_start"][anchor_id])
    end = int(h5["/csr/anchor_start"][anchor_id + 1])
    if end <= start:
        raise RuntimeError("Empty anchor samples")
    if "/csr/sample_index" in h5:
        ids = np.asarray(h5["/csr/sample_index"][start:end], dtype=np.int64)
    else:
        ids = np.arange(start, end, dtype=np.int64)

    quat = np.asarray(h5["/samples/quat"][ids], dtype=np.float32)
    phi = np.asarray(h5["/samples/phi"][ids], dtype=np.float32)
    label = np.asarray(h5["/samples/label"][ids], dtype=np.int8)
    method = np.asarray(h5["/samples/method"][ids], dtype=np.uint8)
    bucket_r = np.asarray(h5["/samples/bucket_r"][ids], dtype=np.int64) if "/samples/bucket_r" in h5 else None
    # normalize label to +/-1
    uniq = set(np.unique(label).tolist())
    if uniq <= {0, 1, 2}:
        label = np.where(label == 0, -1, 1).astype(np.int8)
    elif uniq <= {-1, 1}:
        label = label.astype(np.int8)
    else:
        label = np.where(label >= 0, 1, -1).astype(np.int8)
    return {
        "anchor_pos": anchor_pos,
        "quat": quat,
        "phi": phi,
        "label": label,
        "method": method,
        "bucket_r": bucket_r,
    }


def _render_pyvista_global_v3(
    out_path: Path,
    html_path: Path | None,
    pos_h5: Path,
    orient_h5: Path,
    *,
    num_anchors: int,
    seed: int,
    tool_axis: str,
    B_dir: int,
    N_psi: int,
    min_count: int,
    max_boundary_points: int,
    boundary_stride: int,
    sphere_radius: float,
    point_size: float,
    anchor_point_size: float,
    boundary_point_size: float,
    sphere_alpha: float,
    export_html: bool,
    html_backend: str,
    show: bool,
) -> None:
    try:
        import pyvista as pv
    except ImportError:
        print("Missing dependency: pyvista (skip global 3D)")
        return
    import h5py

    rng = np.random.default_rng(int(seed))
    origin, dims, boundary_pts, _ = _collect_boundary_points(
        pos_h5, max_points=max_boundary_points, stride=boundary_stride, seed=seed
    )

    centers = _fibonacci_sphere(B_dir)

    # pick anchors with method3 and method2
    anchor_ids = []
    with h5py.File(orient_h5, "r") as h5:
        anchors_total = h5["/anchors/pos"].shape[0]
        s_v_norm = None
        if "/anchors/s_v_norm" in h5:
            s_v_norm = np.asarray(h5["/anchors/s_v_norm"], dtype=np.float32)
        elif "/anchors/s_v" in h5:
            s_v_norm = np.asarray(h5["/anchors/s_v"], dtype=np.float32)
        attempts = 0
        while len(anchor_ids) < num_anchors and attempts < anchors_total * 3:
            a = int(rng.integers(0, anchors_total))
            attempts += 1
            try:
                data = _load_anchor_samples_from_h5(h5, a)
            except RuntimeError:
                continue
            method = data["method"]
            if np.any(method == 2) and np.any(method == 3):
                anchor_ids.append(a)
        if len(anchor_ids) == 0:
            print("[global] no valid anchors found")
            return
        if s_v_norm is not None:
            anchor_ids = sorted(anchor_ids, key=lambda x: float(s_v_norm[x]), reverse=True)

        anchor_data = []
        for a in anchor_ids:
            data = _load_anchor_samples_from_h5(h5, a)
            quat = data["quat"]
            label = data["label"]
            method = data["method"]
            n_dir, x_axis = _compute_tool_dir(quat, tool_axis)
            dir_bin = _dir_bins(n_dir, centers)
            total = np.bincount(dir_bin, minlength=B_dir)
            mask_reach = label > 0
            reach = np.bincount(dir_bin[mask_reach], minlength=B_dir)
            psi = _compute_twist(n_dir, x_axis)
            psi_bin = np.floor((psi + math.pi) / (2.0 * math.pi) * N_psi).astype(np.int64)
            psi_bin = np.clip(psi_bin, 0, N_psi - 1)
            cover = np.zeros((B_dir, N_psi), dtype=bool)
            cover[dir_bin[mask_reach], psi_bin[mask_reach]] = True
            cov_ratio = cover.sum(axis=1) / float(N_psi)

            mask_valid = total >= int(min_count)
            pts = data["anchor_pos"][None, :] + sphere_radius * centers[mask_valid]
            anchor_data.append({
                "anchor_id": a,
                "anchor_pos": data["anchor_pos"],
                "points": pts.astype(np.float32),
                "cov_ratio": cov_ratio[mask_valid].astype(np.float32),
            })

    plotter = pv.Plotter()
    plotter.set_background("white")
    if boundary_pts.size > 0:
        plotter.add_points(
            boundary_pts,
            color=(0.7, 0.7, 0.7),
            point_size=boundary_point_size,
            render_points_as_spheres=True,
        )

    # create actors per anchor
    point_actors = []
    anchor_actors = []
    sphere_actors = []
    for item in anchor_data:
        a_pos = item["anchor_pos"]
        # anchor point
        a_actor = plotter.add_points(
            a_pos.reshape(1, 3),
            color=(1.0, 0.0, 0.0),
            point_size=float(anchor_point_size),
            render_points_as_spheres=True,
        )
        # sphere shell
        sphere = pv.Sphere(radius=float(sphere_radius), center=a_pos.tolist(), theta_resolution=32, phi_resolution=32)
        s_actor = plotter.add_mesh(sphere, color=(0.9, 0.9, 0.9), opacity=float(sphere_alpha), smooth_shading=True)
        # v3 points
        p_actor = plotter.add_points(
            item["points"],
            scalars=item["cov_ratio"],
            cmap="plasma",
            point_size=float(point_size),
            render_points_as_spheres=True,
            clim=[0.0, 1.0],
        )
        point_actors.append(p_actor)
        anchor_actors.append(a_actor)
        sphere_actors.append(s_actor)

    # show first k anchors (cumulative)
    def _set_visible(count: int) -> None:
        count = max(0, min(count, len(point_actors)))
        for i in range(len(point_actors)):
            vis = i < count
            point_actors[i].SetVisibility(vis)
            anchor_actors[i].SetVisibility(vis)
            sphere_actors[i].SetVisibility(vis)

    _set_visible(0)

    def _slider_callback(value: float) -> None:
        idx = int(round(value))
        idx = max(0, min(idx, len(point_actors)))
        _set_visible(idx)

    plotter.add_slider_widget(
        _slider_callback,
        rng=[0, max(0, len(point_actors))],
        value=0,
        title="anchor count (0..N)",
        pointa=(0.2, 0.05),
        pointb=(0.8, 0.05),
        fmt="%0.0f",
    )
    plotter.add_text("Global V3 slider (sorted by s_v_norm)", font_size=12)
    plotter.add_axes(line_width=1.0)
    if export_html and html_path is not None and str(html_backend).lower() == "pyvista":
        try:
            if html_path.exists():
                html_path.unlink()
        except Exception:
            pass
        exported = False
        for kwargs in ({"offline": True}, {"inline": True}, {}):
            try:
                plotter.export_html(str(html_path), **kwargs)
                exported = True
                break
            except TypeError:
                continue
            except Exception as exc:
                print(f"[global] export_html failed: {exc}")
                break
        if not exported:
            print("[global] export_html not generated (missing trame or incompatible pyvista).")
    plotter.show(screenshot=str(out_path), auto_close=not show)
    if export_html and html_path is not None and str(html_backend).lower() == "plotly":
        try:
            import plotly.graph_objects as go
        except Exception as exc:
            print(f"[global] plotly not available: {exc}")
            return

        traces = []
        if boundary_pts.size > 0:
            traces.append(go.Scatter3d(
                x=boundary_pts[:, 0], y=boundary_pts[:, 1], z=boundary_pts[:, 2],
                mode="markers",
                marker=dict(size=2, color="rgba(180,180,180,0.25)"),
                showlegend=False,
                name="boundary",
            ))

        for item in anchor_data:
            a_pos = item["anchor_pos"]
            pts = item["points"]
            cov = item["cov_ratio"]
            traces.append(go.Scatter3d(
                x=[a_pos[0]], y=[a_pos[1]], z=[a_pos[2]],
                mode="markers",
                marker=dict(size=6, color="red"),
                showlegend=False,
                name=f"anchor {item['anchor_id']}",
            ))
            traces.append(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode="markers",
                marker=dict(size=3, color=cov, colorscale="Plasma", cmin=0.0, cmax=1.0),
                showlegend=False,
                name=f"v3 {item['anchor_id']}",
            ))

        fig = go.Figure(data=traces)
        fig.update_layout(
            scene=dict(aspectmode="data"),
            title="Global V3 slider (sorted by s_v_norm)",
            margin=dict(l=0, r=0, t=40, b=0),
        )

        n_anchor = len(anchor_data)
        n_boundary = 1 if boundary_pts.size > 0 else 0
        steps = []
        for k in range(n_anchor + 1):
            vis = [True] * n_boundary
            for i in range(n_anchor):
                vis.extend([i < k, i < k])
            steps.append(dict(
                method="update",
                args=[{"visible": vis}],
                label=str(k),
            ))

        fig.update_layout(
            sliders=[dict(
                active=0,
                currentvalue={"prefix": "anchor count: "},
                steps=steps,
            )]
        )
        fig.write_html(str(html_path), include_plotlyjs="inline")


def main() -> int:
    args = SimpleNamespace(**CONFIG)

    if args.global_mode:
        if not args.show_pyvista:
            print("[global] show_pyvista=false, skip")
            return 0
        out_dir = Path(args.out_dir) / "global"
        _ensure_dir(out_dir)
        _render_pyvista_global_v3(
            out_dir / args.global_out_name,
            html_path=(out_dir / args.global_html_name) if args.global_export_html else None,
            pos_h5=Path(args.pos_h5),
            orient_h5=Path(args.orient_h5),
            num_anchors=int(args.global_num_anchors),
            seed=int(args.global_seed),
            tool_axis=str(args.tool_axis),
            B_dir=int(args.global_B_dir),
            N_psi=int(args.global_N_psi),
            min_count=int(args.global_min_count),
            max_boundary_points=int(args.global_max_boundary_points),
            boundary_stride=int(args.global_boundary_stride),
            sphere_radius=float(args.global_pv_sphere_radius),
            point_size=float(args.global_pv_point_size),
            anchor_point_size=float(args.global_anchor_point_size),
            boundary_point_size=float(args.pv_boundary_point_size),
            sphere_alpha=float(args.pv_sphere_alpha),
            export_html=bool(args.global_export_html),
            html_backend=str(args.global_html_backend),
            show=bool(args.show),
        )
        return 0

    if args.backend:
        import matplotlib
        matplotlib.use(args.backend)
    import matplotlib.pyplot as plt

    out_dir = Path(args.out_dir) / f"anchor_{args.anchor_id:05d}"
    _ensure_dir(out_dir)

    data = _load_anchor_samples(Path(args.orient_h5), args.anchor_id)
    anchor_pos = data["anchor_pos"]
    quat = data["quat"]
    phi = data["phi"]
    label = data["label"]
    method = data["method"]
    bucket_r = data["bucket_r"]

    # V0 anchor context
    origin, dims, boundary_pts, voxel = _collect_boundary_points(
        Path(args.pos_h5),
        max_points=args.max_boundary_points,
        stride=args.boundary_stride,
        seed=args.seed,
    )
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    if boundary_pts.size > 0:
        ax.scatter(boundary_pts[:, 0], boundary_pts[:, 1], boundary_pts[:, 2], s=1.0, alpha=args.boundary_alpha, c="#BBBBBB")
    ax.scatter([anchor_pos[0]], [anchor_pos[1]], [anchor_pos[2]], s=80, c="red")
    ax.text(anchor_pos[0], anchor_pos[1], anchor_pos[2], f"anchor {args.anchor_id}")
    if args.show_aabb:
        aabb_min = origin
        aabb_max = origin + dims.astype(np.float64) * voxel
        xs = [aabb_min[0], aabb_max[0]]
        ys = [aabb_min[1], aabb_max[1]]
        zs = [aabb_min[2], aabb_max[2]]
        corners = np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype=np.float64)
        edges = [
            (0, 1), (0, 2), (0, 4),
            (7, 6), (7, 5), (7, 3),
            (1, 3), (1, 5),
            (2, 3), (2, 6),
            (4, 5), (4, 6),
        ]
        for a, b in edges:
            ax.plot([corners[a, 0], corners[b, 0]], [corners[a, 1], corners[b, 1]], [corners[a, 2], corners[b, 2]], c="#888888", lw=1)
    ax.set_title("V0 Anchor Context")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fig.tight_layout()
    fig.savefig(out_dir / "V0_anchor_context.png", dpi=200)

    # V1 sanity check
    stats = _sanity_check(phi, label, method)
    _save_sanity(stats, out_dir / "V1_sanity_table.csv")

    # V2/V3: direction bins
    centers = _fibonacci_sphere(args.B_dir)
    n_dir, x_axis = _compute_tool_dir(quat, args.tool_axis)
    dir_bin = _dir_bins(n_dir, centers)

    total = np.bincount(dir_bin, minlength=args.B_dir)
    reach = np.bincount(dir_bin[label > 0], minlength=args.B_dir)
    with np.errstate(divide="ignore", invalid="ignore"):
        p_reach = np.where(total > 0, reach / total, 0.0)
    mask_valid = total >= args.min_count_per_dir_bin

    # V2 plot
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    sizes = np.log1p(total) * 12.0
    sc = ax.scatter(
        centers[mask_valid, 0], centers[mask_valid, 1], centers[mask_valid, 2],
        c=p_reach[mask_valid], s=sizes[mask_valid], cmap="viridis", vmin=0.0, vmax=1.0
    )
    cb = fig.colorbar(sc, ax=ax, shrink=0.7)
    cb.set_label("reach ratio")
    ax.set_title("V2 Direction Reachability")
    fig.tight_layout()
    fig.savefig(out_dir / "V2_dir_reach_3d.png", dpi=200)

    # V3 twist coverage
    psi = _compute_twist(n_dir, x_axis)
    psi_bin = np.floor((psi + math.pi) / (2.0 * math.pi) * args.N_psi).astype(np.int64)
    psi_bin = np.clip(psi_bin, 0, args.N_psi - 1)
    cover = np.zeros((args.B_dir, args.N_psi), dtype=bool)
    mask_reach = label > 0
    cover[dir_bin[mask_reach], psi_bin[mask_reach]] = True
    cov_ratio = cover.sum(axis=1) / float(args.N_psi)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        centers[mask_valid, 0], centers[mask_valid, 1], centers[mask_valid, 2],
        c=cov_ratio[mask_valid], s=sizes[mask_valid], cmap="plasma", vmin=0.0, vmax=1.0
    )
    cb = fig.colorbar(sc, ax=ax, shrink=0.7)
    cb.set_label("twist coverage")
    ax.set_title("V3 Direction Twist Coverage")
    fig.tight_layout()
    fig.savefig(out_dir / "V3_dir_twist_3d.png", dpi=200)

    # V4 roll slices (mollweide)
    lon = np.arctan2(centers[:, 1], centers[:, 0])
    lat = np.arcsin(np.clip(centers[:, 2], -1.0, 1.0))
    fig, axes = plt.subplots(2, int(math.ceil(args.N_slice / 2)), subplot_kw={"projection": "mollweide"}, figsize=(14, 6))
    axes = np.array(axes).reshape(-1)
    for k in range(args.N_slice):
        ax = axes[k]
        psi_lo = -math.pi + 2.0 * math.pi * k / args.N_slice
        psi_hi = -math.pi + 2.0 * math.pi * (k + 1) / args.N_slice
        mask = (psi >= psi_lo) & (psi < psi_hi)
        if np.any(mask):
            total_k = np.bincount(dir_bin[mask], minlength=args.B_dir)
            reach_k = np.bincount(dir_bin[mask & (label > 0)], minlength=args.B_dir)
            with np.errstate(divide="ignore", invalid="ignore"):
                p_reach_k = np.where(total_k > 0, reach_k / total_k, 0.0)
            mask_k = total_k >= args.min_count_per_dir_bin
            sc = ax.scatter(lon[mask_k], lat[mask_k], c=p_reach_k[mask_k], s=8, cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_title(f"slice {k}")
    for k in range(args.N_slice, axes.size):
        axes[k].axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "V4_roll_slices.png", dpi=200)

    # V5 rotvec ball
    rotvec = _rotvec_from_quat(quat)
    rng = np.random.default_rng(int(args.seed))
    if rotvec.shape[0] > args.max_rotvec_points:
        idx = rng.choice(rotvec.shape[0], size=args.max_rotvec_points, replace=False)
    else:
        idx = np.arange(rotvec.shape[0])
    rotvec = rotvec[idx]
    phi_sub = phi[idx]
    label_sub = label[idx]
    method_sub = method[idx]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    if args.rotvec_color == "phi":
        sc = ax.scatter(rotvec[:, 0], rotvec[:, 1], rotvec[:, 2], c=phi_sub, s=6, cmap="coolwarm", vmin=-math.pi, vmax=math.pi)
        cb = fig.colorbar(sc, ax=ax, shrink=0.7)
        cb.set_label("phi (rad)")
    else:
        colors = np.where(label_sub > 0, "#2ca02c", "#d62728")
        ax.scatter(rotvec[:, 0], rotvec[:, 1], rotvec[:, 2], c=colors, s=6)
    if np.any(method_sub == 2):
        rb = rotvec[method_sub == 2]
        ax.scatter(rb[:, 0], rb[:, 1], rb[:, 2], c="black", s=10)
    ax.set_title("V5 Rotvec Ball")
    fig.tight_layout()
    fig.savefig(out_dir / "V5_rotvec_ball.png", dpi=200)

    # V6 bucket histogram
    if bucket_r is not None:
        k_r = int(np.max(bucket_r))
        hist_all = np.bincount(bucket_r - 1, minlength=k_r)
        hist_reach = np.bincount(bucket_r[label > 0] - 1, minlength=k_r)
        hist_unreach = np.bincount(bucket_r[label < 0] - 1, minlength=k_r)
        p = hist_all / max(np.sum(hist_all), 1)
        entropy = -np.sum(p[p > 0] * np.log(p[p > 0]))
        gini = 1.0 - np.sum(p * p)

        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(k_r)
        ax.plot(x, hist_all, label="all")
        ax.plot(x, hist_reach, label="reach")
        ax.plot(x, hist_unreach, label="unreach")
        ax.set_title(f"V6 bucket_r hist (H={entropy:.3f}, G={gini:.3f})")
        ax.set_xlabel("bucket id")
        ax.set_ylabel("count")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "V6_bucket_hist.png", dpi=200)

    if args.show:
        plt.show()
    else:
        plt.close("all")

    # PyVista 3D window (V0 + V3 overlay)
    if args.show_pyvista:
        _render_pyvista_v0_v2_v3(
            out_dir / "V0V2V3_pyvista.png",
            anchor_pos=anchor_pos,
            boundary_pts=boundary_pts,
            centers=centers,
            p_reach=p_reach,
            cov_ratio=cov_ratio,
            total=total,
            sphere_radius=float(args.pv_sphere_radius),
            point_size=float(args.pv_point_size),
            boundary_point_size=float(args.pv_boundary_point_size),
            sphere_alpha=float(args.pv_sphere_alpha),
            min_count=int(args.pv_min_count),
            show=bool(args.show),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
