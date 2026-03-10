#!/usr/bin/env python3
"""Visualize prepared position datasets, boundary voxels, and sampling buckets.

This script inspects the processed position field using PyVista and configurable grid filters.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import argparse

import numpy as np

# Edit this config block directly. CLI has no options.
CONFIG = {
    # 数据集路径：训练输入的 dataset_pos.h5
    "path": "data/aubo_i5_3mm/dataset_pos.h5",
    # 随机种子：影响 subsample 的随机性（可重复）
    "seed": 42,
    # ===== boundary 可视化 =====
    # 若 dataset_pos 无 /samples/boundary，则使用 label==0 体素中心作为边界点
    "show_boundary": True,
    # 若存在 /samples/boundary，显示 inside / outside / star 三类点
    "show_boundary_tags": False,
    # 边界点最大显示数量（过大会卡）
    "max_boundary": 200000,
    # inside 点颜色（RGB 0~1）
    "inside_color": [0.2, 0.9, 0.3],
    # outside 点颜色（RGB 0~1）
    "outside_color": [0.95, 0.3, 0.3],
    # star（边界中心）颜色（RGB 0~1）
    "star_color": [0.2, 0.7, 1.0],
    # 边界点透明度（0~1）
    "boundary_opacity": 0.6,
    # 边界点大小（数值越大越粗）
    "boundary_size": 3.0,
    # ===== grid 体素中心可视化 =====
    # 是否显示 grid 体素中心点（按 s / label / bucket 过滤）
    "show_grid": True,
    # grid 最大显示点数（过大会卡）
    "max_grid": 200000,
    # 读取 grid 的 x 方向分块大小（越小越省内存，但更慢）
    "chunk_x": 5,
    # s 过滤范围：只显示 s ∈ [s_min, s_max] 的体素中心
    # None 表示不限制。推荐：看边界附近可设 [-0.01, 0.01]
    "s_min": None,
    "s_max": None,
    # 精确值过滤：显示 |s - s_value| <= s_eps
    # 当你想看“某一层距离”的体素时用
    "s_value": None,
    "s_eps": 0.001,
    # label 过滤：0=boundary,1=inside,2=outside,None 表示不过滤
    "label": None,
    # b_s 过滤（log-|s| 桶）：None 表示不过滤；也可在 CLI 输入 "1,2,3" 或 "1-4"
    "b_s": None,
    # b_xyz 过滤：例如 "2,4,1"；支持 "*" 作为通配
    "b_xyz": None,
    # grid 正样本颜色（仅当 grid_color_mode=sign）
    "pos_color": [0.2, 0.7, 1.0],
    # grid 负样本颜色（仅当 grid_color_mode=sign）
    "neg_color": [0.9, 0.3, 0.3],
    # grid 上色方式：
    # - "sign": 正负两色
    # - "label": label 三色
    # - "b_s": 按 b_s 上色
    # - "s": 按 s 连续上色
    "grid_color_mode": "b_s",
    # b_s / s 上色使用的 colormap（matplotlib 名称）
    "grid_cmap": "viridis",
    # grid 点透明度（0~1）
    "grid_opacity": 0.4,
    # grid 点大小
    "grid_size": 2.0,
    # ===== 切面显示（4 图） =====
    # 是否显示 XYZ 三切面（2x2 子图）
    "show_slices": True,
    # 切面厚度（米）；太小会看不到点，太大切面变厚
    "slice_thickness": 0.02,
    # 切面中心（x,y,z）。None 时用点云均值
    "slice_center": None,
    # 是否显示坐标轴
    "show_axes": False,
    # 窗口标题
    "title": "dataset_pos.h5",
}


def _subsample(rng: np.random.Generator, pts: np.ndarray, max_n: int) -> np.ndarray:
    if max_n <= 0 or pts.shape[0] <= max_n:
        return pts
    idx = rng.choice(pts.shape[0], size=max_n, replace=False)
    return pts[idx]


def _parse_range_list(text: str | None, min_v: int, max_v: int) -> set[int] | None:
    if text is None:
        return None
    txt = str(text).strip().lower()
    if txt in ("", "all", "*"):
        return None
    out: set[int] = set()
    for part in txt.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            try:
                a_i = int(a)
                b_i = int(b)
            except ValueError:
                continue
            lo = min(a_i, b_i)
            hi = max(a_i, b_i)
            for v in range(lo, hi + 1):
                if min_v <= v <= max_v:
                    out.add(v)
        else:
            try:
                v = int(part)
            except ValueError:
                continue
            if min_v <= v <= max_v:
                out.add(v)
    return out if out else None


def _parse_b_xyz(text: str | None, bx: int, by: int, bz: int) -> tuple[set[int] | None, set[int] | None, set[int] | None]:
    if text is None:
        return None, None, None
    txt = str(text).strip().lower()
    if txt in ("", "all", "*"):
        return None, None, None
    parts = [p.strip() for p in txt.split(",")]
    if len(parts) != 3:
        return None, None, None
    sel_x = _parse_range_list(parts[0], 0, bx - 1)
    sel_y = _parse_range_list(parts[1], 0, by - 1)
    sel_z = _parse_range_list(parts[2], 0, bz - 1)
    return sel_x, sel_y, sel_z


def _parse_args() -> SimpleNamespace:
    parser = argparse.ArgumentParser(description="Visualize dataset_pos.h5 (new bucket format).")
    parser.add_argument("--path", type=str, default=CONFIG["path"])
    parser.add_argument("--seed", type=int, default=CONFIG["seed"])
    parser.add_argument("--show-boundary", action="store_true", default=CONFIG["show_boundary"])
    parser.add_argument("--show-boundary-tags", action="store_true", default=CONFIG["show_boundary_tags"])
    parser.add_argument("--max-boundary", type=int, default=CONFIG["max_boundary"])
    parser.add_argument("--show-grid", action="store_true", default=CONFIG["show_grid"])
    parser.add_argument("--max-grid", type=int, default=CONFIG["max_grid"])
    parser.add_argument("--chunk-x", type=int, default=CONFIG["chunk_x"])
    parser.add_argument("--s-min", type=float, default=CONFIG["s_min"])
    parser.add_argument("--s-max", type=float, default=CONFIG["s_max"])
    parser.add_argument("--s-value", type=float, default=CONFIG["s_value"])
    parser.add_argument("--s-eps", type=float, default=CONFIG["s_eps"])
    parser.add_argument("--label", type=int, default=CONFIG["label"])
    parser.add_argument("--b-s", type=str, default=CONFIG["b_s"])
    parser.add_argument("--b-xyz", type=str, default=CONFIG["b_xyz"])
    parser.add_argument("--grid-color-mode", type=str, default=CONFIG["grid_color_mode"])
    parser.add_argument("--grid-cmap", type=str, default=CONFIG["grid_cmap"])
    parser.add_argument("--grid-opacity", type=float, default=CONFIG["grid_opacity"])
    parser.add_argument("--grid-size", type=float, default=CONFIG["grid_size"])
    parser.add_argument("--show-slices", action="store_true", default=CONFIG["show_slices"])
    parser.add_argument("--slice-thickness", type=float, default=CONFIG["slice_thickness"])
    parser.add_argument("--slice-center", type=str, default=None)
    parser.add_argument("--show-axes", action="store_true", default=CONFIG["show_axes"])
    parser.add_argument("--title", type=str, default=CONFIG["title"])
    parser.add_argument("--list-buckets", action="store_true", default=False)
    parser.add_argument("--interactive", action="store_true", default=False)
    args = parser.parse_args()
    return SimpleNamespace(**vars(args))


def _apply_slice_mask(pts: np.ndarray, center: np.ndarray, thickness: float, axis: int) -> np.ndarray:
    if pts.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    mask = np.abs(pts[:, axis] - center[axis]) <= thickness
    return pts[mask]


def main() -> int:
    args = _parse_args()
    rng = np.random.default_rng(int(args.seed))
    path = Path(args.path)

    if args.slice_center is not None and isinstance(args.slice_center, str):
        parts = [p.strip() for p in args.slice_center.split(",")]
        if len(parts) == 3:
            try:
                args.slice_center = [float(parts[0]), float(parts[1]), float(parts[2])]
            except ValueError:
                args.slice_center = None

    if not args.show_boundary and not args.show_boundary_tags and not args.show_grid:
        args.show_boundary = True

    try:
        import h5py
    except ImportError:
        print("Missing dependency: h5py")
        print("Install with: python -m pip install h5py")
        return 1

    try:
        import pyvista as pv
    except ImportError:
        print("Missing dependency: pyvista")
        print("Install with: python -m pip install pyvista")
        return 1

    boundary = None
    with h5py.File(path, "r") as h5:
        if "/samples/boundary" in h5:
            boundary = np.asarray(h5["/samples/boundary"], dtype=np.float32)

        # grid info
        origin = np.asarray(h5["/grid/origin"], dtype=np.float64).reshape(3)
        dims = np.asarray(h5["/grid/dims"], dtype=np.int64).reshape(3)
        voxel = float(np.asarray(h5["/grid/voxel_size"], dtype=np.float64).reshape(-1)[0])
        label_ds = h5["/grid/label"]
        sdf_ds = h5["/grid/sdf"]
        b_s_ds = h5["/grid/b_s"]
        b_x_ds = h5["/grid/b_x"]
        b_y_ds = h5["/grid/b_y"]
        b_z_ds = h5["/grid/b_z"]
        num_s = int(np.asarray(h5["/grid/bucket_s_num"], dtype=np.int64).reshape(-1)[0]) if "/grid/bucket_s_num" in h5 else 1
        if "/grid/bucket_spatial_dims" in h5:
            bucket_dims = np.asarray(h5["/grid/bucket_spatial_dims"], dtype=np.int64).reshape(3)
        else:
            bucket_dims = np.asarray([1, 1, 1], dtype=np.int64)
        bx_dim, by_dim, bz_dim = [int(x) for x in bucket_dims]

        if args.list_buckets:
            print(f"bucket_s: 1..{num_s}, bucket_xyz dims=({bx_dim},{by_dim},{bz_dim})")
            return 0

        if args.interactive:
            print(f"[vis] bucket_s: 1..{num_s}, bucket_xyz dims=({bx_dim},{by_dim},{bz_dim})")
            b_s_in = input("选择 b_s (例如 1,2 或 1-4 或 all): ").strip()
            b_xyz_in = input("选择 b_xyz (例如 2,4,1 或 1,*,* 或 all): ").strip()
            args.b_s = b_s_in if b_s_in else args.b_s
            args.b_xyz = b_xyz_in if b_xyz_in else args.b_xyz

        # prepare boundary points
        p_in = p_out = p_star = None
        use_boundary_samples = boundary is not None and boundary.size > 0
        if boundary is not None and boundary.size > 0:
            boundary = _subsample(rng, boundary, int(args.max_boundary))
            p_in = boundary[:, 0:3]
            p_out = boundary[:, 3:6]
            p_star = boundary[:, 6:9]
        elif args.show_boundary or args.show_boundary_tags:
            args.show_boundary_tags = False
            p_star = None

        # collect grid points
        grid_pts = None
        grid_tiers = None
        grid_signs = None
        grid_labels = None
        grid_b_s = None
        grid_s = None
        grid_keys = None
        if args.show_grid:
            nx, ny, nz = [int(x) for x in dims]
            chunk_x = max(1, int(args.chunk_x))
            max_grid = int(args.max_grid)
            use_limit = max_grid > 0
            s_min = args.s_min
            s_max = args.s_max
            s_val = args.s_value
            s_eps = float(args.s_eps)
            label_filter = args.label if args.label is not None else None
            b_s_set = _parse_range_list(args.b_s, 1, num_s)
            sel_bx, sel_by, sel_bz = _parse_b_xyz(args.b_xyz, bx_dim, by_dim, bz_dim)

            for x0 in range(0, nx, chunk_x):
                x1 = min(nx, x0 + chunk_x)
                sl = np.s_[x0:x1, :, :]
                sdf = np.asarray(sdf_ds[sl], dtype=np.float32)
                label = np.asarray(label_ds[sl], dtype=np.uint8)
                b_s = np.asarray(b_s_ds[sl], dtype=np.uint8)
                b_x = np.asarray(b_x_ds[sl], dtype=np.uint16)
                b_y = np.asarray(b_y_ds[sl], dtype=np.uint16)
                b_z = np.asarray(b_z_ds[sl], dtype=np.uint16)

                mask = np.ones_like(sdf, dtype=bool)
                if s_min is not None:
                    mask &= sdf >= float(s_min)
                if s_max is not None:
                    mask &= sdf <= float(s_max)
                if s_val is not None:
                    mask &= np.abs(sdf - float(s_val)) <= s_eps
                if label_filter is not None:
                    mask &= label == int(label_filter)
                if b_s_set is not None:
                    mask &= np.isin(b_s, np.array(sorted(b_s_set), dtype=np.uint8))
                if sel_bx is not None:
                    mask &= np.isin(b_x, np.array(sorted(sel_bx), dtype=np.uint16))
                if sel_by is not None:
                    mask &= np.isin(b_y, np.array(sorted(sel_by), dtype=np.uint16))
                if sel_bz is not None:
                    mask &= np.isin(b_z, np.array(sorted(sel_bz), dtype=np.uint16))

                idx = np.argwhere(mask)
                if idx.size == 0:
                    continue
                xs = idx[:, 0] + x0
                ys = idx[:, 1]
                zs = idx[:, 2]
                pts = origin + voxel * (np.stack([xs, ys, zs], axis=1) + 0.5)
                s_sel = sdf[idx[:, 0], idx[:, 1], idx[:, 2]]
                labels_sel = label[idx[:, 0], idx[:, 1], idx[:, 2]]
                b_s_sel = b_s[idx[:, 0], idx[:, 1], idx[:, 2]]
                pts = pts.astype(np.float32)
                signs = (s_sel < 0).astype(np.uint8)

                if not use_limit:
                    if grid_pts is None:
                        grid_pts = [pts]
                        grid_tiers = [b_s_sel.astype(np.uint8)]
                        grid_signs = [signs]
                        grid_labels = [labels_sel.astype(np.uint8)]
                        grid_b_s = [b_s_sel.astype(np.uint8)]
                        grid_s = [s_sel.astype(np.float32)]
                    else:
                        grid_pts.append(pts)
                        grid_tiers.append(b_s_sel.astype(np.uint8))
                        grid_signs.append(signs)
                        grid_labels.append(labels_sel.astype(np.uint8))
                        grid_b_s.append(b_s_sel.astype(np.uint8))
                        grid_s.append(s_sel.astype(np.float32))
                    continue

                keys_new = rng.random(pts.shape[0], dtype=np.float32)
                if grid_pts is None:
                    if pts.shape[0] > max_grid:
                        sel = np.argpartition(keys_new, max_grid - 1)[:max_grid]
                        grid_pts = pts[sel]
                        grid_tiers = b_s_sel[sel].astype(np.uint8)
                        grid_signs = signs[sel]
                        grid_labels = labels_sel[sel].astype(np.uint8)
                        grid_b_s = b_s_sel[sel].astype(np.uint8)
                        grid_s = s_sel[sel].astype(np.float32)
                        grid_keys = keys_new[sel]
                    else:
                        grid_pts = pts
                        grid_tiers = b_s_sel.astype(np.uint8)
                        grid_signs = signs
                        grid_labels = labels_sel.astype(np.uint8)
                        grid_b_s = b_s_sel.astype(np.uint8)
                        grid_s = s_sel.astype(np.float32)
                        grid_keys = keys_new
                else:
                    all_pts = np.vstack([grid_pts, pts])
                    all_tiers = np.concatenate([grid_tiers, b_s_sel.astype(np.uint8)])
                    all_signs = np.concatenate([grid_signs, signs])
                    all_labels = np.concatenate([grid_labels, labels_sel.astype(np.uint8)])
                    all_b_s = np.concatenate([grid_b_s, b_s_sel.astype(np.uint8)])
                    all_s = np.concatenate([grid_s, s_sel.astype(np.float32)])
                    all_keys = np.concatenate([grid_keys, keys_new])
                    if all_keys.shape[0] > max_grid:
                        sel = np.argpartition(all_keys, max_grid - 1)[:max_grid]
                        grid_pts = all_pts[sel]
                        grid_tiers = all_tiers[sel]
                        grid_signs = all_signs[sel]
                        grid_labels = all_labels[sel]
                        grid_b_s = all_b_s[sel]
                        grid_s = all_s[sel]
                        grid_keys = all_keys[sel]
                    else:
                        grid_pts = all_pts
                        grid_tiers = all_tiers
                        grid_signs = all_signs
                        grid_labels = all_labels
                        grid_b_s = all_b_s
                        grid_s = all_s
                        grid_keys = all_keys

        if grid_pts is None:
            grid_pts = np.zeros((0, 3), dtype=np.float32)
            grid_tiers = np.zeros((0,), dtype=np.uint8)
            grid_signs = np.zeros((0,), dtype=np.uint8)
            grid_labels = np.zeros((0,), dtype=np.uint8)
            grid_b_s = np.zeros((0,), dtype=np.uint8)
            grid_s = np.zeros((0,), dtype=np.float32)
        elif isinstance(grid_pts, list):
            grid_pts = np.vstack(grid_pts)
            grid_tiers = np.concatenate(grid_tiers)
            grid_signs = np.concatenate(grid_signs)
            grid_labels = np.concatenate(grid_labels)
            grid_b_s = np.concatenate(grid_b_s)
            grid_s = np.concatenate(grid_s)

        if (p_star is None or p_star.size == 0) and (args.show_boundary or args.show_boundary_tags):
            nx, ny, nz = [int(x) for x in dims]
            chunk_x = max(1, int(args.chunk_x))
            max_boundary = int(args.max_boundary)
            use_limit = max_boundary > 0
            boundary_pts = None
            boundary_keys = None
            for x0 in range(0, nx, chunk_x):
                x1 = min(nx, x0 + chunk_x)
                sl = np.s_[x0:x1, :, :]
                label = np.asarray(label_ds[sl], dtype=np.uint8)
                mask = label == 0
                idx = np.argwhere(mask)
                if idx.size == 0:
                    continue
                xs = idx[:, 0] + x0
                ys = idx[:, 1]
                zs = idx[:, 2]
                pts = origin + voxel * (np.stack([xs, ys, zs], axis=1) + 0.5)
                pts = pts.astype(np.float32)
                if not use_limit:
                    if boundary_pts is None:
                        boundary_pts = [pts]
                    else:
                        boundary_pts.append(pts)
                    continue
                keys_new = rng.random(pts.shape[0], dtype=np.float32)
                if boundary_pts is None:
                    if pts.shape[0] > max_boundary:
                        sel = np.argpartition(keys_new, max_boundary - 1)[:max_boundary]
                        boundary_pts = pts[sel]
                        boundary_keys = keys_new[sel]
                    else:
                        boundary_pts = pts
                        boundary_keys = keys_new
                else:
                    all_pts = np.vstack([boundary_pts, pts])
                    all_keys = np.concatenate([boundary_keys, keys_new])
                    if all_keys.shape[0] > max_boundary:
                        sel = np.argpartition(all_keys, max_boundary - 1)[:max_boundary]
                        boundary_pts = all_pts[sel]
                        boundary_keys = all_keys[sel]
                    else:
                        boundary_pts = all_pts
                        boundary_keys = all_keys
            if boundary_pts is None:
                p_star = np.zeros((0, 3), dtype=np.float32)
            elif isinstance(boundary_pts, list):
                p_star = np.vstack(boundary_pts)
            else:
                p_star = boundary_pts
            p_in = p_out = None

    if args.show_slices:
        plotter = pv.Plotter(shape=(2, 2))
        plotter.subplot(0, 0)
    else:
        plotter = pv.Plotter()

    def add_cloud(points: np.ndarray, color, opacity: float, size: float) -> None:
        if points is None or points.size == 0:
            return
        cloud = pv.PolyData(points)
        plotter.add_mesh(
            cloud,
            color=tuple(color),
            opacity=float(opacity),
            point_size=float(size),
            render_points_as_spheres=True,
        )

    def add_grid(points: np.ndarray, labels: np.ndarray, b_s: np.ndarray, signs: np.ndarray, s_vals: np.ndarray) -> None:
        if points is None or points.size == 0:
            return
        mode = str(args.grid_color_mode).lower()
        if mode == "b_s":
            cloud = pv.PolyData(points)
            cloud["b_s"] = b_s.astype(np.float32)
            plotter.add_mesh(
                cloud,
                scalars="b_s",
                cmap=args.grid_cmap,
                opacity=float(args.grid_opacity),
                point_size=float(args.grid_size),
                render_points_as_spheres=True,
                scalar_bar_args={"title": "b_s"},
            )
        elif mode == "s":
            cloud = pv.PolyData(points)
            cloud["s"] = s_vals.astype(np.float32)
            plotter.add_mesh(
                cloud,
                scalars="s",
                cmap=args.grid_cmap,
                opacity=float(args.grid_opacity),
                point_size=float(args.grid_size),
                render_points_as_spheres=True,
                scalar_bar_args={"title": "s"},
            )
        elif mode == "label":
            cloud = pv.PolyData(points)
            cloud["label"] = labels.astype(np.float32)
            plotter.add_mesh(
                cloud,
                scalars="label",
                cmap="tab10",
                opacity=float(args.grid_opacity),
                point_size=float(args.grid_size),
                render_points_as_spheres=True,
                scalar_bar_args={"title": "label"},
            )
        else:
            pos_mask = signs == 0
            if np.any(pos_mask):
                add_cloud(points[pos_mask], args.pos_color, args.grid_opacity, args.grid_size)
            if np.any(~pos_mask):
                add_cloud(points[~pos_mask], args.neg_color, args.grid_opacity, args.grid_size)

    # full view
    if args.show_boundary_tags:
        add_cloud(p_in, args.inside_color, args.boundary_opacity, args.boundary_size)
        add_cloud(p_out, args.outside_color, args.boundary_opacity, args.boundary_size)
        add_cloud(p_star, args.star_color, args.boundary_opacity, args.boundary_size)
    elif args.show_boundary:
        add_cloud(p_star, args.star_color, args.boundary_opacity, args.boundary_size)

    if args.show_grid:
        add_grid(grid_pts, grid_labels, grid_b_s, grid_signs, grid_s)

    if args.show_slices:
        if args.slice_center is None:
            pts_for_center = p_star if p_star is not None and p_star.size > 0 else grid_pts
            center = np.mean(pts_for_center, axis=0) if pts_for_center is not None and pts_for_center.size > 0 else np.zeros(3)
        else:
            center = np.asarray(args.slice_center, dtype=np.float32)
        th = float(args.slice_thickness)

        def draw_slice(subplot_idx, view_func, title, axis):
            plotter.subplot(*subplot_idx)
            if args.show_boundary_tags:
                add_cloud(_apply_slice_mask(p_in, center, th, axis), args.inside_color, args.boundary_opacity, args.boundary_size)
                add_cloud(_apply_slice_mask(p_out, center, th, axis), args.outside_color, args.boundary_opacity, args.boundary_size)
                add_cloud(_apply_slice_mask(p_star, center, th, axis), args.star_color, args.boundary_opacity, args.boundary_size)
            elif args.show_boundary:
                add_cloud(_apply_slice_mask(p_star, center, th, axis), args.star_color, args.boundary_opacity, args.boundary_size)
            if args.show_grid:
                pts = _apply_slice_mask(grid_pts, center, th, axis)
                if pts.size > 0:
                    mask = np.abs(grid_pts[:, axis] - center[axis]) <= th
                    add_grid(pts, grid_labels[mask], grid_b_s[mask], grid_signs[mask], grid_s[mask])
            view_func()
            plotter.add_title(title)

        draw_slice((0, 1), plotter.view_xy, f"XY @ z={center[2]:.3f}", 2)
        draw_slice((1, 0), plotter.view_xz, f"XZ @ y={center[1]:.3f}", 1)
        draw_slice((1, 1), plotter.view_yz, f"YZ @ x={center[0]:.3f}", 0)
        plotter.subplot(0, 0)

    if args.show_axes:
        plotter.show_axes()
    plotter.add_title(args.title)
    plotter.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
