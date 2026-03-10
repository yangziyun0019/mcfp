"""Run Position-SDF inference for a query point.

This script evaluates a trained position model and can iteratively project a query position to reachability.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from mcfp.data.morph_spec_io import load_morphology_spec
from mcfp.models.encodings import position_encoding
from mcfp.models.morph_encoder import MorphologyEncoder
from mcfp.models.pos_sdf import PositionSDFModel
from mcfp.utils.config import load_config


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _get_path(cfg: Any, key: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in key.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            if part not in cur:
                return default
            cur = cur[part]
        else:
            if not hasattr(cur, part):
                return default
            cur = getattr(cur, part)
    return cur


def _resolve_path(path: str | Path, repo_root: Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        return (repo_root / p).resolve()
    return p.resolve()


def _parse_pos(values: Sequence[float]) -> np.ndarray:
    if len(values) != 3:
        raise ValueError("Position must have exactly 3 values: x y z")
    return np.asarray(values, dtype=np.float32)


def _load_aabb_from_h5(path: Path, scale_if_missing: float = 2.0) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        import h5py
    except ImportError:
        print("Missing dependency: h5py (python -m pip install h5py)")
        return None
    if path is None or not Path(path).exists():
        return None
    with h5py.File(path, "r") as h5:
        if "/outer_grid/origin" in h5 and "/outer_grid/dims" in h5 and "/outer_grid/voxel_size" in h5:
            origin = np.asarray(h5["/outer_grid/origin"], dtype=np.float64).reshape(3)
            dims = np.asarray(h5["/outer_grid/dims"], dtype=np.float64).reshape(3)
            voxel = float(np.asarray(h5["/outer_grid/voxel_size"], dtype=np.float64).reshape(-1)[0])
            p_min = origin
            p_max = origin + dims * voxel
            return p_min, p_max
        if "/grid/origin" in h5 and "/grid/dims" in h5 and "/grid/voxel_size" in h5:
            origin = np.asarray(h5["/grid/origin"], dtype=np.float64).reshape(3)
            dims = np.asarray(h5["/grid/dims"], dtype=np.float64).reshape(3)
            voxel = float(np.asarray(h5["/grid/voxel_size"], dtype=np.float64).reshape(-1)[0])
            p_min = origin
            p_max = origin + dims * voxel
            if scale_if_missing is not None and float(scale_if_missing) > 1.0:
                center = 0.5 * (p_min + p_max)
                half = 0.5 * (p_max - p_min) * float(scale_if_missing)
                p_min = center - half
                p_max = center + half
            return p_min, p_max
    return None


def _project_to_sphere(p: np.ndarray, center: np.ndarray, radius: float) -> tuple[np.ndarray, bool]:
    v = p - center
    dist = float(np.linalg.norm(v))
    if dist <= radius:
        return p, False
    if dist < 1e-9:
        return center + np.array([radius, 0.0, 0.0], dtype=np.float32), True
    return (center + v / dist * radius).astype(np.float32), True




def _visualize_path(
    path: np.ndarray,
    title: str,
    out_png: str | None,
    boundary: np.ndarray | None,
    boundary_alpha: float,
    boundary_size: float,
    boundary_surface: bool,
    boundary_surface_alpha: float,
    boundary_surface_alpha_shape: float,
) -> None:
    try:
        import pyvista as pv
    except ImportError:
        print("Missing dependency: pyvista (python -m pip install pyvista)")
        return
    if path.size == 0:
        return

    plotter = pv.Plotter()

    if boundary is not None and boundary.size > 0:
        b_cloud = pv.PolyData(boundary)
        if boundary_surface:
            try:
                surf = b_cloud.delaunay_3d(alpha=float(boundary_surface_alpha)).extract_surface()
                plotter.add_mesh(
                    surf,
                    color=(0.40, 0.78, 1.0),
                    opacity=float(boundary_surface_alpha_shape),
                    show_edges=False,
                )
            except Exception as exc:
                print(f"[vis] surface reconstruction failed: {exc}")
        plotter.add_mesh(
            b_cloud,
            color=(0.40, 0.78, 1.0),
            opacity=float(boundary_alpha),
            point_size=float(boundary_size),
            render_points_as_spheres=True,
        )

    if path.shape[0] >= 2:
        line = pv.lines_from_points(path)
        plotter.add_mesh(line, color=(1.0, 0.6, 0.0), line_width=3.0)

    cloud = pv.PolyData(path)
    plotter.add_mesh(
        cloud,
        color=(1.0, 0.6, 0.0),
        point_size=6.0,
        render_points_as_spheres=True,
    )

    start = pv.PolyData(path[:1])
    plotter.add_mesh(
        start,
        color=(1.0, 0.0, 0.0),
        point_size=12.0,
        render_points_as_spheres=True,
        label="start",
    )
    end = pv.PolyData(path[-1:])
    plotter.add_mesh(
        end,
        color=(0.0, 0.8, 0.0),
        point_size=12.0,
        render_points_as_spheres=True,
        label="end",
    )

    plotter.add_title(title)
    if out_png:
        plotter.show(screenshot=out_png)
    else:
        plotter.show()




def _load_boundary_points(path: Path, max_points: int, seed: int) -> np.ndarray:
    try:
        import h5py
    except ImportError:
        print("Missing dependency: h5py (python -m pip install h5py)")
        return np.zeros((0, 3), dtype=np.float32)
    if path is None or not Path(path).exists():
        print(f"[vis] boundary h5 not found: {path}")
        return np.zeros((0, 3), dtype=np.float32)
    with h5py.File(path, "r") as h5:
        if "/samples/boundary" in h5:
            boundary = np.asarray(h5["/samples/boundary"], dtype=np.float32)
            pts = boundary[:, 6:9]
            if max_points is not None and max_points > 0 and pts.shape[0] > max_points:
                rng = np.random.default_rng(int(seed))
                idx = rng.choice(pts.shape[0], size=int(max_points), replace=False)
                pts = pts[idx]
            return pts
        if "/grid/label" in h5:
            origin = np.asarray(h5["/grid/origin"], dtype=np.float64).reshape(3)
            dims = np.asarray(h5["/grid/dims"], dtype=np.int64).reshape(3)
            voxel = float(np.asarray(h5["/grid/voxel_size"], dtype=np.float64).reshape(-1)[0])
            label = np.asarray(h5["/grid/label"], dtype=np.uint8)
            idx = np.argwhere(label == 0)
            if idx.size == 0:
                print(f"[vis] no label==0 voxels in {path}")
                return np.zeros((0, 3), dtype=np.float32)
            if max_points is not None and max_points > 0 and idx.shape[0] > max_points:
                rng = np.random.default_rng(int(seed))
                sel = rng.choice(idx.shape[0], size=int(max_points), replace=False)
                idx = idx[sel]
            pts = origin + voxel * (idx.astype(np.float64) + 0.5)
            return pts.astype(np.float32)
        print(f"[vis] boundary data not found in {path}")
        return np.zeros((0, 3), dtype=np.float32)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer position reachability with trained Position-SDF.")
    parser.add_argument("--config", type=str, default="configs/infer_pos.yaml", help="Config path.")
    parser.add_argument("--pos", type=float, nargs=3, default=None, help="Position x y z (meters).")
    parser.add_argument("--vis", action="store_true", help="Show 3D path visualization.")
    parser.add_argument("--vis-out", type=str, default=None, help="Save visualization PNG path.")
    parser.add_argument("--vis-boundary-h5", type=str, default=None, help="Boundary H5 path (dataset_pos.h5).")
    parser.add_argument("--vis-boundary-max", type=int, default=None, help="Max boundary points to show.")
    parser.add_argument("--vis-boundary-alpha", type=float, default=None, help="Boundary point alpha.")
    parser.add_argument("--vis-boundary-size", type=float, default=None, help="Boundary point size.")
    parser.add_argument("--vis-boundary-seed", type=int, default=None, help="Boundary subsample seed.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    repo_root = Path(_get_path(cfg, "paths.repo_root", ".")).resolve()

    device = torch.device(str(_get_path(cfg, "run.device", "cpu")))

    vis_cfg = _get_path(cfg, "infer_vis", None)
    vis_enabled = bool(_get(vis_cfg, "enabled", False)) or bool(args.vis)
    vis_out = args.vis_out or _get(vis_cfg, "output_png", None)

    vis_boundary_h5 = args.vis_boundary_h5 or _get(vis_cfg, "boundary_h5", None) or _get_path(cfg, "paths.position_h5", None)
    if vis_boundary_h5 is not None:
        vis_boundary_h5 = _resolve_path(vis_boundary_h5, repo_root)
    vis_boundary_max = args.vis_boundary_max
    if vis_boundary_max is None:
        vis_boundary_max = int(_get(vis_cfg, "boundary_max_points", 50000))
    vis_boundary_alpha = args.vis_boundary_alpha
    if vis_boundary_alpha is None:
        vis_boundary_alpha = float(_get(vis_cfg, "boundary_alpha", 0.08))
    vis_boundary_size = args.vis_boundary_size
    if vis_boundary_size is None:
        vis_boundary_size = float(_get(vis_cfg, "boundary_size", 2.0))
    vis_boundary_seed = args.vis_boundary_seed
    if vis_boundary_seed is None:
        vis_boundary_seed = int(_get(vis_cfg, "boundary_seed", 42))
    vis_boundary_surface = bool(_get(vis_cfg, "boundary_surface", False))
    vis_boundary_surface_alpha = float(_get(vis_cfg, "boundary_surface_alpha", 0.05))
    vis_boundary_surface_alpha_shape = float(_get(vis_cfg, "boundary_surface_opacity", 0.15))

    position_h5_cfg = _get_path(cfg, "paths.position_h5", None) or vis_boundary_h5
    position_h5 = _resolve_path(position_h5_cfg, repo_root) if position_h5_cfg is not None else None

    morph_path = _resolve_path(_get_path(cfg, "paths.morphology_spec"), repo_root)
    ckpt_path = _resolve_path(_get_path(cfg, "paths.position_checkpoint"), repo_root)
    morph_spec = load_morphology_spec(morph_path)
    l_ref = float(morph_spec.l_ref)

    # position input
    if args.pos is not None:
        pos = _parse_pos(args.pos)
    else:
        pos_cfg = _get_path(cfg, "input.position", None)
        if pos_cfg is None:
            raise ValueError("No position provided. Use --pos x y z or set input.position in config.")
        pos = _parse_pos(pos_cfg)

    pos_raw = pos.copy()
    moved = False
    clamp_sphere = bool(_get_path(cfg, "infer.clamp_sphere", True))
    sphere_scale = float(_get_path(cfg, "infer.sphere_scale", 1.0))
    sphere_center_cfg = _get_path(cfg, "infer.sphere_center", [0.0, 0.0, 0.0])
    sphere_center = np.asarray(sphere_center_cfg, dtype=np.float32).reshape(3)
    sphere_radius = float(l_ref) * float(sphere_scale)
    if clamp_sphere and sphere_radius > 0:
        pos_proj, moved = _project_to_sphere(pos, sphere_center, sphere_radius)
        if moved:
            print(f"[infer] input outside sphere, move to {pos_proj.tolist()}")
            pos = pos_proj

    # build models
    morph_cfg = _get_path(cfg, "model.morph", None)
    k_p = int(_get_path(cfg, "model.position.k_p", 10))
    morph_encoder = MorphologyEncoder(
        d_model=int(_get(morph_cfg, "d_model", 256)),
        depth_emb_dim=int(_get(morph_cfg, "depth_emb_dim", 16)),
        num_layers=int(_get(morph_cfg, "num_layers", 6)),
        num_heads=int(_get(morph_cfg, "num_heads", 8)),
        dropout=float(_get(morph_cfg, "dropout", 0.1)),
    ).to(device)
    pos_model = PositionSDFModel(
        in_dim=3 + 2 * 3 * k_p,
        hidden_dim=int(_get_path(cfg, "model.position.hidden_dim", 256)),
        num_layers=int(_get_path(cfg, "model.position.num_layers", 5)),
        w0_first=float(_get_path(cfg, "model.position.w0_first", 30.0)),
        w0=float(_get_path(cfg, "model.position.w0", 1.0)),
        cond_dim=int(_get(morph_cfg, "d_model", 256)),
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    if "morph_encoder" in ckpt:
        morph_encoder.load_state_dict(ckpt["morph_encoder"])
    if "pos_model" in ckpt:
        pos_model.load_state_dict(ckpt["pos_model"])
    else:
        pos_model.load_state_dict(ckpt)

    morph_encoder.eval()
    pos_model.eval()

    # precompute morphology embedding
    with torch.no_grad():
        morph_emb, _ = morph_encoder(morph_spec, device)

    # inference + gradient
    path_points = [pos_raw.astype(np.float32)]
    if moved and not np.allclose(pos_raw, pos, atol=1e-6):
        path_points.append(pos.astype(np.float32))
    p = torch.from_numpy(pos[None, :]).to(device)
    p_norm = (p / l_ref).requires_grad_(True)
    e_p = position_encoding(p_norm, k_p=k_p)
    s_pred = pos_model(e_p, morph_emb)  # normalized SDF

    s_norm = float(s_pred.item())
    s_meter = float(s_norm * l_ref)
    tol_cfg = float(_get_path(cfg, "infer.tol", 0.0))
    tol = abs(tol_cfg)
    target_norm = _get_path(cfg, "infer.target_s_norm", None)
    target_meter = _get_path(cfg, "infer.target_s_meter", None)
    if target_meter is not None:
        target_norm = float(target_meter) / l_ref
    elif target_norm is None:
        target_norm = 0.0
    target_norm = float(target_norm)
    reachable = s_norm >= (target_norm - tol)

    print("=== Position Reachability ===")
    print(f"pos = {pos.tolist()} (m)")
    print(f"sdf_norm = {s_norm:.6f}, sdf_meter = {s_meter:.6f}")
    print(f"reachable = {bool(reachable)} (target_s_norm={target_norm:.6f}, tol={tol:.6f})")

    if not reachable:
        step_scale = float(_get_path(cfg, "infer.step_scale", 1.0))
        max_step = _get_path(cfg, "infer.max_step", None)
        min_step = float(_get_path(cfg, "infer.min_step", 1e-4))
        eps = float(_get_path(cfg, "infer.eps", 1e-8))
        max_iters = int(_get_path(cfg, "infer.max_iters", 10))
        tol_cfg = float(_get_path(cfg, "infer.tol", 0.0))
        tol = abs(tol_cfg)
        backtrack = bool(_get_path(cfg, "infer.backtrack", True))
        backtrack_factor = float(_get_path(cfg, "infer.backtrack_factor", 0.5))
        backtrack_iters = int(_get_path(cfg, "infer.backtrack_iters", 5))
        verbose = bool(_get_path(cfg, "infer.verbose", True))

        p_cur = p.detach()
        for it in range(1, max_iters + 1):
            p_cur = p_cur.detach().requires_grad_(True)
            p_norm = p_cur / l_ref
            e_p = position_encoding(p_norm, k_p=k_p)
            s_pred = pos_model(e_p, morph_emb)  # normalized SDF
            s_norm = float(s_pred.item())
            s_meter = float(s_norm * l_ref)
            if s_norm >= (target_norm - tol):
                print(
                    f"Reached target at iter={it}: sdf_norm={s_norm:.6f}, "
                    f"sdf_meter={s_meter:.6f}, target_norm={target_norm:.6f}"
                )
                break

            grad = torch.autograd.grad(s_pred.sum(), p_cur, create_graph=False)[0]
            grad_norm = torch.linalg.norm(grad, dim=1, keepdim=True)
            grad_unit = grad / torch.clamp(grad_norm, min=eps)
            step = ((target_norm - s_pred) * l_ref) * step_scale
            if max_step is not None:
                step = torch.clamp(step, max=float(max_step))
            step = torch.clamp(step, min=min_step)

            if backtrack:
                step_try = step
                for _ in range(backtrack_iters):
                    p_try = (p_cur + grad_unit * step_try.unsqueeze(-1)).detach()
                    with torch.no_grad():
                        e_try = position_encoding(p_try / l_ref, k_p=k_p)
                        s_try = pos_model(e_try, morph_emb)
                    if float(s_try.item()) > s_norm:
                        step = step_try
                        break
                    step_try = step_try * backtrack_factor

            delta = grad_unit * step.unsqueeze(-1)
            p_next = (p_cur + delta).detach()
            if clamp_sphere and sphere_radius > 0:
                p_next_np = p_next.detach().cpu().numpy().reshape(-1)
                p_next_np, _ = _project_to_sphere(p_next_np, sphere_center, sphere_radius)
                p_next = torch.from_numpy(p_next_np.reshape(1, 3)).to(device)

            path_points.append(p_next.detach().cpu().numpy().reshape(-1).astype(np.float32))

            if verbose:
                print(f"--- iter {it} ---")
                print(f"sdf_norm={s_norm:.6f}, sdf_meter={s_meter:.6f}")
                print(f"grad_unit={grad_unit.detach().cpu().numpy().reshape(-1).tolist()}")
                print(f"step={float(step.item()):.6f} (m)")
                print(f"delta={delta.detach().cpu().numpy().reshape(-1).tolist()} (m)")
                print(f"new_pos={p_next.detach().cpu().numpy().reshape(-1).tolist()} (m)")

            p_cur = p_next
        else:
            print(f"Reached max_iters={max_iters}, last sdf_norm={s_norm:.6f}, sdf_meter={s_meter:.6f}")

    if vis_enabled:
        boundary_pts = None
        if vis_boundary_h5 is not None:
            boundary_pts = _load_boundary_points(Path(vis_boundary_h5), vis_boundary_max, vis_boundary_seed)
        _visualize_path(
            np.asarray(path_points, dtype=np.float32),
            "Position Reachability Path",
            vis_out,
            boundary_pts,
            vis_boundary_alpha,
            vis_boundary_size,
            vis_boundary_surface,
            vis_boundary_surface_alpha,
            vis_boundary_surface_alpha_shape,
        )


if __name__ == "__main__":
    main()
