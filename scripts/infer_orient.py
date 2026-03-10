"""Run Orientation-SDF inference for a fixed anchor position.

This script evaluates a trained orientation model and can iteratively project a query attitude to reachability.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from mcfp.data.morph_spec_io import load_morphology_spec
from mcfp.models.encodings import position_encoding, ReferenceQuaternionEncoder
from mcfp.models.morph_encoder import MorphologyEncoder
from mcfp.models.orient_sdf import OrientationSDFModel
from mcfp.utils.config import load_config
from mcfp.utils.quat import exp_quat, quat_mul, quat_normalize


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


def _load_q_ref_from_h5(path: Path) -> np.ndarray | None:
    try:
        import h5py
    except ImportError:
        print("Missing dependency: h5py (python -m pip install h5py)")
        return None
    if not path.exists():
        return None
    with h5py.File(path, "r") as h5:
        if "/meta/q_ref" in h5:
            return np.asarray(h5["/meta/q_ref"], dtype=np.float32)
        if "/q_ref" in h5:
            return np.asarray(h5["/q_ref"], dtype=np.float32)
        if "/meta/ref_quat" in h5:
            return np.asarray(h5["/meta/ref_quat"], dtype=np.float32)
    return None


def _infer_orient_arch(state: dict[str, torch.Tensor]) -> dict[str, int]:
    experts = 0
    layers = 0
    orient_in_dim = None
    hidden_dim = None
    context_dim = None
    cond_in_dim = None
    for k, v in state.items():
        if k.startswith("decoders.") and ".layers." in k and k.endswith(".weight"):
            parts = k.split(".")
            try:
                dec_idx = int(parts[1])
                layer_idx = int(parts[3])
            except Exception:
                continue
            experts = max(experts, dec_idx + 1)
            layers = max(layers, layer_idx + 1)
            if parts[1] == "0" and parts[3] == "0":
                orient_in_dim = int(v.shape[1])
                hidden_dim = int(v.shape[0])
        if k == "context.2.weight":
            context_dim = int(v.shape[0])
        if k == "context.0.weight":
            cond_in_dim = int(v.shape[1])

    out = {
        "experts": experts if experts > 0 else 4,
        "num_layers": layers if layers > 0 else 4,
    }
    if orient_in_dim is not None:
        out["orient_in_dim"] = orient_in_dim
    if hidden_dim is not None:
        out["hidden_dim"] = hidden_dim
    if context_dim is not None:
        out["context_dim"] = context_dim
    if cond_in_dim is not None:
        out["cond_in_dim"] = cond_in_dim
    return out


def _parse_pos(values: Sequence[float]) -> np.ndarray:
    if len(values) != 3:
        raise ValueError("Position must have exactly 3 values: x y z")
    return np.asarray(values, dtype=np.float32)


def _axis_angle_to_quat(axis: np.ndarray, angle: float) -> torch.Tensor:
    axis = torch.as_tensor(axis, dtype=torch.float32)
    if torch.linalg.norm(axis) < 1e-9:
        return torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
    axis = axis / torch.linalg.norm(axis)
    delta = axis * float(angle)
    q = exp_quat(delta[None, :])[0]
    return q


def _euler_to_quat(
    angles_deg: Sequence[float],
    order: str = "xyz",
    intrinsic: bool = True,
) -> torch.Tensor:
    if len(angles_deg) != len(order):
        raise ValueError("Euler angles length must match order")
    angles = [math.radians(float(a)) for a in angles_deg]
    axes = {
        "x": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "y": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "z": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    }
    q = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
    for ax, ang in zip(order.lower(), angles):
        if ax not in axes:
            raise ValueError(f"Invalid axis in order: {ax}")
        q_axis = _axis_angle_to_quat(axes[ax], ang)
        if intrinsic:
            # body-fixed: q = q * q_axis
            q = quat_mul(q, q_axis)
        else:
            # space-fixed: q = q_axis * q
            q = quat_mul(q_axis, q)
    return quat_normalize(q)


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    # q: (4,), v: (3,)
    x, y, z, w = q
    q_xyz = np.array([x, y, z], dtype=np.float32)
    t = 2.0 * np.cross(q_xyz, v)
    return v + w * t + np.cross(q_xyz, t)


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


def _add_axis(
    plotter,
    origin: np.ndarray,
    direction: np.ndarray,
    length: float,
    color: tuple[float, float, float],
    line_width: float,
    dashed: bool,
    dash_segments: int,
) -> None:
    import pyvista as pv

    if length <= 0:
        return
    d = direction / (np.linalg.norm(direction) + 1e-9)
    if not dashed:
        line = pv.Line(origin, origin + d * length)
        plotter.add_mesh(line, color=color, line_width=line_width)
        return

    segs = max(2, int(dash_segments))
    for i in range(segs):
        if i % 2 == 1:
            continue
        a = origin + d * (length * i / segs)
        b = origin + d * (length * (i + 1) / segs)
        line = pv.Line(a, b)
        plotter.add_mesh(line, color=color, line_width=max(1.0, line_width - 0.5))


def _add_axes(
    plotter,
    origin: np.ndarray,
    quat: np.ndarray,
    length: float,
    color: tuple[float, float, float],
    line_width: float,
    dashed: bool,
    dash_segments: int,
    show_z_arrow: bool,
    z_arrow_scale: float,
) -> None:
    import pyvista as pv

    x = _quat_rotate(quat, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    y = _quat_rotate(quat, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    z = _quat_rotate(quat, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    _add_axis(plotter, origin, x, length, color, line_width, dashed, dash_segments)
    _add_axis(plotter, origin, y, length, color, line_width, dashed, dash_segments)
    _add_axis(plotter, origin, z, length, color, line_width, dashed, dash_segments)
    if show_z_arrow and length > 0:
        d = z / (np.linalg.norm(z) + 1e-9)
        tip = origin + d * length
        z_cone = pv.Cone(
            center=tip - d * (0.08 * length),
            direction=d,
            height=0.16 * length,
            radius=0.06 * length * float(z_arrow_scale),
            resolution=16,
        )
        plotter.add_mesh(z_cone, color=color)


def _visualize(
    anchor_pos: np.ndarray,
    q_list: list[np.ndarray],
    start_reachable: bool,
    reached: bool,
    boundary: np.ndarray | None,
    cfg: Any,
) -> None:
    try:
        import pyvista as pv
    except ImportError:
        print("Missing dependency: pyvista (python -m pip install pyvista)")
        return

    vis_cfg = _get_path(cfg, "vis", None)
    axis_len = float(_get(vis_cfg, "axis_length", 0.08))
    axis_width = float(_get(vis_cfg, "axis_width", 2.0))
    dash_segments = int(_get(vis_cfg, "dash_segments", 10))
    show_z_arrow = bool(_get(vis_cfg, "show_z_arrow", True))
    z_arrow_scale = float(_get(vis_cfg, "z_arrow_scale", 1.0))

    color_init = tuple(_get(vis_cfg, "color_init", [1.0, 0.0, 0.0]))
    color_mid = tuple(_get(vis_cfg, "color_mid", [0.7, 0.7, 0.7]))
    color_final = tuple(_get(vis_cfg, "color_final", [0.0, 0.8, 0.0]))
    anchor_color = tuple(_get(vis_cfg, "anchor_color", [1.0, 0.0, 0.0]))
    boundary_alpha = float(_get(vis_cfg, "boundary_alpha", 0.08))
    boundary_size = float(_get(vis_cfg, "boundary_size", 2.0))

    plotter = pv.Plotter()

    if boundary is not None and boundary.size > 0:
        cloud = pv.PolyData(boundary)
        plotter.add_mesh(
            cloud,
            color=(0.40, 0.78, 1.0),
            opacity=boundary_alpha,
            point_size=boundary_size,
            render_points_as_spheres=True,
        )

    anchor = pv.PolyData(anchor_pos.reshape(1, 3))
    plotter.add_mesh(
        anchor,
        color=anchor_color,
        point_size=12.0,
        render_points_as_spheres=True,
    )

    if q_list:
        # initial
        _add_axes(
            plotter,
            anchor_pos,
            q_list[0],
            axis_len,
            color_final if start_reachable else color_init,
            axis_width,
            dashed=False,
            dash_segments=dash_segments,
            show_z_arrow=show_z_arrow,
            z_arrow_scale=z_arrow_scale,
        )
        # intermediates
        for q in q_list[1:-1]:
            _add_axes(
                plotter,
                anchor_pos,
                q,
                axis_len,
                color_mid,
                axis_width,
                dashed=True,
                dash_segments=dash_segments,
                show_z_arrow=show_z_arrow,
                z_arrow_scale=z_arrow_scale,
            )
        # final
        if len(q_list) > 1:
            _add_axes(
                plotter,
                anchor_pos,
                q_list[-1],
                axis_len,
                color_final if reached else color_init,
                axis_width,
                dashed=False,
                dash_segments=dash_segments,
                show_z_arrow=show_z_arrow,
                z_arrow_scale=z_arrow_scale,
            )

    plotter.add_title("Orientation Pullback (anchor-fixed)")
    screenshot = _get(vis_cfg, "screenshot", None)
    if screenshot:
        plotter.show(screenshot=screenshot)
    else:
        plotter.show()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer Orientation-SDF at a fixed anchor.")
    parser.add_argument("--config", type=str, default="configs/infer_orient.yaml", help="Config path.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    repo_root = Path(_get_path(cfg, "paths.repo_root", ".")).resolve()

    device = torch.device(str(_get_path(cfg, "run.device", "cpu")))

    # paths
    orient_h5 = _resolve_path(_get_path(cfg, "paths.orientation_h5"), repo_root)
    morph_path = _resolve_path(_get_path(cfg, "paths.morphology_spec"), repo_root)
    ckpt_path = _resolve_path(_get_path(cfg, "paths.orient_checkpoint"), repo_root)

    # anchor
    anchor_id = _get_path(cfg, "input.anchor_id", None)
    anchor_pos_cfg = _get_path(cfg, "input.anchor_pos", None)
    anchor_pos = None
    anchor_s = None

    try:
        import h5py
    except ImportError:
        print("Missing dependency: h5py (python -m pip install h5py)")
        return

    with h5py.File(orient_h5, "r") as h5:
        if anchor_pos_cfg is not None:
            anchor_pos = _parse_pos(anchor_pos_cfg)
        elif anchor_id is not None:
            anchor_id = int(anchor_id)
            anchor_pos = np.asarray(h5["/anchors/pos"][anchor_id], dtype=np.float32)
        else:
            raise ValueError("Set input.anchor_id or input.anchor_pos in config.")

        if "/anchors/s_v_norm" in h5:
            if anchor_id is None:
                anchor_id = int(_get_path(cfg, "input.anchor_id", 0))
            anchor_s = float(np.asarray(h5["/anchors/s_v_norm"][anchor_id]))
        else:
            if anchor_id is None:
                anchor_id = int(_get_path(cfg, "input.anchor_id", 0))
            anchor_s = float(np.asarray(h5["/anchors/s_v"][anchor_id]))

    anchor_s = abs(anchor_s)

    # euler to quaternion
    euler_deg = _get_path(cfg, "input.euler_deg", [0.0, 0.0, 0.0])
    euler_order = str(_get_path(cfg, "input.euler_order", "xyz"))
    euler_intrinsic = bool(_get_path(cfg, "input.euler_intrinsic", True))
    q0 = _euler_to_quat(euler_deg, order=euler_order, intrinsic=euler_intrinsic)
    q0 = quat_normalize(q0).to(device)

    # morphology
    morph_spec = load_morphology_spec(morph_path)
    l_ref = float(morph_spec.l_ref)

    # model
    q_ref = _load_q_ref_from_h5(orient_h5)
    if q_ref is None:
        raise RuntimeError("q_ref not found in orientation dataset; cannot infer.")
    q_ref_t = torch.from_numpy(q_ref).to(device)
    ref_encoder = ReferenceQuaternionEncoder(q_ref=q_ref_t).to(device)

    k_p = int(_get_path(cfg, "model.position.k_p", 10))

    morph_cfg = _get_path(cfg, "model.morph", None)
    morph_encoder = MorphologyEncoder(
        d_model=int(_get(morph_cfg, "d_model", 256)),
        depth_emb_dim=int(_get(morph_cfg, "depth_emb_dim", 16)),
        num_layers=int(_get(morph_cfg, "num_layers", 6)),
        num_heads=int(_get(morph_cfg, "num_heads", 8)),
        dropout=float(_get(morph_cfg, "dropout", 0.1)),
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    if "morph_encoder" in ckpt:
        morph_encoder.load_state_dict(ckpt["morph_encoder"])
    orient_state = ckpt["orient_model"] if "orient_model" in ckpt else ckpt
    arch = _infer_orient_arch(orient_state)
    orient_in_dim = arch.get("orient_in_dim", 2 * int(q_ref.shape[0]))
    if orient_in_dim != 2 * int(q_ref.shape[0]):
        raise RuntimeError(
            f"q_ref size mismatch: orient_in_dim={orient_in_dim} expects k_r={orient_in_dim//2}, "
            f"but dataset has k_r={int(q_ref.shape[0])}. Use matching dataset/ckpt."
        )
    orient_model = OrientationSDFModel(
        orient_in_dim=orient_in_dim,
        context_dim=int(arch.get("context_dim", _get_path(cfg, "model.orientation.context_dim", 128))),
        hidden_dim=int(arch.get("hidden_dim", _get_path(cfg, "model.orientation.hidden_dim", 256))),
        num_layers=int(arch.get("num_layers", _get_path(cfg, "model.orientation.num_layers", 4))),
        experts=int(arch.get("experts", _get_path(cfg, "model.orientation.experts", 4))),
        cond_in_dim=int(arch.get("cond_in_dim", int(_get(morph_cfg, "d_model", 256)) + (3 + 2 * 3 * k_p) + 1)),
    ).to(device)
    orient_model.load_state_dict(orient_state)

    morph_encoder.eval()
    orient_model.eval()

    with torch.no_grad():
        morph_emb, _ = morph_encoder(morph_spec, device)

    # fixed anchor cond
    p = torch.from_numpy(anchor_pos[None, :]).to(device)
    p_norm = p / l_ref
    e_p = position_encoding(p_norm, k_p=k_p)
    s_in = torch.tensor([[float(anchor_s)]], device=device).clamp(-1.0, 1.0)
    cond = torch.cat([morph_emb.expand(1, -1), e_p, s_in], dim=1)

    tau = float(_get_path(cfg, "infer.tau", 0.05))

    def _predict(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if q.ndim == 1:
            q_in = q[None, :]
        else:
            q_in = q
        e_r = ref_encoder(q_in)
        u_raw, u = orient_model(e_r, cond.expand(q_in.shape[0], -1), tau=tau)
        phi_raw = math.pi * u
        phi_pred = math.pi * torch.tanh(phi_raw / math.pi)
        return phi_raw, phi_pred

    q_list: list[np.ndarray] = [q0.detach().cpu().numpy()]
    phi_raw, phi_pred = _predict(q0)
    phi_raw_val = float(phi_raw.item())
    phi_pred_val = float(phi_pred.item())

    print("=== Orientation Reachability (anchor-fixed) ===")
    print(f"anchor_pos = {anchor_pos.tolist()} (m)")
    print(f"anchor_s = {anchor_s:.6f}")
    print(f"euler_deg = {list(euler_deg)} (order={euler_order}, intrinsic={euler_intrinsic})")
    print(f"phi_raw = {phi_raw_val:.6f} rad, phi = {phi_pred_val:.6f} rad")

    start_reachable = phi_raw_val >= 0.0
    reached = start_reachable
    max_iters = int(_get_path(cfg, "infer.max_iters", 20))
    eta = float(_get_path(cfg, "infer.step_scale", 1.0))
    max_step = float(_get_path(cfg, "infer.max_step", 0.5))
    min_step = float(_get_path(cfg, "infer.min_step", 1e-4))
    eps = float(_get_path(cfg, "infer.eps", 1e-8))
    target_phi = float(_get_path(cfg, "infer.target_phi_raw", 0.0))
    stop_phi = float(_get_path(cfg, "infer.stop_phi_raw", 0.0))
    step_mode = str(_get_path(cfg, "infer.step_mode", "target")).lower()
    verbose = bool(_get_path(cfg, "infer.verbose", True))

    reached = phi_raw_val >= target_phi
    reached = phi_raw_val >= stop_phi
    if not reached:
        q_cur = q0
        for it in range(1, max_iters + 1):
            delta = torch.zeros((1, 3), device=device, requires_grad=True)
            q_new = quat_mul(exp_quat(delta), q_cur[None, :]).squeeze(0)
            phi_raw_it, phi_pred_it = _predict(q_new)
            phi_raw_val = float(phi_raw_it.item())
            phi_pred_val = float(phi_pred_it.item())
            if phi_raw_val >= stop_phi:
                reached = True
                print(
                    f"Reached at iter={it}: phi_raw={phi_raw_val:.6f} "
                    f"(target={target_phi:.6f}, stop={stop_phi:.6f})"
                )
                q_list.append(q_new.detach().cpu().numpy())
                break

            grad = torch.autograd.grad(phi_raw_it.sum(), delta, create_graph=False)[0]
            g = grad.reshape(-1)
            g_norm = torch.linalg.norm(g) + eps
            if step_mode == "zero":
                step = eta * ((-phi_raw_it) / g_norm)
            else:
                step = eta * ((target_phi - phi_raw_it) / g_norm)
            step = torch.clamp(step, min=min_step, max=max_step)
            delta_step = g * step
            q_next = quat_mul(exp_quat(delta_step[None, :]), q_cur[None, :]).squeeze(0)
            q_next = quat_normalize(q_next)
            q_list.append(q_next.detach().cpu().numpy())

            if verbose:
                print(f"--- iter {it} ---")
                print(f"phi_raw={phi_raw_val:.6f}, phi={phi_pred_val:.6f}")
                print(f"grad_norm={float(g_norm.item()):.6f}, step={float(step.item()):.6f}")

            q_cur = q_next
        if not reached:
            print(f"Reached max_iters={max_iters}, last phi_raw={phi_raw_val:.6f}")

    vis_enabled = bool(_get_path(cfg, "vis.enabled", True))
    if vis_enabled:
        boundary = None
        boundary_h5 = _get_path(cfg, "vis.boundary_h5", None) or _get_path(cfg, "paths.position_h5", None)
        if boundary_h5 is not None:
            boundary = _load_boundary_points(
                _resolve_path(boundary_h5, repo_root),
                int(_get_path(cfg, "vis.boundary_max_points", 50000)),
                int(_get_path(cfg, "vis.boundary_seed", 42)),
            )
        _visualize(anchor_pos, q_list, start_reachable, reached, boundary, cfg)


if __name__ == "__main__":
    main()
