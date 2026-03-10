"""Run joint pose inference by chaining position and orientation projection.

This script first repairs position reachability and then refines orientation at the resulting anchor.
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
from mcfp.models.pos_sdf import PositionSDFModel
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


def _parse_pose(values: Sequence[float]) -> tuple[np.ndarray, Sequence[float]]:
    if len(values) != 6:
        raise ValueError("Pose must have 6 values: x y z roll pitch yaw")
    pos = np.asarray(values[:3], dtype=np.float32)
    euler = [float(values[3]), float(values[4]), float(values[5])]
    return pos, euler


def _project_to_sphere(p: np.ndarray, center: np.ndarray, radius: float) -> tuple[np.ndarray, bool]:
    v = p - center
    dist = float(np.linalg.norm(v))
    if dist <= radius:
        return p, False
    if dist < 1e-9:
        return center + np.array([radius, 0.0, 0.0], dtype=np.float32), True
    return (center + v / dist * radius).astype(np.float32), True


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


def _infer_pos_arch(state: dict[str, torch.Tensor]) -> dict[str, int]:
    layers = 0
    in_dim = None
    hidden_dim = None
    cond_dim = None
    for k, v in state.items():
        if k.startswith("decoder.layers.") and k.endswith(".weight"):
            parts = k.split(".")
            try:
                layer_idx = int(parts[2])
            except Exception:
                continue
            layers = max(layers, layer_idx + 1)
            if layer_idx == 0:
                in_dim = int(v.shape[1])
                hidden_dim = int(v.shape[0])
        if k == "decoder.film.0.weight":
            cond_dim = int(v.shape[1])

    out = {"num_layers": layers if layers > 0 else 5}
    if in_dim is not None:
        out["in_dim"] = in_dim
    if hidden_dim is not None:
        out["hidden_dim"] = hidden_dim
    if cond_dim is not None:
        out["cond_dim"] = cond_dim
    return out


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
            q = quat_mul(q, q_axis)
        else:
            q = quat_mul(q_axis, q)
    return quat_normalize(q)


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
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


def _add_z_axis_only(
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

    z = _quat_rotate(quat, np.array([0.0, 0.0, 1.0], dtype=np.float32))
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


def _visualize_pose_result(
    cfg: Any,
    boundary: np.ndarray | None,
    pos_path: np.ndarray,
    pos_reachable: bool,
    q_pos_stage: np.ndarray,
    q_orient_path: list[np.ndarray],
    orient_start_reachable: bool,
    orient_reached: bool,
) -> None:
    try:
        import pyvista as pv
    except ImportError:
        print("Missing dependency: pyvista (python -m pip install pyvista)")
        return

    vis_cfg = _get_path(cfg, "vis", None)
    boundary_alpha = float(_get(vis_cfg, "boundary_alpha", 0.08))
    boundary_size = float(_get(vis_cfg, "boundary_size", 2.0))
    boundary_surface = bool(_get(vis_cfg, "boundary_surface", True))
    boundary_surface_alpha = float(_get(vis_cfg, "boundary_surface_alpha", 0.05))
    boundary_surface_opacity = float(_get(vis_cfg, "boundary_surface_opacity", 0.15))
    point_size = float(_get(vis_cfg, "path_point_size", 8.0))
    pos_axis_length = float(_get(vis_cfg, "pos_axis_length", 0.04))
    orient_axis_length = float(_get(vis_cfg, "orient_axis_length", 0.10))
    orient_mid_stride = max(1, int(_get(vis_cfg, "orient_mid_stride", 5)))
    axis_width = float(_get(vis_cfg, "axis_width", 2.0))
    dash_segments = int(_get(vis_cfg, "dash_segments", 10))
    show_z_arrow = bool(_get(vis_cfg, "show_z_arrow", True))
    z_arrow_scale = float(_get(vis_cfg, "z_arrow_scale", 1.0))

    c_reach = tuple(_get(vis_cfg, "color_reachable", [0.0, 0.8, 0.0]))
    c_unreach = tuple(_get(vis_cfg, "color_unreachable", [1.0, 0.0, 0.0]))
    c_mid = tuple(_get(vis_cfg, "color_mid", [0.7, 0.7, 0.7]))
    c_path = tuple(_get(vis_cfg, "color_path", [1.0, 0.6, 0.0]))

    plotter = pv.Plotter()

    if boundary is not None and boundary.size > 0:
        cloud = pv.PolyData(boundary)
        if boundary_surface:
            try:
                surf = cloud.delaunay_3d(alpha=boundary_surface_alpha).extract_surface()
                plotter.add_mesh(
                    surf,
                    color=(0.40, 0.78, 1.0),
                    opacity=boundary_surface_opacity,
                    show_edges=False,
                )
            except Exception as exc:
                print(f"[vis] boundary surface reconstruction failed: {exc}")
        plotter.add_mesh(
            cloud,
            color=(0.40, 0.78, 1.0),
            opacity=boundary_alpha,
            point_size=boundary_size,
            render_points_as_spheres=True,
        )

    if pos_path.shape[0] >= 2:
        line = pv.lines_from_points(pos_path)
        plotter.add_mesh(line, color=c_path, line_width=3.0)

    # position points: start/mid/final
    start = pv.PolyData(pos_path[:1])
    plotter.add_mesh(
        start,
        color=c_unreach,
        point_size=point_size + 4.0,
        render_points_as_spheres=True,
    )
    if pos_path.shape[0] > 2:
        mid = pv.PolyData(pos_path[1:-1])
        plotter.add_mesh(
            mid,
            color=c_path,
            point_size=point_size,
            render_points_as_spheres=True,
        )
    final = pv.PolyData(pos_path[-1:])
    plotter.add_mesh(
        final,
        color=c_reach if pos_reachable else c_unreach,
        point_size=point_size + 4.0,
        render_points_as_spheres=True,
    )

    # gray small axes on position stage points (before orientation stage)
    pos_axis_points = pos_path[:-1] if pos_path.shape[0] > 1 else pos_path
    for p in pos_axis_points:
        _add_axes(
            plotter,
            p,
            q_pos_stage,
            pos_axis_length,
            c_mid,
            max(1.0, axis_width - 0.5),
            dashed=False,
            dash_segments=dash_segments,
            show_z_arrow=show_z_arrow,
            z_arrow_scale=z_arrow_scale,
        )

    # orientation stage axes at final position
    anchor_pos = pos_path[-1]
    if q_orient_path:
        _add_axes(
            plotter,
            anchor_pos,
            q_orient_path[0],
            orient_axis_length,
            c_reach if orient_start_reachable else c_unreach,
            axis_width,
            dashed=False,
            dash_segments=dash_segments,
            show_z_arrow=show_z_arrow,
            z_arrow_scale=z_arrow_scale,
        )
        for i, q in enumerate(q_orient_path[1:-1], start=1):
            if i % orient_mid_stride != 0:
                continue
            _add_z_axis_only(
                plotter,
                anchor_pos,
                q,
                orient_axis_length,
                c_mid,
                axis_width,
                dashed=True,
                dash_segments=dash_segments,
                show_z_arrow=show_z_arrow,
                z_arrow_scale=z_arrow_scale,
            )
        if len(q_orient_path) > 1:
            _add_axes(
                plotter,
                anchor_pos,
                q_orient_path[-1],
                orient_axis_length,
                c_reach if orient_reached else c_unreach,
                axis_width,
                dashed=False,
                dash_segments=dash_segments,
                show_z_arrow=show_z_arrow,
                z_arrow_scale=z_arrow_scale,
            )

    plotter.add_title("Joint Pose Inference (Position -> Orientation)")
    screenshot = _get(vis_cfg, "screenshot", None)
    if screenshot:
        plotter.show(screenshot=screenshot)
    else:
        plotter.show()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Joint position+orientation inference.")
    parser.add_argument("--config", type=str, default="configs/infer_pose.yaml", help="Config path.")
    parser.add_argument("--pose", type=float, nargs=6, default=None, help="x y z roll pitch yaw")
    parser.add_argument("--vis", action="store_true", help="Enable visualization")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    repo_root = Path(_get_path(cfg, "paths.repo_root", ".")).resolve()
    device = torch.device(str(_get_path(cfg, "run.device", "cpu")))

    # paths
    pos_h5 = _resolve_path(_get_path(cfg, "paths.position_h5"), repo_root)
    orient_h5 = _resolve_path(_get_path(cfg, "paths.orientation_h5"), repo_root)
    morph_path = _resolve_path(_get_path(cfg, "paths.morphology_spec"), repo_root)
    pos_ckpt_path = _resolve_path(_get_path(cfg, "paths.position_checkpoint"), repo_root)
    orient_ckpt_path = _resolve_path(_get_path(cfg, "paths.orient_checkpoint"), repo_root)

    # input pose
    if args.pose is not None:
        pos_in, euler_deg = _parse_pose(args.pose)
    else:
        pose_cfg = _get_path(cfg, "input.pose", None)
        if pose_cfg is None:
            raise ValueError("Set input.pose in config or pass --pose.")
        pos_in, euler_deg = _parse_pose(pose_cfg)
    euler_order = str(_get_path(cfg, "input.euler_order", "xyz"))
    euler_intrinsic = bool(_get_path(cfg, "input.euler_intrinsic", True))
    q0 = quat_normalize(_euler_to_quat(euler_deg, order=euler_order, intrinsic=euler_intrinsic)).to(device)

    # morphology and scale
    morph_spec = load_morphology_spec(morph_path)
    l_ref = float(morph_spec.l_ref)

    # position models
    pos_morph_cfg = _get_path(cfg, "model.position_morph", _get_path(cfg, "model.morph", None))
    pos_model_cfg = _get_path(cfg, "model.position", None)
    k_p = int(_get(pos_model_cfg, "k_p", 10))
    pos_morph_encoder = MorphologyEncoder(
        d_model=int(_get(pos_morph_cfg, "d_model", 256)),
        depth_emb_dim=int(_get(pos_morph_cfg, "depth_emb_dim", 16)),
        num_layers=int(_get(pos_morph_cfg, "num_layers", 6)),
        num_heads=int(_get(pos_morph_cfg, "num_heads", 8)),
        dropout=float(_get(pos_morph_cfg, "dropout", 0.1)),
    ).to(device)
    pos_ckpt = torch.load(pos_ckpt_path, map_location=device)
    if "morph_encoder" in pos_ckpt:
        pos_morph_encoder.load_state_dict(pos_ckpt["morph_encoder"])
    pos_state = pos_ckpt["pos_model"] if "pos_model" in pos_ckpt else pos_ckpt
    pos_arch = _infer_pos_arch(pos_state)
    in_dim = int(pos_arch.get("in_dim", 3 + 2 * 3 * k_p))
    expected_in_dim = 3 + 2 * 3 * k_p
    if in_dim != expected_in_dim:
        raise RuntimeError(
            f"Position model input dim mismatch: ckpt in_dim={in_dim}, expected={expected_in_dim}. "
            "Check k_p / position model config."
        )
    pos_model = PositionSDFModel(
        in_dim=in_dim,
        hidden_dim=int(pos_arch.get("hidden_dim", _get(pos_model_cfg, "hidden_dim", 256))),
        num_layers=int(pos_arch.get("num_layers", _get(pos_model_cfg, "num_layers", 5))),
        w0_first=float(_get(pos_model_cfg, "w0_first", 30.0)),
        w0=float(_get(pos_model_cfg, "w0", 1.0)),
        cond_dim=int(pos_arch.get("cond_dim", _get(pos_morph_cfg, "d_model", 256))),
    ).to(device)
    pos_model.load_state_dict(pos_state)
    pos_morph_encoder.eval()
    pos_model.eval()
    with torch.no_grad():
        pos_morph_emb, _ = pos_morph_encoder(morph_spec, device)

    # orientation models
    q_ref = _load_q_ref_from_h5(orient_h5)
    if q_ref is None:
        raise RuntimeError("q_ref not found in orientation dataset.")
    ref_encoder = ReferenceQuaternionEncoder(q_ref=torch.from_numpy(q_ref).to(device)).to(device)
    orient_morph_cfg = _get_path(cfg, "model.orientation_morph", _get_path(cfg, "model.morph", None))
    orient_morph_encoder = MorphologyEncoder(
        d_model=int(_get(orient_morph_cfg, "d_model", 256)),
        depth_emb_dim=int(_get(orient_morph_cfg, "depth_emb_dim", 16)),
        num_layers=int(_get(orient_morph_cfg, "num_layers", 6)),
        num_heads=int(_get(orient_morph_cfg, "num_heads", 8)),
        dropout=float(_get(orient_morph_cfg, "dropout", 0.1)),
    ).to(device)
    orient_ckpt = torch.load(orient_ckpt_path, map_location=device)
    if "morph_encoder" in orient_ckpt:
        orient_morph_encoder.load_state_dict(orient_ckpt["morph_encoder"])
    orient_state = orient_ckpt["orient_model"] if "orient_model" in orient_ckpt else orient_ckpt
    arch = _infer_orient_arch(orient_state)
    orient_in_dim = int(arch.get("orient_in_dim", 2 * int(q_ref.shape[0])))
    if orient_in_dim != 2 * int(q_ref.shape[0]):
        raise RuntimeError(
            f"q_ref size mismatch: orient_in_dim={orient_in_dim}, dataset q_ref K={int(q_ref.shape[0])}."
        )
    orient_model = OrientationSDFModel(
        orient_in_dim=orient_in_dim,
        context_dim=int(arch.get("context_dim", 128)),
        hidden_dim=int(arch.get("hidden_dim", 256)),
        num_layers=int(arch.get("num_layers", 4)),
        experts=int(arch.get("experts", 4)),
        cond_in_dim=int(arch.get("cond_in_dim", int(_get(orient_morph_cfg, "d_model", 256)) + (3 + 2 * 3 * k_p) + 1)),
    ).to(device)
    orient_model.load_state_dict(orient_state)
    orient_morph_encoder.eval()
    orient_model.eval()
    with torch.no_grad():
        orient_morph_emb, _ = orient_morph_encoder(morph_spec, device)

    # ---------- Stage A: position pullback ----------
    pos_raw = pos_in.copy()
    clamp_sphere = bool(_get_path(cfg, "position_infer.clamp_sphere", True))
    sphere_scale = float(_get_path(cfg, "position_infer.sphere_scale", 1.0))
    sphere_center_cfg = _get_path(cfg, "position_infer.sphere_center", [0.0, 0.0, 0.0])
    sphere_center = np.asarray(sphere_center_cfg, dtype=np.float32).reshape(3)
    sphere_radius = float(l_ref) * sphere_scale
    moved = False
    pos = pos_raw
    if clamp_sphere and sphere_radius > 0:
        pos, moved = _project_to_sphere(pos, sphere_center, sphere_radius)
        if moved:
            print(f"[joint] input position outside sphere, projected to {pos.tolist()}")

    pos_path: list[np.ndarray] = [pos_raw.astype(np.float32)]
    if moved and not np.allclose(pos_raw, pos, atol=1e-6):
        pos_path.append(pos.astype(np.float32))

    p = torch.from_numpy(pos[None, :]).to(device)
    p_norm = (p / l_ref).requires_grad_(True)
    e_p = position_encoding(p_norm, k_p=k_p)
    s_pred = pos_model(e_p, pos_morph_emb)
    s_norm = float(s_pred.item())
    s_meter = s_norm * l_ref
    tol = abs(float(_get_path(cfg, "position_infer.tol", 0.0)))
    target_s_meter = _get_path(cfg, "position_infer.target_s_meter", 0.0)
    target_s_norm = float(target_s_meter) / l_ref
    pos_reachable = s_norm >= (target_s_norm - tol)

    print("=== Stage A: Position Reachability ===")
    print(f"input_pos = {pos_raw.tolist()} (m)")
    print(f"final_eval_pos = {pos.tolist()} (m)")
    print(f"sdf_norm = {s_norm:.6f}, sdf_meter = {s_meter:.6f}")
    print(f"reachable = {bool(pos_reachable)} (target_s_norm={target_s_norm:.6f}, tol={tol:.6f})")

    if not pos_reachable:
        step_scale = float(_get_path(cfg, "position_infer.step_scale", 1.0))
        max_step = _get_path(cfg, "position_infer.max_step", None)
        min_step = float(_get_path(cfg, "position_infer.min_step", 1e-4))
        eps = float(_get_path(cfg, "position_infer.eps", 1e-8))
        max_iters = int(_get_path(cfg, "position_infer.max_iters", 50))
        backtrack = bool(_get_path(cfg, "position_infer.backtrack", True))
        backtrack_factor = float(_get_path(cfg, "position_infer.backtrack_factor", 0.5))
        backtrack_iters = int(_get_path(cfg, "position_infer.backtrack_iters", 5))
        verbose = bool(_get_path(cfg, "position_infer.verbose", True))

        p_cur = p.detach()
        for it in range(1, max_iters + 1):
            p_cur = p_cur.detach().requires_grad_(True)
            e_cur = position_encoding(p_cur / l_ref, k_p=k_p)
            s_pred = pos_model(e_cur, pos_morph_emb)
            s_norm = float(s_pred.item())
            s_meter = s_norm * l_ref
            if s_norm >= (target_s_norm - tol):
                pos_reachable = True
                print(f"Reached position target at iter={it}: sdf_norm={s_norm:.6f}, sdf_meter={s_meter:.6f}")
                break

            grad = torch.autograd.grad(s_pred.sum(), p_cur, create_graph=False)[0]
            grad_norm = torch.linalg.norm(grad, dim=1, keepdim=True)
            grad_unit = grad / torch.clamp(grad_norm, min=eps)
            step = ((target_s_norm - s_pred) * l_ref) * step_scale
            if max_step is not None:
                step = torch.clamp(step, max=float(max_step))
            step = torch.clamp(step, min=min_step)

            if backtrack:
                step_try = step
                for _ in range(backtrack_iters):
                    p_try = (p_cur + grad_unit * step_try.unsqueeze(-1)).detach()
                    with torch.no_grad():
                        s_try = pos_model(position_encoding(p_try / l_ref, k_p=k_p), pos_morph_emb)
                    if float(s_try.item()) > s_norm:
                        step = step_try
                        break
                    step_try = step_try * backtrack_factor

            p_next = (p_cur + grad_unit * step.unsqueeze(-1)).detach()
            if clamp_sphere and sphere_radius > 0:
                p_next_np = p_next.detach().cpu().numpy().reshape(-1)
                p_next_np, _ = _project_to_sphere(p_next_np, sphere_center, sphere_radius)
                p_next = torch.from_numpy(p_next_np.reshape(1, 3)).to(device)

            pos_path.append(p_next.detach().cpu().numpy().reshape(-1).astype(np.float32))
            if verbose:
                print(f"[pos][iter {it}] sdf_norm={s_norm:.6f}, step={float(step.item()):.6f}")
            p_cur = p_next

        p = p_cur.detach()
        with torch.no_grad():
            s_pred = pos_model(position_encoding(p / l_ref, k_p=k_p), pos_morph_emb)
        s_norm = float(s_pred.item())
        s_meter = s_norm * l_ref
    pos_final = p.detach().cpu().numpy().reshape(-1).astype(np.float32)
    if not np.allclose(pos_path[-1], pos_final):
        pos_path.append(pos_final)

    # ---------- Stage B: orientation pullback at final position ----------
    tau = float(_get_path(cfg, "orientation_infer.tau", 0.05))
    phi_target = float(_get_path(cfg, "orientation_infer.target_phi_raw", 0.0))
    phi_stop = float(_get_path(cfg, "orientation_infer.stop_phi_raw", 0.0))
    step_mode = str(_get_path(cfg, "orientation_infer.step_mode", "target")).lower()
    max_iters = int(_get_path(cfg, "orientation_infer.max_iters", 60))
    eta = float(_get_path(cfg, "orientation_infer.step_scale", 1.0))
    max_step = float(_get_path(cfg, "orientation_infer.max_step", 0.5))
    min_step = float(_get_path(cfg, "orientation_infer.min_step", 1e-4))
    eps = float(_get_path(cfg, "orientation_infer.eps", 1e-8))
    verbose = bool(_get_path(cfg, "orientation_infer.verbose", True))

    p_orient = torch.from_numpy(pos_final[None, :]).to(device)
    e_p_orient = position_encoding(p_orient / l_ref, k_p=k_p)
    s_in = torch.tensor([[float(abs(s_norm))]], device=device).clamp(-1.0, 1.0)
    cond = torch.cat([orient_morph_emb.expand(1, -1), e_p_orient, s_in], dim=1)

    def _predict_phi(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q_in = q[None, :] if q.ndim == 1 else q
        e_r = ref_encoder(q_in)
        u_raw, u = orient_model(e_r, cond.expand(q_in.shape[0], -1), tau=tau)
        phi_raw = math.pi * u
        phi = math.pi * torch.tanh(phi_raw / math.pi)
        return phi_raw, phi

    q_orient_path: list[np.ndarray] = [q0.detach().cpu().numpy()]
    phi_raw, phi = _predict_phi(q0)
    phi_raw_val = float(phi_raw.item())
    phi_val = float(phi.item())
    orient_start_reachable = phi_raw_val >= 0.0
    orient_reached = phi_raw_val >= phi_stop

    print("=== Stage B: Orientation Reachability (at final position) ===")
    print(f"pos_final = {pos_final.tolist()} (m)")
    print(f"s_in_from_stageA = {abs(s_norm):.6f} (normalized)")
    print(f"euler_deg = {list(euler_deg)} (order={euler_order}, intrinsic={euler_intrinsic})")
    print(f"phi_raw = {phi_raw_val:.6f} rad, phi = {phi_val:.6f} rad")
    print(f"reachable = {bool(orient_reached)} (stop_phi_raw={phi_stop:.6f})")

    if not orient_reached:
        q_cur = q0
        for it in range(1, max_iters + 1):
            delta = torch.zeros((1, 3), device=device, requires_grad=True)
            q_new = quat_mul(exp_quat(delta), q_cur[None, :]).squeeze(0)
            phi_raw_it, phi_it = _predict_phi(q_new)
            phi_raw_val = float(phi_raw_it.item())
            phi_val = float(phi_it.item())
            if phi_raw_val >= phi_stop:
                orient_reached = True
                q_orient_path.append(q_new.detach().cpu().numpy())
                print(f"Reached orientation target at iter={it}: phi_raw={phi_raw_val:.6f}")
                break

            grad = torch.autograd.grad(phi_raw_it.sum(), delta, create_graph=False)[0]
            g = grad.reshape(-1)
            g_norm = torch.linalg.norm(g) + eps
            if step_mode == "zero":
                step = eta * ((-phi_raw_it) / g_norm)
            else:
                step = eta * ((phi_target - phi_raw_it) / g_norm)
            step = torch.clamp(step, min=min_step, max=max_step)
            delta_step = g * step
            q_next = quat_mul(exp_quat(delta_step[None, :]), q_cur[None, :]).squeeze(0)
            q_next = quat_normalize(q_next)
            q_orient_path.append(q_next.detach().cpu().numpy())

            if verbose:
                print(f"[ori][iter {it}] phi_raw={phi_raw_val:.6f}, step={float(step.item()):.6f}")
            q_cur = q_next

    # ---------- visualization ----------
    vis_enabled = bool(_get_path(cfg, "vis.enabled", True)) or bool(args.vis)
    if vis_enabled:
        boundary_h5 = _get_path(cfg, "vis.boundary_h5", None) or str(pos_h5)
        boundary = _load_boundary_points(
            _resolve_path(boundary_h5, repo_root),
            int(_get_path(cfg, "vis.boundary_max_points", 50000)),
            int(_get_path(cfg, "vis.boundary_seed", 42)),
        )
        _visualize_pose_result(
            cfg=cfg,
            boundary=boundary,
            pos_path=np.asarray(pos_path, dtype=np.float32),
            pos_reachable=pos_reachable,
            q_pos_stage=q0.detach().cpu().numpy(),
            q_orient_path=q_orient_path,
            orient_start_reachable=orient_start_reachable,
            orient_reached=orient_reached,
        )


if __name__ == "__main__":
    main()
