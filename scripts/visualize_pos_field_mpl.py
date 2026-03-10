"""Sample the learned Position-SDF field and render it for inspection.

This script evaluates a trained position model on a cube of query points and visualizes the result.
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
        raise ValueError("center must have 3 values: x y z")
    return np.asarray(values, dtype=np.float32)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample Position-SDF field in a cube and visualize with PyVista.")
    parser.add_argument("--config", type=str, default="configs/infer_pos.yaml", help="Config path.")
    parser.add_argument("--out", type=str, default=None, help="Optional screenshot PNG path.")
    parser.add_argument("--num", type=int, default=None, help="Override number of samples.")
    parser.add_argument("--size", type=float, default=None, help="Override cube size (m).")
    parser.add_argument("--center", type=float, nargs=3, default=None, help="Override cube center x y z.")
    parser.add_argument("--no-show", action="store_true", help="Do not show PyVista window.")
    parser.add_argument("--show-slices", action="store_true", help="Show XYZ slice subplots")
    parser.add_argument("--slice-thickness", type=float, default=None, help="Slice thickness (m)")
    parser.add_argument("--slice-center", type=float, nargs=3, default=None, help="Slice center x y z (m)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    repo_root = Path(_get_path(cfg, "paths.repo_root", ".")).resolve()
    device = torch.device(str(_get_path(cfg, "run.device", "cpu")))

    field_cfg = _get_path(cfg, "field", None)
    n = int(args.num or _get(field_cfg, "num_points", 1000))
    size = float(args.size or _get(field_cfg, "cube_size", 1.0))
    center = _parse_pos(args.center or _get(field_cfg, "cube_center", [0.0, 0.0, 0.0]))
    seed = int(_get(field_cfg, "seed", 42))
    out_png = args.out or _get(field_cfg, "output_png", "runs/pos_sdf/field_sample.png")
    out_png = _resolve_path(out_png, repo_root)
    slice_th = float(args.slice_thickness or _get(field_cfg, "slice_thickness", 0.02))
    slice_center = args.slice_center or _get(field_cfg, "slice_center", center.tolist())

    # models
    morph_path = _resolve_path(_get_path(cfg, "paths.morphology_spec"), repo_root)
    ckpt_path = _resolve_path(_get_path(cfg, "paths.position_checkpoint"), repo_root)
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

    morph_spec = load_morphology_spec(morph_path)
    l_ref = float(morph_spec.l_ref)
    with torch.no_grad():
        morph_emb, _ = morph_encoder(morph_spec, device)

    rng = np.random.default_rng(seed)
    pts = center + (rng.random((n, 3)) - 0.5) * size

    with torch.no_grad():
        p = torch.from_numpy(pts.astype(np.float32)).to(device)
        e_p = position_encoding(p / l_ref, k_p=k_p)
        s_pred = pos_model(e_p, morph_emb)
        sdf = s_pred.detach().cpu().numpy().reshape(-1)

    max_abs = float(_get(field_cfg, "max_abs", 0.0))
    if max_abs <= 0:
        max_abs = float(np.max(np.abs(sdf))) if sdf.size > 0 else 1.0
    max_abs = max(max_abs, 1e-6)
    mag = np.clip(np.abs(sdf) / max_abs, 0.0, 1.0)

    try:
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("pyvista not installed. Run: python -m pip install pyvista") from exc

    pos_mask = sdf >= 0
    neg_mask = ~pos_mask

    point_size = float(_get(field_cfg, "point_size", 3.0))
    min_alpha = float(_get(field_cfg, "min_alpha", 0.05))
    max_alpha = float(_get(field_cfg, "max_alpha", 0.9))
    pos_color = _get(field_cfg, "pos_color", [0.2, 0.6, 1.0])
    neg_color = _get(field_cfg, "neg_color", [1.0, 0.3, 0.3])
    title = _get(field_cfg, "title", "Position-SDF Field Samples (+ blue / - red)")

    if args.show_slices:
        plotter = pv.Plotter(shape=(2, 2))
        plotter.subplot(0, 0)
    else:
        plotter = pv.Plotter()

    def add_cloud(mask, color):
        if not np.any(mask):
            return
        cloud = pv.PolyData(pts[mask])
        rgba = np.zeros((cloud.n_points, 4), dtype=np.uint8)
        rgb = (np.array(color) * 255.0).astype(np.uint8)
        rgba[:, 0] = rgb[0]
        rgba[:, 1] = rgb[1]
        rgba[:, 2] = rgb[2]
        alpha = np.clip(mag[mask], min_alpha, max_alpha)
        rgba[:, 3] = (alpha * 255.0).astype(np.uint8)
        cloud.point_data["rgba"] = rgba
        plotter.add_mesh(
            cloud,
            scalars="rgba",
            rgba=True,
            point_size=point_size,
            render_points_as_spheres=True,
        )

    add_cloud(pos_mask, pos_color)
    add_cloud(neg_mask, neg_color)

    plotter.add_title(title)

    if args.show_slices:
        center_slice = np.asarray(slice_center, dtype=np.float32)
        th = float(slice_th)

        mask_xy = np.abs(pts[:, 2] - center_slice[2]) <= th
        mask_xz = np.abs(pts[:, 1] - center_slice[1]) <= th
        mask_yz = np.abs(pts[:, 0] - center_slice[0]) <= th

        def draw_slice(subplot_idx, mask, view_func, title_text):
            plotter.subplot(*subplot_idx)
            add_cloud(pos_mask & mask, pos_color)
            add_cloud(neg_mask & mask, neg_color)
            plotter.add_title(title_text)
            view_func()

        draw_slice((0, 1), mask_xy, plotter.view_xy, f"XY @ z={center_slice[2]:.2f}")
        draw_slice((1, 0), mask_xz, plotter.view_xz, f"XZ @ y={center_slice[1]:.2f}")
        draw_slice((1, 1), mask_yz, plotter.view_yz, f"YZ @ x={center_slice[0]:.2f}")

    if args.no_show:
        if out_png is not None:
            out_png.parent.mkdir(parents=True, exist_ok=True)
            try:
                plotter.screenshot(str(out_png))
                print(f"[field] saved: {out_png}")
            except Exception:
                pass
    else:
        if out_png is not None:
            out_png.parent.mkdir(parents=True, exist_ok=True)
            plotter.show(screenshot=str(out_png))
            print(f"[field] saved: {out_png}")
        else:
            plotter.show()


if __name__ == "__main__":
    main()
