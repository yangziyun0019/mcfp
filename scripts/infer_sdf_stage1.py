from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from mcfp.data.io import compute_morph_scale, load_morph_spec
from mcfp.models.morph_encoder import build_morph_graph_from_json
from mcfp.models.stage1 import MCFPStage1
from mcfp.sim.collision import (
    HybridSelfCollisionChecker,
    MeshSelfCollisionChecker,
    SelfCollisionChecker,
)
from mcfp.sim.robot_model import RobotModel
from mcfp.utils.config import load_config
from mcfp.utils.logging import setup_logger
from mcfp.utils import se3


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


@contextlib.contextmanager
def _suppress_output() -> None:
    """Suppress stdout/stderr for noisy library initialization."""
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stdout_fd)
        os.close(stderr_fd)
        os.close(devnull_fd)


def _set_logger_level(logger, level: str) -> None:
    level_name = str(level).upper()
    if not hasattr(logging, level_name):
        level_name = "ERROR"
    lvl = getattr(logging, level_name)
    logger.setLevel(lvl)
    for handler in logger.handlers:
        handler.setLevel(lvl)


def _load_meta(dataset_root: Path) -> Dict[str, Any]:
    meta_path = dataset_root / "meta.json"
    if not meta_path.is_file():
        return {}
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _euler_deg_to_quat(euler_deg: np.ndarray, order: str = "xyz") -> np.ndarray:
    euler = np.asarray(euler_deg, dtype=np.float32).reshape(3)
    quat = R.from_euler(order, euler, degrees=True).as_quat()
    return np.asarray(quat, dtype=np.float32)


def _parse_pose(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray]:
    if args.pose is not None:
        if len(args.pose) == 7:
            pose = np.asarray(args.pose, dtype=np.float32).reshape(7)
            return pose[:3], pose[3:]
        if len(args.pose) == 6:
            pose = np.asarray(args.pose, dtype=np.float32).reshape(6)
            pos = pose[:3]
            quat = _euler_deg_to_quat(pose[3:])
            return pos, quat
        raise ValueError("--pose must have 7 values (x y z qx qy qz qw) or 6 values (x y z rx ry rz in deg).")

    if args.pos is None:
        raise ValueError("Provide --pose or --pos (and optional --quat).")
    if len(args.pos) != 3:
        raise ValueError("--pos must have 3 values: x y z")
    pos = np.asarray(args.pos, dtype=np.float32).reshape(3)
    if args.quat is None:
        quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    else:
        if len(args.quat) != 4:
            raise ValueError("--quat must have 4 values: qx qy qz qw")
        quat = np.asarray(args.quat, dtype=np.float32).reshape(4)
    return pos, quat


def _parse_pose_from_cfg(pose_cfg: Any) -> Tuple[np.ndarray, np.ndarray, str]:
    pose_vals = _get(pose_cfg, "pose", None)
    if pose_vals is not None:
        pose = np.asarray(pose_vals, dtype=np.float32).reshape(-1)
        if pose.size != 7:
            raise ValueError("pose.pose must have 7 values: x y z qx qy qz qw")
        return pose[:3], pose[3:], "xyz"

    pos_vals = _get(pose_cfg, "pos", None)
    if pos_vals is None:
        pos_vals = _get(pose_cfg, "position_m", None)
    if pos_vals is None:
        raise ValueError("pose.position_m (or pose.pos) is required in infer config.")
    pos = np.asarray(pos_vals, dtype=np.float32).reshape(3)

    quat_vals = _get(pose_cfg, "quat", None)
    if quat_vals is not None:
        quat = np.asarray(quat_vals, dtype=np.float32).reshape(4)
        return pos, quat, "xyz"

    euler_deg = _get(pose_cfg, "euler_deg", None)
    if euler_deg is None:
        raise ValueError("pose.euler_deg (deg) or pose.quat is required in infer config.")
    order = str(_get(pose_cfg, "euler_order", "xyz"))
    quat = _euler_deg_to_quat(np.asarray(euler_deg, dtype=np.float32), order=order)
    return pos, quat, order


def _compute_s_and_grad(
    model: MCFPStage1,
    w: np.ndarray,
    graph,
    device: torch.device,
) -> Tuple[float, np.ndarray]:
    w_t = torch.from_numpy(w).unsqueeze(0).to(device).requires_grad_(True)
    s = model(w_t, morph_graph=graph)
    grad = torch.autograd.grad(s.sum(), w_t, create_graph=False)[0]
    s_val = float(s.squeeze().detach().cpu())
    grad_val = grad.squeeze(0).detach().cpu().numpy().astype(np.float32)
    return s_val, grad_val


def _quat_to_euler_deg(quat: np.ndarray, order: str) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).reshape(4)
    return R.from_quat(quat).as_euler(order, degrees=True).astype(np.float32)


def _pose_to_w(pos: np.ndarray, quat: np.ndarray, l_ref: float, lambda_val: float) -> np.ndarray:
    quat = se3.quat_normalize(quat)
    axis, angle = se3.quat_to_axis_angle(quat)
    rotvec = axis * float(angle)
    z = np.concatenate([pos / float(l_ref), rotvec], axis=0).astype(np.float32)
    w = z.copy()
    w[3:] = w[3:] / float(lambda_val)
    return w.astype(np.float32)


def _within_limits(q: np.ndarray, limits: np.ndarray, eps: float = 1e-6) -> bool:
    q_arr = np.asarray(q, dtype=np.float32).reshape(-1)
    low = limits[:, 0].astype(np.float32)
    high = limits[:, 1].astype(np.float32)
    return bool(np.all(q_arr >= (low - eps)) and np.all(q_arr <= (high + eps)))


def _sample_rest_pose(rng: np.random.Generator, limits: np.ndarray) -> np.ndarray:
    low = limits[:, 0]
    span = limits[:, 1] - low
    span = np.where(span > 0.0, span, 0.0)
    return (low + rng.random(limits.shape[0]) * span).astype(np.float32)


def _check_reachability(
    robot: RobotModel,
    collision_checker: HybridSelfCollisionChecker,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    *,
    pos_tol_m: float,
    rot_tol_rad: float,
    seeds: int,
    max_iters: int,
    residual_threshold: float,
    rng: np.random.Generator,
) -> Tuple[bool, Optional[np.ndarray]]:
    limits = robot.joint_limits
    if limits.size == 0:
        return False, None

    for _ in range(int(seeds)):
        rest = _sample_rest_pose(rng, limits)
        try:
            q = robot.ik(
                target_pos=target_pos,
                target_quat=target_quat,
                rest_pose=rest,
                max_iters=int(max_iters),
                residual_threshold=float(residual_threshold),
            )
        except Exception:
            continue
        if q is None or not np.all(np.isfinite(q)):
            continue
        if not _within_limits(q, limits):
            continue
        try:
            pos_fk, quat_fk = robot.fk(np.asarray(q, dtype=np.float32))
        except Exception:
            continue
        pos_err = float(np.linalg.norm(pos_fk - target_pos))
        rot_err = float(np.linalg.norm(se3.quat_delta_axis_angle(target_quat, quat_fk)))
        if pos_err > float(pos_tol_m) or rot_err > float(rot_tol_rad):
            continue
        if not collision_checker.is_collision_free(q):
            continue
        return True, np.asarray(q, dtype=np.float32)

    return False, None


def _build_model(cfg: Any, node_feat_dim: int) -> MCFPStage1:
    model_cfg = _get(cfg, "model", None)
    morph_cfg = _get(model_cfg, "morph_encoder", None)
    pose_cfg = _get(model_cfg, "pose_encoder", None)
    backbone_cfg = _get(model_cfg, "backbone", None)

    return MCFPStage1(
        node_feat_dim=int(node_feat_dim),
        morph_dim=int(_get(morph_cfg, "out_dim", 128)),
        pose_dim=int(_get(pose_cfg, "out_dim", 128)),
        pose_hidden_dims=list(_get(pose_cfg, "hidden_dims", [128, 128])),
        pose_fourier_dim=int(_get(pose_cfg, "fourier_dim", 0)),
        pose_fourier_scale=float(_get(pose_cfg, "fourier_scale", 10.0)),
        d_model=int(_get(morph_cfg, "d_model", 128)),
        n_layers=int(_get(morph_cfg, "n_layers", 4)),
        n_heads=int(_get(morph_cfg, "n_heads", 8)),
        d_ff=int(_get(morph_cfg, "d_ff", 512)),
        dropout=float(_get(morph_cfg, "dropout", 0.1)),
        backbone_hidden_dim=int(_get(backbone_cfg, "hidden_dim", 128)),
        backbone_num_layers=int(_get(backbone_cfg, "num_layers", 4)),
        backbone_out_dim=int(_get(backbone_cfg, "out_dim", 128)),
        w0_first=float(_get(backbone_cfg, "w0_first", 30.0)),
        w0=float(_get(backbone_cfg, "w0", 1.0)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="MCFP Stage-1 inference + IK verification.")
    parser.add_argument("--config", type=str, default="configs/train_sdf_stage1.yaml")
    parser.add_argument(
        "--infer-config",
        type=str,
        default="configs/infer_sdf_stage1.yaml",
        help="Inference config YAML with pose and paths.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="runs/sdf_stage1/exp001/checkpoint_final.pt",
        help="Path to checkpoint.",
    )
    parser.add_argument("--spec", type=str, default="", help="Morphology spec JSON path.")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="",
        help="Dataset root containing meta.json (default from config).",
    )
    parser.add_argument("--device", type=str, default="", help="cuda or cpu (default: auto)")
    parser.add_argument("--pose", type=float, nargs="*", default=None, help="Pose: x y z qx qy qz qw")
    parser.add_argument("--pos", type=float, nargs="*", default=None, help="Position: x y z")
    parser.add_argument("--quat", type=float, nargs="*", default=None, help="Quaternion: qx qy qz qw")
    parser.add_argument("--ik-seeds", type=int, default=16, help="IK rest-pose samples.")
    parser.add_argument("--ik-max-iters", type=int, default=150, help="IK max iterations.")
    parser.add_argument("--ik-residual", type=float, default=1e-5, help="IK residual threshold.")
    parser.add_argument("--ik-seed", type=int, default=42, help="RNG seed for IK rest poses.")
    parser.add_argument("--pos-tol-mm", type=float, default=1.0, help="Position tolerance in mm.")
    parser.add_argument("--rot-tol-deg", type=float, default=1.0, help="Rotation tolerance in deg.")
    parser.add_argument("--log-level", type=str, default="", help="Logger level (error/warning/info).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    infer_cfg = None
    infer_cfg_path = Path(args.infer_config)
    if infer_cfg_path.is_file():
        infer_cfg = load_config(infer_cfg_path)
    repo_root = Path(_get_path(cfg, "paths.repo_root", ".")).resolve()
    logger = setup_logger("mcfp.infer_stage1", log_dir=None)
    log_level = args.log_level
    if not log_level and infer_cfg is not None:
        log_level = _get_path(infer_cfg, "run.log_level", "error")
    _set_logger_level(logger, log_level or "error")

    dataset_root = args.dataset_root
    if not dataset_root and infer_cfg is not None:
        dataset_root = _get_path(infer_cfg, "paths.dataset_root", None)
    if not dataset_root:
        dataset_root = _get_path(cfg, "data.datasets", [])[0].get("dataset_root")
    if not dataset_root:
        raise ValueError("dataset_root not provided and missing in config.")
    dataset_root = _resolve_path(dataset_root, repo_root)
    meta = _load_meta(dataset_root)

    spec_path = args.spec
    if not spec_path and infer_cfg is not None:
        spec_path = _get_path(infer_cfg, "paths.spec_path", None)
    if not spec_path:
        spec_path = _get_path(cfg, "data.datasets", [])[0].get("spec_path")
    if not spec_path:
        raise ValueError("spec path not provided and missing in config.")
    spec_path = _resolve_path(spec_path, repo_root)
    spec = load_morph_spec(spec_path)

    l_ref = meta.get("l_ref", None)
    if l_ref is None:
        l_ref = float(compute_morph_scale(spec))
        logger.warning("[infer] meta.json missing l_ref; using computed scale %.6f", l_ref)
    lambda_val = meta.get("lambda", None)
    if lambda_val is None:
        raise ValueError("meta.json missing lambda; cannot build pose encoding.")
    lambda_val = float(lambda_val)

    if args.pose is not None or args.pos is not None:
        pos, quat = _parse_pose(args)
        euler_order = "xyz"
    elif infer_cfg is not None:
        pos, quat, euler_order = _parse_pose_from_cfg(_get(infer_cfg, "pose", None))
    else:
        raise ValueError("No pose provided. Use --pose/--pos or infer config pose block.")
    w = _pose_to_w(pos, quat, l_ref=l_ref, lambda_val=lambda_val)

    device = args.device
    if not device and infer_cfg is not None:
        device = _get_path(infer_cfg, "run.device", None)
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    print(
        f"[pose] initial pos={np.array2string(pos, precision=6)} "
        f"euler_deg={np.array2string(_quat_to_euler_deg(quat, euler_order), precision=3)}"
    )

    graph = build_morph_graph_from_json(spec_path, l_ref=l_ref, device=device)
    model = _build_model(cfg, node_feat_dim=int(graph.x.shape[1]))
    checkpoint_path = args.checkpoint
    if infer_cfg is not None:
        checkpoint_path = _get_path(infer_cfg, "paths.checkpoint", checkpoint_path)
    ckpt = torch.load(_resolve_path(checkpoint_path, repo_root), map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()

    with torch.no_grad():
        w_t = torch.from_numpy(w).unsqueeze(0).to(device)
        s_val = model(w_t, morph_graph=graph).squeeze().item()
    pred = "reachable" if s_val > 0.0 else "unreachable"
    print(f"[infer] s={s_val:.6f} => {pred}")

    meta_urdf = spec.get("meta", {}).get("urdf_path", None)
    if meta_urdf is None:
        raise ValueError("spec.meta.urdf_path missing; cannot run IK verification.")
    urdf_path = _resolve_path(meta_urdf, repo_root)
    base_link = spec.get("meta", {}).get("base_link", None)
    ee_link = spec.get("meta", {}).get("ee_link", None)

    quiet_init = True
    if infer_cfg is not None:
        quiet_init = bool(_get_path(infer_cfg, "run.quiet_init", True))

    init_ctx = _suppress_output() if quiet_init else contextlib.nullcontext()
    with init_ctx:
        robot = RobotModel(
            urdf_path=urdf_path,
            logger=logger,
            base_link=base_link,
            end_effector_link=ee_link,
        )

        collision_mode = str(meta.get("collision_mode", "hybrid")).lower()
        d_check = float(meta.get("collision_d_check_m", 0.015))
        capsule_checker = SelfCollisionChecker.from_robot(
            robot=robot,
            cache_dir=urdf_path.parent,
            logger=logger,
        )
        mesh_checker = None
        if collision_mode in ("hybrid", "mesh_only"):
            mesh_checker = MeshSelfCollisionChecker(
                urdf_path=urdf_path,
                joint_names=robot.joint_names,
                link_edges=robot.link_edges,
                logger=logger,
                use_self_collision=True,
            )
        collision_checker = HybridSelfCollisionChecker(
            capsule_checker=capsule_checker,
            mesh_checker=mesh_checker,
            d_check=d_check,
            mode=collision_mode,
        )

    ik_seeds = args.ik_seeds
    ik_max_iters = args.ik_max_iters
    ik_residual = args.ik_residual
    ik_seed = args.ik_seed
    if infer_cfg is not None:
        ik_cfg = _get(infer_cfg, "ik", None)
        if ik_cfg is not None:
            ik_seeds = int(_get(ik_cfg, "seeds", ik_seeds))
            ik_max_iters = int(_get(ik_cfg, "max_iters", ik_max_iters))
            ik_residual = float(_get(ik_cfg, "residual_threshold", ik_residual))
            ik_seed = int(_get(ik_cfg, "seed", ik_seed))

    pos_tol_mm = args.pos_tol_mm
    rot_tol_deg = args.rot_tol_deg
    if infer_cfg is not None:
        verify_cfg = _get(infer_cfg, "verify", None)
        if verify_cfg is not None:
            pos_tol_mm = float(_get(verify_cfg, "pos_tol_mm", pos_tol_mm))
            rot_tol_deg = float(_get(verify_cfg, "rot_tol_deg", rot_tol_deg))

    rng = np.random.default_rng(int(ik_seed))

    print("[verify] initial pose")
    ok_init, q_init = _check_reachability(
        robot=robot,
        collision_checker=collision_checker,
        target_pos=pos,
        target_quat=se3.quat_normalize(quat),
        pos_tol_m=float(pos_tol_mm) * 1e-3,
        rot_tol_rad=float(rot_tol_deg) * np.pi / 180.0,
        seeds=ik_seeds,
        max_iters=ik_max_iters,
        residual_threshold=ik_residual,
        rng=rng,
    )
    if ok_init:
        print("[verify] IK + limits + collision: reachable")
        print(f"[verify] q={np.array2string(q_init, precision=6)}")
    else:
        print("[verify] IK + limits + collision: unreachable")

    adjust_cfg = _get(infer_cfg, "adjust", None) if infer_cfg is not None else None
    adjust_enabled = bool(_get(adjust_cfg, "enabled", False)) if adjust_cfg is not None else False
    step_size = float(_get(adjust_cfg, "step_size", 0.01)) if adjust_cfg is not None else 0.01
    max_steps = int(_get(adjust_cfg, "max_steps", 50)) if adjust_cfg is not None else 50
    target_abs_s = float(_get(adjust_cfg, "target_abs_s", 0.005)) if adjust_cfg is not None else 0.005
    stop_on_positive = bool(_get(adjust_cfg, "stop_on_positive", False)) if adjust_cfg is not None else False
    normalize_grad = bool(_get(adjust_cfg, "normalize_grad", True)) if adjust_cfg is not None else True
    grad_clip = float(_get(adjust_cfg, "grad_clip", 0.0)) if adjust_cfg is not None else 0.0

    if adjust_enabled:
        cur_pos = pos.copy()
        cur_quat = se3.quat_normalize(quat)
        cur_s, cur_grad = _compute_s_and_grad(model, w, graph, device=device)
        steps_taken = 0
        for step in range(1, max_steps + 1):
            if stop_on_positive:
                if cur_s > 0.0:
                    break
            else:
                if abs(cur_s) <= target_abs_s:
                    break
            grad = cur_grad.astype(np.float32)
            if grad_clip > 0.0:
                grad = np.clip(grad, -grad_clip, grad_clip)
            if normalize_grad:
                n = float(np.linalg.norm(grad))
                if n > 1e-8:
                    grad = grad / n
            direction = -float(np.sign(cur_s)) * grad
            delta_w = float(step_size) * direction

            delta_pos = delta_w[:3] * float(l_ref)
            delta_rot = delta_w[3:] * float(lambda_val)

            cur_pos = cur_pos + delta_pos
            cur_quat = se3.apply_delta_quat(cur_quat, delta_rot)

            w = _pose_to_w(cur_pos, cur_quat, l_ref=l_ref, lambda_val=lambda_val)
            cur_s, cur_grad = _compute_s_and_grad(model, w, graph, device=device)
            euler_deg = _quat_to_euler_deg(cur_quat, euler_order)
            print(
                f"[adjust] step={step} pos={np.array2string(cur_pos, precision=6)} "
                f"euler_deg={np.array2string(euler_deg, precision=3)} s={cur_s:.6f}"
            )
            steps_taken = step

        print(f"[adjust] steps={steps_taken} final_s={cur_s:.6f}")
        print(
            f"[adjust] final pose pos={np.array2string(cur_pos, precision=6)} "
            f"euler_deg={np.array2string(_quat_to_euler_deg(cur_quat, euler_order), precision=3)}"
        )

        print("[verify] final adjusted pose")
        ok_final, q_final = _check_reachability(
            robot=robot,
            collision_checker=collision_checker,
            target_pos=cur_pos,
            target_quat=se3.quat_normalize(cur_quat),
            pos_tol_m=float(pos_tol_mm) * 1e-3,
            rot_tol_rad=float(rot_tol_deg) * np.pi / 180.0,
            seeds=ik_seeds,
            max_iters=ik_max_iters,
            residual_threshold=ik_residual,
            rng=rng,
        )
        if ok_final:
            print("[verify] IK + limits + collision: reachable")
            print(f"[verify] q={np.array2string(q_final, precision=6)}")
        else:
            print("[verify] IK + limits + collision: unreachable")

    return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
