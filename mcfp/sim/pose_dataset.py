# mcfp/sim/pose_dataset.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from mcfp.data.io import save_pose_dataset
from mcfp.sim.collision import SelfCollisionChecker
from mcfp.sim.robot_model import RobotModel
from mcfp.sim.workspace_bounds import apply_aabb_margin, estimate_workspace_bounds
from mcfp.utils.seed import set_seed


def _get_cfg_value(cfg: Any, key: str) -> Any:
    """Retrieve mandatory value from config object or dict."""
    if isinstance(cfg, dict):
        if key not in cfg:
            raise ValueError(f"[pose_dataset] Config missing mandatory key '{key}'.")
        return cfg[key]
    if not hasattr(cfg, key):
        raise ValueError(f"[pose_dataset] Config missing mandatory attr '{key}'.")
    return getattr(cfg, key)


def _get_cfg_val_default(cfg: Any, key: str, default: Any) -> Any:
    """Retrieve optional value from config object or dict."""
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _get_subconfig(cfg: Any, name: str) -> Any:
    """Retrieve sub-config safely."""
    if isinstance(cfg, dict):
        return cfg.get(name, {})
    return getattr(cfg, name, {})



def _sample_uniform_quaternion() -> np.ndarray:
    """Sample a uniform quaternion (x, y, z, w)."""
    u1, u2, u3 = np.random.rand(3)
    s1 = np.sqrt(1.0 - u1)
    s2 = np.sqrt(u1)
    theta1 = 2.0 * np.pi * u2
    theta2 = 2.0 * np.pi * u3
    qx = s1 * np.sin(theta1)
    qy = s1 * np.cos(theta1)
    qz = s2 * np.sin(theta2)
    qw = s2 * np.cos(theta2)
    return np.array([qx, qy, qz, qw], dtype=np.float32)


def _quat_angle_error(q_a: np.ndarray, q_b: np.ndarray) -> float:
    """Compute angular error between two quaternions in radians."""
    qa = np.asarray(q_a, dtype=np.float64).reshape(4)
    qb = np.asarray(q_b, dtype=np.float64).reshape(4)
    na = np.linalg.norm(qa)
    nb = np.linalg.norm(qb)
    if na <= 1e-9 or nb <= 1e-9:
        return float("inf")
    qa /= na
    qb /= nb
    dot = float(np.clip(abs(np.dot(qa, qb)), -1.0, 1.0))
    return float(2.0 * np.arccos(dot))


def _within_limits(q: np.ndarray, limits: np.ndarray, eps: float) -> bool:
    q_arr = np.asarray(q, dtype=np.float32).reshape(-1)
    low = limits[:, 0].astype(np.float32)
    high = limits[:, 1].astype(np.float32)
    return bool(np.all(q_arr >= (low - eps)) and np.all(q_arr <= (high + eps)))


def _sample_rest_pose(limits: np.ndarray) -> np.ndarray:
    low = limits[:, 0]
    span = limits[:, 1] - low
    span = np.where(span > 0.0, span, 0.0)
    return (low + np.random.rand(limits.shape[0]) * span).astype(np.float32)


def _sample_position(
    aabb_min: np.ndarray,
    aabb_max: np.ndarray,
    expanded_min: np.ndarray,
    expanded_max: np.ndarray,
    inside: bool,
) -> np.ndarray:
    if inside:
        return np.random.uniform(aabb_min, aabb_max).astype(np.float32)

    for _ in range(100):
        pos = np.random.uniform(expanded_min, expanded_max).astype(np.float32)
        if np.any(pos < aabb_min) or np.any(pos > aabb_max):
            return pos

    return np.random.uniform(expanded_min, expanded_max).astype(np.float32)


def generate_pose_dataset_ik(
    urdf_path: Path,
    output_path: Path,
    cfg: Any,
    base_link: Optional[str],
    end_effector_link: Optional[str],
    logger,
) -> None:
    """Generate IK-based pose dataset for a single robot."""
    run_cfg = _get_subconfig(cfg, "run")
    seed = _get_cfg_val_default(run_cfg, "seed", None)
    deterministic = bool(_get_cfg_val_default(run_cfg, "deterministic", True))
    if seed is not None:
        set_seed(int(seed), deterministic=deterministic)

    robot = RobotModel(
        urdf_path=urdf_path,
        logger=logger,
        base_link=base_link,
        end_effector_link=end_effector_link,
    )

    self_cfg = _get_subconfig(cfg, "self_collision")
    self_checker = SelfCollisionChecker.from_robot(
        robot=robot,
        cache_dir=urdf_path.parent,
        logger=logger,
        radius_cfg=self_cfg,
    )

    if bool(_get_cfg_val_default(self_cfg, "auto_calibrate", False)):
        samples = int(_get_cfg_val_default(self_cfg, "calibration_samples", 500))
        self_checker.find_static_collisions(
            sample_fn=lambda: _sample_rest_pose(robot.joint_limits),
            n_samples=samples,
        )

    aabb_cfg = _get_subconfig(cfg, "aabb")
    bounds_samples = int(_get_cfg_value(aabb_cfg, "bounds_samples"))
    margin = float(_get_cfg_val_default(aabb_cfg, "margin", 1.0))

    xyz_min, xyz_max = estimate_workspace_bounds(
        robot=robot,
        samples=bounds_samples,
        logger=logger,
    )
    aabb_min, aabb_max = apply_aabb_margin(xyz_min, xyz_max, margin=margin)

    sampling_cfg = _get_subconfig(cfg, "sampling")
    total_samples = int(_get_cfg_value(sampling_cfg, "total_samples"))
    outside_ratio = float(_get_cfg_val_default(sampling_cfg, "outside_ratio", 0.1))
    outside_scale = float(_get_cfg_val_default(sampling_cfg, "outside_scale", 1.2))
    fk_positive_ratio = float(_get_cfg_val_default(sampling_cfg, "fk_positive_ratio", 0.3))
    progress_interval = int(_get_cfg_val_default(sampling_cfg, "progress_interval", 50000))

    outside_ratio = np.clip(outside_ratio, 0.0, 1.0)
    fk_positive_ratio = np.clip(fk_positive_ratio, 0.0, 1.0)
    inside_count = int(total_samples * (1.0 - outside_ratio))
    outside_count = int(total_samples - inside_count)
    fk_count = int(inside_count * fk_positive_ratio)
    fk_count = min(fk_count, inside_count)

    center = 0.5 * (aabb_min + aabb_max)
    half = 0.5 * (aabb_max - aabb_min)
    expanded_min = (center - half * outside_scale).astype(np.float32)
    expanded_max = (center + half * outside_scale).astype(np.float32)

    ik_cfg = _get_subconfig(cfg, "ik")
    ik_attempts = int(_get_cfg_val_default(ik_cfg, "attempts", 5))
    pos_tol = float(_get_cfg_val_default(ik_cfg, "pos_tol_m", 0.002))
    ori_tol = float(_get_cfg_val_default(ik_cfg, "ori_tol_deg", 4.0)) * (np.pi / 180.0)
    max_iters = int(_get_cfg_val_default(ik_cfg, "max_iters", 100))
    residual_threshold = float(_get_cfg_val_default(ik_cfg, "residual_threshold", 1e-5))

    limits_cfg = _get_subconfig(cfg, "limits")
    limit_eps = float(_get_cfg_val_default(limits_cfg, "eps", 1e-6))

    if robot.joint_limits.size == 0:
        raise ValueError("[pose_dataset] Robot joint limits are empty.")

    sample_modes = np.zeros(total_samples, dtype=np.int8)
    # 0: inside random pose, 1: inside FK-seeded pose, 2: outside random pose
    sample_modes[:inside_count - fk_count] = 0
    sample_modes[inside_count - fk_count : inside_count] = 1
    sample_modes[inside_count:] = 2
    np.random.shuffle(sample_modes)

    poses = np.zeros((total_samples, 7), dtype=np.float32)
    labels = np.zeros((total_samples,), dtype=np.float32)

    stats: Dict[str, int] = {
        "total": total_samples,
        "inside": 0,
        "outside": 0,
        "fk_seed": 0,
        "ik_fail": 0,
        "limit_fail": 0,
        "self_fail": 0,
        "valid": 0,
    }

    logger.info(
        f"[pose_dataset] Sampling {total_samples} poses "
        f"(inside={inside_count}, outside={outside_count})."
    )

    for i in range(total_samples):
        mode = int(sample_modes[i])
        inside = mode != 2
        if inside:
            stats["inside"] += 1
        else:
            stats["outside"] += 1

        fk_seed = None
        if mode == 1:
            stats["fk_seed"] += 1
            fk_seed = _sample_rest_pose(robot.joint_limits)
            try:
                target_pos, target_quat = robot.fk(fk_seed)
            except Exception:
                fk_seed = None
                target_pos = _sample_position(
                    aabb_min=aabb_min,
                    aabb_max=aabb_max,
                    expanded_min=expanded_min,
                    expanded_max=expanded_max,
                    inside=True,
                )
                target_quat = _sample_uniform_quaternion()
        else:
            target_pos = _sample_position(
                aabb_min=aabb_min,
                aabb_max=aabb_max,
                expanded_min=expanded_min,
                expanded_max=expanded_max,
                inside=inside,
            )
            target_quat = _sample_uniform_quaternion()

        best_q = None
        best_pos = None
        best_quat = None
        best_err = None

        for attempt in range(ik_attempts):
            if attempt == 0 and fk_seed is not None:
                rest_pose = fk_seed
            else:
                rest_pose = _sample_rest_pose(robot.joint_limits)
            q = robot.ik(
                target_pos=target_pos,
                target_quat=target_quat,
                rest_pose=rest_pose,
                max_iters=max_iters,
                residual_threshold=residual_threshold,
            )
            if q is None or not np.all(np.isfinite(q)):
                continue
            try:
                pos_fk, quat_fk = robot.fk(q)
            except Exception:
                continue

            pos_err = float(np.linalg.norm(pos_fk - target_pos))
            ori_err = _quat_angle_error(target_quat, quat_fk)
            if pos_err <= pos_tol and ori_err <= ori_tol:
                err = pos_err + ori_err
                if best_err is None or err < best_err:
                    best_err = err
                    best_q = q
                    best_pos = pos_fk
                    best_quat = quat_fk

        if best_q is None:
            stats["ik_fail"] += 1
            poses[i, :3] = target_pos
            poses[i, 3:] = target_quat
            labels[i] = 0.0
        else:
            label = 1.0
            if not _within_limits(best_q, robot.joint_limits, limit_eps):
                label = 0.0
                stats["limit_fail"] += 1
            else:
                dist = self_checker.min_distance(best_q)
                if not np.isfinite(dist) or dist <= 0.0:
                    label = 0.0
                    stats["self_fail"] += 1

            if label > 0.5:
                stats["valid"] += 1

            poses[i, :3] = np.asarray(best_pos, dtype=np.float32)
            poses[i, 3:] = np.asarray(best_quat, dtype=np.float32)
            labels[i] = label

        if progress_interval > 0 and (i + 1) % progress_interval == 0:
            logger.info(
                f"[pose_dataset] Iter {i+1}/{total_samples} "
                f"valid={stats['valid']} ik_fail={stats['ik_fail']} "
                f"limit_fail={stats['limit_fail']} self_fail={stats['self_fail']}"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_pose_dataset(
        path=output_path,
        poses=poses,
        labels=labels,
    )

    logger.info(
        f"[pose_dataset] Done. Output={output_path} "
        f"valid={stats['valid']}/{total_samples}"
    )
