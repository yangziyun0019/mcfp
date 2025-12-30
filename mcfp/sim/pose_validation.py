from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from mcfp.sim.collision import SelfCollisionChecker
from mcfp.sim.robot_model import RobotModel


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


def build_robot_and_self_checker(
    urdf_path: Path,
    base_link: Optional[str],
    end_effector_link: Optional[str],
    cfg: Any,
    logger,
) -> tuple[RobotModel, SelfCollisionChecker]:
    """Build robot model and self-collision checker."""
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

    return robot, self_checker


def validate_pose_with_ik(
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    robot: RobotModel,
    self_checker: SelfCollisionChecker,
    cfg: Any,
) -> Dict[str, Any]:
    """Validate a pose with IK, joint limits, and self-collision."""
    ik_cfg = _get_subconfig(cfg, "ik")
    ik_attempts = int(_get_cfg_val_default(ik_cfg, "attempts", 5))
    pos_tol = float(_get_cfg_val_default(ik_cfg, "pos_tol_m", 0.002))
    ori_tol = float(_get_cfg_val_default(ik_cfg, "ori_tol_deg", 4.0)) * (np.pi / 180.0)
    max_iters = int(_get_cfg_val_default(ik_cfg, "max_iters", 100))
    residual_threshold = float(_get_cfg_val_default(ik_cfg, "residual_threshold", 1e-5))

    limits_cfg = _get_subconfig(cfg, "limits")
    limit_eps = float(_get_cfg_val_default(limits_cfg, "eps", 1e-6))

    best_q = None
    best_pos = None
    best_quat = None
    best_err = None

    for attempt in range(ik_attempts):
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
        return {
            "ik_success": False,
            "within_limits": False,
            "self_collision_free": False,
            "label": 0.0,
            "pose": np.concatenate([target_pos, target_quat]).astype(np.float32),
            "q": None,
        }

    within_limits = _within_limits(best_q, robot.joint_limits, limit_eps)
    self_ok = False
    if within_limits:
        dist = self_checker.min_distance(best_q)
        self_ok = bool(np.isfinite(dist) and dist > 0.0)

    label = 1.0 if (within_limits and self_ok) else 0.0
    return {
        "ik_success": True,
        "within_limits": bool(within_limits),
        "self_collision_free": bool(self_ok),
        "label": float(label),
        "pose": np.concatenate([best_pos, best_quat]).astype(np.float32),
        "q": np.asarray(best_q, dtype=np.float32),
    }
