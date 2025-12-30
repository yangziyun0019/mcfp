from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from mcfp.sim.robot_model import RobotModel
from mcfp.utils.seed import set_seed


def _get_cfg_value(cfg: Any, key: str) -> Any:
    """Retrieve mandatory value from config object or dict."""
    if isinstance(cfg, dict):
        if key not in cfg:
            raise ValueError(f"[fk_ik_loop] Config missing mandatory key '{key}'.")
        return cfg[key]
    if not hasattr(cfg, key):
        raise ValueError(f"[fk_ik_loop] Config missing mandatory attr '{key}'.")
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


def run_fk_ik_loop(cfg: Any, logger) -> None:
    """Run FK->IK round-trip tests and log results.

    Parameters
    ----------
    cfg:
        Loaded YAML config.
    logger:
        Logger from mcfp.utils.logging.setup_logger.
    """
    run_cfg = _get_subconfig(cfg, "run")
    seed = _get_cfg_val_default(run_cfg, "seed", None)
    deterministic = bool(_get_cfg_val_default(run_cfg, "deterministic", True))
    if seed is not None:
        set_seed(int(seed), deterministic=deterministic)

    robot_cfg = _get_subconfig(cfg, "robot")
    paths_cfg = _get_subconfig(cfg, "paths")
    repo_root = Path(_get_cfg_val_default(paths_cfg, "repo_root", ".")).resolve()
    urdf_path = Path(_get_cfg_value(robot_cfg, "urdf_path"))
    if not urdf_path.is_absolute():
        urdf_path = (repo_root / urdf_path).resolve()
    base_link = _get_cfg_val_default(robot_cfg, "base_link", None)
    end_effector_link = _get_cfg_val_default(robot_cfg, "end_effector_link", None)

    if not urdf_path.exists():
        raise FileNotFoundError(f"[fk_ik_loop] URDF not found: {urdf_path}")

    robot = RobotModel(
        urdf_path=urdf_path,
        logger=logger,
        base_link=base_link,
        end_effector_link=end_effector_link,
    )

    if robot.joint_limits.size == 0:
        raise ValueError("[fk_ik_loop] Robot joint limits are empty.")

    sampling_cfg = _get_subconfig(cfg, "sampling")
    num_samples = int(_get_cfg_value(sampling_cfg, "num_samples"))
    log_every = int(_get_cfg_val_default(sampling_cfg, "log_every", 1))
    show_q = bool(_get_cfg_val_default(sampling_cfg, "show_q", True))
    show_q_ik = bool(_get_cfg_val_default(sampling_cfg, "show_q_ik", True))

    ik_cfg = _get_subconfig(cfg, "ik")
    ik_attempts = int(_get_cfg_val_default(ik_cfg, "attempts", 5))
    use_fk_seed = bool(_get_cfg_val_default(ik_cfg, "use_fk_seed", True))
    pos_tol = float(_get_cfg_val_default(ik_cfg, "pos_tol_m", 0.002))
    ori_tol = float(_get_cfg_val_default(ik_cfg, "ori_tol_deg", 4.0)) * (np.pi / 180.0)
    max_iters = int(_get_cfg_val_default(ik_cfg, "max_iters", 100))
    residual_threshold = float(_get_cfg_val_default(ik_cfg, "residual_threshold", 1e-5))

    limits_cfg = _get_subconfig(cfg, "limits")
    limit_eps = float(_get_cfg_val_default(limits_cfg, "eps", 1e-6))

    stats: Dict[str, int] = {
        "total": num_samples,
        "ik_fail": 0,
        "limit_fail": 0,
        "valid": 0,
    }

    logger.info(
        f"[fk_ik_loop] Running FK->IK loop with {num_samples} samples "
        f"(attempts={ik_attempts}, tol={pos_tol}m/{ori_tol * 180.0 / np.pi:.2f}deg)."
    )

    for i in range(num_samples):
        q_sample = _sample_rest_pose(robot.joint_limits)
        pos_fk, quat_fk = robot.fk(q_sample)

        best_q = None
        best_pos = None
        best_quat = None
        best_err = None
        best_pos_err = None
        best_ori_err = None

        for attempt in range(ik_attempts):
            if attempt == 0 and use_fk_seed:
                rest_pose = q_sample
            else:
                rest_pose = _sample_rest_pose(robot.joint_limits)
            q_ik = robot.ik(
                target_pos=pos_fk,
                target_quat=quat_fk,
                rest_pose=rest_pose,
                max_iters=max_iters,
                residual_threshold=residual_threshold,
            )
            if q_ik is None or not np.all(np.isfinite(q_ik)):
                continue
            try:
                pos_ik, quat_ik = robot.fk(q_ik)
            except Exception:
                continue

            pos_err = float(np.linalg.norm(pos_ik - pos_fk))
            ori_err = _quat_angle_error(quat_fk, quat_ik)
            if pos_err <= pos_tol and ori_err <= ori_tol:
                err = pos_err + ori_err
                if best_err is None or err < best_err:
                    best_err = err
                    best_q = q_ik
                    best_pos = pos_ik
                    best_quat = quat_ik
                    best_pos_err = pos_err
                    best_ori_err = ori_err

        if best_q is None:
            stats["ik_fail"] += 1
            if log_every > 0 and (i + 1) % log_every == 0:
                logger.info(
                    f"[fk_ik_loop] sample {i+1}/{num_samples} ik_success=False "
                    f"fk_pos={pos_fk.tolist()} fk_quat={quat_fk.tolist()}"
                )
            continue

        within_limits = _within_limits(best_q, robot.joint_limits, limit_eps)
        if within_limits:
            stats["valid"] += 1
        else:
            stats["limit_fail"] += 1

        if log_every > 0 and (i + 1) % log_every == 0:
            msg = (
                f"[fk_ik_loop] sample {i+1}/{num_samples} ik_success=True "
                f"pos_err={best_pos_err:.6f} ori_err_deg={best_ori_err * 180.0 / np.pi:.3f} "
                f"within_limits={within_limits} "
                f"fk_pos={pos_fk.tolist()} fk_quat={quat_fk.tolist()} "
                f"ik_pos={best_pos.tolist()} ik_quat={best_quat.tolist()}"
            )
            if show_q:
                msg += f" q_sample={q_sample.tolist()}"
            if show_q_ik:
                msg += f" q_ik={best_q.tolist()}"
            logger.info(msg)

    logger.info(
        f"[fk_ik_loop] Done. valid={stats['valid']}/{num_samples} "
        f"ik_fail={stats['ik_fail']} limit_fail={stats['limit_fail']}"
    )
