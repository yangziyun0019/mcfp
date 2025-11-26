from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from mcfp.sim.robot_model import RobotModel
from mcfp.sim.collision import SelfCollisionChecker
from mcfp.sim.indicators import (
    compute_g_ws,
    compute_g_self,
    compute_g_lim,
    compute_g_man,
)
from mcfp.data.io import save_capability_map


def generate_capability_for_robot(
    urdf_path: Path,
    output_path: Path,
    grid_cfg: Any,
    logger,
) -> None:
    """End-to-end capability generation pipeline for a single robot.

    Parameters
    ----------
    urdf_path:
        Path to the robot URDF file.
    output_path:
        Path to the output capability map file (e.g., .npz).
    grid_cfg:
        Grid-related configuration object (from YAML), containing
        workspace bounds, resolution, sampling parameters, etc.
    logger:
        Logger instance created via mcfp.utils.logging.setup_logger.
    """
    logger.info(f"[grid_builder] Building capability map for URDF: {urdf_path}")

    # 1) Construct robot model
    robot = RobotModel(urdf_path=urdf_path, logger=logger)
    logger.info(
        f"[grid_builder] RobotModel created with {robot.num_joints} joints "
        f"from URDF: {urdf_path.name}"
    )

    # 2) Construct or load self-collision checker
    robot_dir = urdf_path.parent
    self_checker = SelfCollisionChecker.from_robot(
        robot=robot,
        cache_dir=robot_dir,
        logger=logger,
    )
    logger.info("[grid_builder] SelfCollisionChecker initialized.")

    # 3) Build workspace grid
    grid_meta = _build_workspace_grid(grid_cfg, logger)

    # 4) Sample joint space per cell and compute indicators
    cap_data = _evaluate_grid(
        robot=robot,
        self_checker=self_checker,
        grid_meta=grid_meta,
        grid_cfg=grid_cfg,
        logger=logger,
    )

    # 5) Save capability map
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_capability_map(output_path, cap_data)
    logger.info(f"[grid_builder] Capability map saved to {output_path}")


def _build_workspace_grid(grid_cfg: Any, logger) -> Dict[str, np.ndarray]:
    """Construct workspace grid metadata.

    Parameters
    ----------
    grid_cfg:
        Configuration object with workspace bounds and resolution.
    logger:
        Logger instance.

    Returns
    -------
    grid_meta:
        Dictionary containing at least:
        - "cell_centers": (N, 3) array of XYZ centers.
        - "cell_shape": (nx, ny, nz) shape of the grid.
    """
    # TODO: 从 grid_cfg 中读取边界和分辨率，例如：
    #   grid_cfg.xyz_min, grid_cfg.xyz_max, grid_cfg.resolution
    #   当前占位：只构造一个单 cell，在原点附近。
    logger.warning(
        "[grid_builder] Workspace grid construction is a placeholder. "
        "Only a single cell at origin is created."
    )

    cell_centers = np.zeros((1, 3), dtype=np.float32)
    cell_shape = (1, 1, 1)

    return {
        "cell_centers": cell_centers,
        "cell_shape": np.array(cell_shape, dtype=int),
    }


def _evaluate_grid(
    robot: RobotModel,
    self_checker: SelfCollisionChecker,
    grid_meta: Dict[str, np.ndarray],
    grid_cfg: Any,
    logger,
) -> Dict[str, np.ndarray]:
    """Sample joint space for each grid cell and compute indicators.

    Parameters
    ----------
    robot:
        Robot model instance.
    self_checker:
        Self-collision checker.
    grid_meta:
        Workspace grid metadata returned by _build_workspace_grid.
    grid_cfg:
        Grid-related configuration (sampling parameters, thresholds).
    logger:
        Logger instance.

    Returns
    -------
    cap_data:
        Dictionary of capability arrays, ready to be saved.
    """
    cell_centers = grid_meta["cell_centers"]
    num_cells = cell_centers.shape[0]
    logger.info(f"[grid_builder] Evaluating {num_cells} workspace cells.")

    # TODO: 从 grid_cfg 中读取每个 cell 的采样个数、收敛阈值等
    # 为了先跑通，当前每个 cell 只采样一个随机 q。
    num_samples_per_cell = 1

    # Allocate arrays
    g_ws_all = np.zeros((num_cells,), dtype=np.int32)
    g_self_all = np.zeros((num_cells,), dtype=np.float32)
    g_lim_all = np.zeros((num_cells,), dtype=np.float32)
    g_man_all = np.zeros((num_cells,), dtype=np.float32)
    u_all = np.zeros((num_cells,), dtype=np.float32)

    joint_limits = robot.joint_limits
    if not joint_limits:
        logger.warning(
            "[grid_builder] Joint limits are empty. "
            "Joint limit indicator will be zero."
        )

    for idx in range(num_cells):
        center = cell_centers[idx]
        logger.debug(f"[grid_builder] Evaluating cell {idx} at {center}.")

        # TODO: 在该 cell 内多次采样 q，并根据指标收敛情况选择最佳 q*
        # 当前为了跑通，只采样一次简单的均匀分布 q。
        q = _sample_random_q(joint_limits)

        g_ws_val = compute_g_ws(q, robot, self_checker)
        g_self_val = compute_g_self(q, self_checker)
        g_lim_val = compute_g_lim(q, joint_limits)
        g_man_val = compute_g_man(q, robot)

        # TODO: 设计更合理的 utility 函数，这里先简单相加
        u_val = float(g_ws_val) + g_self_val + g_lim_val + g_man_val

        g_ws_all[idx] = g_ws_val
        g_self_all[idx] = g_self_val
        g_lim_all[idx] = g_lim_val
        g_man_all[idx] = g_man_val
        u_all[idx] = u_val

    cap_data = {
        "cell_centers": cell_centers,
        "g_ws": g_ws_all,
        "g_self": g_self_all,
        "g_lim": g_lim_all,
        "g_man": g_man_all,
        "u": u_all,
    }

    return cap_data


def _sample_random_q(joint_limits) -> np.ndarray:
    """Sample a random joint configuration within given limits.

    Parameters
    ----------
    joint_limits:
        List of (lower, upper) joint bounds.

    Returns
    -------
    q:
        Random joint configuration as a 1D array.
    """
    if not joint_limits:
        # TODO: 考虑在没有限位信息时如何处理
        return np.zeros((0,), dtype=np.float32)

    lows = np.array([lo for lo, _ in joint_limits], dtype=np.float32)
    highs = np.array([hi for _, hi in joint_limits], dtype=np.float32)
    return lows + (highs - lows) * np.random.rand(len(joint_limits)).astype(
        np.float32
    )
