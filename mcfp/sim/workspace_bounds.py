# mcfp/sim/workspace_bounds.py

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from mcfp.sim.robot_model import RobotModel


def estimate_workspace_bounds(
    robot: RobotModel,
    samples: int,
    logger,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate 3D workspace AABB via random FK sampling."""
    samples = int(samples)
    if samples <= 0:
        raise ValueError("[workspace_bounds] bounds_samples must be positive.")

    positions = []
    chunk_size = 10000
    num_chunks = (samples + chunk_size - 1) // chunk_size

    low = robot.joint_limits[:, 0]
    span = robot.joint_limits[:, 1] - low

    for _ in range(num_chunks):
        q_chunk = low + np.random.rand(chunk_size, robot.num_joints) * span
        for q in q_chunk:
            try:
                pos = robot.fk_position(q)
                if np.all(np.isfinite(pos)):
                    positions.append(pos)
            except Exception:
                continue

    if not positions:
        raise RuntimeError("[workspace_bounds] AABB estimation failed: no valid FK samples.")

    pos_arr = np.asarray(positions, dtype=np.float32)
    xyz_min = pos_arr.min(axis=0)
    xyz_max = pos_arr.max(axis=0)

    logger.info(
        f"[workspace_bounds] AABB estimated. Min={np.round(xyz_min, 4)}, "
        f"Max={np.round(xyz_max, 4)}"
    )

    return xyz_min, xyz_max


def apply_aabb_margin(
    xyz_min: np.ndarray,
    xyz_max: np.ndarray,
    margin: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Expand AABB bounds by a symmetric margin."""
    center = 0.5 * (xyz_min + xyz_max)
    half = 0.5 * (xyz_max - xyz_min) * float(margin)
    return (center - half).astype(np.float32), (center + half).astype(np.float32)


def compute_urdf_aabb(
    urdf_path: Path,
    base_link: Optional[str],
    end_effector_link: Optional[str],
    samples: int,
    margin: float,
    logger,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute AABB bounds for a URDF using FK sampling."""
    robot = RobotModel(
        urdf_path=urdf_path,
        logger=logger,
        base_link=base_link,
        end_effector_link=end_effector_link,
    )
    xyz_min, xyz_max = estimate_workspace_bounds(robot=robot, samples=samples, logger=logger)
    return apply_aabb_margin(xyz_min=xyz_min, xyz_max=xyz_max, margin=margin)
