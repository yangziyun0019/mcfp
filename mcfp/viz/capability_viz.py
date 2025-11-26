# mcfp/viz/capability_viz.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from mcfp.sim.robot_model import RobotModel
from mcfp.utils.logging import setup_logger


def load_capability_map(path: Path) -> Dict[str, np.ndarray]:
    """Load capability map from a .npz file into a plain dict."""
    data = np.load(path)
    return {k: data[k] for k in data.files}


def _compute_workspace_bounds(cap_data: Dict[str, np.ndarray]) -> np.ndarray:
    """Compute workspace AABB from capability data.

    Prefer explicit bounds_min/bounds_max if available; otherwise use
    min/max over cell_centers.
    """
    if "bounds_min" in cap_data and "bounds_max" in cap_data:
        xyz_min = np.asarray(cap_data["bounds_min"], dtype=np.float32).reshape(3,)
        xyz_max = np.asarray(cap_data["bounds_max"], dtype=np.float32).reshape(3,)
        return np.stack([xyz_min, xyz_max], axis=0)

    centers = np.asarray(cap_data["cell_centers"], dtype=np.float32)
    xyz_min = centers.min(axis=0)
    xyz_max = centers.max(axis=0)
    return np.stack([xyz_min, xyz_max], axis=0)


def _plot_workspace_box(ax, xyz_min: np.ndarray, xyz_max: np.ndarray) -> None:
    """Draw axis-aligned workspace bounding box."""
    x0, y0, z0 = xyz_min
    x1, y1, z1 = xyz_max

    corners = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=np.float32,
    )

    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]

    segs = [(corners[i], corners[j]) for i, j in edges]
    lc = Line3DCollection(segs, linewidths=1.0, linestyles="dashed", alpha=0.6)
    ax.add_collection3d(lc)


def _plot_robot_points(
    ax,
    robot: RobotModel,
    q: np.ndarray,
    color: str = "black",
    size: float = 15.0,
) -> np.ndarray:
    """Plot robot link positions as points and a simple skeleton line.

    Returns
    -------
    pts:
        Array of shape (L, 3) with link positions. Empty array if nothing is plotted.
    """
    poses = robot.link_poses(q)
    print("[DEBUG] link_poses count:", len(poses), "names:", list(poses.keys()))

    poses = robot.link_poses(q)
    if not poses:
        return np.zeros((0, 3), dtype=np.float32)

    positions = []
    for pos, _ori in poses.values():
        positions.append(np.asarray(pos, dtype=np.float32).reshape(3,))

    pts = np.stack(positions, axis=0)

    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=size, c=color, depthshade=True)

    segments = [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    lc = Line3DCollection(segments, colors=color, linewidths=2.0, alpha=0.9)
    ax.add_collection3d(lc)

    return pts




def plot_capability_with_robot(
    urdf_path: Path,
    cap_path: Path,
    base_link: Optional[str] = None,
    end_effector_link: Optional[str] = None,
    value_key: str = "u",
    log_dir: Optional[Path] = None,
) -> None:
    """Visualize capability map together with a static robot pose.

    Parameters
    ----------
    urdf_path:
        Path to the URDF file of the robot.
    cap_path:
        Path to the .npz capability map file.
    base_link:
        Optional base link name for RobotModel.
    end_effector_link:
        Optional end-effector link name for RobotModel.
    value_key:
        Which scalar field to use for colouring workspace cells.
        Typical choices: "u", "g_ws", "g_self", "g_lim", "g_man".
    log_dir:
        Directory for log files. If None, logging is disabled.
    """
    logger_name = "mcfp.viz.capability"
    if log_dir is not None:
        logger = setup_logger(name=logger_name, log_dir=str(log_dir))
    else:
        logger = setup_logger(name=logger_name, log_dir=None)

    urdf_path = urdf_path.resolve()
    cap_path = cap_path.resolve()

    logger.info("[capability_viz] Loading capability map from %s", cap_path)
    cap_data = load_capability_map(cap_path)

    centers = np.asarray(cap_data["cell_centers"], dtype=np.float32)
    values = np.asarray(cap_data[value_key], dtype=np.float32)

    logger.info(
        "[capability_viz] Loaded %d cells; plotting with value_key='%s'.",
        centers.shape[0],
        value_key,
    )

        # ----- filter cells by value threshold -----
    threshold = 1.0  # 只保留得分 >= 5 的 cell
    mask = values >= threshold

    if not np.any(mask):
        logger.warning(
            "[capability_viz] No cells with %s >= %.3f; nothing to plot.",
            value_key,
            threshold,
        )
        return

    centers = centers[mask]
    values = values[mask]

    logger.info(
        "[capability_viz] After thresholding (%s >= %.3f), kept %d cells.",
        value_key,
        threshold,
        centers.shape[0],
    )
    # -------------------------------------------
    # Workspace bounds
    bounds = _compute_workspace_bounds(cap_data)
    xyz_min, xyz_max = bounds[0], bounds[1]
    logger.info(
        "[capability_viz] Workspace bounds: xyz_min=%s, xyz_max=%s",
        xyz_min,
        xyz_max,
    )

    # Construct robot model and pick a simple joint configuration
    logger.info("[capability_viz] Loading robot from URDF: %s", urdf_path)
    robot = RobotModel(
        urdf_path=urdf_path,
        logger=logger,
        base_link=base_link,
        end_effector_link=end_effector_link,
    )

    joint_limits = robot.joint_limits
    if joint_limits.size == 0:
        q_plot = np.zeros((robot.num_joints,), dtype=np.float32)
    else:
        lows = joint_limits[:, 0]
        highs = joint_limits[:, 1]
        q_plot = 0.5 * (lows + highs)
        q_plot = q_plot.astype(np.float32)

    # Prepare figure
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title(f"Capability map coloured by '{value_key}'")

    # Plot capability cells coloured by the selected scalar value
    sc = ax.scatter(
        centers[:, 0],
        centers[:, 1],
        centers[:, 2],
        c=values,
        cmap="viridis",
        s=5.0,
        alpha=0.9,
        depthshade=True,
    )
    fig.colorbar(sc, ax=ax, shrink=0.8, label=value_key)

    # Plot workspace AABB
    _plot_workspace_box(ax, xyz_min, xyz_max)

    # Plot robot links for a simple pose
    robot_pts = _plot_robot_points(ax, robot, q_plot, color="black", size=18.0)

    # 如果机器人点超出 workspace，把范围扩展到同时包括两者
    if robot_pts.size > 0:
        r_min = robot_pts.min(axis=0)
        r_max = robot_pts.max(axis=0)

        all_min = np.minimum(xyz_min, r_min)
        all_max = np.maximum(xyz_max, r_max)
    else:
        all_min = xyz_min
        all_max = xyz_max

    ax.set_xlim(all_min[0], all_max[0])
    ax.set_ylim(all_min[1], all_max[1])
    ax.set_zlim(all_min[2], all_max[2])


    ax.view_init(elev=30.0, azim=45.0)

    plt.tight_layout()
    plt.show()
