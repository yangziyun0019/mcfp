# mcfp/viz/capability_viz.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

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


def _plot_robot(ax, robot: RobotModel, q_vis: np.ndarray) -> np.ndarray:
    """Plot a simple kinematic skeleton of the robot.

    Prefer URDF link_edges if available. If not, fall back to a
    heuristic chain built by sorting link names such as 'Link1' ... 'Link6'.

    Returns
    -------
    points:
        Array of shape (N, 3) containing all link positions that were
        actually plotted. Empty array if nothing was drawn.
    """
    poses = robot.link_poses(q_vis)
    if poses is None or not poses:
        return np.zeros((0, 3), dtype=float)

    plotted_points: list[np.ndarray] = []

    
    edges = getattr(robot, "link_edges", None)
    if edges:
        for parent_link, child_link in edges:
            if parent_link not in poses or child_link not in poses:
                continue

            parent_pos, _ = poses[parent_link]
            child_pos, _ = poses[child_link]

            p = np.asarray(parent_pos, dtype=float).reshape(3,)
            c = np.asarray(child_pos, dtype=float).reshape(3,)

            ax.plot(
                [p[0], c[0]],
                [p[1], c[1]],
                [p[2], c[2]],
                linewidth=2.5,
                color="k",
                alpha=0.9,
            )
            ax.scatter([c[0]], [c[1]], [c[2]], s=18, color="k")
            plotted_points.append(p)
            plotted_points.append(c)
    else:

        import re

        link_names = list(poses.keys())

        def sort_key(name: str):
            """Sort by trailing integer if present, otherwise by name."""
            m = re.search(r"(\d+)$", name)
            if m:
                return (0, int(m.group(1)))
            return (1, name)

        link_names.sort(key=sort_key)


        for i in range(len(link_names) - 1):
            parent_link = link_names[i]
            child_link = link_names[i + 1]

            parent_pos, _ = poses[parent_link]
            child_pos, _ = poses[child_link]

            p = np.asarray(parent_pos, dtype=float).reshape(3,)
            c = np.asarray(child_pos, dtype=float).reshape(3,)

            ax.plot(
                [p[0], c[0]],
                [p[1], c[1]],
                [p[2], c[2]],
                linewidth=2.5,
                color="k",
                alpha=0.9,
            )
            ax.scatter([c[0]], [c[1]], [c[2]], s=18, color="k")
            plotted_points.append(p)
            plotted_points.append(c)

    if robot.end_effector_link is not None and robot.end_effector_link in poses:
        ee_pos, _ = poses[robot.end_effector_link]
        ee_pos = np.asarray(ee_pos, dtype=float).reshape(3,)
        ax.scatter(
            [ee_pos[0]],
            [ee_pos[1]],
            [ee_pos[2]],
            s=40,
            marker="*",
            color="r",
        )
        plotted_points.append(ee_pos)

    if not plotted_points:
        return np.zeros((0, 3), dtype=float)

    return np.vstack(plotted_points)


def plot_capability_with_robot(
    urdf_path: Path,
    cap_path: Path,
    base_link: Optional[str] = None,
    end_effector_link: Optional[str] = None,
    value_key: str = "u",
    log_dir: Optional[Path] = None,
    custom_values: Optional[np.ndarray] = None,
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
        Which scalar field to use for colouring workspace cells when
        custom_values is None. Typical choices: "u", "g_ws",
        "g_self", "g_self_qstar", "g_lim", "g_man".
    log_dir:
        Directory for log files. If None, logging is disabled.
    custom_values:
        Optional array of scalar values to use for colouring workspace
        cells. If provided, it must have length equal to the number of
        cells in the capability map and will override value_key.
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

    # Use best-sample positions if available; otherwise fall back to cell centers
    if "x_pos" in cap_data:
        positions = np.asarray(cap_data["x_pos"], dtype=np.float32)
        logger.info(
            "[capability_viz] Using x_pos as point locations for visualisation."
        )
    else:
        positions = np.asarray(cap_data["cell_centers"], dtype=np.float32)
        logger.info(
            "[capability_viz] x_pos not found; using cell_centers for visualisation."
        )

    # Decide which scalar field to use
    if custom_values is not None:
        values = np.asarray(custom_values, dtype=np.float32).reshape(-1)
        logger.info(
            "[capability_viz] Using custom_values for visualisation (len=%d).",
            values.shape[0],
        )
        if values.shape[0] != positions.shape[0]:
            raise ValueError(
                f"[capability_viz] custom_values has length {values.shape[0]}, "
                f"but positions has {positions.shape[0]} cells."
            )
        value_name_for_log = "custom"
    else:
        if value_key not in cap_data:
            raise KeyError(
                f"[capability_viz] value_key='{value_key}' not found in capability map."
            )
        values = np.asarray(cap_data[value_key], dtype=np.float32)
        value_name_for_log = value_key

    g_ws = (
        np.asarray(cap_data.get("g_ws", None), dtype=np.float32)
        if "g_ws" in cap_data
        else None
    )

    num_cells = positions.shape[0]
    logger.info(
        "[capability_viz] Loaded %d cells; plotting with value='%s'.",
        num_cells,
        value_name_for_log,
    )

    # ----- filter cells by validity and value threshold -----
    # Valid cells: have a finite value, and (if g_ws is present) g_ws > 0
    valid_mask = np.isfinite(values)
    if g_ws is not None:
        valid_mask &= g_ws > 0.0

    # Simple value threshold: keep cells with value >= this
    value_threshold = 0.0
    valid_mask &= values >= value_threshold

    if not np.any(valid_mask):
        logger.warning(
            "[capability_viz] No cells with valid %s >= %.3f; nothing to plot.",
            value_name_for_log,
            value_threshold,
        )
        return

    positions = positions[valid_mask]
    values = values[valid_mask]

    logger.info(
        "[capability_viz] After masking and thresholding (%s >= %.3f), kept %d cells.",
        value_name_for_log,
        value_threshold,
        positions.shape[0],
    )

    # Workspace bounds (still based on grid meta)
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
        urdf_path=str(urdf_path),
        logger=logger,
        base_link=base_link,
        end_effector_link=end_effector_link,
    )

    q_plot = np.zeros(robot.num_joints, dtype=float)
    logger.info(
        "[capability_viz] Plotting robot at q=%s (zeros configuration).",
        q_plot,
    )

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Scatter plot of capability values
    sc = ax.scatter(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        c=values,
        cmap="plasma",
        s=8.0,
        alpha=0.9,
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.7)
    cbar.set_label(value_name_for_log)

    # Plot the robot skeleton
    _plot_robot_skeleton(ax, robot, q_plot)

    # Draw workspace bounding box
    _plot_workspace_box(ax, xyz_min, xyz_max)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Extend axes to include both workspace and robot geometry, with padding
    robot_pts = _collect_robot_points(robot, q_plot)
    if robot_pts.size > 0:
        r_min = robot_pts.min(axis=0)
        r_max = robot_pts.max(axis=0)

        all_min = np.minimum(xyz_min, r_min)
        all_max = np.maximum(xyz_max, r_max)
    else:
        all_min = xyz_min
        all_max = xyz_max

    span = all_max - all_min
    padding = 0.2 * span
    ax.set_xlim(all_min[0] - padding[0], all_max[0] + padding[0])
    ax.set_ylim(all_min[1] - padding[1], all_max[1] + padding[1])
    ax.set_zlim(all_min[2] - padding[2], all_max[2] + padding[2])

    ax.view_init(elev=30.0, azim=45.0)

    plt.tight_layout()
    plt.show()

def _plot_robot_skeleton(
    ax: plt.Axes,
    robot: RobotModel,
    q: np.ndarray,
) -> None:
    """Plot a simple placeholder for the robot skeleton.

    This stub intentionally does nothing. It keeps the visualisation
    pipeline simple and avoids hard dependencies on the internal robot
    representation. The workspace point cloud is still shown correctly.

    Parameters
    ----------
    ax : matplotlib 3D axes
        Target axes for drawing.
    robot : RobotModel
        Robot model instance (currently unused).
    q : np.ndarray
        Joint configuration used for plotting (currently unused).
    """
    # If you later want a real skeleton, implement it here by querying
    # link positions from the kinematics backend and drawing line
    # segments between successive links.
    return


def _collect_robot_points(
    robot: RobotModel,
    q: np.ndarray,
) -> np.ndarray:
    """Collect robot points for axis scaling.

    This stub returns an empty array, so axis limits will be determined
    purely by the workspace bounds. It is sufficient for debugging
    capability fields.

    Parameters
    ----------
    robot : RobotModel
        Robot model instance (currently unused).
    q : np.ndarray
        Joint configuration used for plotting (currently unused).

    Returns
    -------
    pts : np.ndarray
        Array of shape (0, 3). A non-empty implementation could return
        link frame origins for better axis scaling.
    """
    return np.empty((0, 3), dtype=float)



