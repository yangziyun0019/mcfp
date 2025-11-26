from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import math
import numpy as np

from mcfp.sim.robot_model import RobotModel
from mcfp.sim.collision import SelfCollisionChecker
from mcfp.sim.indicators import (
    compute_g_ws,
    compute_g_self,
    compute_g_lim,
    compute_g_man,
)
from mcfp.data.io import save_capability_map, save_workspace_samples


def _get_cfg_value(cfg: Any, key: str) -> Any:
    """Fetch a value from a config object that may be a dict or have attributes."""
    if isinstance(cfg, dict):
        if key not in cfg:
            raise ValueError(f"[grid_builder] Config has no field '{key}'.")
        return cfg[key]
    if not hasattr(cfg, key):
        raise ValueError(f"[grid_builder] Config has no attribute '{key}'.")
    return getattr(cfg, key)


def _get_subconfig(cfg: Any, name: str) -> Any:
    """Return a nested sub-config, compatible with dict and attribute access."""
    if isinstance(cfg, dict):
        return cfg.get(name, {})
    return getattr(cfg, name, {})


def _compute_utility(
    g_ws: float,
    g_self: float,
    g_lim: float,
    g_man: float,
    grid_cfg: Any,
) -> float:
    """Compute utility score u(q) as a weighted sum of indicators.

    The weights are read from grid_cfg.utility with sensible defaults.
    """
    utility_cfg = _get_subconfig(grid_cfg, "utility")

    def _get_weight(key: str, default: float) -> float:
        if isinstance(utility_cfg, dict):
            return float(utility_cfg.get(key, default))
        return float(getattr(utility_cfg, key, default))

    w_ws = _get_weight("w_ws", 10.0)
    w_self = _get_weight("w_self", 3.0)
    w_lim = _get_weight("w_lim", 2.0)
    w_man = _get_weight("w_man", 1.0)

    return (
        w_ws * float(g_ws)
        + w_self * float(g_self)
        + w_lim * float(g_lim)
        + w_man * float(g_man)
    )


def generate_capability_for_robot(
    urdf_path: Path,
    output_path: Path,
    grid_cfg: Any,
    base_link: Optional[str],
    end_effector_link: Optional[str],
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
        Configuration for workspace grid, sampling, and utility weights.
    base_link:
        Optional name of the base link in URDF.
    end_effector_link:
        Optional name of the end-effector link in URDF.
    logger:
        Logger instance.
    """
    logger.info(f"[grid_builder] Loading robot from URDF: {urdf_path}")

    robot = RobotModel(
        urdf_path=urdf_path,
        logger=logger,
        base_link=base_link,
        end_effector_link=end_effector_link,
    )

    robot_dir = urdf_path.parent  
    self_checker = SelfCollisionChecker.from_robot(
        robot=robot,
        cache_dir=robot_dir,
        logger=logger,
    )

    logger.info(
        f"[grid_builder] Robot has {robot.num_joints} joints with limits: "
        f"{robot.joint_limits}."
    )

    # 1) Estimate workspace bounds via FK sampling if required
    mode = _get_cfg_value(grid_cfg, "mode")
    if mode not in ("auto", "explicit"):
        raise ValueError(f"[grid_builder] Unsupported grid.mode='{mode}'.")

    if mode == "auto":
        (
            xyz_min,
            xyz_max,
            fk_positions,
            fk_qs,
        ) = _estimate_workspace_bounds(robot=robot, grid_cfg=grid_cfg, logger=logger)

        setattr(grid_cfg, "xyz_min", xyz_min.tolist())
        setattr(grid_cfg, "xyz_max", xyz_max.tolist())


        logger.info(
            "[grid_builder] Using auto-estimated workspace bounds: "
            f"xyz_min={xyz_min}, xyz_max={xyz_max}."
        )

        # Optionally save raw FK samples for debugging
        save_workspace_samples(
            output_path.with_suffix(".fk_samples.npz"),
            fk_positions,
            fk_qs,
        )
    else:
        xyz_min = np.asarray(_get_cfg_value(grid_cfg, "xyz_min"), dtype=np.float32)
        xyz_max = np.asarray(_get_cfg_value(grid_cfg, "xyz_max"), dtype=np.float32)

        if xyz_min.shape != (3,) or xyz_max.shape != (3,):
            raise ValueError(
                "[grid_builder] explicit xyz_min/xyz_max must be length-3 lists."
            )

        fk_positions = np.empty((0, 3), dtype=np.float32)
        fk_qs = np.empty((0, robot.num_joints), dtype=np.float32)

        logger.info(
            "[grid_builder] Using explicit workspace bounds: "
            f"xyz_min={xyz_min}, xyz_max={xyz_max}."
        )

    # 2) Build workspace grid metadata
    grid_meta = _build_workspace_grid(grid_cfg=grid_cfg, logger=logger)
    logger.info(
        "[grid_builder] Workspace grid has "
        f"{grid_meta['cell_centers'].shape[0]} cells."
    )

    # 3) Evaluate joint space sampling for each grid cell
    cap_data = _evaluate_grid(
        robot=robot,
        self_checker=self_checker,
        grid_meta=grid_meta,
        grid_cfg=grid_cfg,
        logger=logger,
    )

    # 4) Save capability map
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_capability_map(output_path, cap_data)
    logger.info(f"[grid_builder] Capability map saved to {output_path}")


def _estimate_workspace_bounds(
    robot,
    grid_cfg: Any,
    logger,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate workspace bounds by random FK sampling.

    The method samples joint configurations uniformly within joint limits,
    evaluates end-effector positions, and builds an axis-aligned bounding box
    around the valid samples, then expands it by a safety margin.

    Parameters
    ----------
    robot:
        RobotModel instance with joint_limits and fk_position available.
    grid_cfg:
        Config with fields bounds_samples and bounds_margin.
    logger:
        Logger instance.

    Returns
    -------
    xyz_min:
        Lower corner of the workspace bounding box.
    xyz_max:
        Upper corner of the workspace bounding box.
    positions:
        Array of sampled end-effector positions, shape (N, 3).
    joint_samples:
        Array of sampled joint configurations, shape (N, num_joints).
    """
    samples = int(_get_cfg_value(grid_cfg, "bounds_samples"))
    margin = float(_get_cfg_value(grid_cfg, "bounds_margin"))

    joint_limits = robot.joint_limits
    if joint_limits.ndim != 2 or joint_limits.shape[1] != 2:
        raise ValueError(
            f"[grid_builder] joint_limits must have shape (num_joints, 2), "
            f"got {joint_limits.shape}."
        )

    num_joints = joint_limits.shape[0]
    lower = joint_limits[:, 0]
    upper = joint_limits[:, 1]
    span = upper - lower

    logger.info(
        "[grid_builder] Estimating workspace bounds via FK sampling: "
        f"samples={samples}, margin={margin}."
    )

    positions: list[np.ndarray] = []
    joint_samples: list[np.ndarray] = []

    for i in range(samples):
        # Uniform sampling in joint space within limits
        u = np.random.rand(num_joints)
        q = lower + u * span

        try:
            pos = robot.fk_position(q)
        except Exception as exc:
            # Skip failed FK evaluations
            if i == 0:
                logger.warning(
                    "[grid_builder] fk_position failed during bounds sampling: "
                    f"{exc}. Skipping this sample."
                )
            continue

        pos = np.asarray(pos, dtype=float).reshape(-1)
        if pos.size != 3 or not np.all(np.isfinite(pos)):
            continue

        positions.append(pos.astype(np.float32))
        joint_samples.append(q.astype(np.float32))

    if not positions:
        raise RuntimeError(
            "[grid_builder] No valid FK samples found while estimating bounds."
        )

    pos_arr = np.stack(positions, axis=0)  # (N, 3)
    q_arr = np.stack(joint_samples, axis=0)  # (N, num_joints)

    # Axis-aligned bounding box of sample points
    xyz_min = pos_arr.min(axis=0)
    xyz_max = pos_arr.max(axis=0)

    # Expand by margin factor
    center = 0.5 * (xyz_min + xyz_max)
    half = 0.5 * (xyz_max - xyz_min)
    half_expanded = half * float(margin)

    xyz_min = center - half_expanded
    xyz_max = center + half_expanded

    logger.info(
        "[grid_builder] Estimated workspace bounds: "
        f"xyz_min={xyz_min}, xyz_max={xyz_max}."
    )

    return (
        xyz_min.astype(np.float32),
        xyz_max.astype(np.float32),
        pos_arr,
        q_arr,
    )


def _build_workspace_grid(grid_cfg: Any, logger) -> Dict[str, np.ndarray]:
    """Construct workspace grid metadata from config.

    Parameters
    ----------
    grid_cfg:
        Configuration object with workspace bounds and resolution.
        Expected fields:
        - xyz_min: [x_min, y_min, z_min]
        - xyz_max: [x_max, y_max, z_max]
        - resolution: either a scalar or [dx, dy, dz].
    logger:
        Logger instance.

    Returns
    -------
    grid_meta:
        Dictionary with:
        - cell_centers: (N, 3) array of cell centers.
        - cell_shape: (3,) array with [nx, ny, nz].
        - bounds_min: (3,) array of workspace lower bounds.
        - bounds_max: (3,) array of workspace upper bounds.
        - resolution: (3,) array of cell sizes.
    """
    xyz_min = np.asarray(_get_cfg_value(grid_cfg, "xyz_min"), dtype=np.float32)
    xyz_max = np.asarray(_get_cfg_value(grid_cfg, "xyz_max"), dtype=np.float32)
    resolution = _get_cfg_value(grid_cfg, "resolution")

    if np.isscalar(resolution):
        resolution = np.array([resolution, resolution, resolution], dtype=float)
    else:
        resolution = np.asarray(resolution, dtype=float).reshape(3)

    if xyz_min.shape != (3,) or xyz_max.shape != (3,):
        raise ValueError(
            "[grid_builder] xyz_min/xyz_max must be length-3 sequences."
        )

    logger.info(
        "[grid_builder] Building workspace grid with "
        f"xyz_min={xyz_min}, xyz_max={xyz_max}, resolution={resolution}."
    )

    # Compute number of cells along each axis
    spans = xyz_max - xyz_min
    if np.any(spans <= 0.0):
        raise ValueError("[grid_builder] Invalid workspace bounds: spans <= 0.")

    n_xyz = np.floor(spans / resolution).astype(int)
    n_xyz = np.maximum(n_xyz, 1)

    nx, ny, nz = int(n_xyz[0]), int(n_xyz[1]), int(n_xyz[2])

    xs = np.linspace(xyz_min[0], xyz_max[0], nx, endpoint=False) + 0.5 * resolution[0]
    ys = np.linspace(xyz_min[1], xyz_max[1], ny, endpoint=False) + 0.5 * resolution[1]
    zs = np.linspace(xyz_min[2], xyz_max[2], nz, endpoint=False) + 0.5 * resolution[2]

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    cell_centers = np.stack([X, Y, Z], axis=-1).reshape(-1, 3).astype(np.float32)

    cell_shape = np.array([nx, ny, nz], dtype=int)

    grid_meta = {
        "cell_centers": cell_centers,
        "cell_shape": cell_shape,
        "bounds_min": xyz_min.astype(np.float32),
        "bounds_max": xyz_max.astype(np.float32),
        "resolution": resolution.astype(np.float32),
    }

    return grid_meta


def _get_position_cell_bounds(
    cell_index: int,
    grid_meta: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return axis-aligned bounds [min, max] for a given workspace cell.

    The bounds are computed around the stored cell center using half the
    resolution as radius along each axis.
    """
    cell_shape = grid_meta.get("cell_shape")
    if cell_shape is None or cell_shape.size != 3:
        raise ValueError("[grid_builder] grid_meta.cell_shape must be a (3,) array.")

    nx, ny, nz = int(cell_shape[0]), int(cell_shape[1]), int(cell_shape[2])

    if cell_index < 0 or cell_index >= nx * ny * nz:
        raise IndexError(
            f"[grid_builder] cell_index {cell_index} out of range for shape {cell_shape}."
        )

    # Decode flat index into 3D indices (ix, iy, iz)
    iz = cell_index % nz
    iy = (cell_index // nz) % ny
    ix = cell_index // (ny * nz)

    centers = grid_meta["cell_centers"].reshape(nx, ny, nz, 3)
    center = centers[ix, iy, iz]

    resolution = grid_meta["resolution"].astype(np.float32).reshape(3,)
    half = resolution * 0.5

    xyz_min_cell = center - half
    xyz_max_cell = center + half

    return xyz_min_cell, xyz_max_cell


def _position_in_cell(
    pos: np.ndarray,
    xyz_min_cell: np.ndarray,
    xyz_max_cell: np.ndarray,
) -> bool:
    """Check whether a position lies inside a cell's axis-aligned bounds."""
    pos = np.asarray(pos, dtype=np.float32).reshape(3,)
    return bool(np.all(pos >= xyz_min_cell) and np.all(pos <= xyz_max_cell))


def _quat_to_euler(orientation: np.ndarray) -> np.ndarray:
    """Convert quaternion (x, y, z, w) to Euler angles (roll, pitch, yaw).

    The returned angles follow the XYZ (roll-pitch-yaw) convention.
    This helper is only used for coarse orientation binning and does not
    attempt to handle all edge cases perfectly.
    """
    q = np.asarray(orientation, dtype=np.float64).reshape(4,)
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw], dtype=np.float32)


def _orientation_within_limits(
    orientation: np.ndarray,
    grid_cfg: Any,
) -> bool:
    """Check whether the given orientation lies within configured limits.

    The limits are read from grid_cfg.orientation as Euler angle ranges:
    - mode: "euler" (currently the only supported mode)
    - min: [roll_min, pitch_min, yaw_min]
    - max: [roll_max, pitch_max, yaw_max]

    If no orientation config is provided, all orientations are accepted.
    """
    ori_cfg = _get_subconfig(grid_cfg, "orientation")
    if not ori_cfg:
        return True

    if isinstance(ori_cfg, dict):
        mode = ori_cfg.get("mode", "euler")
        min_vals = np.asarray(
            ori_cfg.get("min", [-math.pi, -math.pi, -math.pi]), dtype=np.float32
        )
        max_vals = np.asarray(
            ori_cfg.get("max", [math.pi, math.pi, math.pi]), dtype=np.float32
        )
    else:
        mode = getattr(ori_cfg, "mode", "euler")
        min_vals = np.asarray(
            getattr(ori_cfg, "min", [-math.pi, -math.pi, -math.pi]),
            dtype=np.float32,
        )
        max_vals = np.asarray(
            getattr(ori_cfg, "max", [math.pi, math.pi, math.pi]),
            dtype=np.float32,
        )

    mode = str(mode).lower()
    if mode != "euler":
        # Fallback: if an unsupported mode is requested, accept all.
        return True

    euler = _quat_to_euler(orientation)
    return bool(np.all(euler >= min_vals) and np.all(euler <= max_vals))


def _evaluate_grid(
    robot: RobotModel,
    self_checker: SelfCollisionChecker,
    grid_meta: Dict[str, np.ndarray],
    grid_cfg: Any,
    logger,
) -> Dict[str, np.ndarray]:
    """Sample joint space for each grid cell and compute indicators.

    For each workspace cell, this function samples joint configurations
    uniformly in joint space, computes forward kinematics, and keeps only
    those samples whose end-effector pose lies inside the corresponding
    workspace cell (position-wise) and satisfies the global orientation
    limits.

    For each cell, it then evaluates the indicators g_ws, g_self,
    g_lim, g_man and the utility u(q), and keeps the best configuration
    q* according to the utility score.

    Sampling hyperparameters and utility weights are read from grid_cfg.
    """
    joint_limits = robot.joint_limits
    if joint_limits.ndim != 2 or joint_limits.shape[1] != 2:
        raise ValueError(
            f"[grid_builder] joint_limits must have shape (num_joints, 2), "
            f"got {joint_limits.shape}."
        )

    cell_centers = grid_meta["cell_centers"]
    num_cells = cell_centers.shape[0]
    num_joints = joint_limits.shape[0]

    # Sampling hyperparameters
    sampling_cfg = _get_subconfig(grid_cfg, "sampling")

    def _get_sampling_param(key: str, default: float) -> float:
        if isinstance(sampling_cfg, dict):
            return sampling_cfg.get(key, default)
        return getattr(sampling_cfg, key, default)

    min_samples_per_cell = int(_get_sampling_param("min_samples_per_cell", 8))
    max_samples_per_cell = int(_get_sampling_param("max_samples_per_cell", 64))
    improvement_threshold = float(
        _get_sampling_param("improvement_threshold", 1e-3)
    )
    patience = int(_get_sampling_param("patience", 5))

    if max_samples_per_cell < min_samples_per_cell:
        logger.warning(
            "[grid_builder] max_samples_per_cell < min_samples_per_cell; "
            "clamping to be equal."
        )
        max_samples_per_cell = min_samples_per_cell

    logger.info(
        "[grid_builder] Evaluating workspace grid with "
        f"{num_cells} cells, min_samples_per_cell={min_samples_per_cell}, "
        f"max_samples_per_cell={max_samples_per_cell}."
    )

    # Allocate arrays for per-cell best indicators and utility
    g_ws_all = np.zeros(num_cells, dtype=np.float32)
    g_self_all = np.zeros(num_cells, dtype=np.float32)
    g_lim_all = np.zeros(num_cells, dtype=np.float32)
    g_man_all = np.zeros(num_cells, dtype=np.float32)
    u_all = np.zeros(num_cells, dtype=np.float32)

    # Store the best configuration and pose per cell
    q_star_all = np.zeros((num_cells, num_joints), dtype=np.float32)
    x_pos_all = np.zeros((num_cells, 3), dtype=np.float32)
    x_ori_all = np.zeros((num_cells, 4), dtype=np.float32)

    for idx in range(num_cells):
        xyz_min_cell, xyz_max_cell = _get_position_cell_bounds(idx, grid_meta)

        best_u = -np.inf
        best_g_ws = 0.0
        best_g_self = 0.0
        best_g_lim = 0.0
        best_g_man = 0.0
        best_q: Optional[np.ndarray] = None
        best_pos: Optional[np.ndarray] = None
        best_ori: Optional[np.ndarray] = None

        no_improve_count = 0
        valid_samples = 0

        logger.debug(
            f"[grid_builder] Evaluating cell {idx} / {num_cells} "
            f"at center {cell_centers[idx]}."
        )

        for it in range(1, max_samples_per_cell + 1):
            q = _sample_random_q(joint_limits)

            # Forward kinematics to obtain pose
            try:
                pos, ori = robot.fk(q)
            except Exception as exc:
                if it == 1:
                    logger.debug(
                        "[grid_builder] FK failed in cell "
                        f"{idx} on first sample: {exc}."
                    )
                continue

            if pos is None or ori is None:
                continue

            pos_arr = np.asarray(pos, dtype=np.float32).reshape(3,)
            ori_arr = np.asarray(ori, dtype=np.float32).reshape(4,)

            # Check whether the pose belongs to this cell
            if not _position_in_cell(pos_arr, xyz_min_cell, xyz_max_cell):
                continue

            if not _orientation_within_limits(ori_arr, grid_cfg):
                continue

            valid_samples += 1

            # Now evaluate indicators for this (r, q)
            pose_in_cell = True
            g_ws_val = compute_g_ws(q, pose_in_cell, robot, self_checker)
            g_self_val = compute_g_self(q, self_checker)
            g_lim_val = compute_g_lim(q, joint_limits)
            g_man_val = compute_g_man(q, robot)

            u_val = _compute_utility(
                g_ws=g_ws_val,
                g_self=g_self_val,
                g_lim=g_lim_val,
                g_man=g_man_val,
                grid_cfg=grid_cfg,
            )

            if u_val > best_u + improvement_threshold:
                best_u = u_val
                best_g_ws = g_ws_val
                best_g_self = g_self_val
                best_g_lim = g_lim_val
                best_g_man = g_man_val
                best_q = q.copy()
                best_pos = pos_arr.copy()
                best_ori = ori_arr.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Only allow early stopping after we have enough valid samples
            if (
                valid_samples >= min_samples_per_cell
                and no_improve_count >= patience
            ):
                break

        if not np.isfinite(best_u) or best_q is None:
            logger.debug(
                f"[grid_builder] No valid samples found for cell {idx}; "
                "leaving indicators at zero."
            )
            continue

        g_ws_all[idx] = float(best_g_ws)
        g_self_all[idx] = float(best_g_self)
        g_lim_all[idx] = float(best_g_lim)
        g_man_all[idx] = float(best_g_man)
        u_all[idx] = float(best_u)

        q_star_all[idx] = best_q
        x_pos_all[idx] = best_pos
        x_ori_all[idx] = best_ori

    cap_data = {
        "cell_centers": cell_centers,
        "q_star": q_star_all,
        "x_pos": x_pos_all,
        "x_ori": x_ori_all,
        "g_ws": g_ws_all,
        "g_self": g_self_all,
        "g_lim": g_lim_all,
        "g_man": g_man_all,
        "u": u_all,
    }

    return cap_data


def _sample_random_q(joint_limits: np.ndarray) -> np.ndarray:
    """Sample a random joint configuration within given limits.

    joint_limits is an array of shape (num_joints, 2).
    """
    if joint_limits.size == 0:
        # No limits: treat as zero configuration for now.
        # TODO: later consider unbounded joints with custom ranges.
        return np.zeros((0,), dtype=np.float32)

    # Ensure float32 array of shape (num_joints, 2)
    jl = np.asarray(joint_limits, dtype=np.float32)
    if jl.ndim != 2 or jl.shape[1] != 2:
        raise ValueError(
            f"joint_limits is expected to have shape (num_joints, 2), "
            f"got {jl.shape}."
        )

    lows = jl[:, 0]
    highs = jl[:, 1]

    # Guard against degenerate or inverted ranges
    widths = highs - lows
    widths = np.maximum(widths, 1e-6)

    rnd = np.random.rand(jl.shape[0]).astype(np.float32)
    return lows + widths * rnd
