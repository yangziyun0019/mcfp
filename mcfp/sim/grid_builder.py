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

def _position_to_cell_index(
    pos: np.ndarray,
    grid_meta: Dict[str, np.ndarray],
) -> Optional[int]:
    """Map a 3D position to a flat cell index in the workspace grid.

    This uses only the position and the 3D grid defined by bounds_min,
    bounds_max, resolution and cell_shape in grid_meta.

    Parameters
    ----------
    pos:
        End-effector position in workspace coordinates, shape (3,).
    grid_meta:
        Workspace grid metadata as produced by _build_workspace_grid.

    Returns
    -------
    index:
        Flat cell index in [0, num_cells), or None if the position lies
        outside the workspace bounds.
    """
    p = np.asarray(pos, dtype=np.float32).reshape(3,)

    bounds_min = grid_meta["bounds_min"].astype(np.float32).reshape(3,)
    bounds_max = grid_meta["bounds_max"].astype(np.float32).reshape(3,)
    resolution = grid_meta["resolution"].astype(np.float32).reshape(3,)
    cell_shape = grid_meta["cell_shape"]
    if cell_shape.size != 3:
        raise ValueError("[grid_builder] grid_meta.cell_shape must be a (3,) array.")

    nx, ny, nz = int(cell_shape[0]), int(cell_shape[1]), int(cell_shape[2])

    # Fast reject if clearly outside the AABB
    if np.any(p < bounds_min) or np.any(p > bounds_max):
        return None

    # Compute 3D indices by flooring to the nearest lower cell boundary
    rel = (p - bounds_min) / resolution  # in units of cell size
    idx_float = np.floor(rel)
    idx = idx_float.astype(int)

    # Clamp to valid range to deal with numerical edge cases near bounds_max
    idx[0] = max(0, min(idx[0], nx - 1))
    idx[1] = max(0, min(idx[1], ny - 1))
    idx[2] = max(0, min(idx[2], nz - 1))

    ix, iy, iz = int(idx[0]), int(idx[1]), int(idx[2])

    if ix < 0 or ix >= nx or iy < 0 or iy >= ny or iz < 0 or iz >= nz:
        return None

    flat_index = ix * (ny * nz) + iy * nz + iz
    return flat_index

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
    """Sample joint space globally and aggregate best indicators per workspace cell.

    This function performs global random sampling in joint space. For each
    sampled configuration q, it computes forward kinematics, assigns the
    resulting end-effector position to a workspace cell (if inside the
    workspace AABB), and updates that cell's best configuration according
    to the utility score u(q).

    Orientation is currently used only as a global filter via
    _orientation_within_limits. Cell assignment is purely position-based
    so that later upgrades to a 6D pose grid can be implemented by
    extending the cell index mapping without changing the outer logic.

    Sampling hyperparameters and utility weights are read from grid_cfg.

    Sampling behaviour
    ------------------
    Let num_cells be the number of workspace cells. The following
    parameters are used:

    - sampling.min_samples_per_cell (default: 8)
        Target minimum number of *effective* samples per cell on average.
        Used only to derive a reasonable default for the total sampling
        budget if max_total_samples is not provided.

    - sampling.max_samples_per_cell (default: 64)
        Used only as a fallback to define max_total_samples when the
        latter is not explicitly configured.

    - sampling.max_total_samples (optional)
        Hard upper bound on the total number of sampled configurations.
        Default is num_cells * max_samples_per_cell.

    - sampling.target_coverage (default: 0.7)
        Fraction of cells that must have at least one valid sample
        (i.e. a finite utility) before convergence is allowed.

    - sampling.improvement_threshold (default: 1e-3)
        Minimum required improvement in u(q) to be considered a true
        update for a cell.

    - sampling.global_patience (optional)
        Number of *consecutive* samples without any utility improvement
        (in any cell) after which, once target_coverage is reached,
        sampling is stopped early. If not provided, falls back to
        sampling.patience, and finally to 1000.
    """
    joint_limits = robot.joint_limits
    if joint_limits.ndim != 2 or joint_limits.shape[1] != 2:
        raise ValueError(
            f"[grid_builder] joint_limits must have shape (num_joints, 2), "
            f"got {joint_limits.shape}."
        )

    cell_centers = grid_meta["cell_centers"]
    num_cells = int(cell_centers.shape[0])
    num_joints = int(joint_limits.shape[0])

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

    # Derive total sampling budget
    default_max_total = num_cells * max_samples_per_cell
    max_total_samples = int(_get_sampling_param("max_total_samples", default_max_total))

    target_coverage = float(_get_sampling_param("target_coverage", 0.7))
    target_coverage = float(max(0.0, min(target_coverage, 1.0)))

    # Global patience for early stopping after coverage is reached
    global_patience = int(
        _get_sampling_param(
            "global_patience",
            _get_sampling_param("patience", 1000),
        )
    )

    # How often to print progress; 0 or negative disables periodic logs
    progress_interval = int(
        _get_sampling_param(
            "progress_interval",
            max(10000, max_total_samples // 10),
        )
    )


    logger.info(
        "[grid_builder] Evaluating workspace grid with %d cells using "
        "global sampling: max_total_samples=%d, target_coverage=%.3f, "
        "min_samples_per_cell=%d.",
        num_cells,
        max_total_samples,
        target_coverage,
        min_samples_per_cell,
    )

    # Allocate arrays for per-cell best indicators and utility.
    # u_all is initialised to -inf so that any first valid sample wins.
    g_ws_all = np.zeros(num_cells, dtype=np.float32)
    g_self_all = np.zeros(num_cells, dtype=np.float32)
    g_lim_all = np.zeros(num_cells, dtype=np.float32)
    g_man_all = np.zeros(num_cells, dtype=np.float32)
    u_all = np.full(num_cells, -np.inf, dtype=np.float32)

    # Store the best configuration and pose per cell
    q_star_all = np.zeros((num_cells, num_joints), dtype=np.float32)
    x_pos_all = np.zeros((num_cells, 3), dtype=np.float32)
    x_ori_all = np.zeros((num_cells, 4), dtype=np.float32)

    # Per-cell sample counts (for diagnostics and potential future use)
    sample_counts = np.zeros(num_cells, dtype=np.int32)

    # Global convergence bookkeeping
    no_improve_streak = 0

    bounds_min = grid_meta["bounds_min"].astype(np.float32).reshape(3,)
    bounds_max = grid_meta["bounds_max"].astype(np.float32).reshape(3,)

    logger.info(
        "[grid_builder] Workspace bounds for evaluation: xyz_min=%s, xyz_max=%s.",
        bounds_min,
        bounds_max,
    )

    for it in range(1, max_total_samples + 1):
        q = _sample_random_q(joint_limits)

        # Forward kinematics to obtain pose
        try:
            pos, ori = robot.fk(q)
        except Exception as exc:
            if it == 1:
                logger.debug(
                    "[grid_builder] FK failed on first sample: %s.",
                    exc,
                )
            continue

        if pos is None or ori is None:
            continue

        pos_arr = np.asarray(pos, dtype=np.float32).reshape(3,)
        ori_arr = np.asarray(ori, dtype=np.float32).reshape(4,)

        # Reject samples outside workspace AABB
        if np.any(pos_arr < bounds_min) or np.any(pos_arr > bounds_max):
            continue

        # Optional global orientation filter
        if not _orientation_within_limits(ori_arr, grid_cfg):
            continue

        # Map position to a workspace cell
        cell_index = _position_to_cell_index(pos_arr, grid_meta)
        if cell_index is None:
            continue

        sample_counts[cell_index] += 1

        # Evaluate indicators for this (r, q)
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

        prev_u = u_all[cell_index]
        improved = False

        if not np.isfinite(prev_u) or (u_val > prev_u + improvement_threshold):
            # First valid sample for this cell or significant improvement
            u_all[cell_index] = float(u_val)
            g_ws_all[cell_index] = float(g_ws_val)
            g_self_all[cell_index] = float(g_self_val)
            g_lim_all[cell_index] = float(g_lim_val)
            g_man_all[cell_index] = float(g_man_val)
            q_star_all[cell_index] = q.astype(np.float32)
            x_pos_all[cell_index] = pos_arr
            x_ori_all[cell_index] = ori_arr
            improved = True

        if improved:
            no_improve_streak = 0
        else:
            no_improve_streak += 1

        # Convergence check
        valid_cells = int(np.count_nonzero(np.isfinite(u_all)))
        coverage = valid_cells / max(num_cells, 1)

        if progress_interval > 0 and (it % progress_interval == 0):
            logger.info(
                "[grid_builder] Sample %d / %d, coverage=%.3f (%d cells), "
                "no_improve_streak=%d.",
                it,
                max_total_samples,
                coverage,
                valid_cells,
                no_improve_streak,
            )


        if coverage >= target_coverage and no_improve_streak >= global_patience:
            logger.info(
                "[grid_builder] Early stopping at sample %d: "
                "coverage=%.3f (%d cells), no_improve_streak=%d.",
                it,
                coverage,
                valid_cells,
                no_improve_streak,
            )
            break

    # Finalise: cells that never received a valid sample keep all-zero
    # indicators, and their utility is set to 0 for downstream convenience.
    invalid_mask = ~np.isfinite(u_all)
    num_invalid = int(np.count_nonzero(invalid_mask))

    if num_invalid > 0:
        logger.info(
            "[grid_builder] %d cells received no valid samples; "
            "their indicators remain zero and utility is set to 0.",
            num_invalid,
        )
        u_all[invalid_mask] = 0.0

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
