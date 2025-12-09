# mcfp/sim/grid_builder.py

from __future__ import annotations

import collections
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Deque

import numpy as np

from mcfp.sim.robot_model import RobotModel
from mcfp.sim.collision import SelfCollisionChecker
from mcfp.sim.indicators import (
    compute_g_self,
    compute_g_lim,
    compute_g_man,
    compute_g_sigma_min,
    compute_g_iso,
    compute_g_ws_margin,
    compute_g_rot,
)
from mcfp.data.io import save_capability_map
import scipy.ndimage as ndimage


def _get_cfg_value(cfg: Any, key: str) -> Any:
    """Retrieve MANDATORY value from config object or dict. Raises error if missing."""
    if isinstance(cfg, dict):
        if key not in cfg:
            raise ValueError(f"[grid_builder] Config missing mandatory key '{key}'.")
        return cfg[key]
    if not hasattr(cfg, key):
        raise ValueError(f"[grid_builder] Config missing mandatory attr '{key}'.")
    return getattr(cfg, key)


def _get_cfg_val_default(cfg: Any, key: str, default: Any) -> Any:
    """Retrieve OPTIONAL value from config object or dict safely."""
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _get_subconfig(cfg: Any, name: str) -> Any:
    """Retrieve sub-config safely."""
    if isinstance(cfg, dict):
        return cfg.get(name, {})
    return getattr(cfg, name, {})


def generate_capability_for_robot(
    urdf_path: Path,
    output_path: Path,
    grid_cfg: Any,
    base_link: Optional[str],
    end_effector_link: Optional[str],
    logger,
) -> None:
    """Execute the end-to-end capability map generation pipeline.

    Process:
    1. Setup Robot & Collision Checker (Auto-radius estimation).
    2. Phase 1 (Scouting): Estimate workspace AABB via fast FK.
    3. Build Voxel Grid.
    4. Phase 2 (Surveying): Sample configurations using Hybrid Aggregation 
       and Event-based Convergence.
    5. Save results to .npz.
    """
    logger.info(f"[grid_builder] Initializing robot from: {urdf_path}")

    # 1. Initialize System
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
        f"[grid_builder] Robot ready. Joints: {robot.num_joints}. "
        "Starting Phase 1: Workspace Scouting."
    )

    # 2. Phase 1: Scouting (Bounds Estimation)
    mode = _get_cfg_value(grid_cfg, "mode")
    if mode == "auto":
        xyz_min, xyz_max = _estimate_workspace_bounds(robot, grid_cfg, logger)
        # Update config for record keeping (handling both dict and object)
        if isinstance(grid_cfg, dict):
            grid_cfg["xyz_min"] = xyz_min.tolist()
            grid_cfg["xyz_max"] = xyz_max.tolist()
        else:
            setattr(grid_cfg, "xyz_min", xyz_min.tolist())
            setattr(grid_cfg, "xyz_max", xyz_max.tolist())
    else:
        xyz_min = np.asarray(_get_cfg_value(grid_cfg, "xyz_min"), dtype=np.float32)
        xyz_max = np.asarray(_get_cfg_value(grid_cfg, "xyz_max"), dtype=np.float32)

    # 3. Build Grid
    grid_meta = _build_workspace_grid(
        grid_cfg=grid_cfg, xyz_min=xyz_min, xyz_max=xyz_max, logger=logger
    )

    logger.info("[grid_builder] Phase 1.5: Calibrating Collision Matrix (Auto-ACM)...")
    
    low = robot.joint_limits[:, 0]
    span = robot.joint_limits[:, 1] - low
    def sample_q_fn():
        return low + np.random.rand(robot.num_joints) * span

    self_checker.find_static_collisions(sample_fn=sample_q_fn, n_samples=500)

    # 4. Phase 2: Surveying (Main Sampling Loop)
    cap_data = _evaluate_grid_hybrid(
        robot=robot,
        self_checker=self_checker,
        grid_meta=grid_meta,
        grid_cfg=grid_cfg,
        logger=logger,
    )

    # 5. Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_capability_map(output_path, cap_data)
    logger.info(f"[grid_builder] Process complete. Output: {output_path}")


def _estimate_workspace_bounds(
    robot,
    grid_cfg: Any,
    logger,
) -> Tuple[np.ndarray, np.ndarray]:
    """Phase 1: Estimate workspace AABB via fast random FK sampling.
    
    Does not check collisions, purely geometric reach.
    """
    samples = int(_get_cfg_value(grid_cfg, "bounds_samples"))
    margin = float(_get_cfg_value(grid_cfg, "bounds_margin"))
    
    logger.info(f"[grid_builder] Scouting workspace with {samples} samples...")
    
    positions = []
    chunk_size = 10000
    num_chunks = (samples + chunk_size - 1) // chunk_size
    
    low = robot.joint_limits[:, 0]
    span = robot.joint_limits[:, 1] - low
    
    for _ in range(num_chunks):
        q_chunk = low + np.random.rand(chunk_size, robot.num_joints) * span
        
        # Fast FK loop (geometry only)
        for q in q_chunk:
            try:
                pos = robot.fk_position(q)
                if np.all(np.isfinite(pos)):
                    positions.append(pos)
            except Exception:
                pass
    
    if not positions:
        raise RuntimeError("[grid_builder] Scouting failed: No valid FK samples.")

    pos_arr = np.array(positions, dtype=np.float32)
    xyz_min = pos_arr.min(axis=0)
    xyz_max = pos_arr.max(axis=0)

    # Apply margin
    center = 0.5 * (xyz_min + xyz_max)
    half = 0.5 * (xyz_max - xyz_min) * margin
    
    bounds_min = (center - half).astype(np.float32)
    bounds_max = (center + half).astype(np.float32)
    
    logger.info(
        f"[grid_builder] Scouting done. Bounds: "
        f"Min={np.round(bounds_min, 3)}, Max={np.round(bounds_max, 3)}"
    )
    
    return bounds_min, bounds_max


def _build_workspace_grid(
    grid_cfg: Any, 
    xyz_min: np.ndarray, 
    xyz_max: np.ndarray, 
    logger
) -> Dict[str, np.ndarray]:
    """Construct voxel grid metadata based on bounds and resolution."""
    resolution = _get_cfg_value(grid_cfg, "resolution")
    if np.isscalar(resolution):
        resolution = np.array([resolution]*3, dtype=float)
    else:
        resolution = np.asarray(resolution, dtype=float)

    spans = xyz_max - xyz_min
    dims = np.ceil(spans / resolution).astype(int)
    dims = np.maximum(dims, 1)

    # Re-center grid
    real_span = dims * resolution
    center = (xyz_min + xyz_max) / 2
    bounds_min = center - real_span / 2
    bounds_max = center + real_span / 2

    # Generate centers
    xs = np.linspace(bounds_min[0] + resolution[0]/2, bounds_max[0] - resolution[0]/2, dims[0])
    ys = np.linspace(bounds_min[1] + resolution[1]/2, bounds_max[1] - resolution[1]/2, dims[1])
    zs = np.linspace(bounds_min[2] + resolution[2]/2, bounds_max[2] - resolution[2]/2, dims[2])

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    cell_centers = np.stack([X, Y, Z], axis=-1).reshape(-1, 3).astype(np.float32)

    logger.info(
        f"[grid_builder] Grid constructed. Shape: {dims}, "
        f"Total Cells: {len(cell_centers)}, Res: {resolution}"
    )

    return {
        "cell_centers": cell_centers,
        "cell_shape": dims,
        "bounds_min": bounds_min.astype(np.float32),
        "bounds_max": bounds_max.astype(np.float32),
        "resolution": resolution.astype(np.float32),
    }


def _position_to_cell_index(
    pos: np.ndarray, 
    grid_meta: Dict[str, np.ndarray]
) -> Optional[int]:
    """Map 3D position to flat cell index."""
    p = pos.reshape(3)
    bmin = grid_meta["bounds_min"]
    res = grid_meta["resolution"]
    dims = grid_meta["cell_shape"]

    idx = np.floor((p - bmin) / res).astype(int)
    
    if np.any(idx < 0) or np.any(idx >= dims):
        return None
        
    return idx[0] * (dims[1] * dims[2]) + idx[1] * dims[2] + idx[2]


def _evaluate_grid_hybrid(
    robot: RobotModel,
    self_checker: SelfCollisionChecker,
    grid_meta: Dict[str, np.ndarray],
    grid_cfg: Any,
    logger,
) -> Dict[str, np.ndarray]:
    """Phase 2: Main Sampling Loop with Hybrid Aggregation & Event Convergence.

    Aggregation Strategy:
      - Risk (g_self, g_lim): Expectation (Average)
      - Capability (g_man, etc.): Max
      - Density (g_red): Count

    Convergence Strategy:
      - Monitor 'Update Heat': Frequency of (New Cell Found OR Significant Max Improvement).
      - Stop when Heat drops below 'min_update_rate' over 'stability_window'.
    """
    
    # --- 1. Load Config (using robust helpers) ---
    joint_limits = robot.joint_limits
    cell_centers = grid_meta["cell_centers"]
    num_cells = len(cell_centers)
    
    # Self collision d_max
    self_cfg = _get_subconfig(grid_cfg, "self_collision")
    d_max = float(_get_cfg_val_default(self_cfg, "safe_clearance", 0.05))

    # Sampling params
    sampling_cfg = _get_subconfig(grid_cfg, "sampling")
    max_total_samples = int(_get_cfg_val_default(sampling_cfg, "max_total_samples", 1000000))
    target_coverage = float(_get_cfg_val_default(sampling_cfg, "target_coverage", 0.8))
    progress_interval = int(_get_cfg_val_default(sampling_cfg, "progress_interval", 20000))
    
    # Convergence params
    improvement_thresh = float(_get_cfg_val_default(sampling_cfg, "improvement_threshold", 0.01))
    stability_window = int(_get_cfg_val_default(sampling_cfg, "stability_window", 50000))
    min_update_rate = float(_get_cfg_val_default(sampling_cfg, "min_update_rate", 0.0005))

    # Heuristic weights for q* selection (visualization only)
    util_cfg = _get_subconfig(grid_cfg, "utility")
    w_man = float(_get_cfg_val_default(util_cfg, "w_man", 1.0))
    w_self = float(_get_cfg_val_default(util_cfg, "w_self", 1.0))

    # --- 2. Buffers ---
    count_valid = np.zeros(num_cells, dtype=np.int32)
    sum_g_self = np.zeros(num_cells, dtype=np.float32)
    sum_g_lim = np.zeros(num_cells, dtype=np.float32)
    
    max_g_man = np.zeros(num_cells, dtype=np.float32)
    max_g_iso = np.zeros(num_cells, dtype=np.float32)
    max_g_sigma = np.zeros(num_cells, dtype=np.float32)

    best_u_heuristic = np.full(num_cells, -np.inf, dtype=np.float32)
    q_star = np.zeros((num_cells, robot.num_joints), dtype=np.float32)

    # Sliding window for convergence monitoring (1 = significant event, 0 = boring)
    update_events: Deque[int] = collections.deque(maxlen=stability_window)

    cell_quats: List[List[np.ndarray]] = [[] for _ in range(num_cells)]

    # --- 3. Sampling Loop ---
    logger.info(
        f"[grid_builder] Starting Survey. Budget: {max_total_samples}. "
        f"Window: {stability_window}, Min Rate: {min_update_rate:.2%}"
    )

    low = joint_limits[:, 0]
    span = joint_limits[:, 1] - low
    
    total_new_cells = 0
    total_improvements = 0

    for i in range(max_total_samples):
        # A. Sample
        q = low + np.random.rand(robot.num_joints) * span
        
        # B. Check Validity (Self-Collision)
        # Note: radius handled inside checker
        val_g_self = compute_g_self(q, self_checker, d_max=d_max)
        if val_g_self <= 0.0:
            # Collision -> Invalid. Record as non-event (0) unless we skip entirely.
            # To measure "rate per sample attempt", we append 0.
            update_events.append(0)
            continue
            
        # C. FK
        try:
            pos, ori = robot.fk(q) 
        except Exception:
            update_events.append(0)
            continue
            
        idx = _position_to_cell_index(pos, grid_meta)
        if idx is None:
            update_events.append(0)
            continue

        cell_quats[idx].append(np.array(ori, dtype=np.float32))
            
        # D. Compute Indicators
        val_g_lim = compute_g_lim(q, joint_limits)
        val_g_man = compute_g_man(q, robot)
        val_g_iso = compute_g_iso(q, robot)
        val_g_sigma = compute_g_sigma_min(q, robot)
        
        # E. Check "Significance" (Convergence Logic)
        is_significant = False
        
        # E1. Topological Discovery
        if count_valid[idx] == 0:
            is_significant = True
            total_new_cells += 1
            
        # E2. Capability Breakthrough (Max Improvement)
        elif val_g_man > max_g_man[idx] * (1.0 + improvement_thresh):
            is_significant = True
            total_improvements += 1
            
        update_events.append(1 if is_significant else 0)

        # F. Update Aggregators
        count_valid[idx] += 1
        sum_g_self[idx] += val_g_self
        sum_g_lim[idx] += val_g_lim
        
        max_g_man[idx] = max(max_g_man[idx], val_g_man)
        max_g_iso[idx] = max(max_g_iso[idx], val_g_iso)
        max_g_sigma[idx] = max(max_g_sigma[idx], val_g_sigma)
        
        # Update q* (Heuristic)
        u_curr = w_man * val_g_man + w_self * val_g_self
        if u_curr > best_u_heuristic[idx]:
            best_u_heuristic[idx] = u_curr
            q_star[idx] = q

        # G. Logging & Convergence Check
        if (i + 1) % progress_interval == 0:
            n_valid_cells = np.count_nonzero(count_valid)
            coverage = n_valid_cells / num_cells
            
            # Calculate Heat (Update Rate)
            current_heat = 0.0
            if len(update_events) > 0:
                current_heat = sum(update_events) / len(update_events)
            
            # Log Status
            logger.info(
                f"Iter {i+1}/{max_total_samples} | "
                f"Cells: {n_valid_cells} ({coverage:.1%}) | "
                f"Heat: {current_heat:.4%} (Thresh: {min_update_rate:.4%}) | "
                f"Events: +{total_new_cells} New, ^{total_improvements} Impr."
            )
            
            # Check Stop Conditions
            # 1. Enough Topological Coverage
            if coverage >= target_coverage:
                # 2. Window is full (stats are reliable)
                if len(update_events) >= stability_window:
                    # 3. Heat is low (marginal utility is low)
                    if current_heat < min_update_rate:
                        logger.info(
                            f"[grid_builder] Converged! Heat ({current_heat:.5f}) "
                            f"< Threshold ({min_update_rate:.5f}). Stopping."
                        )
                        break

    # --- 4. Finalize Data ---
    mask = count_valid > 0
    num_final_valid = np.count_nonzero(mask)
    
    # Averages
    avg_g_self = np.zeros(num_cells, dtype=np.float32)
    avg_g_lim = np.zeros(num_cells, dtype=np.float32)
    
    if num_final_valid > 0:
        avg_g_self[mask] = sum_g_self[mask] / count_valid[mask]
        avg_g_lim[mask] = sum_g_lim[mask] / count_valid[mask]
    
    # Density (Log-Normalized)
    g_red = np.zeros(num_cells, dtype=np.float32)
    if num_final_valid > 0:
        raw = count_valid.astype(np.float32)
        g_red = np.log1p(raw)
        m_val = g_red.max()
        if m_val > 1e-6:
            g_red = g_red / m_val
            
    # Robust Distance Transform (EDT) for g_margin 
 
    dims = grid_meta["cell_shape"] # (Nx, Ny, Nz)
    valid_grid_3d = mask.reshape(dims)
    res = grid_meta["resolution"]
    
    if num_final_valid > 0:
        dist_grid = ndimage.distance_transform_edt(
            valid_grid_3d, 
            sampling=res  
        )
        
        max_dist = dist_grid.max()
        if max_dist > 1e-6:
            dist_grid = dist_grid / max_dist
        
        g_margin = dist_grid.flatten().astype(np.float32)
    else:
        g_margin = np.zeros(num_cells, dtype=np.float32)

    logger.info("[grid_builder] Computing rotational coverage (g_rot)...")
    g_rot = np.zeros(num_cells, dtype=np.float32)
    
    # Only compute for valid cells to save time
    valid_indices = np.where(mask)[0]
    for idx in valid_indices:
        # compute_g_rot handles the batch processing internally
        g_rot[idx] = compute_g_rot(cell_quats[idx])

    cap_data = {
        "cell_centers": cell_centers,
        "q_star": q_star,
        
        # Core Indicators
        "g_ws": mask.astype(np.float32),      # [0,1] Existence
        "g_red": g_red,                       # [0,1] Density
        "g_margin": g_margin,                 # [0,1] Boundary Dist (Updated via EDT)
        
        "g_self": avg_g_self,                 # [0,1] Avg Safety
        "g_lim": avg_g_lim,                   # [0,1] Avg Margin

        "g_rot": g_rot,                       # [0, 1] Dexterity
        
        "g_man": max_g_man,                   # [0,1] Max Capability
        "g_iso": max_g_iso,                   # [0,1] Max Isotropy
        "g_sigma": max_g_sigma,               # [0,1] Max Singularity
        
        # Debug
        "sample_counts": count_valid
    }
    
    logger.info(
        f"[grid_builder] Survey finished. "
        f"Valid Cells: {num_final_valid}/{num_cells} ({num_final_valid/num_cells:.1%})"
    )
    
    return cap_data