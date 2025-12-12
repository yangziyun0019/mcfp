from __future__ import annotations

from typing import List, Optional, Tuple, Dict, Any, Deque, Sequence

import numpy as np

from mcfp.sim.robot_model import RobotModel
from mcfp.sim.collision import SelfCollisionChecker
from mcfp.utils.logging import setup_logger
from scipy.spatial import KDTree

# Module-level logger. In scripts, you still initialise the main logger
# with cfg.logging.log_dir; here we use None as a safe default.
logger = setup_logger(name="mcfp.sim.indicators", log_dir=None)



def _as_array(q: Sequence[float], expected_dim: int) -> np.ndarray:
    """Convert q to a float32 array and enforce dimensionality.

    This helper keeps all indicator functions consistent. Any mismatch
    in degrees of freedom is treated as a hard error.
    """
    q_arr = np.asarray(q, dtype=np.float32).reshape(-1)
    if q_arr.shape[0] != expected_dim:
        msg = (
            f"[indicators] Got configuration with DOF={q_arr.shape[0]}, "
            f"but expected DOF={expected_dim}."
        )
        logger.error(msg)
        raise ValueError(msg)
    return q_arr



def compute_g_ws(
    q: Sequence[float],
    pose_in_cell: bool,
    robot: RobotModel,
    checker: SelfCollisionChecker,
) -> int:
    """Compute workspace reachability indicator g_ws(r, q).

    A configuration is counted as reachable for a given workspace cell r if:
    - The end-effector pose lies inside that cell (pose_in_cell is True).
    - Forward kinematics for the end-effector succeeds.
    - The configuration is self-collision free.
    - All joints stay within limits.

    Returns 1 for reachable, 0 otherwise.
    """
    if not pose_in_cell:
        return 0

    q_arr = _as_array(q, robot.num_joints)

    # 1) Joint limits
    g_lim = compute_g_lim(q_arr, robot.joint_limits)
    if g_lim <= 0.0:
        return 0

    # 2) Self-collision
    dist = checker.min_distance(q_arr)
    if not np.isfinite(dist) or dist <= 0.0:
        return 0

    # 3) Forward kinematics
    try:
        pos, ori = robot.fk(q_arr)
    except Exception:
        return 0

    if pos is None or ori is None:
        return 0

    return 1


def compute_g_self(
    q: Sequence[float],
    checker: SelfCollisionChecker,
    d_max: float,
) -> float:
    """Compute self-collision clearance indicator g_self(r, q).

    The raw quantity is the approximate minimum distance between different
    parts of the arm, as returned by SelfCollisionChecker.min_distance(..).
    The distance is mapped into [0, 1] with a saturating linear function:

        d <= 0         -> 0
        0 < d < d_max  -> d / d_max
        d >= d_max     -> 1

    The exact value of d_max is robot- and scale-dependent and should be
    provided from configuration.
    """
    q_arr = np.asarray(q, dtype=np.float32).reshape(-1)
    dist = checker.min_distance(q_arr)

    if not np.isfinite(dist):
        return 0.0
    if dist <= 0.0:
        return 0.0

    # Guard against invalid configuration.
    if d_max <= 0.0:
        d_max = 0.10

    return float(np.clip(dist / float(d_max), 0.0, 1.0))



def compute_g_lim(
    q: Sequence[float],
    joint_limits: np.ndarray,
) -> float:
    """Compute joint-limit indicator g_lim(r, q) in [0, 1].

    The indicator is high when joints stay away from their limits and low
    when they are near or outside their limits.

    For each joint i with limits [l_i, u_i]:
    - If q_i is outside [l_i, u_i], its contribution is 0.
    - Otherwise, the margin to the closest bound is normalised by half
      the range width. The final score is the average over all joints.
    """
    if joint_limits.size == 0:
        # No limit information: treat as neutral but non-zero.
        return 1.0

    # Enforce consistent dimensionality and centralised error handling.
    q_arr = _as_array(q, expected_dim=joint_limits.shape[0])

    lower = joint_limits[:, 0].astype(np.float32)
    upper = joint_limits[:, 1].astype(np.float32)
    widths = upper - lower

    # Guard against degenerate ranges.
    widths[widths <= 0.0] = 1e-6

    margins_low = q_arr - lower
    margins_high = upper - q_arr

    inside = (margins_low >= 0.0) & (margins_high >= 0.0)
    margin = np.minimum(margins_low, margins_high)
    margin[~inside] = 0.0

    # Normalise: margin == width / 2 -> 1, margin == 0 -> 0.
    half_width = widths * 0.5
    norm = np.clip(margin / half_width, 0.0, 1.0)

    return float(norm.mean())



def compute_g_man(
    q: Sequence[float],
    robot: RobotModel,
    k_man: float = 1.0,
) -> float:
    """Compute manipulability indicator g_man(r, q) in [0, 1].

    This uses a Yoshikawa-style manipulability measure based on
    sqrt(det(J J^T)) and maps it into [0, 1] via a saturating scale.

    Parameters
    ----------
    q : Sequence[float]
        Joint configuration.
    robot : RobotModel
        Robot model providing the Jacobian.
    k_man : float, optional
        Positive scale parameter controlling how fast the indicator
        saturates; typically read from configuration. Default is 1.0.

    Returns
    -------
    g_man : float
        Manipulability score in [0, 1], 0 for singular or invalid Jacobian.
    """
    q_arr = _as_array(q, expected_dim=robot.num_joints)

    try:
        J = robot.jacobian(q_arr)
    except Exception:
        return 0.0

    J = np.asarray(J, dtype=np.float64)
    if J.ndim != 2:
        return 0.0
    
    col_norms = np.linalg.norm(J, axis=0)
    valid_cols_mask = col_norms > 1e-6
    J = J[:, valid_cols_mask]
    
    # Use linear part only for underactuated arms
    rows, cols = J.shape
    if cols < 6 and rows == 6:
        J = J[:3, :]  

    JJ = J @ J.T
    det_val = float(np.linalg.det(JJ))
    if not np.isfinite(det_val) or det_val <= 0.0:
        return 0.0

    w = float(np.sqrt(det_val))

    # Normalisation scale; configuration should provide a positive value.
    if k_man <= 0.0:
        k_man = 1.0

    g_man = w / (w + float(k_man))
    return float(np.clip(g_man, 0.0, 1.0))


def compute_g_sigma_min(
    q: Sequence[float],
    robot: RobotModel,
    k_sigma: float = 0.1,
) -> float:
    """Compute minimum singular value indicator g_sigma_min(r, q) in [0, 1].

    The indicator is based on the smallest singular value of the Jacobian
    and measures how far the configuration is from kinematic singularity.

    Parameters
    ----------
    q : Sequence[float]
        Joint configuration.
    robot : RobotModel
        Robot model providing the Jacobian.
    k_sigma : float, optional
        Positive scale parameter for normalisation; typically read from
        configuration. Default is 0.1.

    Returns
    -------
    g_sigma_min : float
        Normalised minimum singular value in [0, 1].
    """
    q_arr = _as_array(q, expected_dim=robot.num_joints)

    try:
        J = robot.jacobian(q_arr)
    except Exception:
        return 0.0

    J = np.asarray(J, dtype=np.float64)
    if J.ndim != 2:
        return 0.0
    
    col_norms = np.linalg.norm(J, axis=0)
    J = J[:, col_norms > 1e-6]
    # Use linear part only for underactuated arms
    rows, cols = J.shape
    if cols < 6 and rows == 6:
        J = J[:3, :]

    try:
        s = np.linalg.svd(J, compute_uv=False)
    except np.linalg.LinAlgError:
        return 0.0

    if s.size == 0:
        return 0.0

    sigma_min = float(s.min())
    if not np.isfinite(sigma_min) or sigma_min <= 0.0:
        return 0.0

    if k_sigma <= 0.0:
        k_sigma = 0.1

    g = sigma_min / (sigma_min + float(k_sigma))
    return float(np.clip(g, 0.0, 1.0))

def compute_g_iso(
    q: Sequence[float],
    robot: RobotModel,
) -> float:
    """Isotropy index g_iso(q) in [0, 1], defined as sigma_min / sigma_max."""
    q_arr = _as_array(q, robot.num_joints)

    try:
        J = robot.jacobian(q_arr)
    except Exception:
        return 0.0

    if J is None:
        return 0.0

    J = np.asarray(J, dtype=np.float32)

    col_norms = np.linalg.norm(J, axis=0)
    J = J[:, col_norms > 1e-6]
    # Use linear part only for underactuated arms
    rows, cols = J.shape
    if cols < 6 and rows == 6:
        J = J[:3, :]

    try:
        s = np.linalg.svd(J, compute_uv=False)
    except np.linalg.LinAlgError:
        return 0.0

    s = np.asarray(s, dtype=np.float32).reshape(-1)
    if s.size == 0 or not np.all(np.isfinite(s)):
        return 0.0

    sigma_min = float(s.min())
    sigma_max = float(s.max())
    if sigma_max <= 0.0:
        return 0.0

    g = sigma_min / sigma_max
    return float(np.clip(g, 0.0, 1.0))

def compute_g_ws_margin(
    pos: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
) -> np.ndarray:
    """Workspace boundary margin for positions in AABB.

    Parameters
    ----------
    pos : array_like, shape (..., 3)
        Positions in workspace coordinates (e.g., cell centres).
    bounds_min, bounds_max : array_like, shape (3,)
        Workspace AABB used to build the grid.

    Returns
    -------
    g_ws_margin : np.ndarray, shape (...,)
        0 at the boundary, ~1 near the centre.
    """
    p = np.asarray(pos, dtype=np.float32)
    bmin = np.asarray(bounds_min, dtype=np.float32).reshape(1, 3)
    bmax = np.asarray(bounds_max, dtype=np.float32).reshape(1, 3)

    spans = np.maximum(bmax - bmin, 1e-6)
    margins = np.minimum(p - bmin, bmax - p)  # distance to 6 faces per axis
    raw = margins.min(axis=-1)                # (...,)

    raw = np.clip(raw, 0.0, None)

    ref = 0.5 * float(np.min(spans))
    if ref <= 0.0:
        return np.zeros_like(raw, dtype=np.float32)

    g = raw / ref
    return np.clip(g, 0.0, 1.0).astype(np.float32)

def compute_g_red(ik_counts: np.ndarray) -> np.ndarray:
    """Normalise per-cell IK solution counts to redundancy g_red in [0, 1].

    Parameters
    ----------
    ik_counts : array_like, shape (num_cells,)
        Number of valid IK samples per cell (e.g., g_ws >= 0.5).

    Returns
    -------
    g_red : np.ndarray, shape (num_cells,)
        Redundancy index per cell, 0 for no IK, 1 for highest redundancy.
    """
    counts = np.asarray(ik_counts, dtype=np.float32).reshape(-1)
    max_ik = float(counts.max())
    if max_ik <= 0.0:
        return np.zeros_like(counts, dtype=np.float32)

    g = np.log1p(counts) / np.log1p(max_ik)
    return np.clip(g, 0.0, 1.0).astype(np.float32)
# ------------------------------------------------------------------
# Rotational Coverage (Dexterity) Indicator
# ------------------------------------------------------------------

class SphereCoverageEvaluator:
    """Evaluates how well a set of approach vectors covers the unit sphere.
    
    Uses a Fibonacci lattice to generate uniformly distributed bins on a sphere.
    The coverage score is the ratio of unique bins occupied by the Z-axis 
    of the end-effector orientations.
    """
    
    def __init__(self, num_bins: int = 64):
        self.num_bins = num_bins
        # Generate reference points (bin centers)
        self.ref_points = self._fibonacci_sphere(num_bins)
        # KDTree for fast nearest-neighbor lookup
        self.tree = KDTree(self.ref_points)

    @staticmethod
    def _fibonacci_sphere(samples: int) -> np.ndarray:
        """Generate N uniformly distributed points on a sphere."""
        points = []
        phi = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle
        
        for i in range(samples):
            y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)
            theta = phi * i 
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])
            
        return np.array(points, dtype=np.float32)

    def compute_coverage(self, quaternions: np.ndarray) -> float:
        """Calculate coverage ratio for a batch of quaternions.

        Parameters
        ----------
        quaternions : np.ndarray
            Shape (N, 4) array of (x, y, z, w).

        Returns
        -------
        coverage : float
            Fraction of sphere bins covered [0.0, 1.0].
        """
        if quaternions.shape[0] == 0:
            return 0.0
            
        # 1. Extract Z-axis (Approach) vectors from quaternions
        # Rotating v=[0,0,1] by q=[x,y,z,w]:
        # x' = 2(xz + wy)
        # y' = 2(yz - wx)
        # z' = 1 - 2(xx + yy)
        
        x = quaternions[:, 0]
        y = quaternions[:, 1]
        z = quaternions[:, 2]
        w = quaternions[:, 3]
        
        vec_x = 2.0 * (x * z + w * y)
        vec_y = 2.0 * (y * z - w * x)
        vec_z = 1.0 - 2.0 * (x * x + y * y)
        
        z_vectors = np.stack([vec_x, vec_y, vec_z], axis=1) # (N, 3)
        
        # 2. Map vectors to nearest bins
        # k=1 returns (distances, indices)
        _, bin_indices = self.tree.query(z_vectors, k=1)
        
        # 3. Count unique occupied bins
        unique_bins = np.unique(bin_indices)
        
        return float(len(unique_bins) / self.num_bins)


# Global singleton to avoid rebuilding KDTree for every cell
_GLOBAL_SPHERE_EVALUATOR: Optional[SphereCoverageEvaluator] = None

def compute_g_rot(quaternions: List[np.ndarray]) -> float:
    """Compute Rotational Coverage g_rot for a list of quaternions.
    
    Parameters
    ----------
    quaternions : List[np.ndarray]
        List of (4,) arrays representing orientations in a cell.
        
    Returns
    -------
    g_rot : float
        Dexterity index in [0, 1].
    """
    global _GLOBAL_SPHERE_EVALUATOR
    if _GLOBAL_SPHERE_EVALUATOR is None:
        _GLOBAL_SPHERE_EVALUATOR = SphereCoverageEvaluator(num_bins=64)
        
    if not quaternions:
        return 0.0
        
    # Convert list of arrays to (N, 4) matrix
    q_arr = np.vstack(quaternions)
    return _GLOBAL_SPHERE_EVALUATOR.compute_coverage(q_arr)
