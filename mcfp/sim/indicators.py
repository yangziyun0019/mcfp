from __future__ import annotations

from typing import Sequence

import numpy as np

from mcfp.sim.robot_model import RobotModel
from mcfp.sim.collision import SelfCollisionChecker


def _as_array(q: Sequence[float], expected_dim: int) -> np.ndarray:
    """Convert q to a float32 array and check dimensionality.

    This helper keeps all indicator functions consistent.
    """
    q_arr = np.asarray(q, dtype=np.float32).reshape(-1)
    if q_arr.shape[0] != expected_dim:
        # Shape mismatch usually means an upstream bug, but we do not raise
        # here and simply return an all-zero vector so indicators become zero.
        return np.zeros((expected_dim,), dtype=np.float32)
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
) -> float:
    """Compute self-collision clearance indicator g_self(r, q).

    The raw quantity is the minimum distance between any self-collision pair.
    It is mapped into [0, 1] with a simple saturating linear function:

        d <= 0         -> 0
        0 < d < d_max  -> d / d_max
        d >= d_max     -> 1

    The exact value of d_max is a heuristic and can be moved to config later.
    """
    q_arr = np.asarray(q, dtype=np.float32).reshape(-1)
    dist = checker.min_distance(q_arr)

    if not np.isfinite(dist):
        return 0.0
    if dist <= 0.0:
        return 0.0

    # Heuristic saturation distance (meters).  TODO: move threshold to config.
    d_max = 0.10
    return float(np.clip(dist / d_max, 0.0, 1.0))


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

    q_arr = np.asarray(q, dtype=np.float32).reshape(-1)
    if q_arr.shape[0] != joint_limits.shape[0]:
        raise ValueError(
            f"[indicators] q has dimension {q_arr.shape[0]} "
            f"but joint_limits has {joint_limits.shape[0]}."
        )

    lows = joint_limits[:, 0].astype(np.float32)
    highs = joint_limits[:, 1].astype(np.float32)

    widths = np.maximum(highs - lows, 1e-6)
    margins_low = q_arr - lows
    margins_high = highs - q_arr

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
) -> float:
    """Compute manipulability indicator g_man(r, q) in [0, 1].

    This uses a Yoshikawa-style manipulability measure based on the
    geometric Jacobian J(q):

        w = sqrt(det(J J^T))

    The raw scalar w is then mapped to [0, 1] with a simple normalisation:

        g_man = w / (w + k)

    where k is a positive scale parameter.
    """
    q_arr = _as_array(q, robot.num_joints)

    try:
        J = robot.jacobian(q_arr)
    except Exception:
        return 0.0

    if J is None:
        return 0.0

    J = np.asarray(J, dtype=np.float32)
    JJ = J @ J.T

    try:
        det_val = float(np.linalg.det(JJ))
    except np.linalg.LinAlgError:
        det_val = 0.0

    if not np.isfinite(det_val) or det_val <= 0.0:
        return 0.0

    w = float(np.sqrt(det_val))

    # Heuristic scale.  TODO: move normalisation scale to config.
    k = 1.0
    g_man = w / (w + k)
    return float(np.clip(g_man, 0.0, 1.0))
