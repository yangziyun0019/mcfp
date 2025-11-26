from __future__ import annotations

from typing import Sequence

import numpy as np

from mcfp.sim.robot_model import RobotModel
from mcfp.sim.collision import SelfCollisionChecker


def compute_g_ws(
    q: Sequence[float],
    robot: RobotModel,
    checker: SelfCollisionChecker,
) -> int:
    """Compute workspace reachability indicator g_ws(r, q).

    Parameters
    ----------
    q:
        Joint configuration.
    robot:
        Robot model instance.
    checker:
        Self-collision checker.

    Returns
    -------
    g_ws:
        1 if configuration is considered reachable and collision-free,
        0 otherwise. Currently always returns 1 as a placeholder.
    """
    # TODO: 根据自碰、关节限位等条件判断是否可达
    _ = robot
    _ = checker
    return 1


def compute_g_self(
    q: Sequence[float],
    checker: SelfCollisionChecker,
) -> float:
    """Compute self-collision clearance g_self(r, q).

    Parameters
    ----------
    q:
        Joint configuration.
    checker:
        Self-collision checker.

    Returns
    -------
    g_self:
        Minimum distance between any self-collision link pair.
        Placeholder uses checker.min_distance.
    """
    # TODO: 决定是否需要进行归一化或截断
    return float(checker.min_distance(q))


def compute_g_lim(
    q: Sequence[float],
    joint_limits,
) -> float:
    """Compute joint limit clearance g_lim(r, q).

    Parameters
    ----------
    q:
        Joint configuration.
    joint_limits:
        List of (lower, upper) joint bounds.

    Returns
    -------
    g_lim:
        Minimum normalized distance to joint limits in [0, 1].
        Returns 0.0 if limits are not available.
    """
    q = np.asarray(q, dtype=float)
    if not joint_limits:
        # TODO: 实现真正的限位裕度，当前没有限位信息时直接给 0
        return 0.0

    distances = []
    for i, (lower, upper) in enumerate(joint_limits):
        width = upper - lower
        if width <= 0:
            continue
        d = min(upper - q[i], q[i] - lower) / width
        distances.append(d)

    if not distances:
        return 0.0

    return float(np.clip(min(distances), 0.0, 1.0))


def compute_g_man(
    q: Sequence[float],
    robot: RobotModel,
) -> float:
    """Compute manipulability index g_man(r, q), e.g. Yoshikawa.

    Parameters
    ----------
    q:
        Joint configuration.
    robot:
        Robot model instance.

    Returns
    -------
    g_man:
        Manipulability measure. Currently returns 0.0 as a placeholder.
    """
    # TODO: 使用 robot.jacobian(q) 计算 Yoshikawa 或其他操控性指标
    _ = q
    _ = robot
    return 0.0
