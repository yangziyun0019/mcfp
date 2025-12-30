from __future__ import annotations

from typing import Tuple

import numpy as np


def quat_normalize(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize a quaternion to unit length.

    Parameters
    ----------
    q:
        Quaternion [x, y, z, w].
    eps:
        Small epsilon for numerical stability.

    Returns
    -------
    qn:
        Normalized quaternion [x, y, z, w].
    """
    q = np.asarray(q, dtype=np.float32).reshape(4)
    n = float(np.linalg.norm(q))
    if n <= eps:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return (q / n).astype(np.float32)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Return quaternion conjugate."""
    q = np.asarray(q, dtype=np.float32).reshape(4)
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions."""
    x1, y1, z1, w1 = np.asarray(q1, dtype=np.float32).reshape(4)
    x2, y2, z2, w2 = np.asarray(q2, dtype=np.float32).reshape(4)
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return np.array([x, y, z, w], dtype=np.float32)


def axis_angle_to_quat(axis_angle: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Convert axis-angle vector to quaternion.

    Parameters
    ----------
    axis_angle:
        Axis-angle vector (axis * angle) of shape (3,).
    eps:
        Small epsilon for numerical stability.

    Returns
    -------
    q:
        Quaternion [x, y, z, w].
    """
    aa = np.asarray(axis_angle, dtype=np.float32).reshape(3)
    angle = float(np.linalg.norm(aa))
    if angle <= eps:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    axis = aa / max(angle, eps)
    half = 0.5 * angle
    s = float(np.sin(half))
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, float(np.cos(half))], dtype=np.float32)


def quat_to_axis_angle(q: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, float]:
    """Convert quaternion to axis-angle representation.

    Parameters
    ----------
    q:
        Quaternion [x, y, z, w].
    eps:
        Small epsilon for numerical stability.

    Returns
    -------
    axis:
        Unit axis (3,).
    angle:
        Rotation angle in radians.
    """
    qn = quat_normalize(q, eps=eps)
    w = float(np.clip(qn[3], -1.0, 1.0))
    angle = float(2.0 * np.arccos(w))
    s = float(np.sqrt(max(1.0 - w * w, 0.0)))
    if s <= eps or angle <= eps:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32), 0.0
    axis = qn[:3] / s
    return axis.astype(np.float32), angle


def quat_delta_axis_angle(q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
    """Compute axis-angle delta from q_from to q_to.

    Parameters
    ----------
    q_from:
        Source quaternion [x, y, z, w].
    q_to:
        Target quaternion [x, y, z, w].

    Returns
    -------
    axis_angle:
        Axis-angle vector (axis * angle).
    """
    qf = quat_normalize(q_from)
    qt = quat_normalize(q_to)
    q_delta = quat_multiply(qt, quat_conjugate(qf))
    axis, angle = quat_to_axis_angle(q_delta)
    return axis * angle


def apply_delta_quat(q: np.ndarray, delta_axis_angle: np.ndarray) -> np.ndarray:
    """Apply axis-angle delta to a quaternion.

    Parameters
    ----------
    q:
        Base quaternion [x, y, z, w].
    delta_axis_angle:
        Axis-angle vector (axis * angle) to apply.

    Returns
    -------
    q_out:
        Updated quaternion [x, y, z, w].
    """
    q = quat_normalize(q)
    q_delta = axis_angle_to_quat(delta_axis_angle)
    q_out = quat_multiply(q_delta, q)
    return quat_normalize(q_out)
