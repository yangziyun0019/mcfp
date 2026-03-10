"""Quaternion utilities used by orientation training and inference.

These functions implement the torch-based operations needed for iterative rotation updates.
"""

from __future__ import annotations

import torch


def quat_normalize(q: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return q / torch.clamp(torch.linalg.norm(q, dim=-1, keepdim=True), min=eps)


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product, q = q1 * q2.

    q format: (...,4) as [x, y, z, w].
    """
    x1, y1, z1, w1 = q1.unbind(-1)
    x2, y2, z2, w2 = q2.unbind(-1)
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return torch.stack([x, y, z, w], dim=-1)


def exp_quat(delta_omega: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Exponential map from axis-angle to quaternion.

    delta_omega: (...,3) axis-angle (rad).
    returns quaternion (...,4) [x,y,z,w].
    """
    theta = torch.linalg.norm(delta_omega, dim=-1, keepdim=True)
    half = 0.5 * theta
    small = theta < eps

    # sin(theta/2)/theta with Taylor for small angles
    sin_half = torch.sin(half)
    scale = torch.where(small, 0.5 - (theta ** 2) / 48.0, sin_half / torch.clamp(theta, min=1e-9))

    xyz = delta_omega * scale
    w = torch.where(small, 1.0 - (theta ** 2) / 8.0, torch.cos(half))
    return torch.cat([xyz, w], dim=-1)
