"""Feature encodings for positions and reference quaternions.

These functions and modules provide the input representations used by MCFP networks.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


def position_encoding(p_norm: torch.Tensor, k_p: int = 10) -> torch.Tensor:
    """Fourier position encoding.

    Args:
        p_norm: (B,3) normalized positions.
        k_p: number of frequency bands.
    Returns:
        (B, 3 + 2*3*k_p) encoded positions.
    """
    if p_norm.ndim != 2 or p_norm.shape[1] != 3:
        raise ValueError(f"p_norm must be (B,3). Got {p_norm.shape}.")
    device = p_norm.device
    freq = (2.0 ** torch.arange(k_p, device=device, dtype=p_norm.dtype)) * math.pi
    # (B,3,K)
    angles = p_norm.unsqueeze(-1) * freq
    sin = torch.sin(angles)
    cos = torch.cos(angles)
    feat = torch.cat([sin, cos], dim=1)  # (B,6,K)
    feat = feat.reshape(p_norm.shape[0], -1)
    return torch.cat([p_norm, feat], dim=1)


def sample_ref_quaternions(k: int, seed: int = 20260126, device: Optional[torch.device] = None) -> torch.Tensor:
    """Sample K reference quaternions uniformly on SO(3), with w>=0."""
    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed))
    u1 = torch.rand(k, generator=rng)
    u2 = torch.rand(k, generator=rng)
    u3 = torch.rand(k, generator=rng)
    s1 = torch.sqrt(1.0 - u1)
    s2 = torch.sqrt(u1)
    theta1 = 2.0 * math.pi * u2
    theta2 = 2.0 * math.pi * u3
    qx = s1 * torch.sin(theta1)
    qy = s1 * torch.cos(theta1)
    qz = s2 * torch.sin(theta2)
    qw = s2 * torch.cos(theta2)
    q = torch.stack([qx, qy, qz, qw], dim=1)
    # enforce w >= 0
    sign = torch.where(q[:, 3:4] < 0, -1.0, 1.0)
    q = q * sign
    if device is not None:
        q = q.to(device)
    return q


def orientation_encoding(
    q: torch.Tensor,
    q_ref: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Quaternion geodesic surrogate encoding.

    Args:
        q: (B,4) quaternions.
        q_ref: (K,4) reference quaternions.
    Returns:
        (B, 2*K) encoding.
    """
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"q must be (B,4). Got {q.shape}.")
    q = q / torch.clamp(torch.linalg.norm(q, dim=1, keepdim=True), min=1e-9)
    dots = torch.einsum("bd,kd->bk", q, q_ref)
    dots = torch.abs(dots)
    c = torch.sqrt(dots * dots + eps)
    s = torch.sqrt(1.0 - c * c + eps)
    enc = torch.stack([c, s], dim=-1).reshape(q.shape[0], -1)
    return enc


def scalar_encoding(s: torch.Tensor) -> torch.Tensor:
    """Clip scalar input to [-1,1]."""
    return torch.clamp(s, -1.0, 1.0)


class ReferenceQuaternionEncoder(nn.Module):
    def __init__(
        self,
        k: int = 64,
        seed: int = 20260126,
        eps: float = 1e-8,
        q_ref: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.eps = float(eps)
        if q_ref is None:
            self.k = int(k)
            self.seed = int(seed)
            q_ref = sample_ref_quaternions(self.k, seed=self.seed)
        else:
            if q_ref.ndim != 2 or q_ref.shape[1] != 4:
                raise ValueError(f"q_ref must be (K,4). Got {q_ref.shape}.")
            q_ref = q_ref.to(dtype=torch.float32)
            q_ref = q_ref / torch.clamp(torch.linalg.norm(q_ref, dim=1, keepdim=True), min=1e-9)
            self.k = int(q_ref.shape[0])
            self.seed = int(seed)
        self.register_buffer("q_ref", q_ref)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        return orientation_encoding(q, self.q_ref, eps=self.eps)
