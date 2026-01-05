from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import json
import math

import numpy as np
import torch

from mcfp.utils import se3


@dataclass(frozen=True)
class BatchRatios:
    """Sampling ratios for each sample group."""
    boundary: float
    rot_far: float
    pos_far: float
    eikonal: float


def _load_meta(path: Path) -> Dict[str, Any]:
    """Load meta.json from a dataset root."""
    path = Path(path)
    meta_path = path / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"meta.json not found: {meta_path}")
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    """Load arrays from a NPZ file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"NPZ not found: {path}")
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def _split_counts(total: int, parts: int) -> List[int]:
    """Split total into near-equal parts."""
    if parts <= 0:
        return []
    base = int(total // parts)
    rem = int(total % parts)
    out = [base + (1 if i < rem else 0) for i in range(parts)]
    return out


def _choice_indices(rng: np.random.Generator, n_total: int, n: int) -> np.ndarray:
    """Sample indices with replacement if needed."""
    n_total = int(n_total)
    n = int(n)
    if n_total <= 0:
        return np.zeros((0,), dtype=np.int64)
    replace = n > n_total
    return rng.choice(n_total, size=n, replace=replace).astype(np.int64)


def _sample_uniform_quaternion(rng: np.random.Generator) -> np.ndarray:
    """Sample a uniform quaternion (x, y, z, w)."""
    u1, u2, u3 = rng.random(3)
    s1 = math.sqrt(1.0 - u1)
    s2 = math.sqrt(u1)
    theta1 = 2.0 * math.pi * u2
    theta2 = 2.0 * math.pi * u3
    qx = s1 * math.sin(theta1)
    qy = s1 * math.cos(theta1)
    qz = s2 * math.sin(theta2)
    qw = s2 * math.cos(theta2)
    return np.array([qx, qy, qz, qw], dtype=np.float32)


def _quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    """Convert quaternion to axis-angle vector."""
    axis, angle = se3.quat_to_axis_angle(q)
    return (axis * float(angle)).astype(np.float32)


def _to_w(z: np.ndarray, lambda_val: float) -> np.ndarray:
    """Convert z=[p/L_ref, xi] to w=[p/L_ref, xi/lambda]."""
    z = np.asarray(z, dtype=np.float32)
    if z.ndim != 2 or z.shape[1] != 6:
        raise ValueError(f"z must be (N,6). Got {z.shape}.")
    w = np.zeros_like(z, dtype=np.float32)
    w[:, :3] = z[:, :3]
    w[:, 3:] = z[:, 3:] / float(lambda_val)
    return w


class SDFDataset:
    """Container for one morphology's SDF samples."""

    def __init__(
        self,
        root: Path | str,
        *,
        require_boundary: bool = True,
    ) -> None:
        self.root = Path(root)
        self.meta = _load_meta(self.root)

        if "l_ref" not in self.meta or "lambda" not in self.meta:
            raise ValueError("[SDFDataset] meta.json must contain l_ref and lambda.")

        self.l_ref = float(self.meta["l_ref"])
        self.lambda_val = float(self.meta["lambda"])
        self.delta = float(self.meta.get("delta", 0.0))

        aabb_min = self.meta.get("aabb_min", None)
        aabb_max = self.meta.get("aabb_max", None)
        self.aabb_min = np.asarray(aabb_min, dtype=np.float32) if aabb_min is not None else None
        self.aabb_max = np.asarray(aabb_max, dtype=np.float32) if aabb_max is not None else None

        boundary = _load_npz(self.root / "boundary_samples.npz")
        rot_far = _load_npz(self.root / "rot_far_samples.npz")
        pos_far = _load_npz(self.root / "pos_far_samples.npz")

        if "z" not in boundary or "y" not in boundary:
            raise ValueError("[SDFDataset] boundary_samples.npz must include z and y.")
        if "z" not in rot_far:
            raise ValueError("[SDFDataset] rot_far_samples.npz must include z.")
        if "z" not in pos_far:
            raise ValueError("[SDFDataset] pos_far_samples.npz must include z.")

        z_nb = boundary["z"]
        y_nb = boundary["y"].astype(np.float32).reshape(-1)

        w_nb = _to_w(z_nb, self.lambda_val)
        tol = 1e-6
        idx_b = np.isclose(y_nb, 0.0, atol=tol)
        idx_in = y_nb > 0.0
        idx_out = y_nb < 0.0

        self.w_b = w_nb[idx_b]
        self.w_in = w_nb[idx_in]
        self.w_out = w_nb[idx_out]

        if require_boundary and self.w_b.shape[0] == 0:
            raise ValueError("[SDFDataset] No boundary points (y==0).")

        self.w_rot_far = _to_w(rot_far["z"], self.lambda_val)
        self.w_pos_far = _to_w(pos_far["z"], self.lambda_val)

    def sample_boundary(
        self,
        n_total: int,
        rng: np.random.Generator,
    ) -> Dict[str, np.ndarray]:
        """Sample boundary neighborhood points."""
        n_total = int(n_total)
        n_in, n_out, n_b = _split_counts(n_total, 3)
        idx_in = _choice_indices(rng, self.w_in.shape[0], n_in)
        idx_out = _choice_indices(rng, self.w_out.shape[0], n_out)
        idx_b = _choice_indices(rng, self.w_b.shape[0], n_b)
        return {
            "w_in": self.w_in[idx_in],
            "w_out": self.w_out[idx_out],
            "w_b": self.w_b[idx_b],
        }

    def sample_far(
        self,
        n_rot_far: int,
        n_pos_far: int,
        rng: np.random.Generator,
    ) -> Dict[str, np.ndarray]:
        """Sample far negative points."""
        n_rot_far = int(n_rot_far)
        n_pos_far = int(n_pos_far)
        idx_rot = _choice_indices(rng, self.w_rot_far.shape[0], n_rot_far)
        idx_pos = _choice_indices(rng, self.w_pos_far.shape[0], n_pos_far)
        return {
            "w_rot_far": self.w_rot_far[idx_rot],
            "w_pos_far": self.w_pos_far[idx_pos],
        }

    def sample_eikonal(
        self,
        w_ref: np.ndarray,
        n_jitter: int,
        n_cover: int,
        rng: np.random.Generator,
        *,
        sigma: float,
        cover_scale: float = 1.2,
    ) -> np.ndarray:
        """Sample eikonal points from jittered boundary and AABB coverage."""
        w_ref = np.asarray(w_ref, dtype=np.float32)
        if w_ref.ndim != 2 or w_ref.shape[1] != 6:
            raise ValueError(f"w_ref must be (N,6). Got {w_ref.shape}.")

        n_jitter = int(n_jitter)
        n_cover = int(n_cover)

        jitter = np.zeros((0, 6), dtype=np.float32)
        if n_jitter > 0 and w_ref.shape[0] > 0:
            idx = _choice_indices(rng, w_ref.shape[0], n_jitter)
            noise = rng.normal(scale=float(sigma), size=(n_jitter, 6)).astype(np.float32)
            jitter = w_ref[idx] + noise

        cover = np.zeros((0, 6), dtype=np.float32)
        if n_cover > 0:
            if self.aabb_min is None or self.aabb_max is None:
                raise ValueError("[SDFDataset] AABB bounds required for coverage sampling.")
            center = 0.5 * (self.aabb_min + self.aabb_max)
            half = 0.5 * (self.aabb_max - self.aabb_min) * float(cover_scale)
            pos = rng.uniform(center - half, center + half, size=(n_cover, 3)).astype(np.float32)
            rotvec = np.zeros((n_cover, 3), dtype=np.float32)
            for i in range(n_cover):
                q = _sample_uniform_quaternion(rng)
                rotvec[i] = _quat_to_rotvec(q)
            w_pos = pos / float(self.l_ref)
            w_rot = rotvec / float(self.lambda_val)
            cover = np.concatenate([w_pos, w_rot], axis=1).astype(np.float32)

        if jitter.shape[0] == 0:
            return cover
        if cover.shape[0] == 0:
            return jitter
        return np.concatenate([jitter, cover], axis=0)


@dataclass(frozen=True)
class MorphDatasetEntry:
    """Dataset entry binding a morphology spec to a dataset root."""
    dataset_root: Path
    spec_path: Path


class MultiMorphSDFDataset:
    """Multi-morph dataset wrapper with fixed-ratio batch sampling."""

    def __init__(
        self,
        entries: List[MorphDatasetEntry],
        *,
        require_boundary: bool = True,
    ) -> None:
        if not entries:
            raise ValueError("[MultiMorphSDFDataset] entries must be non-empty.")
        self.entries = list(entries)
        self.datasets = [
            SDFDataset(e.dataset_root, require_boundary=require_boundary)
            for e in self.entries
        ]

    def sample_batch(
        self,
        batch_size: int,
        ratios: BatchRatios,
        rng: np.random.Generator,
        *,
        eikonal_sigma: Optional[float] = None,
        eikonal_sigma_scale: float = 0.5,
        eikonal_cover_ratio: float = 0.05,
        cover_scale: float = 1.2,
        morph_sampling: str = "uniform",
    ) -> Dict[str, Any]:
        """Sample a mixed batch from one morphology."""
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("[MultiMorphSDFDataset] batch_size must be positive.")

        weights = None
        if morph_sampling == "proportional":
            sizes = np.array([ds.w_b.shape[0] for ds in self.datasets], dtype=np.float64)
            sizes = np.where(sizes > 0, sizes, 1.0)
            weights = sizes / float(np.sum(sizes))

        morph_id = int(rng.choice(len(self.datasets), p=weights))
        ds = self.datasets[morph_id]
        entry = self.entries[morph_id]

        raw_counts = np.array(
            [
                ratios.boundary,
                ratios.rot_far,
                ratios.pos_far,
                ratios.eikonal,
            ],
            dtype=np.float64,
        )
        raw_counts = np.clip(raw_counts, 0.0, None)
        if float(np.sum(raw_counts)) <= 0.0:
            raise ValueError("[MultiMorphSDFDataset] ratios sum to zero.")
        raw_counts = raw_counts / float(np.sum(raw_counts))
        counts = np.round(raw_counts * float(batch_size)).astype(int).tolist()
        diff = batch_size - int(sum(counts))
        if diff != 0:
            counts[0] += diff

        n_boundary, n_rot_far, n_pos_far, n_eik = counts
        boundary = ds.sample_boundary(n_boundary, rng=rng)
        far = ds.sample_far(n_rot_far, n_pos_far, rng=rng)

        w_ref = np.concatenate([boundary["w_b"], boundary["w_in"], boundary["w_out"]], axis=0)
        n_cover = int(round(float(n_eik) * float(eikonal_cover_ratio)))
        n_jitter = max(0, int(n_eik) - n_cover)
        sigma = eikonal_sigma
        if sigma is None:
            sigma = float(ds.delta) * float(eikonal_sigma_scale)
        w_eik = ds.sample_eikonal(
            w_ref=w_ref,
            n_jitter=n_jitter,
            n_cover=n_cover,
            rng=rng,
            sigma=float(sigma),
            cover_scale=float(cover_scale),
        )

        return {
            "morph_id": morph_id,
            "spec_path": str(entry.spec_path),
            "w_b": torch.from_numpy(boundary["w_b"]),
            "w_in": torch.from_numpy(boundary["w_in"]),
            "w_out": torch.from_numpy(boundary["w_out"]),
            "w_rot_far": torch.from_numpy(far["w_rot_far"]),
            "w_pos_far": torch.from_numpy(far["w_pos_far"]),
            "w_eik": torch.from_numpy(w_eik),
            "meta": {
                "l_ref": ds.l_ref,
                "lambda": ds.lambda_val,
                "delta": ds.delta,
            },
        }
