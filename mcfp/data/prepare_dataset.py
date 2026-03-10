"""Convert raw generated HDF5 assets into training-ready MCFP datasets.

This module builds processed position and orientation data, index tables, and split metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple
import json
import os
import shutil

import numpy as np

try:
    import h5py
except ImportError as exc:  # pragma: no cover
    raise ImportError("h5py is required for dataset preparation") from exc


_SPLIT_TRAIN = 0
_SPLIT_VAL = 1
_SPLIT_TEST = 2


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _get_path(cfg: Any, key: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in key.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            if part not in cur:
                return default
            cur = cur[part]
        else:
            if not hasattr(cur, part):
                return default
            cur = getattr(cur, part)
    return cur


def _resolve_path(path: str | Path, repo_root: Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        return (repo_root / p).resolve()
    return p.resolve()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _compute_voxel_id(
    points: np.ndarray,
    origin: np.ndarray,
    voxel_size: float,
    dims: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute voxel_id for points. Returns (voxel_id, valid_mask)."""
    pts = np.asarray(points, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64).reshape(1, 3)
    dims = np.asarray(dims, dtype=np.int64).reshape(3)

    idx = np.floor((pts - origin) / float(voxel_size)).astype(np.int64)
    valid = np.all((idx >= 0) & (idx < dims.reshape(1, 3)), axis=1)

    voxel_id = idx[:, 0] * (dims[1] * dims[2]) + idx[:, 1] * dims[2] + idx[:, 2]
    voxel_id = voxel_id.astype(np.int64)
    voxel_id[~valid] = -1
    return voxel_id, valid


def _build_split_array(total: int, ratios: Tuple[float, float, float], seed: int) -> np.ndarray:
    """Build split array of size total with 0=train,1=val,2=test."""
    total = int(total)
    ratios = tuple(float(r) for r in ratios)
    if total <= 0:
        return np.zeros((0,), dtype=np.uint8)
    if not np.isclose(sum(ratios), 1.0, atol=1e-6):
        raise ValueError(f"split ratios must sum to 1.0, got {ratios}")

    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(total)

    n_train = int(round(total * ratios[0]))
    n_val = int(round(total * ratios[1]))
    n_test = total - n_train - n_val
    if n_test < 0:
        n_test = 0

    split = np.full((total,), _SPLIT_TEST, dtype=np.uint8)
    if n_train > 0:
        split[perm[:n_train]] = _SPLIT_TRAIN
    if n_val > 0:
        split[perm[n_train:n_train + n_val]] = _SPLIT_VAL
    if n_test > 0:
        split[perm[n_train + n_val:]] = _SPLIT_TEST
    return split


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    q = q / np.clip(norm, 1e-9, None)
    return q


def _ensure_w_positive(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    sign = np.where(q[:, 3:4] < 0.0, -1.0, 1.0)
    return q * sign


def _sample_quat_pool(
    size: int,
    seed: int,
    method: str = "gaussian",
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    method = str(method).lower()
    if method == "gaussian":
        q = rng.standard_normal(size=(size, 4)).astype(np.float32)
        q = _normalize_quat(q)
        return q
    if method == "shoemake":
        u1 = rng.random(size)
        u2 = rng.random(size)
        u3 = rng.random(size)
        s1 = np.sqrt(1.0 - u1)
        s2 = np.sqrt(u1)
        theta1 = 2.0 * np.pi * u2
        theta2 = 2.0 * np.pi * u3
        qx = s1 * np.sin(theta1)
        qy = s1 * np.cos(theta1)
        qz = s2 * np.sin(theta2)
        qw = s2 * np.cos(theta2)
        q = np.stack([qx, qy, qz, qw], axis=1).astype(np.float32)
        q = _normalize_quat(q)
        return q
    if method == "sobol":
        try:
            from scipy.stats import qmc  # type: ignore
        except Exception:
            # fallback to shoemake if sobol is unavailable
            return _sample_quat_pool(size, seed=seed, method="shoemake")
        engine = qmc.Sobol(d=3, scramble=True, seed=int(seed))
        u = engine.random(size)
        u1, u2, u3 = u[:, 0], u[:, 1], u[:, 2]
        s1 = np.sqrt(1.0 - u1)
        s2 = np.sqrt(u1)
        theta1 = 2.0 * np.pi * u2
        theta2 = 2.0 * np.pi * u3
        qx = s1 * np.sin(theta1)
        qy = s1 * np.cos(theta1)
        qz = s2 * np.sin(theta2)
        qw = s2 * np.cos(theta2)
        q = np.stack([qx, qy, qz, qw], axis=1).astype(np.float32)
        q = _normalize_quat(q)
        return q
    # default fallback
    return _sample_quat_pool(size, seed=seed, method="gaussian")


def _farthest_point_sampling(pool: np.ndarray, k: int, seed: int) -> np.ndarray:
    pool = _normalize_quat(pool)
    m = pool.shape[0]
    if k >= m:
        return pool
    rng = np.random.default_rng(int(seed))
    first = int(rng.integers(0, m))
    selected = np.empty((k, 4), dtype=np.float32)
    selected[0] = pool[first]
    dots = np.abs(pool @ selected[0])
    min_dist = 1.0 - dots
    for i in range(1, k):
        idx = int(np.argmax(min_dist))
        selected[i] = pool[idx]
        dots = np.abs(pool @ selected[i])
        dist = 1.0 - dots
        min_dist = np.minimum(min_dist, dist)
    return selected


def _compute_bucket_r(
    quat: np.ndarray,
    q_ref: np.ndarray,
    chunk: int = 200000,
) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    q_ref = _normalize_quat(q_ref.astype(np.float32))
    n = quat.shape[0]
    out = np.empty((n,), dtype=np.uint16)
    for start in range(0, n, int(chunk)):
        end = min(n, start + int(chunk))
        q = quat[start:end]
        q = _normalize_quat(q)
        dots = np.abs(q @ q_ref.T)
        out[start:end] = 1 + np.argmax(dots, axis=1).astype(np.uint16)
    return out



# Hash-based split for streamable assignment
def _hash_split_ids(ids: np.ndarray, seed: int, ratios: Tuple[float, float, float]) -> np.ndarray:
    if not np.isclose(sum(ratios), 1.0, atol=1e-6):
        raise ValueError(f"split ratios must sum to 1.0, got {ratios}")
    t0 = float(ratios[0])
    t1 = float(ratios[0] + ratios[1])
    ids_u = ids.astype(np.uint64)
    x = (ids_u + np.uint64(seed)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9) & np.uint64(0xFFFFFFFFFFFFFFFF)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94d049bb133111eb) & np.uint64(0xFFFFFFFFFFFFFFFF)
    x = x ^ (x >> np.uint64(31))
    u = (x >> np.uint64(11)).astype(np.float64) * (1.0 / float(1 << 53))
    split = np.full(u.shape, _SPLIT_TEST, dtype=np.uint8)
    split[u < t1] = _SPLIT_VAL
    split[u < t0] = _SPLIT_TRAIN
    return split


# Worker for chunk processing (pass 2 inner ids)
def _chunk_inner_worker(args: Tuple[str, int, int, np.ndarray, float, int, int, int, int, Tuple[float, float, float], bool, bool, Tuple[int, int, int]]):
    raw_pos, x0, x1, dims, voxel_size, bx, by, bz, split_seed, split_ratios, has_tier, has_bucket, bucket_dims = args
    import h5py
    import numpy as np

    nx, ny, nz = int(dims[0]), int(dims[1]), int(dims[2])
    plane = np.arange(ny * nz, dtype=np.int64)

    with h5py.File(raw_pos, "r") as h5:
        sdf_chunk = np.asarray(h5["/grid/sdf"][x0:x1, :, :], dtype=np.float32)
        if has_tier:
            tier_chunk = np.asarray(h5["/grid/farfield_tier"][x0:x1, :, :], dtype=np.uint8)
        else:
            v = float(voxel_size)
            abs_s = np.abs(sdf_chunk)
            tier_chunk = np.zeros_like(sdf_chunk, dtype=np.uint8)
            tier_chunk[(abs_s >= 1 * v) & (abs_s < 3 * v)] = 1
            tier_chunk[(abs_s >= 3 * v) & (abs_s < 5 * v)] = 2
            tier_chunk[(abs_s >= 5 * v) & (abs_s < 7 * v)] = 3
            tier_chunk[(abs_s >= 7 * v) & (abs_s < 9 * v)] = 4
            tier_chunk[(abs_s >= 9 * v)] = 5

        if has_bucket:
            bucket_chunk = np.asarray(h5["/grid/farfield_bucket"][x0:x1, :, :], dtype=np.uint8)
        else:
            bx_all, by_all, bz_all = bucket_dims
            bx_idx = np.minimum((np.arange(nx) * bx_all) // nx, bx_all - 1).astype(np.uint16)
            by_idx = np.minimum((np.arange(ny) * by_all) // ny, by_all - 1).astype(np.uint16)
            bz_idx = np.minimum((np.arange(nz) * bz_all) // nz, bz_all - 1).astype(np.uint16)
            bx_chunk = np.broadcast_to(bx_idx[x0:x1][:, None, None], (x1 - x0, ny, nz))
            by_chunk = np.broadcast_to(by_idx[None, :, None], (x1 - x0, ny, nz))
            bz_chunk = np.broadcast_to(bz_idx[None, None, :], (x1 - x0, ny, nz))
            bucket_chunk = np.stack([bx_chunk, by_chunk, bz_chunk], axis=-1).astype(np.uint8)

    base_x = (np.arange(x0, x1, dtype=np.int64) * (ny * nz))
    voxel_ids = (base_x[:, None] + plane[None, :]).reshape(-1)
    split_chunk = _hash_split_ids(voxel_ids, split_seed, split_ratios)

    tier_flat = tier_chunk.reshape(-1)
    sdf_flat = sdf_chunk.reshape(-1)
    bucket_flat = bucket_chunk.reshape(-1, 3)
    valid_bucket = np.all(bucket_flat != 255, axis=1)
    valid_inner = (tier_flat >= 1) & (tier_flat <= 5) & valid_bucket
    if not np.any(valid_inner):
        return x0, np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.uint8)

    voxel_ids_valid = voxel_ids[valid_inner].astype(np.int64)
    bucket_ids = bucket_flat[valid_inner]
    tiers = tier_flat[valid_inner].astype(np.int64)
    signs = (sdf_flat[valid_inner] < 0).astype(np.int64)
    bucket_id = (((tiers - 1) * bx + bucket_ids[:, 0].astype(np.int64)) * by + bucket_ids[:, 1].astype(np.int64)) * bz + bucket_ids[:, 2].astype(np.int64)
    bucket_id = bucket_id * 2 + signs
    split_valid = split_chunk[valid_inner]
    return x0, bucket_id.astype(np.int64), voxel_ids_valid, split_valid


def _allocate_counts(total: int, ratios: Iterable[float]) -> np.ndarray:
    ratios_arr = np.asarray(list(ratios), dtype=np.float64)
    if ratios_arr.size == 0:
        raise ValueError("ratios must not be empty")
    if np.any(ratios_arr < 0):
        raise ValueError("ratios must be non-negative")
    if float(ratios_arr.sum()) <= 0:
        ratios_arr = np.ones_like(ratios_arr, dtype=np.float64)
    ratios_arr = ratios_arr / float(ratios_arr.sum())
    counts = np.floor(ratios_arr * float(total)).astype(np.int64)
    remainder = int(total - int(counts.sum()))
    if remainder > 0:
        for i in range(remainder):
            counts[i % counts.size] += 1
    return counts


def _sample_uniform_from_grid(
    sdf_grid: np.ndarray,
    origin: np.ndarray,
    voxel_size: float,
    dims: np.ndarray,
    *,
    total_points: int,
    bin_edges_voxel: Iterable[float],
    bin_ratios: Iterable[float],
    sign_ratios: Iterable[float],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Sample uniform points from grid SDF by sign and |s| bins."""
    total_points = int(total_points)
    if total_points <= 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            {"total": 0},
        )

    sdf = np.asarray(sdf_grid, dtype=np.float32)
    sdf_flat = sdf.reshape(-1)
    abs_s = np.abs(sdf_flat)

    sign_ratios_arr = np.asarray(list(sign_ratios), dtype=np.float64)
    if sign_ratios_arr.size != 2:
        raise ValueError("sign_ratios must have 2 values: [pos, neg]")
    sign_counts = _allocate_counts(total_points, sign_ratios_arr)
    n_pos = int(sign_counts[0])
    n_neg = int(sign_counts[1])

    bin_edges = np.asarray(list(bin_edges_voxel), dtype=np.float64)
    if bin_edges.size < 2:
        raise ValueError("bin_edges_voxel must have at least 2 values")
    bin_edges_m = bin_edges * float(voxel_size)

    bin_ratios_arr = np.asarray(list(bin_ratios), dtype=np.float64)
    if bin_ratios_arr.size != bin_edges.size:
        raise ValueError("bin_ratios length must equal bin_edges length")

    rng = np.random.default_rng(int(seed))
    ny = int(dims[1])
    nz = int(dims[2])

    def sample_for_sign(sign_mask: np.ndarray, target: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        counts = _allocate_counts(target, bin_ratios_arr)
        picked: list[np.ndarray] = []
        stats: Dict[str, Any] = {"target": int(target), "picked": 0, "bins": []}
        for i, count in enumerate(counts):
            lo = bin_edges_m[i]
            hi = bin_edges_m[i + 1] if i + 1 < bin_edges_m.size else np.inf
            mask = sign_mask & (abs_s >= lo) & (abs_s < hi)
            idx = np.flatnonzero(mask)
            if idx.size == 0:
                stats["bins"].append({"range": [float(lo), float(hi)], "picked": 0, "avail": 0})
                continue
            if count <= 0 or idx.size <= count:
                choice = idx
            else:
                choice = rng.choice(idx, size=int(count), replace=False)
            picked.append(choice)
            stats["bins"].append({"range": [float(lo), float(hi)], "picked": int(choice.size), "avail": int(idx.size)})
        if picked:
            all_idx = np.concatenate(picked, axis=0)
        else:
            all_idx = np.zeros((0,), dtype=np.int64)
        stats["picked"] = int(all_idx.size)
        return all_idx, stats

    pos_mask = sdf_flat > 0
    neg_mask = sdf_flat < 0

    pos_idx, pos_stats = sample_for_sign(pos_mask, n_pos)
    neg_idx, neg_stats = sample_for_sign(neg_mask, n_neg)

    if pos_idx.size or neg_idx.size:
        all_idx = np.concatenate([pos_idx, neg_idx], axis=0)
    else:
        all_idx = np.zeros((0,), dtype=np.int64)

    if all_idx.size == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            {"total": 0, "pos": pos_stats, "neg": neg_stats},
        )

    i = all_idx // (ny * nz)
    rem = all_idx % (ny * nz)
    j = rem // nz
    k = rem % nz
    coords = np.stack([i, j, k], axis=1).astype(np.float32)
    points = origin.reshape(1, 3).astype(np.float32) + (coords + 0.5) * float(voxel_size)

    s_vals = sdf_flat[all_idx].astype(np.float32)
    voxel_ids = all_idx.astype(np.int64)

    stats = {
        "total_target": int(total_points),
        "total_picked": int(all_idx.size),
        "pos": pos_stats,
        "neg": neg_stats,
    }
    return points, s_vals, voxel_ids, stats


def prepare_position_dataset(
    cfg: Any,
    logger,
    *,
    repo_root: Path,
    l_ref: Optional[float] = None,
) -> Path:
    """Prepare Position-SDF dataset (dataset_pos.h5) for Stage A training."""
    raw_pos = _resolve_path(_get_path(cfg, "paths.raw_position_h5"), repo_root)
    out_pos = _resolve_path(_get_path(cfg, "paths.position_out"), repo_root)
    meta_dir = _resolve_path(_get_path(cfg, "paths.meta_dir"), repo_root)
    _ensure_dir(out_pos.parent)
    _ensure_dir(meta_dir)

    split_seed = int(_get_path(cfg, "splits.voxel_seed", 42))
    split_ratios = tuple(_get_path(cfg, "splits.voxel", [0.9, 0.05, 0.05]))

    bucket_cfg = _get_path(cfg, "buckets", None)
    num_s = int(_get(bucket_cfg, "num_s", 10))
    spatial_dims = _get(bucket_cfg, "spatial_dims", [8, 8, 8])
    eps = float(_get(bucket_cfg, "eps", 1e-6))
    bx = max(int(spatial_dims[0]), 1)
    by = max(int(spatial_dims[1]), 1)
    bz = max(int(spatial_dims[2]), 1)

    chunk_x = int(_get_path(cfg, "prepare.chunk_x", 4))
    chunk_x = max(1, chunk_x)

    if not np.isclose(sum(split_ratios), 1.0, atol=1e-6):
        raise ValueError(f"split ratios must sum to 1.0, got {split_ratios}")
    if num_s <= 0:
        raise ValueError(f"buckets.num_s must be positive, got {num_s}")

    logger.info(f"[prepare_position] raw={raw_pos}")
    logger.info(f"[prepare_position] out={out_pos}")

    with h5py.File(raw_pos, "r") as h5:
        if "/grid/label" not in h5 or "/grid/sdf" not in h5:
            raise ValueError("raw_position_h5 missing /grid/label or /grid/sdf")
        origin = np.asarray(h5["/grid/origin"], dtype=np.float64).reshape(3)
        dims = np.asarray(h5["/grid/dims"], dtype=np.int64).reshape(3)
        voxel_size = float(np.asarray(h5["/grid/voxel_size"], dtype=np.float64).reshape(-1)[0])
        label_ds = h5["/grid/label"]
        sdf_ds = h5["/grid/sdf"]
        nx, ny, nz = int(dims[0]), int(dims[1]), int(dims[2])

        # Pass 1: estimate a_max for non-boundary voxels
        a_max = 0.0
        for x0 in range(0, nx, chunk_x):
            x1 = min(nx, x0 + chunk_x)
            sl = np.s_[x0:x1, :, :]
            label_chunk = np.asarray(label_ds[sl], dtype=np.uint8)
            mask = (label_chunk == 1) | (label_chunk == 2)
            if not np.any(mask):
                continue
            sdf_chunk = np.asarray(sdf_ds[sl], dtype=np.float32)
            a_max = max(a_max, float(np.max(np.abs(sdf_chunk[mask]))))

    if a_max > 0.0 and np.isfinite(a_max):
        denom = float(np.log(a_max / float(voxel_size) + eps))
        if not np.isfinite(denom) or denom <= 0.0:
            denom = None
    else:
        denom = None

    grid_size = int(dims[0] * dims[1] * dims[2])
    splits_path = meta_dir / "splits_voxel.npy"
    splits_mm = np.lib.format.open_memmap(str(splits_path), mode="w+", dtype=np.uint8, shape=(grid_size,))

    # bucket index grids (use voxel centers)
    bx_idx = np.floor(((np.arange(nx) + 0.5) * bx) / max(nx, 1)).astype(np.int64)
    by_idx = np.floor(((np.arange(ny) + 0.5) * by) / max(ny, 1)).astype(np.int64)
    bz_idx = np.floor(((np.arange(nz) + 0.5) * bz) / max(nz, 1)).astype(np.int64)
    bx_idx = np.clip(bx_idx, 0, bx - 1).astype(np.uint16)
    by_idx = np.clip(by_idx, 0, by - 1).astype(np.uint16)
    bz_idx = np.clip(bz_idx, 0, bz - 1).astype(np.uint16)

    bucket_spatial = int(bx * by * bz)
    bucket_count_nb = int(num_s * bucket_spatial * 2)
    nb_counts = np.zeros((3, bucket_count_nb), dtype=np.uint64)
    bd_counts = np.zeros((3, bucket_spatial), dtype=np.uint64)

    plane = np.arange(ny * nz, dtype=np.int64)

    def _compute_b_s(abs_s: np.ndarray) -> np.ndarray:
        if denom is None:
            b = np.ones_like(abs_s, dtype=np.uint8)
        else:
            u = np.log(abs_s / float(voxel_size) + eps) / float(denom)
            u = np.clip(u, 0.0, 1.0)
            b = 1 + np.floor(num_s * u).astype(np.int64)
            b = np.clip(b, 1, num_s).astype(np.uint8)
        return b

    def _scatter_append(bucket_ids: np.ndarray, voxel_ids: np.ndarray, start: np.ndarray, cursor: np.ndarray, out_ids: np.ndarray) -> None:
        if bucket_ids.size == 0:
            return
        order = np.argsort(bucket_ids)
        b_sorted = bucket_ids[order]
        v_sorted = voxel_ids[order]
        unique, counts = np.unique(b_sorted, return_counts=True)
        offset = 0
        for u, c in zip(unique, counts):
            pos = int(cursor[u])
            c = int(c)
            end = pos + c
            out_ids[pos:end] = v_sorted[offset:offset + c]
            cursor[u] = end
            offset += c

    with h5py.File(raw_pos, "r") as h5_in, h5py.File(out_pos, "w") as out:
        label_ds = h5_in["/grid/label"]
        sdf_ds = h5_in["/grid/sdf"]

        grid_grp = out.create_group("grid")
        grid_grp.create_dataset("origin", data=origin.astype(np.float64))
        grid_grp.create_dataset("dims", data=dims.astype(np.uint64))
        grid_grp.create_dataset("voxel_size", data=np.asarray([voxel_size], dtype=np.float64))
        grid_grp.create_dataset("bucket_s_num", data=np.asarray([num_s], dtype=np.uint16))
        grid_grp.create_dataset("bucket_spatial_dims", data=np.asarray([bx, by, bz], dtype=np.uint16))

        out_label = grid_grp.create_dataset("label", shape=label_ds.shape, dtype=np.uint8, compression="gzip", shuffle=True)
        out_sdf = grid_grp.create_dataset("sdf", shape=sdf_ds.shape, dtype=np.float32, compression="gzip", shuffle=True)
        out_b_s = grid_grp.create_dataset("b_s", shape=sdf_ds.shape, dtype=np.uint8, compression="gzip", shuffle=True)
        out_bx = grid_grp.create_dataset("b_x", shape=sdf_ds.shape, dtype=np.uint16, compression="gzip", shuffle=True)
        out_by = grid_grp.create_dataset("b_y", shape=sdf_ds.shape, dtype=np.uint16, compression="gzip", shuffle=True)
        out_bz = grid_grp.create_dataset("b_z", shape=sdf_ds.shape, dtype=np.uint16, compression="gzip", shuffle=True)

        # Pass 2: write grid data + count index buckets
        for x0 in range(0, nx, chunk_x):
            x1 = min(nx, x0 + chunk_x)
            sl = np.s_[x0:x1, :, :]
            label_chunk = np.asarray(label_ds[sl], dtype=np.uint8)
            sdf_chunk = np.asarray(sdf_ds[sl], dtype=np.float32)
            abs_s = np.abs(sdf_chunk)
            b_s_chunk = _compute_b_s(abs_s)
            b_s_chunk[label_chunk == 0] = 0

            bx_chunk = np.broadcast_to(bx_idx[x0:x1][:, None, None], (x1 - x0, ny, nz))
            by_chunk = np.broadcast_to(by_idx[None, :, None], (x1 - x0, ny, nz))
            bz_chunk = np.broadcast_to(bz_idx[None, None, :], (x1 - x0, ny, nz))

            out_label[sl] = label_chunk
            out_sdf[sl] = sdf_chunk
            out_b_s[sl] = b_s_chunk
            out_bx[sl] = bx_chunk
            out_by[sl] = by_chunk
            out_bz[sl] = bz_chunk

            base_x = (np.arange(x0, x1, dtype=np.int64) * (ny * nz))
            voxel_ids = (base_x[:, None] + plane[None, :]).reshape(-1)
            split_chunk = _hash_split_ids(voxel_ids, split_seed, split_ratios)
            splits_mm[voxel_ids] = split_chunk

            label_flat = label_chunk.reshape(-1)
            b_s_flat = b_s_chunk.reshape(-1).astype(np.int64)
            bx_flat = bx_chunk.reshape(-1).astype(np.int64)
            by_flat = by_chunk.reshape(-1).astype(np.int64)
            bz_flat = bz_chunk.reshape(-1).astype(np.int64)

            nb_mask = (label_flat == 1) | (label_flat == 2)
            if np.any(nb_mask):
                b_s_nb = b_s_flat[nb_mask]
                sigma = (label_flat[nb_mask] == 2).astype(np.int64)
                bucket_id = (
                    (((b_s_nb - 1) * bx + bx_flat[nb_mask]) * by + by_flat[nb_mask]) * bz
                    + bz_flat[nb_mask]
                )
                bucket_id = bucket_id * 2 + sigma
                split_valid = split_chunk[nb_mask]
                for split_id in (_SPLIT_TRAIN, _SPLIT_VAL, _SPLIT_TEST):
                    mask = split_valid == split_id
                    if np.any(mask):
                        np.add.at(nb_counts[split_id], bucket_id[mask], 1)

            bd_mask = label_flat == 0
            if np.any(bd_mask):
                bucket_id = (bx_flat[bd_mask] * by + by_flat[bd_mask]) * bz + bz_flat[bd_mask]
                split_valid = split_chunk[bd_mask]
                for split_id in (_SPLIT_TRAIN, _SPLIT_VAL, _SPLIT_TEST):
                    mask = split_valid == split_id
                    if np.any(mask):
                        np.add.at(bd_counts[split_id], bucket_id[mask], 1)

        splits_mm.flush()

        # build CSR for non-boundary and boundary buckets
        nb_start = {}
        nb_ids = {}
        nb_cursor = {}
        bd_start = {}
        bd_ids = {}
        bd_cursor = {}
        for split_id in (_SPLIT_TRAIN, _SPLIT_VAL, _SPLIT_TEST):
            start = np.zeros((bucket_count_nb + 1,), dtype=np.uint64)
            start[1:] = np.cumsum(nb_counts[split_id])
            nb_start[split_id] = start
            nb_ids[split_id] = np.empty((int(start[-1]),), dtype=np.uint64)
            nb_cursor[split_id] = start.copy()

            start_bd = np.zeros((bucket_spatial + 1,), dtype=np.uint64)
            start_bd[1:] = np.cumsum(bd_counts[split_id])
            bd_start[split_id] = start_bd
            bd_ids[split_id] = np.empty((int(start_bd[-1]),), dtype=np.uint64)
            bd_cursor[split_id] = start_bd.copy()

        # Pass 3: fill nb_ids and bd_ids
        for x0 in range(0, nx, chunk_x):
            x1 = min(nx, x0 + chunk_x)
            sl = np.s_[x0:x1, :, :]
            label_chunk = np.asarray(label_ds[sl], dtype=np.uint8)
            sdf_chunk = np.asarray(sdf_ds[sl], dtype=np.float32)
            abs_s = np.abs(sdf_chunk)
            b_s_chunk = _compute_b_s(abs_s)
            b_s_chunk[label_chunk == 0] = 0

            bx_chunk = np.broadcast_to(bx_idx[x0:x1][:, None, None], (x1 - x0, ny, nz))
            by_chunk = np.broadcast_to(by_idx[None, :, None], (x1 - x0, ny, nz))
            bz_chunk = np.broadcast_to(bz_idx[None, None, :], (x1 - x0, ny, nz))

            base_x = (np.arange(x0, x1, dtype=np.int64) * (ny * nz))
            voxel_ids = (base_x[:, None] + plane[None, :]).reshape(-1)
            split_chunk = _hash_split_ids(voxel_ids, split_seed, split_ratios)

            label_flat = label_chunk.reshape(-1)
            b_s_flat = b_s_chunk.reshape(-1).astype(np.int64)
            bx_flat = bx_chunk.reshape(-1).astype(np.int64)
            by_flat = by_chunk.reshape(-1).astype(np.int64)
            bz_flat = bz_chunk.reshape(-1).astype(np.int64)

            nb_mask = (label_flat == 1) | (label_flat == 2)
            if np.any(nb_mask):
                b_s_nb = b_s_flat[nb_mask]
                sigma = (label_flat[nb_mask] == 2).astype(np.int64)
                bucket_id = (
                    (((b_s_nb - 1) * bx + bx_flat[nb_mask]) * by + by_flat[nb_mask]) * bz
                    + bz_flat[nb_mask]
                )
                bucket_id = bucket_id * 2 + sigma
                split_valid = split_chunk[nb_mask]
                for split_id in (_SPLIT_TRAIN, _SPLIT_VAL, _SPLIT_TEST):
                    mask = split_valid == split_id
                    if np.any(mask):
                        _scatter_append(
                            bucket_id[mask],
                            voxel_ids[nb_mask][mask].astype(np.uint64),
                            nb_start[split_id],
                            nb_cursor[split_id],
                            nb_ids[split_id],
                        )

            bd_mask = label_flat == 0
            if np.any(bd_mask):
                bucket_id = (bx_flat[bd_mask] * by + by_flat[bd_mask]) * bz + bz_flat[bd_mask]
                split_valid = split_chunk[bd_mask]
                for split_id in (_SPLIT_TRAIN, _SPLIT_VAL, _SPLIT_TEST):
                    mask = split_valid == split_id
                    if np.any(mask):
                        _scatter_append(
                            bucket_id[mask],
                            voxel_ids[bd_mask][mask].astype(np.uint64),
                            bd_start[split_id],
                            bd_cursor[split_id],
                            bd_ids[split_id],
                        )

        grp_index = out.create_group("index")
        for split_id in (_SPLIT_TRAIN, _SPLIT_VAL, _SPLIT_TEST):
            grp_split = grp_index.create_group(f"split{split_id}")
            grp_split.create_dataset("nb_start", data=nb_start[split_id], compression="gzip", shuffle=True)
            grp_split.create_dataset("nb_ids", data=nb_ids[split_id], compression="gzip", shuffle=True)
            grp_split.create_dataset("bd_start", data=bd_start[split_id], compression="gzip", shuffle=True)
            grp_split.create_dataset("bd_ids", data=bd_ids[split_id], compression="gzip", shuffle=True)

        meta = {
            "raw_position_h5": str(raw_pos),
            "position_out": str(out_pos),
            "grid_size": int(grid_size),
            "voxel_split_seed": int(split_seed),
            "voxel_split_ratios": list(split_ratios),
            "buckets": {
                "num_s": int(num_s),
                "spatial_dims": [int(bx), int(by), int(bz)],
                "eps": float(eps),
                "a_max": float(a_max),
            },
            "prepare": {"chunk_x": int(chunk_x)},
        }
        if l_ref is not None:
            meta["l_ref"] = float(l_ref)
        dt_meta = h5py.string_dtype("utf-8")
        out.create_dataset("meta/json", data=np.asarray(json.dumps(meta, ensure_ascii=True), dtype=dt_meta))

    logger.info(
        f"[prepare_position] a_max={a_max:.6f} "
        f"bucket_s={num_s} bucket_spatial=({bx},{by},{bz}) chunk_x={chunk_x}"
    )
    logger.info(f"[prepare_position] splits saved to {splits_path}")
    return out_pos
def prepare_orientation_dataset(
    cfg: Any,
    logger,
    *,
    repo_root: Path,
    l_ref: Optional[float] = None,
) -> Path:
    """Prepare Orientation-SDF dataset for training."""
    raw_orient = _resolve_path(_get_path(cfg, "paths.raw_orientation_h5"), repo_root)
    out_orient = _resolve_path(_get_path(cfg, "paths.orientation_out"), repo_root)
    meta_dir = _resolve_path(_get_path(cfg, "paths.meta_dir"), repo_root)
    _ensure_dir(out_orient.parent)
    _ensure_dir(meta_dir)

    split_seed = int(_get_path(cfg, "splits.anchor_seed", 42))
    split_ratios = tuple(_get_path(cfg, "splits.anchor", [0.9, 0.05, 0.05]))

    logger.info(f"[prepare_orient] raw={raw_orient}")
    logger.info(f"[prepare_orient] out={out_orient}")

    with h5py.File(raw_orient, "r") as h5:
        anchors = {
            "voxel_id": np.asarray(h5["/anchors/voxel_id"], dtype=np.uint64),
            "pos": np.asarray(h5["/anchors/pos"], dtype=np.float32),
            "s_v": np.asarray(h5["/anchors/s_v"], dtype=np.float32),
            "c_v": np.asarray(h5["/anchors/c_v"], dtype=np.float32) if "/anchors/c_v" in h5 else None,
            "g_v": np.asarray(h5["/anchors/g_v"], dtype=np.float32) if "/anchors/g_v" in h5 else None,
            "n_seed": np.asarray(h5["/anchors/n_seed"], dtype=np.uint32) if "/anchors/n_seed" in h5 else None,
        }
        ds_quat = h5["/samples/quat"]
        ds_phi = h5["/samples/phi"]
        ds_label = h5["/samples/label"] if "/samples/label" in h5 else None
        ds_method = h5["/samples/method"] if "/samples/method" in h5 else None

        q_ref = None
        if "/meta/q_ref" in h5:
            q_ref = np.asarray(h5["/meta/q_ref"], dtype=np.float32)
        elif "/q_ref" in h5:
            q_ref = np.asarray(h5["/q_ref"], dtype=np.float32)
        elif "/meta/ref_quat" in h5:
            q_ref = np.asarray(h5["/meta/ref_quat"], dtype=np.float32)

        q_ref_cfg = _get_path(cfg, "prepare.orientation.q_ref", None)
        k_r = int(_get(q_ref_cfg, "k_r", 64))
        pool_size = int(_get(q_ref_cfg, "pool_size", 4096))
        q_seed = int(_get(q_ref_cfg, "seed", 20260126))
        q_method = str(_get(q_ref_cfg, "method", "gaussian"))
        use_existing = bool(_get(q_ref_cfg, "use_existing", True))
        q_ref_source = "raw"
        if use_existing:
            if q_ref is None:
                raise RuntimeError(
                    "q_ref not found in raw orientation dataset. "
                    "Set prepare.orientation.q_ref.use_existing=false to generate a new q_ref."
                )
            q_ref = _ensure_w_positive(_normalize_quat(q_ref))
            k_r = int(q_ref.shape[0])
        else:
            pool = _sample_quat_pool(pool_size, seed=q_seed, method=q_method)
            q_ref = _farthest_point_sampling(pool, k_r, seed=q_seed + 17)
            q_ref = _ensure_w_positive(q_ref)
            q_ref_source = "generated"

        bucket_chunk = int(_get_path(cfg, "prepare.orientation.bucket_chunk", 200000))
        log_every = int(_get_path(cfg, "prepare.orientation.log_every", 50))

        num_anchors = int(anchors["pos"].shape[0])
        num_samples = int(ds_quat.shape[0])

        anchor_split = _build_split_array(num_anchors, split_ratios, split_seed)
        np.save(meta_dir / "splits_anchor.npy", anchor_split)

        s_v_norm = None
        if l_ref is not None and l_ref > 0:
            s_v_norm = anchors["s_v"] / float(l_ref)

        with h5py.File(out_orient, "w") as out:
            grp_a = out.create_group("anchors")
            grp_a.create_dataset("voxel_id", data=anchors["voxel_id"], compression="gzip", shuffle=True)
            grp_a.create_dataset("pos", data=anchors["pos"], compression="gzip", shuffle=True)
            grp_a.create_dataset("s_v", data=anchors["s_v"], compression="gzip", shuffle=True)
            if s_v_norm is not None:
                grp_a.create_dataset("s_v_norm", data=s_v_norm, compression="gzip", shuffle=True)
            if anchors["c_v"] is not None:
                grp_a.create_dataset("c_v", data=anchors["c_v"], compression="gzip", shuffle=True)
            if anchors["g_v"] is not None:
                grp_a.create_dataset("g_v", data=anchors["g_v"], compression="gzip", shuffle=True)
            if anchors["n_seed"] is not None:
                grp_a.create_dataset("n_seed", data=anchors["n_seed"], compression="gzip", shuffle=True)
            grp_a.create_dataset("split", data=anchor_split.astype(np.uint8), compression="gzip", shuffle=True)

            grp_s = out.create_group("samples")
            chunk_n = int(min(bucket_chunk, max(num_samples, 1)))
            out_quat = grp_s.create_dataset(
                "quat",
                shape=ds_quat.shape,
                dtype=np.float32,
                chunks=(chunk_n, ds_quat.shape[1]),
                compression="gzip",
                shuffle=True,
            )
            out_phi = grp_s.create_dataset(
                "phi",
                shape=ds_phi.shape,
                dtype=np.float32,
                chunks=(chunk_n,),
                compression="gzip",
                shuffle=True,
            )
            out_label = None
            if ds_label is not None:
                out_label = grp_s.create_dataset(
                    "label",
                    shape=ds_label.shape,
                    dtype=np.int8,
                    chunks=(chunk_n,),
                    compression="gzip",
                    shuffle=True,
                )
            out_method = None
            if ds_method is not None:
                out_method = grp_s.create_dataset(
                    "method",
                    shape=ds_method.shape,
                    dtype=np.uint8,
                    chunks=(chunk_n,),
                    compression="gzip",
                    shuffle=True,
                )
            out_bucket = grp_s.create_dataset(
                "bucket_r",
                shape=(num_samples,),
                dtype=np.uint16,
                chunks=(chunk_n,),
                compression="gzip",
                shuffle=True,
            )

            # Copy sample arrays + compute bucket_r in chunks to avoid OOM.
            for idx, start in enumerate(range(0, num_samples, bucket_chunk)):
                end = min(num_samples, start + bucket_chunk)
                quat_chunk = np.asarray(ds_quat[start:end], dtype=np.float32)
                out_quat[start:end] = quat_chunk
                out_phi[start:end] = np.asarray(ds_phi[start:end], dtype=np.float32)
                if out_method is not None:
                    out_method[start:end] = np.asarray(ds_method[start:end], dtype=np.uint8)
                if out_label is not None:
                    label_chunk = np.asarray(ds_label[start:end], dtype=np.int8)
                    if np.any(label_chunk < 0):
                        label_chunk = np.where(label_chunk >= 0, 1, -1).astype(np.int8)
                    else:
                        label_chunk = np.where(label_chunk == 0, -1, 1).astype(np.int8)
                    out_label[start:end] = label_chunk
                qn = _normalize_quat(quat_chunk)
                dots = np.abs(qn @ q_ref.T)
                out_bucket[start:end] = 1 + np.argmax(dots, axis=1).astype(np.uint16)

                if log_every > 0 and (idx % log_every == 0 or end == num_samples):
                    logger.info(f"[prepare_orient] samples {end}/{num_samples}")

            grp_csr = out.create_group("csr")
            out.copy(h5["/csr/anchor_start"], grp_csr, name="anchor_start")
            out.copy(h5["/csr/sample_index"], grp_csr, name="sample_index")

            meta = {
                "raw_orientation_h5": str(raw_orient),
                "orientation_out": str(out_orient),
                "anchor_count": int(num_anchors),
                "sample_count": int(num_samples),
                "anchor_split_seed": int(split_seed),
                "anchor_split_ratios": list(split_ratios),
            }
            if l_ref is not None:
                meta["l_ref"] = float(l_ref)
            meta["q_ref_k"] = int(k_r)
            meta["q_ref_pool"] = int(pool_size)
            meta["q_ref_seed"] = int(q_seed)
            meta["q_ref_method"] = str(q_method)
            meta["q_ref_source"] = q_ref_source
            meta["bucket_r_field"] = "samples/bucket_r"
            meta["bucket_r_index_base"] = 1
            meta["bucket_r_def"] = "argmax_abs_dot(q, q_ref)"
            dt_meta = h5py.string_dtype("utf-8")
            out.create_dataset("meta/json", data=np.asarray(json.dumps(meta, ensure_ascii=True), dtype=dt_meta))
            if q_ref is not None:
                out.create_dataset("meta/q_ref", data=q_ref, compression="gzip", shuffle=True)

    logger.info(f"[prepare_orient] anchors={num_anchors} samples={num_samples}")
    logger.info(f"[prepare_orient] splits saved to {meta_dir / 'splits_anchor.npy'}")
    return out_orient


def prepare_dataset(cfg: Any, logger, *, repo_root: Path, l_ref: Optional[float] = None) -> Dict[str, Path]:
    """Run position + orientation dataset preparation and record meta files."""
    meta_dir = _resolve_path(_get_path(cfg, "paths.meta_dir"), repo_root)
    _ensure_dir(meta_dir)

    outputs: Dict[str, Path] = {}
    do_pos = bool(_get_path(cfg, "run.prepare_position", True))
    do_orient = bool(_get_path(cfg, "run.prepare_orientation", True))

    if do_pos:
        outputs["position"] = prepare_position_dataset(cfg, logger, repo_root=repo_root, l_ref=l_ref)
    else:
        logger.info("[prepare_dataset] skip position (run.prepare_position=false)")

    if do_orient:
        outputs["orientation"] = prepare_orientation_dataset(cfg, logger, repo_root=repo_root, l_ref=l_ref)
    else:
        logger.info("[prepare_dataset] skip orientation (run.prepare_orientation=false)")

    cfg_path = _resolve_path(_get_path(cfg, "paths.config_path"), repo_root)
    if cfg_path.is_file():
        dst = meta_dir / "data_pipeline.yaml"
        shutil.copyfile(cfg_path, dst)
        logger.info(f"[prepare_dataset] config snapshot: {dst}")

    return outputs
