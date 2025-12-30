# mcfp/data/pose_neighbors.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from scipy.spatial import KDTree

from mcfp.data.io import save_pose_deltas


def _get_cfg_value(cfg: Any, key: str) -> Any:
    """Retrieve mandatory value from config object or dict."""
    if isinstance(cfg, dict):
        if key not in cfg:
            raise ValueError(f"[pose_neighbors] Config missing mandatory key '{key}'.")
        return cfg[key]
    if not hasattr(cfg, key):
        raise ValueError(f"[pose_neighbors] Config missing mandatory attr '{key}'.")
    return getattr(cfg, key)


def _get_cfg_val_default(cfg: Any, key: str, default: Any) -> Any:
    """Retrieve optional value from config object or dict."""
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _get_subconfig(cfg: Any, name: str) -> Any:
    """Retrieve sub-config safely."""
    if isinstance(cfg, dict):
        return cfg.get(name, {})
    return getattr(cfg, name, {})


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    qn = np.asarray(q, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(qn))
    if n <= 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return qn / n


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    qn = np.asarray(q, dtype=np.float64).reshape(4)
    return np.array([-qn[0], -qn[1], -qn[2], qn[3]], dtype=np.float64)


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return np.array([x, y, z, w], dtype=np.float64)


def _quat_to_axis_angle(q: np.ndarray) -> Tuple[np.ndarray, float]:
    qn = _quat_normalize(q)
    w = float(np.clip(qn[3], -1.0, 1.0))
    angle = float(2.0 * np.arccos(w))
    s = float(np.sqrt(max(1.0 - w * w, 0.0)))
    if s < 1e-9 or angle <= 1e-9:
        return np.array([0.0, 0.0, 0.0], dtype=np.float64), 0.0
    axis = qn[:3] / s
    return axis, angle


def _delta_rot_axis_angle(q_from: np.ndarray, q_to: np.ndarray) -> Tuple[np.ndarray, float]:
    qf = _quat_normalize(q_from)
    qt = _quat_normalize(q_to)
    q_delta = _quat_multiply(qt, _quat_conjugate(qf))
    axis, angle = _quat_to_axis_angle(q_delta)
    return axis * angle, angle


def build_pose_deltas(
    input_path: Path,
    output_path: Path,
    cfg: Any,
    logger,
) -> None:
    """Build nearest-positive deltas for pose samples."""
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"[pose_neighbors] Input not found: {input_path}")

    with np.load(input_path) as data:
        poses = np.asarray(data["poses"], dtype=np.float32)
        labels = np.asarray(data["labels"], dtype=np.float32)

    if poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError(f"[pose_neighbors] poses must be (N,7), got {poses.shape}.")
    if labels.ndim != 1 or labels.shape[0] != poses.shape[0]:
        raise ValueError("[pose_neighbors] labels shape mismatch.")

    pos = poses[:, :3]
    quat = poses[:, 3:]

    pos_mask = labels > 0.5
    neg_mask = ~pos_mask

    pos_indices = np.where(pos_mask)[0]
    neg_indices = np.where(neg_mask)[0]

    if pos_indices.size == 0:
        raise ValueError("[pose_neighbors] No positive samples found.")

    nearest_cfg = _get_subconfig(cfg, "nearest")
    k = int(_get_cfg_val_default(nearest_cfg, "k", 50))
    w_pos = float(_get_cfg_val_default(nearest_cfg, "w_pos", 1.0))
    w_rot = float(_get_cfg_val_default(nearest_cfg, "w_rot", 1.0))
    metric = str(_get_cfg_val_default(nearest_cfg, "metric", "l2")).lower()
    max_distance = _get_cfg_val_default(nearest_cfg, "max_distance", None)
    truncate_mode = str(_get_cfg_val_default(nearest_cfg, "truncate_mode", "mask")).lower()

    if k <= 0:
        raise ValueError("[pose_neighbors] nearest.k must be positive.")

    if metric not in ("l2", "l1"):
        raise ValueError("[pose_neighbors] nearest.metric must be 'l2' or 'l1'.")

    if max_distance is not None:
        max_distance = float(max_distance)
        if max_distance <= 0.0:
            raise ValueError("[pose_neighbors] nearest.max_distance must be positive.")

    if truncate_mode not in ("mask", "clip"):
        raise ValueError("[pose_neighbors] nearest.truncate_mode must be 'mask' or 'clip'.")

    tree = KDTree(pos[pos_indices])
    k_eff = min(k, pos_indices.size)
    query_pos = pos[neg_indices]
    dists, idxs = tree.query(query_pos, k=k_eff)

    if k_eff == 1:
        dists = dists.reshape(-1, 1)
        idxs = idxs.reshape(-1, 1)

    delta_pos = np.zeros((poses.shape[0], 3), dtype=np.float32)
    delta_rot = np.zeros((poses.shape[0], 3), dtype=np.float32)
    delta_dist = np.zeros((poses.shape[0],), dtype=np.float32)
    delta_mask = np.zeros((poses.shape[0],), dtype=np.float32)

    logger.info(f"[pose_neighbors] Negatives: {len(neg_indices)} | Positives: {len(pos_indices)}")

    for row_idx, neg_idx in enumerate(neg_indices):
        cand_ids = pos_indices[idxs[row_idx]]
        best_dist = None
        best_dp = None
        best_dr = None
        for cand in np.atleast_1d(cand_ids):
            dp = pos[cand] - pos[neg_idx]
            dr, angle = _delta_rot_axis_angle(quat[neg_idx], quat[cand])
            if metric == "l2":
                dist = float(np.sqrt((w_pos * np.linalg.norm(dp)) ** 2 + (w_rot * angle) ** 2))
            else:
                dist = float(w_pos * np.linalg.norm(dp) + w_rot * angle)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_dp = dp
                best_dr = dr

        delta_dist[neg_idx] = float(best_dist)
        if max_distance is None:
            delta_pos[neg_idx] = np.asarray(best_dp, dtype=np.float32)
            delta_rot[neg_idx] = np.asarray(best_dr, dtype=np.float32)
            delta_mask[neg_idx] = 1.0
        elif truncate_mode == "mask":
            if best_dist <= max_distance:
                delta_pos[neg_idx] = np.asarray(best_dp, dtype=np.float32)
                delta_rot[neg_idx] = np.asarray(best_dr, dtype=np.float32)
                delta_mask[neg_idx] = 1.0
        elif truncate_mode == "clip":
            clipped_dp = np.clip(best_dp, -max_distance, max_distance)
            delta_pos[neg_idx] = np.asarray(clipped_dp, dtype=np.float32)
            delta_rot[neg_idx] = np.asarray(best_dr, dtype=np.float32)
            delta_mask[neg_idx] = 1.0

        if (row_idx + 1) % 50000 == 0:
            logger.info(f"[pose_neighbors] Processed {row_idx + 1}/{len(neg_indices)} negatives.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_pose_deltas(
        path=output_path,
        poses=poses,
        labels=labels,
        delta_pos=delta_pos,
        delta_rot=delta_rot,
        delta_dist=delta_dist,
        delta_mask=delta_mask,
    )

    logger.info(f"[pose_neighbors] Output saved: {output_path}")
