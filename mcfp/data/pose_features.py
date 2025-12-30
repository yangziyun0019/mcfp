from __future__ import annotations

from typing import Tuple

import numpy as np

from mcfp.data.io import compute_morph_scale
from mcfp.utils import se3


def _get_cfg_value(cfg, key: str, default: str) -> str:
    """Retrieve a config string value with fallbacks."""
    if isinstance(cfg, dict):
        if key in cfg:
            return str(cfg[key])
        return str(default)
    if hasattr(cfg, key):
        return str(getattr(cfg, key))
    return str(default)


def get_workspace_aabb(spec: dict, fallback_scale: float) -> Tuple[np.ndarray, np.ndarray]:
    """Get workspace AABB from spec or fallback to a cube around origin.

    Parameters
    ----------
    spec:
        Morphology spec dict.
    fallback_scale:
        Scale for fallback cube if workspace is missing.

    Returns
    -------
    aabb_min, aabb_max:
        Arrays of shape (3,).
    """
    ws = spec.get("workspace", {}) if isinstance(spec, dict) else {}
    aabb_min = ws.get("aabb_min", None)
    aabb_max = ws.get("aabb_max", None)
    if aabb_min is not None and aabb_max is not None:
        try:
            aabb_min = np.asarray(aabb_min, dtype=np.float32).reshape(3)
            aabb_max = np.asarray(aabb_max, dtype=np.float32).reshape(3)
            return aabb_min, aabb_max
        except Exception:
            pass

    scale = float(fallback_scale) if np.isfinite(fallback_scale) and fallback_scale > 0 else 1.0
    aabb_min = np.array([-scale, -scale, -scale], dtype=np.float32)
    aabb_max = np.array([scale, scale, scale], dtype=np.float32)
    return aabb_min, aabb_max


def compute_pose_features(
    pose: np.ndarray,
    pose_cfg,
    aabb_min: np.ndarray,
    aabb_max: np.ndarray,
    morph_scale: float,
) -> np.ndarray:
    """Compute pose feature vector from pose and morphology metadata."""
    pose = np.asarray(pose, dtype=np.float32).reshape(7)
    pos = pose[:3]
    quat = pose[3:]

    bmin = np.asarray(aabb_min, dtype=np.float32).reshape(3)
    bmax = np.asarray(aabb_max, dtype=np.float32).reshape(3)
    extent = np.maximum(bmax - bmin, float(pose_cfg.eps)).astype(np.float32)

    aabb_ratio = (pos - bmin) / extent
    aabb_centered = 2.0 * aabb_ratio - 1.0
    morph_scaled = pos / max(float(morph_scale), float(pose_cfg.eps))
    raw_pos = pos

    mapping = {
        "aabb_ratio": aabb_ratio,
        "aabb_centered": aabb_centered,
        "morph_scale": morph_scaled,
        "raw": raw_pos,
    }

    primary_key = str(pose_cfg.primary_pos).lower()
    if primary_key not in mapping:
        raise ValueError(f"[pose_features] Invalid primary_pos='{primary_key}'")
    primary = mapping[primary_key]

    feats = [primary.astype(np.float32)]
    if pose_cfg.include_aabb_ratio and primary_key != "aabb_ratio":
        feats.append(aabb_ratio.astype(np.float32))
    if pose_cfg.include_aabb_centered and primary_key != "aabb_centered":
        feats.append(aabb_centered.astype(np.float32))
    if pose_cfg.include_morph_scale and primary_key != "morph_scale":
        feats.append(morph_scaled.astype(np.float32))
    if pose_cfg.include_raw_pos and primary_key != "raw":
        feats.append(raw_pos.astype(np.float32))

    if pose_cfg.include_quat:
        if pose_cfg.quat_normalize:
            quat = se3.quat_normalize(quat, eps=float(pose_cfg.eps))
        feats.append(np.asarray(quat, dtype=np.float32).reshape(4))

    return np.concatenate(feats, axis=0).astype(np.float32)


def normalize_delta(
    delta_pos: np.ndarray,
    delta_rot: np.ndarray,
    delta_cfg,
    aabb_min: np.ndarray,
    aabb_max: np.ndarray,
    morph_scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize delta vectors using config settings."""
    dp = np.asarray(delta_pos, dtype=np.float32).reshape(3)
    dr = np.asarray(delta_rot, dtype=np.float32).reshape(3)

    pos_mode = _get_cfg_value(delta_cfg, "pos_norm", _get_cfg_value(delta_cfg, "pos", "aabb")).lower()
    if pos_mode == "none":
        dp_norm = dp
    elif pos_mode == "aabb":
        extent = np.maximum(aabb_max - aabb_min, float(delta_cfg.eps)).astype(np.float32)
        dp_norm = dp / extent
    elif pos_mode == "morph_scale":
        dp_norm = dp / max(float(morph_scale), float(delta_cfg.eps))
    else:
        raise ValueError(f"[pose_features] Unknown delta pos_norm='{pos_mode}'")

    rot_mode = _get_cfg_value(delta_cfg, "rot_norm", _get_cfg_value(delta_cfg, "rot", "pi")).lower()
    if rot_mode == "none":
        dr_norm = dr
    elif rot_mode == "pi":
        dr_norm = dr / np.pi
    else:
        raise ValueError(f"[pose_features] Unknown delta rot_norm='{rot_mode}'")

    return dp_norm.astype(np.float32), dr_norm.astype(np.float32)


def denormalize_delta(
    delta_pos_norm: np.ndarray,
    delta_rot_norm: np.ndarray,
    delta_cfg,
    aabb_min: np.ndarray,
    aabb_max: np.ndarray,
    morph_scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Denormalize delta vectors using config settings."""
    dp = np.asarray(delta_pos_norm, dtype=np.float32).reshape(3)
    dr = np.asarray(delta_rot_norm, dtype=np.float32).reshape(3)

    pos_mode = _get_cfg_value(delta_cfg, "pos_norm", _get_cfg_value(delta_cfg, "pos", "aabb")).lower()
    if pos_mode == "none":
        dp_den = dp
    elif pos_mode == "aabb":
        extent = np.maximum(aabb_max - aabb_min, float(delta_cfg.eps)).astype(np.float32)
        dp_den = dp * extent
    elif pos_mode == "morph_scale":
        dp_den = dp * max(float(morph_scale), float(delta_cfg.eps))
    else:
        raise ValueError(f"[pose_features] Unknown delta pos_norm='{pos_mode}'")

    rot_mode = _get_cfg_value(delta_cfg, "rot_norm", _get_cfg_value(delta_cfg, "rot", "pi")).lower()
    if rot_mode == "none":
        dr_den = dr
    elif rot_mode == "pi":
        dr_den = dr * np.pi
    else:
        raise ValueError(f"[pose_features] Unknown delta rot_norm='{rot_mode}'")

    return dp_den.astype(np.float32), dr_den.astype(np.float32)


def compute_morph_meta(spec: dict) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute morphology scale and workspace AABB from spec."""
    scale = compute_morph_scale(spec)
    aabb_min, aabb_max = get_workspace_aabb(spec, fallback_scale=scale)
    return float(scale), aabb_min.astype(np.float32), aabb_max.astype(np.float32)
