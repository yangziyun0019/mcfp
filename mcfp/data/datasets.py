from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from mcfp.data.io import load_morph_spec, load_pose_samples
from mcfp.data.pose_features import compute_morph_meta, compute_pose_features, normalize_delta


@dataclass(frozen=True)
class PoseFeatureConfig:
    """Configuration for pose feature construction.

    Attributes
    ----------
    primary_pos:
        Name of the primary position feature used for Fourier encoding.
        One of: "aabb_ratio", "aabb_centered", "morph_scale", "raw".
    include_aabb_ratio:
        Append normalized ratio (pos - min) / (max - min).
    include_aabb_centered:
        Append centered ratio in [-1, 1].
    include_morph_scale:
        Append position normalized by morphology scale L_r.
    include_raw_pos:
        Append raw XYZ position.
    include_quat:
        Append quaternion [x, y, z, w] as-is.
    quat_normalize:
        If True, normalize quaternion to unit length before appending.
    eps:
        Small epsilon for numerical stability.
    """

    primary_pos: str = "aabb_centered"
    include_aabb_ratio: bool = True
    include_aabb_centered: bool = True
    include_morph_scale: bool = True
    include_raw_pos: bool = False
    include_quat: bool = True
    quat_normalize: bool = False
    eps: float = 1e-8


@dataclass(frozen=True)
class DeltaFeatureConfig:
    """Configuration for delta normalization.

    Attributes
    ----------
    pos_norm:
        Position delta normalization mode: "none", "aabb", or "morph_scale".
    rot_norm:
        Rotation delta normalization mode: "none" or "pi".
    eps:
        Small epsilon for numerical stability.
    """

    pos_norm: str = "aabb"
    rot_norm: str = "pi"
    eps: float = 1e-8




class PoseDeltaDataset(Dataset):
    """Dataset for (morphology, pose) -> reachability + delta targets.

    Each sample returns:
      - pose_feats: Tensor[F]
      - labels: Tensor[K] aligned with label_keys
      - delta_mask: float Tensor scalar (1 for valid delta targets)
      - ws_mask: float Tensor scalar (1 if g_ws>0.5)
      - morph_spec: dict
      - variant_id/family/dof: metadata
    """

    def __init__(
        self,
        repo_root: Path,
        manifest_records: List[Dict[str, Any]],
        variant_ids: List[str],
        label_keys: List[str],
        pose_cfg: PoseFeatureConfig,
        delta_cfg: DeltaFeatureConfig,
        sample_indices_by_variant: Optional[Dict[str, List[int]]] = None,
        cache_pose: bool = True,
        cache_specs: bool = True,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.records_by_id = {str(r["variant_id"]): r for r in manifest_records}
        self.variant_ids = [str(v) for v in variant_ids]
        self.label_keys = [str(k) for k in label_keys]
        self.pose_cfg = pose_cfg
        self.delta_cfg = delta_cfg
        self.sample_indices_by_variant = sample_indices_by_variant
        self.cache_pose = bool(cache_pose)
        self.cache_specs = bool(cache_specs)

        if len(self.label_keys) == 0:
            raise ValueError("[PoseDeltaDataset] label_keys is empty.")
        if "g_ws" not in self.label_keys:
            raise ValueError("[PoseDeltaDataset] label_keys must include 'g_ws'.")
        valid_keys = {
            "g_ws",
            "delta_pos_x",
            "delta_pos_y",
            "delta_pos_z",
            "delta_rot_x",
            "delta_rot_y",
            "delta_rot_z",
        }
        unknown = [k for k in self.label_keys if k not in valid_keys]
        if unknown:
            raise ValueError(f"[PoseDeltaDataset] Unknown label_keys: {unknown}")

        self._pose_cache: Dict[str, Dict[str, np.ndarray]] = {}
        self._spec_cache: Dict[str, Dict[str, Any]] = {}
        self._aabb_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._scale_cache: Dict[str, float] = {}

        self._index: List[Tuple[str, int]] = []
        self._build_index()
        if len(self._index) == 0:
            raise ValueError("[PoseDeltaDataset] Empty dataset. Check splits or sample indices.")

    def _build_index(self) -> None:
        self._index.clear()
        for vid in self.variant_ids:
            rec = self.records_by_id.get(vid, None)
            if rec is None:
                raise KeyError(f"[PoseDeltaDataset] Missing record for variant_id={vid}")
            if self.sample_indices_by_variant is not None and vid in self.sample_indices_by_variant:
                indices = self.sample_indices_by_variant[vid]
                for i in indices:
                    self._index.append((vid, int(i)))
                continue
            n = int(rec.get("num_samples", -1))
            if n <= 0:
                pose = self._load_pose(vid)
                n = int(pose["poses"].shape[0])
            for i in range(n):
                self._index.append((vid, i))

    def __len__(self) -> int:
        return len(self._index)

    def _load_pose(self, vid: str) -> Dict[str, np.ndarray]:
        if self.cache_pose and vid in self._pose_cache:
            return self._pose_cache[vid]
        rec = self.records_by_id[vid]
        pose_path = (self.repo_root / rec["pose_path"]).resolve()
        data = load_pose_samples(pose_path)
        required = ["poses", "labels", "delta_pos", "delta_rot", "delta_mask"]
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"[PoseDeltaDataset] Missing keys {missing} in {pose_path}")
        poses = np.asarray(data["poses"], dtype=np.float32)
        labels = np.asarray(data["labels"], dtype=np.float32).reshape(-1)
        delta_pos = np.asarray(data["delta_pos"], dtype=np.float32)
        delta_rot = np.asarray(data["delta_rot"], dtype=np.float32)
        delta_mask = np.asarray(data["delta_mask"], dtype=np.float32).reshape(-1)
        n = int(poses.shape[0]) if poses.ndim == 2 else -1
        if poses.ndim != 2 or poses.shape[1] != 7:
            raise ValueError(f"[PoseDeltaDataset] poses must be (N,7), got {poses.shape} in {pose_path}")
        if labels.shape[0] != n:
            raise ValueError(f"[PoseDeltaDataset] labels length mismatch in {pose_path}")
        if delta_pos.shape != (n, 3):
            raise ValueError(f"[PoseDeltaDataset] delta_pos shape mismatch in {pose_path}")
        if delta_rot.shape != (n, 3):
            raise ValueError(f"[PoseDeltaDataset] delta_rot shape mismatch in {pose_path}")
        if delta_mask.shape[0] != n:
            raise ValueError(f"[PoseDeltaDataset] delta_mask length mismatch in {pose_path}")
        if self.cache_pose:
            self._pose_cache[vid] = data
        return data

    def _load_spec(self, vid: str) -> Dict[str, Any]:
        if self.cache_specs and vid in self._spec_cache:
            return self._spec_cache[vid]
        rec = self.records_by_id[vid]
        spec_path = (self.repo_root / rec["spec_path"]).resolve()
        spec = load_morph_spec(spec_path)
        if self.cache_specs:
            self._spec_cache[vid] = spec
        return spec

    def _get_scale(self, vid: str, spec: Dict[str, Any]) -> float:
        if vid in self._scale_cache:
            return self._scale_cache[vid]
        Lr, _, _ = compute_morph_meta(spec)
        self._scale_cache[vid] = float(Lr)
        return float(Lr)

    def _get_aabb(self, vid: str, spec: Dict[str, Any], Lr: float) -> Tuple[np.ndarray, np.ndarray]:
        if vid in self._aabb_cache:
            return self._aabb_cache[vid]
        _, aabb_min, aabb_max = compute_morph_meta(spec)
        self._aabb_cache[vid] = (aabb_min, aabb_max)
        return aabb_min, aabb_max

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        vid, sample_idx = self._index[idx]
        rec = self.records_by_id[vid]

        pose_data = self._load_pose(vid)
        spec = self._load_spec(vid)

        poses = np.asarray(pose_data["poses"], dtype=np.float32)
        labels = np.asarray(pose_data["labels"], dtype=np.float32).reshape(-1)
        delta_pos = np.asarray(pose_data["delta_pos"], dtype=np.float32)
        delta_rot = np.asarray(pose_data["delta_rot"], dtype=np.float32)
        delta_mask = np.asarray(pose_data["delta_mask"], dtype=np.float32).reshape(-1)

        if sample_idx < 0 or sample_idx >= poses.shape[0]:
            raise IndexError("[PoseDeltaDataset] sample_idx out of range.")

        pose = poses[sample_idx]
        g_ws = float(labels[sample_idx])
        dp = delta_pos[sample_idx]
        dr = delta_rot[sample_idx]
        dmask = float(delta_mask[sample_idx])

        Lr = self._get_scale(vid, spec)
        aabb_min, aabb_max = self._get_aabb(vid, spec, Lr)

        pose_feats = compute_pose_features(
            pose=pose,
            pose_cfg=self.pose_cfg,
            aabb_min=aabb_min,
            aabb_max=aabb_max,
            morph_scale=Lr,
        )

        dp_norm, dr_norm = normalize_delta(
            delta_pos=dp,
            delta_rot=dr,
            delta_cfg=self.delta_cfg,
            aabb_min=aabb_min,
            aabb_max=aabb_max,
            morph_scale=Lr,
        )

        label_map = {
            "g_ws": g_ws,
            "delta_pos_x": float(dp_norm[0]),
            "delta_pos_y": float(dp_norm[1]),
            "delta_pos_z": float(dp_norm[2]),
            "delta_rot_x": float(dr_norm[0]),
            "delta_rot_y": float(dr_norm[1]),
            "delta_rot_z": float(dr_norm[2]),
        }

        y = np.asarray([label_map[k] for k in self.label_keys], dtype=np.float32)

        ws_mask = 1.0 if g_ws > 0.5 else 0.0

        return {
            "variant_id": vid,
            "family": rec.get("family", None),
            "dof": int(rec.get("dof", -1)),
            "pose_feats": torch.from_numpy(pose_feats),
            "labels": torch.from_numpy(y),
            "ws_mask": torch.tensor(ws_mask, dtype=torch.float32),
            "delta_mask": torch.tensor(dmask, dtype=torch.float32),
            "morph_spec": spec,
        }
