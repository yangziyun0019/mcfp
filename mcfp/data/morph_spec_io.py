"""Load serialized morphology specifications into lightweight Python objects.

These helpers provide the morphology structure consumed by model and training code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import json
import numpy as np
import h5py


@dataclass
class MorphologySpec:
    link_names: List[str]
    depth: np.ndarray  # (N,)
    points: np.ndarray  # (N, P, 6)
    bbox_min: np.ndarray  # (N,3)
    bbox_max: np.ndarray  # (N,3)
    mass: np.ndarray  # (N,)
    inertia: np.ndarray  # (N,6)
    joint_parent: np.ndarray  # (J,)
    joint_child: np.ndarray  # (J,)
    joint_type: List[str]
    joint_origin_xyz: np.ndarray  # (J,3)
    joint_origin_rpy: np.ndarray  # (J,3)
    joint_axis: np.ndarray  # (J,3)
    joint_limit_lower: np.ndarray  # (J,)
    joint_limit_upper: np.ndarray  # (J,)
    acm_allowed: np.ndarray  # (M,2)
    l_ref: float

    tree_dist: np.ndarray  # (N,N)
    kin_mask: np.ndarray  # (N,N) bool
    col_mask: np.ndarray  # (N,N) bool
    acm_mask: np.ndarray  # (N,N) bool


def load_morphology_spec(path: Path) -> MorphologySpec:
    path = Path(path)
    with h5py.File(path, "r") as h5:
        link_names = [n.decode("utf-8") if isinstance(n, bytes) else str(n) for n in h5["/links/name"][...]]
        depth = np.asarray(h5["/links/depth"], dtype=np.int64)
        points = np.asarray(h5["/collision/points"], dtype=np.float32)
        bbox_min = np.asarray(h5["/collision/bbox_min"], dtype=np.float32)
        bbox_max = np.asarray(h5["/collision/bbox_max"], dtype=np.float32)
        mass = np.asarray(h5["/inertial/mass"], dtype=np.float32)
        inertia = np.asarray(h5["/inertial/inertia"], dtype=np.float32)

        joint_parent = h5["/joints/parent"][...]
        joint_child = h5["/joints/child"][...]
        joint_type = [t.decode("utf-8") if isinstance(t, bytes) else str(t) for t in h5["/joints/type"][...]]
        joint_origin_xyz = np.asarray(h5["/joints/origin_xyz"], dtype=np.float32)
        joint_origin_rpy = np.asarray(h5["/joints/origin_rpy"], dtype=np.float32)
        joint_axis = np.asarray(h5["/joints/axis"], dtype=np.float32)
        joint_limit_lower = np.asarray(h5["/joints/limit_lower"], dtype=np.float32)
        joint_limit_upper = np.asarray(h5["/joints/limit_upper"], dtype=np.float32)

        acm_allowed = np.asarray(h5["/acm/allowed_pairs"], dtype=np.int64)

        meta = h5["/meta/json"][()]
        if isinstance(meta, bytes):
            meta = meta.decode("utf-8")
        meta_obj = json.loads(meta) if isinstance(meta, str) else meta
        l_ref = float(meta_obj.get("l_ref", 1.0))

    # Build index map
    name_to_idx = {n: i for i, n in enumerate(link_names)}
    parent_idx = np.array([name_to_idx.get(p.decode("utf-8") if isinstance(p, bytes) else str(p), -1) for p in joint_parent], dtype=np.int64)
    child_idx = np.array([name_to_idx.get(c.decode("utf-8") if isinstance(c, bytes) else str(c), -1) for c in joint_child], dtype=np.int64)

    n = len(link_names)
    kin_mask = np.zeros((n, n), dtype=bool)
    for p, c in zip(parent_idx, child_idx):
        if p >= 0 and c >= 0:
            kin_mask[p, c] = True
            kin_mask[c, p] = True

    # tree distance via BFS
    adj = [[] for _ in range(n)]
    for p, c in zip(parent_idx, child_idx):
        if p >= 0 and c >= 0:
            adj[p].append(c)
            adj[c].append(p)
    tree_dist = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        dist = np.full((n,), -1, dtype=np.int64)
        dist[i] = 0
        queue = [i]
        for u in queue:
            for v in adj[u]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        tree_dist[i] = dist

    # collision edges: full pairwise (excluding self)
    col_mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(col_mask, False)

    acm_mask = np.zeros((n, n), dtype=bool)
    for u, v in acm_allowed:
        if u >= 0 and v >= 0 and u < n and v < n:
            acm_mask[u, v] = True
            acm_mask[v, u] = True

    return MorphologySpec(
        link_names=link_names,
        depth=depth,
        points=points,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        mass=mass,
        inertia=inertia,
        joint_parent=parent_idx,
        joint_child=child_idx,
        joint_type=joint_type,
        joint_origin_xyz=joint_origin_xyz,
        joint_origin_rpy=joint_origin_rpy,
        joint_axis=joint_axis,
        joint_limit_lower=joint_limit_lower,
        joint_limit_upper=joint_limit_upper,
        acm_allowed=acm_allowed,
        l_ref=l_ref,
        tree_dist=tree_dist,
        kin_mask=kin_mask,
        col_mask=col_mask,
        acm_mask=acm_mask,
    )
