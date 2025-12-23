# mcfp/models/morph_graph.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import math

import torch


@dataclass(frozen=True)
class GraphData:
    """A minimal graph container for morphology encoding.

    Attributes:
        x: Node features, shape (N, D).
        edge_index: Directed edges in COO format, shape (2, E).
        edge_attr: Optional edge features, shape (E, F).
        batch: Graph id for each node, shape (N,). Optional for single-graph input.
        node_names: Link names aligned with node order, length N.
        meta: Optional metadata for debugging or analysis (non-tensor).
    """
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: Optional[torch.Tensor] = None
    batch: Optional[torch.Tensor] = None
    node_names: List[str] = None
    meta: Dict[str, Any] = None


def _safe_float(v: Any, default: float = 0.0) -> float:
    """Convert a value to float with a safe fallback."""
    try:
        if v is None:
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _extract_links(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract ordered link records from a morphology spec."""
    chain = spec.get("chain", None)
    if chain is None:
        raise ValueError("morph_spec missing key: 'chain'")
    links = chain.get("links", None)
    if links is None:
        raise ValueError("morph_spec missing key: 'chain.links'")
    if not isinstance(links, list) or len(links) == 0:
        raise ValueError("morph_spec['chain']['links'] must be a non-empty list")

    # Prefer explicit index ordering if present.
    if all(isinstance(lk, dict) and "index" in lk for lk in links):
        links_sorted = sorted(links, key=lambda d: int(d["index"]))
        return links_sorted
    return links


def _build_edges_from_joints(
    links: List[Dict[str, Any]],
    joints: List[Dict[str, Any]],
    bidirectional: bool,
) -> Tuple[List[int], List[int], List[Dict[str, Any]]]:
    """Build edges from joint parent/child relations.

    Returns:
        src_list, dst_list, joint_records aligned to edges (parent->child only).
    """
    name_to_idx: Dict[str, int] = {}
    for i, lk in enumerate(links):
        name = str(lk.get("name", f"link_{i}"))
        name_to_idx[name] = i

    src_list: List[int] = []
    dst_list: List[int] = []

    edge_joints: List[Dict[str, Any]] = []
    for jt in joints:
        if not isinstance(jt, dict):
            continue
        p = jt.get("parent_link", None)
        c = jt.get("child_link", None)
        if p is None or c is None:
            continue
        p = str(p)
        c = str(c)
        if p not in name_to_idx or c not in name_to_idx:
            continue

        u = name_to_idx[p]
        v = name_to_idx[c]
        src_list.append(u)
        dst_list.append(v)
        edge_joints.append(jt)
        if bidirectional:
            src_list.append(v)
            dst_list.append(u)
            edge_joints.append(jt)

    return src_list, dst_list, edge_joints


def _build_edges_fallback_chain(
    n_links: int, bidirectional: bool
) -> Tuple[List[int], List[int], List[Dict[str, Any]]]:
    """Fallback edge builder if joints are missing: connect consecutive links."""
    src_list: List[int] = []
    dst_list: List[int] = []
    edge_joints: List[Dict[str, Any]] = []
    if n_links <= 1:
        return src_list, dst_list, edge_joints

    for i in range(n_links - 1):
        src_list.append(i)
        dst_list.append(i + 1)
        edge_joints.append({})
        if bidirectional:
            src_list.append(i + 1)
            dst_list.append(i)
            edge_joints.append({})

    return src_list, dst_list, edge_joints


def build_link_graph(
    spec: Dict[str, Any],
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    bidirectional: bool = True,
    use_link_index_feature: bool = True,
) -> GraphData:
    """Build a link-graph (links as nodes) from a morphology spec.

    Node features default to:
        [length_estimate, radius_estimate, link_index_norm,
         depth_norm, path_length_norm, is_base, is_ee, is_actuated_link]
        (if use_link_index_feature=False, link_index_norm is 0)

    Edge features default to:
        [joint_type_onehot(4), axis(3), origin_xyz(3), origin_rpy(3),
         limit_lower, limit_upper, limit_range, limit_mid, is_actuated, edge_dir]

    Args:
        spec: Morphology spec dict loaded from JSON.
        device: Target torch device.
        dtype: Floating dtype for node features.
        bidirectional: If True, add reverse edges for each kinematic edge.
        use_link_index_feature: If True, include normalized link index as a feature.

    Returns:
        GraphData with node features and edge_index.
    """
    links = _extract_links(spec)
    n = len(links)

    node_names: List[str] = []
    lengths: List[float] = []
    radii: List[float] = []
    idx_norm: List[float] = []

    denom = float(max(n - 1, 1))
    for i, lk in enumerate(links):
        name = str(lk.get("name", f"link_{i}"))
        node_names.append(name)

        lengths.append(_safe_float(lk.get("length_estimate", 0.0), default=0.0))
        radii.append(_safe_float(lk.get("radius_estimate", 0.0), default=0.0))

        if use_link_index_feature:
            raw_idx = lk.get("index", i)
            try:
                raw_idx = int(raw_idx)
            except Exception:
                raw_idx = int(i)
            idx_norm.append(float(raw_idx) / denom)

    chain = spec.get("chain", {})
    joints = chain.get("joints", [])
    meta = spec.get("meta", {})

    name_to_idx = {str(n): i for i, n in enumerate(node_names)}

    # Build parent/child mapping for depth/path length
    parent_of: Dict[str, str] = {}
    children_of: Dict[str, List[str]] = {}
    for jt in joints:
        if not isinstance(jt, dict):
            continue
        p = jt.get("parent_link", None)
        c = jt.get("child_link", None)
        if p is None or c is None:
            continue
        p = str(p)
        c = str(c)
        if p not in name_to_idx or c not in name_to_idx:
            continue
        if c not in parent_of:
            parent_of[c] = p
        children_of.setdefault(p, []).append(c)

    base_name = str(meta.get("base_link")) if meta.get("base_link") is not None else None
    if base_name is None or base_name not in name_to_idx:
        roots = [nm for nm in node_names if nm not in parent_of]
        base_name = roots[0] if roots else node_names[0]

    # BFS for depth/path_length
    depth: Dict[str, int] = {base_name: 0}
    path_len: Dict[str, float] = {base_name: lengths[name_to_idx[base_name]]}
    stack = [base_name]
    while stack:
        cur = stack.pop(0)
        for ch in children_of.get(cur, []):
            depth[ch] = depth[cur] + 1
            path_len[ch] = path_len[cur] + lengths[name_to_idx[ch]]
            stack.append(ch)

    max_depth = max(depth.values()) if depth else 0
    total_len = float(sum(lengths))
    total_len = total_len if total_len > 1e-8 else 1.0

    # Actuated link flags: mark child link if joint is actuated
    actuated_joint_names = set(
        [str(n) for n in spec.get("morphology", {}).get("actuated_joint_names", [])]
    )
    actuated_link = {nm: 0.0 for nm in node_names}
    for jt in joints:
        if not isinstance(jt, dict):
            continue
        c = jt.get("child_link", None)
        if c is None:
            continue
        c = str(c)
        if c not in name_to_idx:
            continue
        is_act = bool(jt.get("is_actuated", False)) or (str(jt.get("name", "")) in actuated_joint_names)
        if is_act:
            actuated_link[c] = 1.0

    depth_norm = []
    path_norm = []
    depth_sin = []
    depth_cos = []
    path_sin = []
    path_cos = []
    is_base = []
    is_ee = []
    is_act = []
    ee_name = str(meta.get("ee_link")) if meta.get("ee_link") is not None else None

    for nm in node_names:
        d = depth.get(nm, 0)
        dn = float(d) / float(max_depth if max_depth > 0 else 1)
        depth_norm.append(dn)
        pl = path_len.get(nm, lengths[name_to_idx[nm]])
        pn = float(pl) / float(total_len)
        path_norm.append(pn)
        depth_sin.append(math.sin(2.0 * math.pi * dn))
        depth_cos.append(math.cos(2.0 * math.pi * dn))
        path_sin.append(math.sin(2.0 * math.pi * pn))
        path_cos.append(math.cos(2.0 * math.pi * pn))
        is_base.append(1.0 if nm == base_name else 0.0)
        is_ee.append(1.0 if (ee_name is not None and nm == ee_name) else 0.0)
        is_act.append(float(actuated_link.get(nm, 0.0)))

    feat_cols: List[List[float]] = [lengths, radii]
    if use_link_index_feature:
        feat_cols.append(idx_norm)
    else:
        feat_cols.append([0.0 for _ in node_names])
    feat_cols.extend([depth_norm, path_norm, depth_sin, depth_cos, path_sin, path_cos, is_base, is_ee, is_act])

    x_np = list(zip(*feat_cols))
    x = torch.tensor(x_np, dtype=dtype, device=device)

    src_list: List[int]
    dst_list: List[int]
    edge_joints: List[Dict[str, Any]]
    if isinstance(joints, list) and len(joints) > 0:
        src_list, dst_list, edge_joints = _build_edges_from_joints(links, joints, bidirectional)
    else:
        src_list, dst_list, edge_joints = _build_edges_fallback_chain(n, bidirectional)

    if len(src_list) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    else:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long, device=device)

    # Edge features
    type_list = ["revolute", "prismatic", "fixed", "continuous"]
    edge_feat_list: List[List[float]] = []
    for jt in edge_joints:
        jtype = str(jt.get("type", "")).lower()
        type_onehot = [1.0 if jtype == t else 0.0 for t in type_list]

        axis = jt.get("axis", [0.0, 0.0, 0.0])
        origin_xyz = jt.get("origin_xyz", [0.0, 0.0, 0.0])
        origin_rpy = jt.get("origin_rpy", [0.0, 0.0, 0.0])
        origin_xyz = [float(x) / float(total_len) for x in origin_xyz]
        origin_rpy = [float(x) / math.pi for x in origin_rpy]
        limit_lower = _safe_float(jt.get("limit_lower", 0.0), default=0.0)
        limit_upper = _safe_float(jt.get("limit_upper", 0.0), default=0.0)
        limit_lower = float(limit_lower) / math.pi
        limit_upper = float(limit_upper) / math.pi
        limit_range = limit_upper - limit_lower
        limit_mid = 0.5 * (limit_upper + limit_lower)
        is_act = 1.0 if bool(jt.get("is_actuated", False)) else 0.0

        edge_feat_list.append(
            type_onehot
            + [float(x) for x in axis]
            + [float(x) for x in origin_xyz]
            + [float(x) for x in origin_rpy]
            + [float(limit_lower), float(limit_upper), float(limit_range), float(limit_mid)]
            + [float(is_act)]
            # edge_dir is appended below to match edge_index order
        )

    # Add direction sign (+1 for forward, -1 for reverse).
    edge_dir: List[float] = []
    if edge_index.numel() > 0:
        for i in range(edge_index.shape[1]):
            if i % 2 == 1 and bidirectional:
                edge_dir.append(-1.0)
            else:
                edge_dir.append(1.0)

    if len(edge_feat_list) > 0:
        edge_feat_dim = len(edge_feat_list[0]) + 1
        edge_attr = torch.tensor(
            [ef + [edge_dir[i]] for i, ef in enumerate(edge_feat_list)],
            dtype=dtype,
            device=device,
        )
    else:
        # Default edge feature dimension for empty graphs.
        edge_feat_dim = 4 + 3 + 3 + 3 + 4 + 1 + 1
        edge_attr = torch.empty((0, edge_feat_dim), dtype=dtype, device=device)

    meta = {
        "family": meta.get("family", None),
        "variant_id": meta.get("variant_id", None),
        "num_links": n,
        "bidirectional": bidirectional,
        "node_feat_dim": int(x.shape[1]),
        "num_edges": int(edge_index.shape[1]),
        "edge_feat_dim": int(edge_attr.shape[1]) if edge_attr is not None else 0,
    }

    batch = torch.zeros((n,), dtype=torch.long, device=device)

    return GraphData(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        batch=batch,
        node_names=node_names,
        meta=meta,
    )
