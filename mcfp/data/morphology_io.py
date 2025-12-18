# mcfp/data/morphology_io.py

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import random


def _load_urdf_root(urdf_path: Path) -> ET.Element:
    """Load URDF file and return the XML root element."""
    tree = ET.parse(urdf_path)
    return tree.getroot()


def _collect_links(root: ET.Element) -> Dict[str, ET.Element]:
    """Collect link elements from URDF."""
    link_map: Dict[str, ET.Element] = {}
    for link in root.findall(".//link"):
        name = link.attrib.get("name")
        if name:
            link_map[name] = link
    return link_map


def _collect_joints(root: ET.Element) -> List[ET.Element]:
    """Collect joint elements from URDF."""
    joints: List[ET.Element] = []
    for joint in root.findall(".//joint"):
        name = joint.attrib.get("name")
        if not name:
            continue
        joints.append(joint)
    return joints


def _detect_base_and_ee(
    link_map: Dict[str, ET.Element], joints: List[ET.Element]
) -> Tuple[str, str]:
    """Detect base link and end-effector link heuristically."""
    parents = set()
    children = set()
    for j in joints:
        parent = j.find("parent")
        child = j.find("child")
        if parent is not None:
            parents.add(parent.attrib.get("link"))
        if child is not None:
            children.add(child.attrib.get("link"))

    # Base link: appears as parent but never as child.
    candidate_bases = [l for l in parents if l and l not in children]
    base_link = candidate_bases[0] if candidate_bases else next(iter(link_map.keys()))

    # End-effector: appears as child but never as parent.
    candidate_ees = [l for l in children if l and l not in parents]
    ee_link = candidate_ees[0] if candidate_ees else next(reversed(list(link_map.keys())))

    return base_link, ee_link


def _parent_child_from_joint(j: ET.Element) -> Tuple[Optional[str], Optional[str]]:
    """Extract parent and child link names from a joint element."""
    p = j.find("parent")
    c = j.find("child")
    return (p.attrib.get("link") if p is not None else None,
            c.attrib.get("link") if c is not None else None)


def _build_adjacency(joints: List[ET.Element]) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """Build parent->children adjacency and child->parent mapping."""
    children_of: Dict[str, List[str]] = {}
    parent_of: Dict[str, str] = {}
    for j in joints:
        p, c = _parent_child_from_joint(j)
        if not p or not c:
            continue
        children_of.setdefault(p, []).append(c)
        parent_of[c] = p
    return children_of, parent_of


def _path_between(parent_of: Dict[str, str], base: str, ee: str) -> List[str]:
    """Return link path from base to ee (inclusive)."""
    # Walk up from ee to root
    chain = [ee]
    cur = ee
    while cur != base and cur in parent_of:
        cur = parent_of[cur]
        chain.append(cur)
    chain.reverse()
    if not chain or chain[0] != base:
        # Fallback: not connected; return base only.
        return [base]
    return chain


def _extract_main_chain(
    root: ET.Element,
    link_map: Dict[str, ET.Element],
    joints: List[ET.Element],
    base_link: str,
    ee_link: str,
) -> Tuple[List[str], List[ET.Element]]:
    """Extract the main kinematic chain between base and end-effector."""
    _, parent_of = _build_adjacency(joints)
    chain_links = _path_between(parent_of=parent_of, base=base_link, ee=ee_link)

    chain_joints: List[ET.Element] = []
    chain_set = set(chain_links)
    for j in joints:
        p, c = _parent_child_from_joint(j)
        if p in chain_set and c in chain_set:
            chain_joints.append(j)

    # Keep joints ordered along the chain (parent->child)
    link_index = {name: i for i, name in enumerate(chain_links)}
    chain_joints.sort(key=lambda jj: link_index.get(_parent_child_from_joint(jj)[0] or "", 0))

    return chain_links, chain_joints


def _get_joint_origin_xyz_rpy(j: ET.Element) -> Tuple[List[float], List[float]]:
    """Extract joint origin xyz/rpy from joint element."""
    origin = j.find("origin")
    if origin is None:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    xyz = origin.attrib.get("xyz", "0 0 0").split()
    rpy = origin.attrib.get("rpy", "0 0 0").split()
    xyz_f = [float(x) for x in xyz[:3]] + [0.0] * max(0, 3 - len(xyz))
    rpy_f = [float(x) for x in rpy[:3]] + [0.0] * max(0, 3 - len(rpy))
    return xyz_f[:3], rpy_f[:3]


def _get_joint_axis(j: ET.Element) -> List[float]:
    """Extract joint axis from joint element."""
    axis = j.find("axis")
    if axis is None:
        return [0.0, 0.0, 1.0]
    xyz = axis.attrib.get("xyz", "0 0 1").split()
    vals = [float(x) for x in xyz[:3]] + [0.0] * max(0, 3 - len(xyz))
    return vals[:3]


def _get_joint_limit(j: ET.Element) -> Tuple[Optional[float], Optional[float]]:
    """Extract joint limits from joint element."""
    limit = j.find("limit")
    if limit is None:
        return None, None
    lo = limit.attrib.get("lower")
    hi = limit.attrib.get("upper")
    if lo is None or hi is None:
        return None, None
    return float(lo), float(hi)


def _estimate_link_lengths(chain_joints: List[ET.Element]) -> Dict[str, float]:
    """Estimate link lengths using joint origin translation magnitude."""
    lengths: Dict[str, float] = {}
    for j in chain_joints:
        _, child = _parent_child_from_joint(j)
        if not child:
            continue
        xyz, _ = _get_joint_origin_xyz_rpy(j)
        l = float(math.sqrt(xyz[0] ** 2 + xyz[1] ** 2 + xyz[2] ** 2))
        if l <= 1e-8:
            l = 0.0
        lengths[child] = l
    return lengths


def _extract_inertial_matrix(link_elem: ET.Element) -> Optional[List[float]]:
    """Extract inertia matrix diagonal and products from URDF link inertial."""
    inertial = link_elem.find("inertial")
    if inertial is None:
        return None
    inertia = inertial.find("inertia")
    if inertia is None:
        return None
    # URDF inertia fields
    fields = ["ixx", "iyy", "izz", "ixy", "ixz", "iyz"]
    vals = []
    for f in fields:
        v = inertia.attrib.get(f)
        if v is None:
            return None
        vals.append(float(v))
    return vals


def _estimate_link_radii_from_inertia(
    root: ET.Element,
    chain_links: List[str],
    link_lengths: Dict[str, float],
) -> Dict[str, float]:
    """Estimate a proxy collision radius for each link based on inertia.

    This is an intentionally rough proxy. It is used to build capsule-like
    approximations when collision geometry is missing or unsuitable.
    """
    link_map = _collect_links(root)
    radii: Dict[str, float] = {}

    DEFAULT_RADIUS = 0.019
    MIN_RADIUS = 0.003
    MAX_RADIUS = 0.019

    for link_name in chain_links:
        link_elem = link_map.get(link_name)
        if link_elem is None:
            radii[link_name] = DEFAULT_RADIUS
            continue

        inertia_vals = _extract_inertial_matrix(link_elem)
        if inertia_vals is None:
            radii[link_name] = DEFAULT_RADIUS
            continue

        ixx, iyy, izz, ixy, ixz, iyz = inertia_vals
        # Use a simple scale proxy: larger principal inertia -> larger radius.
        diag_mean = max(0.0, (abs(ixx) + abs(iyy) + abs(izz)) / 3.0)
        base = math.sqrt(diag_mean) if diag_mean > 0.0 else DEFAULT_RADIUS

        # Link length constraint (avoid extreme aspect ratios).
        L = float(link_lengths.get(link_name, 0.0))
        if L > 1e-6:
            base = min(base, 0.5 * L)

        # Clamp to robust bounds.
        r = float(max(MIN_RADIUS, min(MAX_RADIUS, base)))
        radii[link_name] = r

    return radii


def _build_link_features(
    root: ET.Element,
    chain_links: List[str],
    link_map: Dict[str, ET.Element],
    link_lengths: Dict[str, float],
    link_radii: Dict[str, float],
) -> List[Dict[str, Any]]:
    """Build link feature dicts for JSON spec."""
    features: List[Dict[str, Any]] = []
    for idx, name in enumerate(chain_links):
        L = float(link_lengths.get(name, 0.0))
        R = float(link_radii.get(name, 0.03))
        features.append(
            {
                "index": int(idx),
                "name": name,
                "length_estimate": float(L),
                "radius_estimate": float(R),
            }
        )
    return features


def _build_joint_features(chain_joints: List[ET.Element]) -> List[Dict[str, Any]]:
    """Build joint feature dicts for JSON spec."""
    jfs: List[Dict[str, Any]] = []
    for j in chain_joints:
        name = j.attrib.get("name", "")
        jtype = j.attrib.get("type", "revolute")
        p, c = _parent_child_from_joint(j)
        xyz, rpy = _get_joint_origin_xyz_rpy(j)
        axis = _get_joint_axis(j)
        lo, hi = _get_joint_limit(j)

        is_actuated = jtype in ["revolute", "continuous", "prismatic"]

        # Provide reasonable fallbacks for missing limits.
        if is_actuated:
            if jtype == "continuous":
                lo, hi = -math.pi, math.pi
            elif lo is None or hi is None:
                # Conservative fallback limits.
                lo, hi = -math.pi, math.pi

        jfs.append(
            {
                "name": name,
                "type": jtype,
                "parent_link": p,
                "child_link": c,
                "origin_xyz": [float(x) for x in xyz],
                "origin_rpy": [float(x) for x in rpy],
                "axis": [float(a) for a in axis],
                "limit_lower": None if lo is None else float(lo),
                "limit_upper": None if hi is None else float(hi),
                "is_actuated": bool(is_actuated),
            }
        )
    return jfs


def save_morph_json(morph_dict: Dict[str, Any], output_path: Path | str) -> None:
    """Save morphology dict to JSON file."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(morph_dict, f, indent=2)


def urdf_to_morph_dict(
    urdf_path: Path | str,
    robot_name: Optional[str] = None,
    family: Optional[str] = None,
    source: str = "real",
    base_link: Optional[str] = None,
    ee_link: Optional[str] = None,
    variant_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert a URDF file into a morphology JSON dictionary.

    The output is designed to serve two distinct purposes:
    1) Capability map generation (kinematics + proxy collision).
    2) Morphology encoding (graph-based learning).

    Args:
        urdf_path: Path to the URDF file.
        robot_name: Logical robot name used in metadata (defaults to URDF stem).
        family: Optional family/group name for organizing variants.
        source: A short tag describing the source (e.g., "real", "synthetic").
        base_link: Optional base link override; if None, a heuristic is used.
        ee_link: Optional end-effector override; if None, a heuristic is used.
        variant_id: Optional variant id to write into meta/morphology. If None,
            defaults to "base".

    Returns:
        A JSON-serializable dict containing:
            - meta
            - chain (links + joints, already linearized to the main chain)
            - morphology (base / identity parameters)
            - graph (edges)
    """
    urdf_path = Path(urdf_path).resolve()
    root = _load_urdf_root(urdf_path)

    link_map = _collect_links(root)
    joints = _collect_joints(root)

    if base_link is None or ee_link is None:
        base_auto, ee_auto = _detect_base_and_ee(link_map=link_map, joints=joints)
        base_link = base_link or base_auto
        ee_link = ee_link or ee_auto

    chain_links, chain_joints = _extract_main_chain(
        root=root, link_map=link_map, joints=joints, base_link=base_link, ee_link=ee_link
    )

    # Proxy geometry for collision: link lengths + capsule radii
    link_lengths = _estimate_link_lengths(chain_joints=chain_joints)
    link_radii = _estimate_link_radii_from_inertia(
        root=root, chain_links=chain_links, link_lengths=link_lengths
    )

    link_features = _build_link_features(
        root=root,
        chain_links=chain_links,
        link_map=link_map,
        link_lengths=link_lengths,
        link_radii=link_radii,
    )
    joint_features = _build_joint_features(chain_joints)

    actuated_joint_names: List[str] = [jf["name"] for jf in joint_features if jf["is_actuated"]]
    dof = len(actuated_joint_names)

    # Graph edges (for EGNN)
    name_to_index = {lf["name"]: lf["index"] for lf in link_features}
    edges: List[Tuple[int, int]] = []
    for jf in joint_features:
        p = jf["parent_link"]
        c = jf["child_link"]
        if p in name_to_index and c in name_to_index:
            edges.append((name_to_index[p], name_to_index[c]))

    base_variant_id = variant_id or "base"

    morphology = {
        "variant_id": base_variant_id,
        "parent_variant_id": None,
        "actuated_joint_names": actuated_joint_names,
        "length_scale": [1.0] * dof,
        "joint_limit_scale": [1.0] * dof,
        "joint_limit_shift": [0.0] * dof,
        "r_vector": None,
        "notes": {
            "generated_from": "urdf",
            "warning": "Base spec uses identity perturbations; variants should materialize chain edits.",
        },
    }

    morph_dict = {
        "meta": {
            "robot_name": robot_name or urdf_path.stem,
            "family": family,
            "variant_id": base_variant_id,
            "parent_variant_id": None,
            "source": source,
            "urdf_path": str(urdf_path),
            "dof": int(dof),
            "base_link": base_link,
            "ee_link": ee_link,
        },
        "chain": {
            "links": link_features,
            "joints": joint_features,
        },
        "graph": {
            "num_nodes": int(len(link_features)),
            "edges": edges,
        },
        "morphology": morphology,
    }

    return morph_dict


def create_variant_from_base(
    base_morph: Dict[str, Any],
    variant_id: str,
    length_scale_range: Tuple[float, float] = (0.7, 1.3),
    joint_limit_scale_range: Tuple[float, float] = (0.5, 1.0),
    joint_limit_shift_fraction: float = 0.2,
    materialize_chain: bool = True,
    perturb_fixed_joints: bool = False,
    min_limit_width: float = 1e-3,
) -> Dict[str, Any]:
    """Create a perturbed morphology variant from a base morphology.

    This function supports two modes:
    - materialize_chain=True:
        Apply perturbations directly to the kinematic chain (joint origins and
        joint limits). This produces a *physically different* morphology that
        will result in different FK / workspace / capability scores.
    - materialize_chain=False:
        Only write perturbation parameters into the "morphology" block (legacy
        behavior). This does NOT change capability results and is kept only for
        backward compatibility.

    Perturbations:
        1) Link length proxy: scale joint origin translation (origin_xyz).
        2) Joint limit proxy: shrink joint range and optionally shift its center.

    Args:
        base_morph: Base morphology dict.
        variant_id: New variant id to write into meta/morphology.
        length_scale_range: Range for length scale sampling (per actuated DOF).
        joint_limit_scale_range: Range for limit range scaling (per actuated DOF).
        joint_limit_shift_fraction: Std of shift as a fraction of original width.
        materialize_chain: Whether to apply perturbations to chain.
        perturb_fixed_joints: If True, also scale origin_xyz for fixed joints
            (not recommended unless the URDF uses fixed joints for main segments).
        min_limit_width: Minimum limit width enforced after perturbation.

    Returns:
        A new morphology dict for the variant.
    """
    morph_variant = deepcopy(base_morph)

    base_meta = morph_variant.get("meta", {})
    base_morphology = morph_variant.get("morphology", {})
    actuated_joint_names: List[str] = base_morphology.get("actuated_joint_names", [])

    chain_joints: List[Dict[str, Any]] = morph_variant["chain"]["joints"]
    joint_by_name = {j["name"]: j for j in chain_joints}

    # Determine DOF order if not provided
    if not actuated_joint_names:
        actuated_joint_names = [j["name"] for j in chain_joints if j.get("is_actuated")]

    # Collect original widths for shift scaling
    limit_widths: List[float] = []
    for name in actuated_joint_names:
        j = joint_by_name.get(name)
        if j is None:
            limit_widths.append(0.0)
            continue
        lo = j.get("limit_lower")
        hi = j.get("limit_upper")
        if lo is None or hi is None:
            limit_widths.append(0.0)
        else:
            limit_widths.append(float(hi) - float(lo))

    # Sample per-DOF perturbation parameters
    per_dof_length_scale: List[float] = []
    per_dof_limit_scale: List[float] = []
    per_dof_limit_shift: List[float] = []

    dof_params: Dict[str, Tuple[float, float, float]] = {}
    for name, width in zip(actuated_joint_names, limit_widths):
        ls = random.uniform(*length_scale_range)
        js = random.uniform(*joint_limit_scale_range)
        if width > 0.0 and joint_limit_shift_fraction > 0.0:
            shift_std = joint_limit_shift_fraction * width
            shift = random.gauss(0.0, shift_std)
        else:
            shift = 0.0
        per_dof_length_scale.append(ls)
        per_dof_limit_scale.append(js)
        per_dof_limit_shift.append(shift)
        dof_params[name] = (ls, js, shift)

    # Update ids
    morph_variant.setdefault("meta", {})
    morph_variant["meta"]["variant_id"] = variant_id
    morph_variant["meta"]["parent_variant_id"] = base_meta.get("variant_id", None)

    morph_variant["morphology"] = {
        "variant_id": variant_id,
        "parent_variant_id": base_meta.get("variant_id", None),
        "actuated_joint_names": actuated_joint_names,
        "length_scale": per_dof_length_scale,
        "joint_limit_scale": per_dof_limit_scale,
        "joint_limit_shift": per_dof_limit_shift,
        "r_vector": None,
        "notes": {
            "generated_from": "perturbation",
            "materialize_chain": bool(materialize_chain),
            "length_scale_range": list(length_scale_range),
            "joint_limit_scale_range": list(joint_limit_scale_range),
            "joint_limit_shift_fraction": float(joint_limit_shift_fraction),
        },
    }

    if not materialize_chain:
        return morph_variant

    # ------------------------------------------------------------------
    # Materialize perturbations into the chain
    # ------------------------------------------------------------------
    eps = 1e-12

    # 1) Apply length scaling on joint origins (origin_xyz)
    for j in chain_joints:
        jname = j.get("name")
        jtype = j.get("type")
        is_act = bool(j.get("is_actuated"))
        if (not is_act) and (not perturb_fixed_joints):
            continue
        if (not is_act) and perturb_fixed_joints and jtype != "fixed":
            # Only fixed joints are optionally perturbed in this mode.
            continue

        ls = dof_params.get(jname, (1.0, 1.0, 0.0))[0]
        xyz = j.get("origin_xyz")
        if isinstance(xyz, list) and len(xyz) == 3:
            vx, vy, vz = float(xyz[0]), float(xyz[1]), float(xyz[2])
            if abs(vx) + abs(vy) + abs(vz) > eps:
                j["origin_xyz"] = [vx * ls, vy * ls, vz * ls]

    # 2) Apply joint limit shrinking + shift for actuated joints
    for j in chain_joints:
        if not bool(j.get("is_actuated")):
            continue
        jname = j.get("name")
        _, js, shift = dof_params.get(jname, (1.0, 1.0, 0.0))
        lo = j.get("limit_lower")
        hi = j.get("limit_upper")
        if lo is None or hi is None:
            continue

        lo0 = float(lo)
        hi0 = float(hi)
        if hi0 <= lo0:
            continue

        center0 = 0.5 * (lo0 + hi0)
        half0 = 0.5 * (hi0 - lo0)

        half_new = max(min_limit_width * 0.5, half0 * js)
        center_new = center0 + shift

        lo_new = center_new - half_new
        hi_new = center_new + half_new

        # Keep the perturbed interval within the original bounds
        lo_new = max(lo0, lo_new)
        hi_new = min(hi0, hi_new)

        # Enforce minimum width if clamping collapses the interval
        if hi_new - lo_new < min_limit_width:
            mid = max(lo0, min(hi0, center_new))
            lo_new = max(lo0, mid - 0.5 * min_limit_width)
            hi_new = min(hi0, mid + 0.5 * min_limit_width)

        # Final guard
        if hi_new <= lo_new:
            # Fallback to original if numerical issues occur
            lo_new, hi_new = lo0, hi0

        j["limit_lower"] = float(lo_new)
        j["limit_upper"] = float(hi_new)

    # 3) Update link proxies: length_estimate and radius_estimate
    chain_links: List[Dict[str, Any]] = morph_variant["chain"]["links"]
    child_to_joint: Dict[str, Dict[str, Any]] = {jj.get("child_link"): jj for jj in chain_joints}

    link_scale_by_name: Dict[str, float] = {}
    for link in chain_links:
        lname = link.get("name")
        incoming = child_to_joint.get(lname)
        if incoming is not None and bool(incoming.get("is_actuated")):
            ls = dof_params.get(incoming.get("name"), (1.0, 1.0, 0.0))[0]
        else:
            ls = 1.0
        link_scale_by_name[lname] = float(ls)

    for link in chain_links:
        lname = link.get("name")
        ls = link_scale_by_name.get(lname, 1.0)

        if isinstance(link.get("length_estimate"), (int, float)):
            link["length_estimate"] = float(link["length_estimate"]) * ls

        if isinstance(link.get("radius_estimate"), (int, float)):
            link["radius_estimate"] = float(link["radius_estimate"]) * ls

    return morph_variant


def generate_variants_from_urdf(
    urdf_path: Path | str,
    output_dir: Path | str,
    family: Optional[str] = None,
    source: str = "real",
    robot_name: Optional[str] = None,
    base_link: Optional[str] = None,
    ee_link: Optional[str] = None,
    num_variants: int = 0,
    length_scale_range: Tuple[float, float] = (0.7, 1.3),
    joint_limit_scale_range: Tuple[float, float] = (0.5, 1.0),
    joint_limit_shift_fraction: float = 0.2,
    materialize_chain: bool = True,
) -> List[Path]:
    """Convert a URDF to a base morphology JSON and generate perturbed variants.

    Args:
        urdf_path: URDF file path.
        output_dir: Output directory for generated JSONs.
        family: Optional family name.
        source: Source tag for metadata.
        robot_name: Logical name; defaults to URDF stem.
        base_link: Optional base link override.
        ee_link: Optional end-effector override.
        num_variants: Number of variants to generate.
        length_scale_range: Per-DOF origin scaling range.
        joint_limit_scale_range: Per-DOF limit scaling range.
        joint_limit_shift_fraction: Shift std fraction (relative to original width).
        materialize_chain: If True, variants will modify the chain (recommended).

    Returns:
        List of JSON file paths (base first, then variants).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    urdf_path = Path(urdf_path)
    base_name = robot_name or urdf_path.stem
    base_variant_id = f"{base_name}_base"

    base_morph = urdf_to_morph_dict(
        urdf_path=urdf_path,
        robot_name=base_name,
        family=family,
        source=source,
        base_link=base_link,
        ee_link=ee_link,
        variant_id=base_variant_id,
    )

    base_json_path = output_dir / f"{base_variant_id}.json"
    save_morph_json(base_morph, base_json_path)

    json_paths: List[Path] = [base_json_path]

    for idx in range(1, num_variants + 1):
        vid = f"{base_name}_v{idx:04d}"
        variant_morph = create_variant_from_base(
            base_morph=base_morph,
            variant_id=vid,
            length_scale_range=length_scale_range,
            joint_limit_scale_range=joint_limit_scale_range,
            joint_limit_shift_fraction=joint_limit_shift_fraction,
            materialize_chain=materialize_chain,
        )
        variant_json_path = output_dir / f"{vid}.json"
        save_morph_json(variant_morph, variant_json_path)
        json_paths.append(variant_json_path)

    return json_paths
