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
    urdf_path = Path(urdf_path).resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")
    tree = ET.parse(urdf_path)
    return tree.getroot()


def _collect_links_and_joints(
    root: ET.Element,
) -> Tuple[Dict[str, ET.Element], List[ET.Element]]:
    """Collect link and joint elements from URDF."""
    link_map: Dict[str, ET.Element] = {}
    for link in root.findall(".//link"):
        name = link.attrib.get("name")
        if name:
            link_map[name] = link

    joints: List[ET.Element] = []
    for joint in root.findall(".//joint"):
        name = joint.attrib.get("name")
        if not name:
            continue
        joints.append(joint)

    return link_map, joints


def _autodetect_base_and_ee(
    link_map: Dict[str, ET.Element],
    joints: List[ET.Element],
) -> Tuple[str, str]:
    """Heuristically detect base and end-effector links from URDF.

    Base:
        Link that never appears as a child.
    EE:
        Leaf link (never a parent) that is farthest in kinematic depth,
        with a preference for names containing typical EE keywords.

    This heuristic is robust for most collaborative arms, but callers
    should override base and EE explicitly if their URDF contains
    grippers or tool attachments with non-standard naming.
    """
    link_names = list(link_map.keys())

    parent_links = set()
    child_links = set()
    for j in joints:
        parent = j.find("parent")
        child = j.find("child")
        if parent is not None and "link" in parent.attrib:
            parent_links.add(parent.attrib["link"])
        if child is not None and "link" in child.attrib:
            child_links.add(child.attrib["link"])

    base_candidates = [ln for ln in link_names if ln not in child_links]
    if not base_candidates:
        base_link = link_names[0]
    else:
        base_link = base_candidates[0]

    # Build adjacency for depth estimation
    children_by_parent: Dict[str, List[str]] = {}
    for j in joints:
        parent = j.find("parent")
        child = j.find("child")
        if parent is None or child is None:
            continue
        p_name = parent.attrib.get("link")
        c_name = child.attrib.get("link")
        if not p_name or not c_name:
            continue
        children_by_parent.setdefault(p_name, []).append(c_name)

    depth: Dict[str, int] = {base_link: 0}
    queue: List[str] = [base_link]
    while queue:
        current = queue.pop(0)
        for child in children_by_parent.get(current, []):
            if child not in depth:
                depth[child] = depth[current] + 1
                queue.append(child)

    leaf_links = [ln for ln in link_names if ln not in children_by_parent]

    ee_keywords = ("tool", "tcp", "ee", "end_effector", "wrist")
    ee_link: Optional[str] = None
    best_score: Tuple[int, int] = (-1, -1)

    for ln in leaf_links:
        d = depth.get(ln, 0)
        name_lower = ln.lower()
        keyword_score = 1 if any(k in name_lower for k in ee_keywords) else 0
        score = (keyword_score, d)
        if score > best_score:
            best_score = score
            ee_link = ln

    if ee_link is None:
        # fallback: deepest leaf
        best_depth = -1
        for ln in leaf_links:
            d = depth.get(ln, 0)
            if d > best_depth:
                best_depth = d
                ee_link = ln

    if ee_link is None:
        ee_link = link_names[-1]

    return base_link, ee_link


def _estimate_link_lengths(root: ET.Element) -> Dict[str, float]:
    """Estimate link length based on distance to child joint origins."""
    link_lengths: Dict[str, float] = {}

    for joint in root.findall(".//joint"):
        parent_elem = joint.find("parent")
        origin_elem = joint.find("origin")
        if parent_elem is None or origin_elem is None:
            continue

        parent_name = parent_elem.attrib.get("link")
        xyz_str = origin_elem.attrib.get("xyz")
        if not parent_name or not xyz_str:
            continue

        try:
            xyz = [float(x) for x in xyz_str.split()]
            dist = math.sqrt(sum(v * v for v in xyz))
        except ValueError:
            continue

        current = link_lengths.get(parent_name, 0.0)
        if dist > current:
            link_lengths[parent_name] = dist

    return link_lengths


def _estimate_link_radii(
    root: ET.Element,
    link_lengths: Dict[str, float],
) -> Dict[str, float]:
    """Estimate collision radius per link using a robust heuristic.

    Strategy:
    1. Use explicit collision primitives (cylinder/sphere radius) if available.
    2. Else, use inertial parameters (mass + inertia tensor) to estimate radius.
    3. Apply aspect-ratio constraint based on estimated link length.
    4. Fallback to a safe default radius for links with geometry; 0 for virtual links.
    """
    MIN_RADIUS = 0.02
    MAX_RADIUS = 0.12
    DEFAULT_RADIUS = 0.03
    SCALE = 0.9

    radii: Dict[str, float] = {}

    for link in root.findall(".//link"):
        name = link.attrib.get("name")
        if not name:
            continue

        radius: Optional[float] = None

        coll = link.find("collision")
        if coll is not None:
            geo = coll.find("geometry")
            if geo is not None:
                cyl = geo.find("cylinder")
                if cyl is not None and "radius" in cyl.attrib:
                    try:
                        radius = float(cyl.attrib["radius"])
                    except ValueError:
                        radius = None
                if radius is None:
                    sph = geo.find("sphere")
                    if sph is not None and "radius" in sph.attrib:
                        try:
                            radius = float(sph.attrib["radius"])
                        except ValueError:
                            radius = None

        if radius is None:
            inertial = link.find("inertial")
            if inertial is not None:
                mass_elem = inertial.find("mass")
                inertia_elem = inertial.find("inertia")
                if mass_elem is not None and "value" in mass_elem.attrib:
                    try:
                        m = float(mass_elem.attrib["value"])
                        if m <= 1e-4:
                            radius = 0.0
                        elif inertia_elem is not None:
                            ixx = float(inertia_elem.attrib.get("ixx", 0.0))
                            iyy = float(inertia_elem.attrib.get("iyy", 0.0))
                            izz = float(inertia_elem.attrib.get("izz", 0.0))
                            moments = [v for v in (ixx, iyy, izz) if v > 1e-6]
                            if moments:
                                min_I = min(moments)
                                r_est = math.sqrt(2.0 * min_I / m)
                                radius = max(MIN_RADIUS, min(r_est, MAX_RADIUS))
                    except ValueError:
                        pass

        link_len = link_lengths.get(name, 0.0)
        if radius is not None and radius > 0.0 and link_len > 0.05:
            max_geom_r = 0.4 * link_len
            if radius > max_geom_r:
                radius = max(MIN_RADIUS, max_geom_r)

        if radius is None:
            if link.find("visual") is None and coll is None:
                radius = 0.0
            else:
                radius = DEFAULT_RADIUS

        radii[name] = float(radius)

    for k in list(radii.keys()):
        radii[k] *= SCALE

    return radii


def _build_chain(
    base_link: str,
    ee_link: str,
    link_map: Dict[str, ET.Element],
    joints: List[ET.Element],
) -> Tuple[List[str], List[ET.Element]]:
    """Extract the kinematic chain (links + joints) between base and EE."""
    from collections import deque

    children_by_parent: Dict[str, List[Tuple[str, ET.Element]]] = {}
    for j in joints:
        parent = j.find("parent")
        child = j.find("child")
        if parent is None or child is None:
            continue
        p_name = parent.attrib.get("link")
        c_name = child.attrib.get("link")
        if not p_name or not c_name:
            continue
        children_by_parent.setdefault(p_name, []).append((c_name, j))

    queue: "deque[str]" = deque([base_link])
    parent_link_of: Dict[str, Optional[str]] = {base_link: None}
    parent_joint_of: Dict[str, Optional[ET.Element]] = {base_link: None}

    while queue:
        current = queue.popleft()
        if current == ee_link:
            break
        for child_link, joint in children_by_parent.get(current, []):
            if child_link in parent_link_of:
                continue
            parent_link_of[child_link] = current
            parent_joint_of[child_link] = joint
            queue.append(child_link)

    if ee_link not in parent_link_of:
        raise RuntimeError(
            f"End-effector link '{ee_link}' is not reachable from base link '{base_link}'."
        )

    chain_links: List[str] = []
    chain_joints: List[ET.Element] = []

    cursor: Optional[str] = ee_link
    while cursor is not None:
        chain_links.append(cursor)
        joint = parent_joint_of.get(cursor)
        if joint is not None:
            chain_joints.append(joint)
        cursor = parent_link_of.get(cursor)

    chain_links.reverse()
    chain_joints.reverse()

    if not chain_links or chain_links[0] != base_link:
        chain_links.insert(0, base_link)

    return chain_links, chain_joints


def _parse_float_list(s: Optional[str], dim: int) -> List[float]:
    """Parse a string of floats into a fixed-length list."""
    if not s:
        return [0.0] * dim
    parts = s.split()
    values: List[float] = []
    for idx in range(dim):
        try:
            values.append(float(parts[idx]))
        except (ValueError, IndexError):
            values.append(0.0)
    return values


def _build_link_features(
    root: ET.Element,
    chain_links: List[str],
    link_map: Dict[str, ET.Element],
    link_lengths: Dict[str, float],
    link_radii: Dict[str, float],
) -> List[Dict[str, Any]]:
    """Build feature dict for each link in the kinematic chain."""
    link_features: List[Dict[str, Any]] = []

    for idx, name in enumerate(chain_links):
        link_elem = link_map.get(name)
        if link_elem is None:
            mass = None
            com = [0.0, 0.0, 0.0]
            inertia = None
        else:
            inertial = link_elem.find("inertial")
            if inertial is None:
                mass = None
                com = [0.0, 0.0, 0.0]
                inertia = None
            else:
                mass_elem = inertial.find("mass")
                origin_elem = inertial.find("origin")
                inertia_elem = inertial.find("inertia")

                if mass_elem is not None and "value" in mass_elem.attrib:
                    try:
                        mass = float(mass_elem.attrib["value"])
                    except ValueError:
                        mass = None
                else:
                    mass = None

                if origin_elem is not None:
                    com = _parse_float_list(origin_elem.attrib.get("xyz"), 3)
                else:
                    com = [0.0, 0.0, 0.0]

                if inertia_elem is not None:
                    inertia = {
                        "ixx": float(inertia_elem.attrib.get("ixx", 0.0)),
                        "iyy": float(inertia_elem.attrib.get("iyy", 0.0)),
                        "izz": float(inertia_elem.attrib.get("izz", 0.0)),
                        "ixy": float(inertia_elem.attrib.get("ixy", 0.0)),
                        "ixz": float(inertia_elem.attrib.get("ixz", 0.0)),
                        "iyz": float(inertia_elem.attrib.get("iyz", 0.0)),
                    }
                else:
                    inertia = None

        length_est = float(link_lengths.get(name, 0.0))
        radius_est = float(link_radii.get(name, 0.0))

        link_features.append(
            {
                "name": name,
                "index": idx,
                "is_base": idx == 0,
                "is_ee": idx == len(chain_links) - 1,
                "mass": mass,
                "com": com,
                "inertia": inertia,
                "length_estimate": length_est,
                "radius_estimate": radius_est,
            }
        )

    return link_features


def _build_joint_features(chain_joints: List[ET.Element]) -> List[Dict[str, Any]]:
    """Build feature dict for joints along the kinematic chain."""
    joint_features: List[Dict[str, Any]] = []

    default_rot_lower = -math.pi
    default_rot_upper = math.pi
    default_lin_lower = -1.0
    default_lin_upper = 1.0

    for idx, joint in enumerate(chain_joints):
        name = joint.attrib.get("name", f"joint_{idx}")
        j_type = joint.attrib.get("type", "fixed")

        parent_elem = joint.find("parent")
        child_elem = joint.find("child")
        parent_link = parent_elem.attrib.get("link") if parent_elem is not None else None
        child_link = child_elem.attrib.get("link") if child_elem is not None else None

        origin_elem = joint.find("origin")
        if origin_elem is not None:
            origin_xyz = _parse_float_list(origin_elem.attrib.get("xyz"), 3)
            origin_rpy = _parse_float_list(origin_elem.attrib.get("rpy"), 3)
        else:
            origin_xyz = [0.0, 0.0, 0.0]
            origin_rpy = [0.0, 0.0, 0.0]

        axis_elem = joint.find("axis")
        if axis_elem is not None and "xyz" in axis_elem.attrib:
            axis = _parse_float_list(axis_elem.attrib.get("xyz"), 3)
        else:
            if j_type in ("revolute", "prismatic", "continuous"):
                axis = [1.0, 0.0, 0.0]
            else:
                axis = [0.0, 0.0, 0.0]

        limit_elem = joint.find("limit")
        lower = None
        upper = None
        if limit_elem is not None:
            lower_str = limit_elem.attrib.get("lower")
            upper_str = limit_elem.attrib.get("upper")
            try:
                if lower_str is not None and upper_str is not None:
                    lower = float(lower_str)
                    upper = float(upper_str)
            except ValueError:
                lower = None
                upper = None

        if lower is None or upper is None:
            if j_type == "prismatic":
                lower, upper = default_lin_lower, default_lin_upper
            elif j_type in ("continuous", "revolute"):
                lower, upper = default_rot_lower, default_rot_upper
            else:
                lower, upper = None, None

        is_actuated = j_type in ("revolute", "prismatic", "continuous")

        joint_features.append(
            {
                "name": name,
                "index": idx,
                "type": j_type,
                "parent_link": parent_link,
                "child_link": child_link,
                "axis": axis,
                "origin_xyz": origin_xyz,
                "origin_rpy": origin_rpy,
                "limit_lower": lower,
                "limit_upper": upper,
                "is_actuated": is_actuated,
            }
        )

    return joint_features


def urdf_to_morph_dict(
    urdf_path: Path | str,
    robot_name: Optional[str] = None,
    family: Optional[str] = None,
    source: str = "real",
    base_link: Optional[str] = None,
    ee_link: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert a URDF file into a morphology JSON dictionary.

    The output can be used both for:
    1) physics-based capability computation, and
    2) EGNN-based morphology encoding.
    """
    urdf_path = Path(urdf_path).resolve()
    root = _load_urdf_root(urdf_path)
    link_map, joints = _collect_links_and_joints(root)

    if base_link is None or ee_link is None:
        auto_base, auto_ee = _autodetect_base_and_ee(link_map, joints)
        base_link = base_link or auto_base
        ee_link = ee_link or auto_ee

    chain_links, chain_joints = _build_chain(
        base_link=base_link,
        ee_link=ee_link,
        link_map=link_map,
        joints=joints,
    )

    link_lengths = _estimate_link_lengths(root)
    link_radii = _estimate_link_radii(root, link_lengths)

    link_features = _build_link_features(
        root=root,
        chain_links=chain_links,
        link_map=link_map,
        link_lengths=link_lengths,
        link_radii=link_radii,
    )
    joint_features = _build_joint_features(chain_joints)

    # Actuated DOFs and mapping
    actuated_joint_names: List[str] = [
        jf["name"] for jf in joint_features if jf["is_actuated"]
    ]
    dof = len(actuated_joint_names)

    # Graph edges (for EGNN)
    name_to_index = {lf["name"]: lf["index"] for lf in link_features}
    edges: List[Tuple[int, int]] = []
    for jf in joint_features:
        p = jf["parent_link"]
        c = jf["child_link"]
        if p in name_to_index and c in name_to_index:
            edges.append((name_to_index[p], name_to_index[c]))

    # Base morphology (identity scales)
    morphology = {
        "variant_id": "base",
        "parent_variant_id": None,
        "actuated_joint_names": actuated_joint_names,
        "length_scale": [1.0] * dof,
        "joint_limit_scale": [1.0] * dof,
        "joint_limit_shift": [0.0] * dof,
        "r_vector": None,
        "notes": {
            "generated_from": "urdf",
        },
    }

    morph_dict: Dict[str, Any] = {
        "meta": {
            "robot_name": robot_name or urdf_path.stem,
            "family": family,
            "variant_id": "base",
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
            "edges": [[int(i), int(j)] for i, j in edges],
            "node_type": [
                ("base" if lf["is_base"] else ("ee" if lf["is_ee"] else "link"))
                for lf in link_features
            ],
        },
        "morphology": morphology,
    }

    return morph_dict


def save_morph_json(morph_dict: Dict[str, Any], output_path: Path | str) -> None:
    """Serialize morphology dict to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(morph_dict, f, indent=2, sort_keys=False)


def create_variant_from_base(
    base_morph: Dict[str, Any],
    variant_id: str,
    length_scale_range: Tuple[float, float] = (0.7, 1.3),
    joint_limit_scale_range: Tuple[float, float] = (0.5, 1.0),
    joint_limit_shift_fraction: float = 0.2,
) -> Dict[str, Any]:
    """Create a perturbed morphology variant from a base morphology.

    The base morphology is assumed to have identity scales (1.0, 1.0, 0.0).
    Only the morphology block and meta.variant_id / meta.parent_variant_id
    are changed; kinematic chain information remains identical.
    """
    morph_variant = deepcopy(base_morph)

    base_meta = morph_variant.get("meta", {})
    base_morphology = morph_variant.get("morphology", {})
    actuated_joint_names: List[str] = base_morphology.get(
        "actuated_joint_names", []
    )
    dof = len(actuated_joint_names)

    # Collect original limits in DOF order for shift magnitude
    joint_by_name = {j["name"]: j for j in morph_variant["chain"]["joints"]}
    limit_widths: List[float] = []
    for name in actuated_joint_names:
        j = joint_by_name.get(name)
        if j is None:
            limit_widths.append(0.0)
            continue
        lower = j.get("limit_lower")
        upper = j.get("limit_upper")
        if lower is None or upper is None:
            limit_widths.append(0.0)
        else:
            limit_widths.append(float(upper) - float(lower))

    length_scales: List[float] = []
    joint_limit_scales: List[float] = []
    joint_limit_shifts: List[float] = []

    for width in limit_widths:
        ls = random.uniform(*length_scale_range)
        js = random.uniform(*joint_limit_scale_range)
        if width > 0.0 and joint_limit_shift_fraction > 0.0:
            std = joint_limit_shift_fraction * width
            shift = random.gauss(0.0, std)
        else:
            shift = 0.0

        length_scales.append(ls)
        joint_limit_scales.append(js)
        joint_limit_shifts.append(shift)

    morph_variant["meta"]["variant_id"] = variant_id
    morph_variant["meta"]["parent_variant_id"] = base_meta.get("variant_id", "base")

    morph_variant["morphology"] = {
        "variant_id": variant_id,
        "parent_variant_id": base_meta.get("variant_id", "base"),
        "actuated_joint_names": actuated_joint_names,
        "length_scale": length_scales,
        "joint_limit_scale": joint_limit_scales,
        "joint_limit_shift": joint_limit_shifts,
        "r_vector": None,
        "notes": {
            "generated_from": "perturbation",
            "length_scale_range": list(length_scale_range),
            "joint_limit_scale_range": list(joint_limit_scale_range),
            "joint_limit_shift_fraction": joint_limit_shift_fraction,
        },
    }

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
) -> List[Path]:
    """Convert URDF to morphology JSON and optionally generate perturbed variants.

    Returns a list of all generated JSON paths (including the base).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_morph = urdf_to_morph_dict(
        urdf_path=urdf_path,
        robot_name=robot_name,
        family=family,
        source=source,
        base_link=base_link,
        ee_link=ee_link,
    )

    base_name = base_morph["meta"]["robot_name"]
    base_json_path = output_dir / f"{base_name}_base.json"
    save_morph_json(base_morph, base_json_path)

    json_paths: List[Path] = [base_json_path]

    for idx in range(1, num_variants + 1):
        variant_id = f"{base_name}_v{idx:04d}"
        variant_morph = create_variant_from_base(
            base_morph=base_morph,
            variant_id=variant_id,
            length_scale_range=length_scale_range,
            joint_limit_scale_range=joint_limit_scale_range,
            joint_limit_shift_fraction=joint_limit_shift_fraction,
        )
        variant_json_path = output_dir / f"{variant_id}.json"
        save_morph_json(variant_morph, variant_json_path)
        json_paths.append(variant_json_path)

    return json_paths
