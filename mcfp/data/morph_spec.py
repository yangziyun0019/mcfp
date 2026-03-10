"""Build HDF5 morphology specifications from URDF and SRDF robot assets.

This module converts robot geometry and kinematic metadata into the morphology format used by MCFP.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import math
import struct
import xml.etree.ElementTree as ET

import numpy as np

try:
    import h5py
except ImportError as exc:  # pragma: no cover
    raise ImportError("h5py is required for morphology spec generation") from exc


@dataclass(frozen=True)
class JointInfo:
    name: str
    joint_type: str
    parent: str
    child: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis: np.ndarray
    limit_lower: Optional[float]
    limit_upper: Optional[float]


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


def _load_xml(path: Path) -> ET.Element:
    tree = ET.parse(path)
    return tree.getroot()


def _collect_links(root: ET.Element) -> Dict[str, ET.Element]:
    link_map: Dict[str, ET.Element] = {}
    for link in root.findall(".//link"):
        name = link.attrib.get("name")
        if name:
            link_map[name] = link
    return link_map


def _collect_joints(root: ET.Element) -> List[ET.Element]:
    joints: List[ET.Element] = []
    for joint in root.findall(".//joint"):
        if joint.attrib.get("name"):
            joints.append(joint)
    return joints


def _parent_child_from_joint(j: ET.Element) -> Tuple[Optional[str], Optional[str]]:
    p = j.find("parent")
    c = j.find("child")
    return (p.attrib.get("link") if p is not None else None,
            c.attrib.get("link") if c is not None else None)


def _detect_base_and_ee(link_map: Dict[str, ET.Element], joints: List[ET.Element]) -> Tuple[str, str]:
    parents = set()
    children = set()
    for j in joints:
        p, c = _parent_child_from_joint(j)
        if p:
            parents.add(p)
        if c:
            children.add(c)
    bases = [l for l in parents if l not in children]
    base = bases[0] if bases else next(iter(link_map.keys()))
    ees = [l for l in children if l not in parents]
    ee = ees[0] if ees else next(reversed(list(link_map.keys())))
    return base, ee


def _build_parent_of(joints: List[ET.Element]) -> Dict[str, str]:
    parent_of: Dict[str, str] = {}
    for j in joints:
        p, c = _parent_child_from_joint(j)
        if p and c:
            parent_of[c] = p
    return parent_of


def _path_between(parent_of: Dict[str, str], base: str, ee: str) -> List[str]:
    chain = [ee]
    cur = ee
    while cur != base and cur in parent_of:
        cur = parent_of[cur]
        chain.append(cur)
    chain.reverse()
    if not chain or chain[0] != base:
        return [base]
    return chain


def _extract_main_chain(
    root: ET.Element,
    link_map: Dict[str, ET.Element],
    joints: List[ET.Element],
    base_link: str,
    ee_link: str,
) -> Tuple[List[str], List[ET.Element]]:
    parent_of = _build_parent_of(joints)
    chain_links = _path_between(parent_of, base_link, ee_link)
    chain_set = set(chain_links)
    chain_joints: List[ET.Element] = []
    for j in joints:
        p, c = _parent_child_from_joint(j)
        if p in chain_set and c in chain_set:
            chain_joints.append(j)
    link_index = {name: i for i, name in enumerate(chain_links)}
    chain_joints.sort(key=lambda jj: link_index.get(_parent_child_from_joint(jj)[0] or "", 0))
    return chain_links, chain_joints


def _get_joint_origin_xyz_rpy(j: ET.Element) -> Tuple[np.ndarray, np.ndarray]:
    origin = j.find("origin")
    if origin is None:
        return np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
    xyz = origin.attrib.get("xyz", "0 0 0").split()
    rpy = origin.attrib.get("rpy", "0 0 0").split()
    xyz_f = [float(x) for x in xyz[:3]] + [0.0] * max(0, 3 - len(xyz))
    rpy_f = [float(x) for x in rpy[:3]] + [0.0] * max(0, 3 - len(rpy))
    return np.asarray(xyz_f[:3], dtype=np.float64), np.asarray(rpy_f[:3], dtype=np.float64)


def _get_joint_axis(j: ET.Element) -> np.ndarray:
    axis = j.find("axis")
    if axis is None:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    xyz = axis.attrib.get("xyz", "0 0 1").split()
    vals = [float(x) for x in xyz[:3]] + [0.0] * max(0, 3 - len(xyz))
    arr = np.asarray(vals[:3], dtype=np.float64)
    norm = np.linalg.norm(arr)
    if norm < 1e-9:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return arr / norm


def _get_joint_limit(j: ET.Element) -> Tuple[Optional[float], Optional[float]]:
    limit = j.find("limit")
    if limit is None:
        return None, None
    lo = limit.attrib.get("lower")
    hi = limit.attrib.get("upper")
    if lo is None or hi is None:
        return None, None
    return float(lo), float(hi)


def _rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return Rz @ Ry @ Rx


def compute_l_ref_from_urdf(
    urdf_path: Path,
    base_link: Optional[str] = None,
    ee_link: Optional[str] = None,
) -> float:
    root = _load_xml(urdf_path)
    link_map = _collect_links(root)
    joints = _collect_joints(root)

    if base_link is None or ee_link is None:
        base_auto, ee_auto = _detect_base_and_ee(link_map, joints)
        base_link = base_link or base_auto
        ee_link = ee_link or ee_auto

    _, chain_joints = _extract_main_chain(root, link_map, joints, base_link, ee_link)
    l_ref = 0.0
    for j in chain_joints:
        xyz, _ = _get_joint_origin_xyz_rpy(j)
        l_ref += float(np.linalg.norm(xyz))
        if j.attrib.get("type") == "prismatic":
            lo, hi = _get_joint_limit(j)
            if lo is not None and hi is not None:
                l_ref += float(abs(hi - lo))
    if l_ref <= 0.0:
        l_ref = 1.0
    return float(l_ref)


def _parse_collision_geometries(link_elem: ET.Element) -> List[Dict[str, Any]]:
    geoms: List[Dict[str, Any]] = []
    for col in link_elem.findall("collision"):
        origin_xyz, origin_rpy = _get_joint_origin_xyz_rpy(col)
        geom = col.find("geometry")
        if geom is None:
            continue
        mesh = geom.find("mesh")
        if mesh is not None:
            filename = mesh.attrib.get("filename")
            scale_str = mesh.attrib.get("scale", "1 1 1")
            scale = [float(x) for x in scale_str.split()[:3]]
            geoms.append({
                "type": "mesh",
                "filename": filename,
                "scale": np.asarray(scale, dtype=np.float64),
                "origin_xyz": origin_xyz,
                "origin_rpy": origin_rpy,
            })
            continue
        box = geom.find("box")
        if box is not None:
            size = [float(x) for x in box.attrib.get("size", "0 0 0").split()[:3]]
            geoms.append({
                "type": "box",
                "size": np.asarray(size, dtype=np.float64),
                "origin_xyz": origin_xyz,
                "origin_rpy": origin_rpy,
            })
            continue
        cyl = geom.find("cylinder")
        if cyl is not None:
            radius = float(cyl.attrib.get("radius", 0.0))
            length = float(cyl.attrib.get("length", 0.0))
            geoms.append({
                "type": "cylinder",
                "radius": radius,
                "length": length,
                "origin_xyz": origin_xyz,
                "origin_rpy": origin_rpy,
            })
            continue
        sph = geom.find("sphere")
        if sph is not None:
            radius = float(sph.attrib.get("radius", 0.0))
            geoms.append({
                "type": "sphere",
                "radius": radius,
                "origin_xyz": origin_xyz,
                "origin_rpy": origin_rpy,
            })
            continue
    return geoms


def _resolve_mesh_path(filename: str, mesh_root: Path, package_map: Dict[str, Path]) -> Path:
    if filename.startswith("package://"):
        rel = filename[len("package://"):]
        parts = rel.split("/", 1)
        if len(parts) == 2:
            pkg, rest = parts
            base = package_map.get(pkg, mesh_root)
            return (Path(base) / rest).resolve()
        return (mesh_root / rel).resolve()
    if filename.startswith("file://"):
        return Path(filename[len("file://"):]).resolve()
    return (mesh_root / filename).resolve()


def _load_stl_triangles(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError(f"STL file too small: {path}")

    header = data[:80]
    num_tri = struct.unpack("<I", data[80:84])[0]
    expected = 84 + num_tri * 50
    if expected == len(data):
        verts = np.zeros((num_tri, 3, 3), dtype=np.float32)
        norms = np.zeros((num_tri, 3), dtype=np.float32)
        offset = 84
        for i in range(num_tri):
            chunk = data[offset:offset + 50]
            n = struct.unpack("<3f", chunk[0:12])
            v1 = struct.unpack("<3f", chunk[12:24])
            v2 = struct.unpack("<3f", chunk[24:36])
            v3 = struct.unpack("<3f", chunk[36:48])
            norms[i] = np.array(n, dtype=np.float32)
            verts[i, 0] = np.array(v1, dtype=np.float32)
            verts[i, 1] = np.array(v2, dtype=np.float32)
            verts[i, 2] = np.array(v3, dtype=np.float32)
            offset += 50
        return verts, norms

    text = data.decode(errors="ignore")
    verts_list: List[List[float]] = []
    norms_list: List[List[float]] = []
    cur_norm = [0.0, 0.0, 1.0]
    cur_verts: List[List[float]] = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("facet normal"):
            parts = line.split()
            if len(parts) >= 5:
                cur_norm = [float(parts[2]), float(parts[3]), float(parts[4])]
        elif line.startswith("vertex"):
            parts = line.split()
            if len(parts) >= 4:
                cur_verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif line.startswith("endfacet"):
            if len(cur_verts) >= 3:
                verts_list.append(cur_verts[:3])
                norms_list.append(cur_norm)
            cur_verts = []
    if not verts_list:
        raise ValueError(f"Failed to parse STL: {path}")
    verts = np.asarray(verts_list, dtype=np.float32)
    norms = np.asarray(norms_list, dtype=np.float32)
    return verts, norms


def _triangle_areas(verts: np.ndarray) -> np.ndarray:
    v1 = verts[:, 0]
    v2 = verts[:, 1]
    v3 = verts[:, 2]
    area = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1), axis=1)
    return area


def _sample_triangles(
    verts: np.ndarray,
    norms: np.ndarray,
    num: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    areas = _triangle_areas(verts)
    if np.sum(areas) <= 0:
        raise ValueError("Mesh has zero surface area")
    probs = areas / np.sum(areas)
    idx = rng.choice(len(verts), size=int(num), replace=True, p=probs)
    v = verts[idx]
    n = norms[idx]

    r1 = rng.random((num, 1))
    r2 = rng.random((num, 1))
    sqrt_r1 = np.sqrt(r1)
    u = 1.0 - sqrt_r1
    v_w = sqrt_r1 * (1.0 - r2)
    w = sqrt_r1 * r2
    pts = u * v[:, 0] + v_w * v[:, 1] + w * v[:, 2]
    n_norm = np.linalg.norm(n, axis=1, keepdims=True)
    n_norm = np.where(n_norm < 1e-9, 1.0, n_norm)
    n = n / n_norm
    return pts.astype(np.float32), n.astype(np.float32)


def _sample_box(size: np.ndarray, num: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    sx, sy, sz = size
    areas = np.array([
        sy * sz, sy * sz,
        sx * sz, sx * sz,
        sx * sy, sx * sy,
    ], dtype=np.float64)
    probs = areas / np.sum(areas)
    face_idx = rng.choice(6, size=int(num), replace=True, p=probs)
    pts = np.zeros((num, 3), dtype=np.float32)
    nrm = np.zeros((num, 3), dtype=np.float32)
    for i in range(6):
        mask = face_idx == i
        if not np.any(mask):
            continue
        count = int(np.sum(mask))
        u = rng.random((count, 1)) - 0.5
        v = rng.random((count, 1)) - 0.5
        if i == 0:
            pts[mask] = np.hstack([np.full((count, 1), sx / 2), u * sy, v * sz])
            nrm[mask] = np.array([1.0, 0.0, 0.0])
        elif i == 1:
            pts[mask] = np.hstack([np.full((count, 1), -sx / 2), u * sy, v * sz])
            nrm[mask] = np.array([-1.0, 0.0, 0.0])
        elif i == 2:
            pts[mask] = np.hstack([u * sx, np.full((count, 1), sy / 2), v * sz])
            nrm[mask] = np.array([0.0, 1.0, 0.0])
        elif i == 3:
            pts[mask] = np.hstack([u * sx, np.full((count, 1), -sy / 2), v * sz])
            nrm[mask] = np.array([0.0, -1.0, 0.0])
        elif i == 4:
            pts[mask] = np.hstack([u * sx, v * sy, np.full((count, 1), sz / 2)])
            nrm[mask] = np.array([0.0, 0.0, 1.0])
        else:
            pts[mask] = np.hstack([u * sx, v * sy, np.full((count, 1), -sz / 2)])
            nrm[mask] = np.array([0.0, 0.0, -1.0])
    return pts, nrm


def _sample_sphere(radius: float, num: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    u = rng.random(num)
    v = rng.random(num)
    theta = 2 * math.pi * u
    phi = np.arccos(2 * v - 1)
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    pts = np.stack([x, y, z], axis=1).astype(np.float32)
    nrm = pts / np.maximum(np.linalg.norm(pts, axis=1, keepdims=True), 1e-9)
    return pts, nrm.astype(np.float32)


def _sample_cylinder(radius: float, length: float, num: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    side_area = 2 * math.pi * radius * length
    cap_area = math.pi * radius * radius
    areas = np.array([side_area, cap_area, cap_area], dtype=np.float64)
    probs = areas / np.sum(areas)
    face_idx = rng.choice(3, size=int(num), replace=True, p=probs)
    pts = np.zeros((num, 3), dtype=np.float32)
    nrm = np.zeros((num, 3), dtype=np.float32)
    mask = face_idx == 0
    if np.any(mask):
        count = int(np.sum(mask))
        theta = 2 * math.pi * rng.random(count)
        z = (rng.random(count) - 0.5) * length
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        pts[mask] = np.stack([x, y, z], axis=1)
        nrm[mask] = np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=1)
    mask = face_idx == 1
    if np.any(mask):
        count = int(np.sum(mask))
        r = radius * np.sqrt(rng.random(count))
        theta = 2 * math.pi * rng.random(count)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.full_like(x, length / 2)
        pts[mask] = np.stack([x, y, z], axis=1)
        nrm[mask] = np.array([0.0, 0.0, 1.0])
    mask = face_idx == 2
    if np.any(mask):
        count = int(np.sum(mask))
        r = radius * np.sqrt(rng.random(count))
        theta = 2 * math.pi * rng.random(count)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.full_like(x, -length / 2)
        pts[mask] = np.stack([x, y, z], axis=1)
        nrm[mask] = np.array([0.0, 0.0, -1.0])
    return pts, nrm


def _geometry_area(geom: Dict[str, Any]) -> float:
    t = geom["type"]
    if t == "mesh":
        path = geom.get("_mesh_path")
        if path is None:
            return 0.0
        verts, _ = _load_stl_triangles(path)
        return float(np.sum(_triangle_areas(verts)))
    if t == "box":
        sx, sy, sz = geom["size"]
        return float(2 * (sx * sy + sy * sz + sx * sz))
    if t == "cylinder":
        r = float(geom["radius"])
        L = float(geom["length"])
        return float(2 * math.pi * r * L + 2 * math.pi * r * r)
    if t == "sphere":
        r = float(geom["radius"])
        return float(4 * math.pi * r * r)
    return 0.0


def _sample_geometry_points(
    geom: Dict[str, Any],
    num: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    t = geom["type"]
    if num <= 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    if t == "mesh":
        verts, norms = _load_stl_triangles(geom["_mesh_path"])
        pts, nrm = _sample_triangles(verts * geom["scale"], norms, num, rng)
    elif t == "box":
        pts, nrm = _sample_box(geom["size"], num, rng)
    elif t == "cylinder":
        pts, nrm = _sample_cylinder(float(geom["radius"]), float(geom["length"]), num, rng)
    elif t == "sphere":
        pts, nrm = _sample_sphere(float(geom["radius"]), num, rng)
    else:
        pts = np.zeros((0, 3), dtype=np.float32)
        nrm = np.zeros((0, 3), dtype=np.float32)

    R = _rpy_to_matrix(geom["origin_rpy"])
    t_vec = geom["origin_xyz"].reshape(3, 1)
    pts = (R @ pts.T).T + t_vec.T
    nrm = (R @ nrm.T).T
    return pts.astype(np.float32), nrm.astype(np.float32)


def _build_link_depths(links: List[str], joints: List[JointInfo], base_link: str) -> Dict[str, int]:
    children: Dict[str, List[str]] = {}
    for j in joints:
        children.setdefault(j.parent, []).append(j.child)
    depth: Dict[str, int] = {base_link: 0}
    stack = [base_link]
    while stack:
        cur = stack.pop()
        for ch in children.get(cur, []):
            depth[ch] = depth[cur] + 1
            stack.append(ch)
    for name in links:
        depth.setdefault(name, 0)
    return depth


def _parse_srdf_acm(srdf_path: Optional[Path]) -> List[Tuple[str, str]]:
    if srdf_path is None or not srdf_path.is_file():
        return []
    root = _load_xml(srdf_path)
    pairs: List[Tuple[str, str]] = []
    for elem in root.findall(".//disable_collisions"):
        l1 = elem.attrib.get("link1")
        l2 = elem.attrib.get("link2")
        if l1 and l2:
            pairs.append((l1, l2))
    return pairs


def build_morphology_spec(
    cfg: Any,
    logger,
    *,
    repo_root: Path,
) -> Path:
    urdf_path = Path(str(_get_path(cfg, "paths.urdf_path"))).expanduser()
    srdf_path = _get_path(cfg, "paths.srdf_path", None)
    mesh_root = _get_path(cfg, "paths.mesh_root", None)

    urdf_path = urdf_path if urdf_path.is_absolute() else (repo_root / urdf_path).resolve()
    srdf_path = Path(srdf_path).expanduser() if srdf_path else None
    if srdf_path is not None and not srdf_path.is_absolute():
        srdf_path = (repo_root / srdf_path).resolve()

    mesh_root = Path(mesh_root).expanduser() if mesh_root else urdf_path.parent
    if not mesh_root.is_absolute():
        mesh_root = (repo_root / mesh_root).resolve()

    out_path = Path(str(_get_path(cfg, "paths.morph_out"))).expanduser()
    if not out_path.is_absolute():
        out_path = (repo_root / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    robot_name = str(_get_path(cfg, "robot.name", urdf_path.stem))
    variant_id = str(_get_path(cfg, "robot.variant", "base"))
    base_link = _get_path(cfg, "robot.base_link", None)
    ee_link = _get_path(cfg, "robot.ee_link", None)

    root = _load_xml(urdf_path)
    link_map = _collect_links(root)
    joint_elems = _collect_joints(root)

    if base_link is None or ee_link is None:
        base_auto, ee_auto = _detect_base_and_ee(link_map, joint_elems)
        base_link = base_link or base_auto
        ee_link = ee_link or ee_auto

    joints: List[JointInfo] = []
    for j in joint_elems:
        name = j.attrib.get("name", "")
        jtype = j.attrib.get("type", "fixed")
        parent, child = _parent_child_from_joint(j)
        if not parent or not child:
            continue
        xyz, rpy = _get_joint_origin_xyz_rpy(j)
        axis = _get_joint_axis(j)
        lo, hi = _get_joint_limit(j)
        joints.append(JointInfo(
            name=name,
            joint_type=jtype,
            parent=parent,
            child=child,
            origin_xyz=xyz,
            origin_rpy=rpy,
            axis=axis,
            limit_lower=lo,
            limit_upper=hi,
        ))

    link_names = list(link_map.keys())
    link_index = {name: i for i, name in enumerate(link_names)}

    depth = _build_link_depths(link_names, joints, base_link)

    l_ref = compute_l_ref_from_urdf(urdf_path, base_link=base_link, ee_link=ee_link)

    acm_pairs = _parse_srdf_acm(srdf_path)
    acm_idx: List[Tuple[int, int]] = []
    for l1, l2 in acm_pairs:
        if l1 in link_index and l2 in link_index:
            acm_idx.append((link_index[l1], link_index[l2]))

    package_map: Dict[str, Path] = {}
    pkg_cfg = _get_path(cfg, "paths.package_map", {})
    if isinstance(pkg_cfg, dict):
        for k, v in pkg_cfg.items():
            package_map[str(k)] = (repo_root / v).resolve() if not Path(v).is_absolute() else Path(v).resolve()

    num_points = int(_get_path(cfg, "morphology.pointcloud.num_points", 1024))
    seed = int(_get_path(cfg, "morphology.pointcloud.seed", 42))
    rng = np.random.default_rng(seed)

    points = np.zeros((len(link_names), num_points, 6), dtype=np.float32)
    has_geom = np.zeros((len(link_names),), dtype=np.uint8)
    bbox_min = np.zeros((len(link_names), 3), dtype=np.float32)
    bbox_max = np.zeros((len(link_names), 3), dtype=np.float32)

    for idx, name in enumerate(link_names):
        link_elem = link_map[name]
        geoms = _parse_collision_geometries(link_elem)
        for g in geoms:
            if g["type"] == "mesh" and g.get("filename"):
                g["_mesh_path"] = _resolve_mesh_path(g["filename"], mesh_root, package_map)
        if not geoms:
            has_geom[idx] = 0
            bbox_min[idx] = 0
            bbox_max[idx] = 0
            continue

        areas = np.array([_geometry_area(g) for g in geoms], dtype=np.float64)
        total_area = float(np.sum(areas))
        if total_area <= 0:
            has_geom[idx] = 0
            bbox_min[idx] = 0
            bbox_max[idx] = 0
            continue

        alloc = np.floor(num_points * areas / total_area).astype(int)
        for i in range(len(alloc)):
            if areas[i] > 0 and alloc[i] == 0:
                alloc[i] = 1
        diff = num_points - int(np.sum(alloc))
        if diff > 0:
            extra_idx = rng.choice(len(alloc), size=diff, replace=True, p=areas / total_area)
            for i in extra_idx:
                alloc[i] += 1
        elif diff < 0:
            for _ in range(-diff):
                j = int(np.argmax(alloc))
                alloc[j] = max(alloc[j] - 1, 0)

        pts_list: List[np.ndarray] = []
        nrm_list: List[np.ndarray] = []
        for g, n in zip(geoms, alloc):
            if n <= 0:
                continue
            p_i, n_i = _sample_geometry_points(g, int(n), rng)
            pts_list.append(p_i)
            nrm_list.append(n_i)
        if pts_list:
            pts = np.vstack(pts_list)
            nrm = np.vstack(nrm_list)
            if pts.shape[0] < num_points:
                pad = num_points - pts.shape[0]
                pts = np.vstack([pts, np.zeros((pad, 3), dtype=np.float32)])
                nrm = np.vstack([nrm, np.zeros((pad, 3), dtype=np.float32)])
            elif pts.shape[0] > num_points:
                sel = rng.choice(pts.shape[0], size=num_points, replace=False)
                pts = pts[sel]
                nrm = nrm[sel]
            points[idx] = np.hstack([pts, nrm]).astype(np.float32)
            has_geom[idx] = 1
            bbox_min[idx] = np.min(pts, axis=0)
            bbox_max[idx] = np.max(pts, axis=0)
        else:
            has_geom[idx] = 0
            bbox_min[idx] = 0
            bbox_max[idx] = 0

    mass = np.zeros((len(link_names),), dtype=np.float32)
    inertia = np.zeros((len(link_names), 6), dtype=np.float32)
    for idx, name in enumerate(link_names):
        link_elem = link_map[name]
        inertial = link_elem.find("inertial")
        if inertial is None:
            mass[idx] = 0.0
            inertia[idx] = 0.0
            continue
        mass_elem = inertial.find("mass")
        if mass_elem is not None and mass_elem.attrib.get("value") is not None:
            mass[idx] = float(mass_elem.attrib.get("value"))
        else:
            mass[idx] = 0.0
        inertia_elem = inertial.find("inertia")
        if inertia_elem is not None:
            vals = [
                inertia_elem.attrib.get("ixx"),
                inertia_elem.attrib.get("iyy"),
                inertia_elem.attrib.get("izz"),
                inertia_elem.attrib.get("ixy"),
                inertia_elem.attrib.get("ixz"),
                inertia_elem.attrib.get("iyz"),
            ]
            if all(v is not None for v in vals):
                inertia[idx] = np.asarray([float(v) for v in vals], dtype=np.float32)
            else:
                inertia[idx] = 0.0
        else:
            inertia[idx] = 0.0

    with h5py.File(out_path, "w") as out:
        meta = {
            "robot_name": robot_name,
            "variant_id": variant_id,
            "base_link": base_link,
            "ee_link": ee_link,
            "urdf_path": str(urdf_path),
            "srdf_path": str(srdf_path) if srdf_path else None,
            "mesh_root": str(mesh_root),
            "l_ref": float(l_ref),
        }
        dt_meta = h5py.string_dtype("utf-8")
        out.create_dataset("meta/json", data=np.asarray(json.dumps(meta, ensure_ascii=True), dtype=dt_meta))

        grp_links = out.create_group("links")
        dt = h5py.string_dtype("utf-8")
        grp_links.create_dataset("name", data=np.asarray(link_names, dtype=object), dtype=dt)
        grp_links.create_dataset("depth", data=np.asarray([depth[n] for n in link_names], dtype=np.int32))
        grp_links.create_dataset("index", data=np.arange(len(link_names), dtype=np.int32))

        grp_joints = out.create_group("joints")
        grp_joints.create_dataset("name", data=np.asarray([j.name for j in joints], dtype=object), dtype=dt)
        grp_joints.create_dataset("type", data=np.asarray([j.joint_type for j in joints], dtype=object), dtype=dt)
        grp_joints.create_dataset("parent", data=np.asarray([j.parent for j in joints], dtype=object), dtype=dt)
        grp_joints.create_dataset("child", data=np.asarray([j.child for j in joints], dtype=object), dtype=dt)
        grp_joints.create_dataset("origin_xyz", data=np.asarray([j.origin_xyz for j in joints], dtype=np.float32))
        grp_joints.create_dataset("origin_rpy", data=np.asarray([j.origin_rpy for j in joints], dtype=np.float32))
        grp_joints.create_dataset("axis", data=np.asarray([j.axis for j in joints], dtype=np.float32))
        limit_lower = [j.limit_lower if j.limit_lower is not None else 0.0 for j in joints]
        limit_upper = [j.limit_upper if j.limit_upper is not None else 0.0 for j in joints]
        grp_joints.create_dataset("limit_lower", data=np.asarray(limit_lower, dtype=np.float32))
        grp_joints.create_dataset("limit_upper", data=np.asarray(limit_upper, dtype=np.float32))

        grp_col = out.create_group("collision")
        grp_col.create_dataset("points", data=points, compression="gzip", shuffle=True)
        grp_col.create_dataset("has_geometry", data=has_geom)
        grp_col.create_dataset("bbox_min", data=bbox_min)
        grp_col.create_dataset("bbox_max", data=bbox_max)

        grp_acm = out.create_group("acm")
        if acm_idx:
            grp_acm.create_dataset("allowed_pairs", data=np.asarray(acm_idx, dtype=np.int32))
        else:
            grp_acm.create_dataset("allowed_pairs", data=np.zeros((0, 2), dtype=np.int32))

        grp_inertial = out.create_group("inertial")
        grp_inertial.create_dataset("mass", data=mass)
        grp_inertial.create_dataset("inertia", data=inertia)

    logger.info(f"[morph_spec] wrote {out_path}")
    return out_path
