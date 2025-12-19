# mcfp/sim/robot_model.py

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Dict, Optional ,Union, Any
from scipy.spatial.transform import Rotation as R
from pathlib import Path

import numpy as np

from mcfp.sim.fk_ik import create_kinematics_backend, SpecKinematics, KinematicsBackend


class RobotModel:
    """Lightweight robot model built from a URDF file.

    This class is responsible for:
    - Parsing the URDF and discovering actuated joints.
    - Exposing joint metadata (names, limits, count).
    - Providing a stable kinematics API (fk / link_poses / jacobian).
    - Robustly estimating link collision geometries for self-collision checks.

    Parameters
    ----------
    urdf_path:
        Path to the URDF file describing the robot.
    logger:
        Logger instance for reporting status and potential issues.
    """

    def __init__(
        self,
        urdf_path: Optional[Union[str, Path]] = None,
        spec: Optional[Dict[str, Any]] = None,
        logger=None,
        base_link: Optional[str] = None,
        end_effector_link: Optional[str] = None,
    ) -> None:
        if logger is None:
            logger.basicConfig(level=logger.INFO)
            self.logger = logger.getLogger("RobotModel")
        else:
            self.logger = logger

        self.name: str = "unknown_robot"
        self._joint_names: List[str] = []
        self._joint_limits: List[Tuple[float, float]] = []
        self._link_edges: List[Tuple[str, str]] = []  
        self._link_radii: Dict[str, float] = {}       
        
        self.base_link: Optional[str] = base_link
        self.end_effector_link: Optional[str] = end_effector_link
        
        self.urdf_path = Path(urdf_path).resolve() if urdf_path else None
        # self.package_root = self.urdf_path.parent.parent
        self._spec = spec

        # =========================================================
        # A: JSON Spec 
        # =========================================================
        if self._spec is not None:
            self.logger.info("[RobotModel] Initializing from Morphology Spec (JSON Mode).")

            self._init_from_spec(base_link, end_effector_link)

            self._kin_backend: KinematicsBackend = SpecKinematics(
                spec=self._spec,
                joint_names=self._joint_names,
                base_link=self.base_link,
                end_effector_link=self.end_effector_link
            )

        # =========================================================
        # B: URDF 
        # =========================================================
        elif self.urdf_path is not None:
            if not self.urdf_path.is_file():
                raise FileNotFoundError(f"URDF file not found: {self.urdf_path}")
            
            self.logger.info(f"[RobotModel] Loading URDF from: {self.urdf_path}")
            self.name = "urdf_robot" 

            self._parse_urdf_joints()

            self._kin_backend: KinematicsBackend = create_kinematics_backend(
                urdf_path=self.urdf_path,
                joint_names=self._joint_names,
                base_link=self.base_link,
                end_effector_link=self.end_effector_link,
                logger=self.logger,
            )
        else:
            raise ValueError("[RobotModel] Must provide either 'urdf_path' or 'spec'.")

        # End Effector logging
        if self.end_effector_link is None:
            self.logger.info(
                "[RobotModel] end_effector_link not provided; "
                "kinematics backend will use its default end-effector."
            )
        else:
            self.logger.info(f"[RobotModel] End-effector link set to: {self.end_effector_link}")

    def _init_from_spec(self, base_link_hint: Optional[str], ee_link_hint: Optional[str]):
        """from JSON Spec get RobotModel """

        meta = self._spec.get('meta', {})
        self.name = meta.get('robot_name', 'unknown_robot')
        self.base_link = base_link_hint or meta.get('base_link', 'base_link')
        self.end_effector_link = ee_link_hint or meta.get('ee_link', 'Link6')

        morph = self._spec.get('morphology', {})
        ordered_names = morph.get('actuated_joint_names', [])

        chain_joints = {j['name']: j for j in self._spec['chain']['joints']}

        if not ordered_names:
            ordered_names = [j['name'] for j in self._spec['chain']['joints'] if j.get('is_actuated')]
            
        for name in ordered_names:
            if name in chain_joints:
                j_data = chain_joints[name]
                self._joint_names.append(name)
                self._joint_limits.append((j_data['limit_lower'], j_data['limit_upper']))
            else:
                self.logger.warning(f"[RobotModel] Joint {name} listed in morphology but missing in chain.")

        for j in self._spec['chain']['joints']:
            p, c = j.get('parent_link'), j.get('child_link')
            if p and c:
                self._link_edges.append((p, c))

        for link in self._spec['chain']['links']:
            r = link.get('radius_estimate', 0.03)
            self._link_radii[link['name']] = r * 0.9

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def num_joints(self) -> int:
        """Return the number of actuated joints."""
        return len(self._joint_names)

    @property
    def joint_names(self) -> List[str]:
        """Return the list of actuated joint names."""
        return list(self._joint_names)

    @property
    def joint_limits(self) -> np.ndarray:
        """Return joint limits as an array of shape (num_joints, 2)."""
        if not self._joint_limits:
            return np.zeros((0, 2), dtype=float)
        return np.asarray(self._joint_limits, dtype=float)
    
    @property
    def link_edges(self) -> List[Tuple[str, str]]:
        """Return list of (parent_link, child_link) edges from URDF."""
        return list(self._link_edges)

    # ------------------------------------------------------------------
    # Geometry Estimation API (The Robust "Fat Robot" Fix)
    # ------------------------------------------------------------------

    def get_link_radii(self) -> Dict[str, float]:
        """
        Estimate collision radius for each link robustly.
        
        In Spec Mode: Returns pre-loaded radii from JSON.
        In URDF Mode: Parses XML to estimate radii (Legacy logic).
        """
        if self._spec is not None:
            return self._link_radii.copy()
        
        radii = {}
        
        try:
            tree = ET.parse(self.urdf_path)
            root = tree.getroot()
        except Exception as exc:
            self.logger.error(f"[RobotModel] Failed to parse URDF for radii: {exc}")
            return {}

        # 1. Pre-calculate Link Lengths (Distance between joints)
        link_lengths = self._estimate_link_lengths(root)

        # 2. Defaults
        MIN_RADIUS = 0.005  # 0.5cm - minimum physical thickness
        MAX_RADIUS = 0.002  # 3cm - max reasonable thickness for arm links
        DEFAULT_RADIUS = 0.03 # 3cm - safe fallback

        for link in root.findall(".//link"):
            name = link.attrib.get("name")
            if not name:
                continue

            radius: Optional[float] = None
            
            # --- Strategy A: Explicit Collision Primitives (Most Reliable) ---
            coll = link.find("collision")
            if coll is not None:
                geo = coll.find("geometry")
                if geo is not None:
                    # Try cylinder
                    cyl = geo.find("cylinder")
                    if cyl is not None and "radius" in cyl.attrib:
                        try:
                            radius = float(cyl.attrib["radius"])
                        except ValueError:
                            pass
                    # Try sphere
                    if radius is None:
                        sph = geo.find("sphere")
                        if sph is not None and "radius" in sph.attrib:
                            try:
                                radius = float(sph.attrib["radius"])
                            except ValueError:
                                pass
            
            # --- Strategy B: Inertial Estimation with Physics Clamping ---
            if radius is None:
                inertial = link.find("inertial")
                if inertial is not None:
                    mass_elem = inertial.find("mass")
                    inertia_elem = inertial.find("inertia")
                    
                    if mass_elem is not None and "value" in mass_elem.attrib:
                        try:
                            m = float(mass_elem.attrib["value"])
                            
                            # Virtual link (mass near 0)
                            if m <= 1e-4:
                                radius = 0.0
                            elif inertia_elem is not None:
                                ixx = float(inertia_elem.attrib.get("ixx", 0))
                                iyy = float(inertia_elem.attrib.get("iyy", 0))
                                izz = float(inertia_elem.attrib.get("izz", 0))
                                
                                # Use minimal non-zero inertia.
                                # Assumption: Smallest I corresponds to rotation around long axis.
                                moments = [v for v in (ixx, iyy, izz) if v > 1e-6]
                                if moments:
                                    min_I = min(moments)
                                    # Basic estimation: r = sqrt(2*I/m)
                                    r_est = math.sqrt(2.0 * min_I / m)
                                    
                                    # Apply Hard Physical Clamp
                                    radius = max(MIN_RADIUS, min(r_est, MAX_RADIUS))
                        except ValueError:
                            pass

            # --- Strategy C: Geometric Aspect Ratio Constraint ---
            # Even if we got a radius from Inertia, it might be "fat" due to hollow shells.
            # We check against the link length. A link shouldn't be drastically wider than it is long.
            
            link_len = link_lengths.get(name, 0.0)
            
            # If we have a non-zero radius and a meaningful length
            if radius is not None and radius > 0 and link_len > 0.05:
                # Aspect Ratio Limit: Radius shouldn't exceed 40% of length for arm segments.
                # This heuristic prevents spheres appearing where cylinders should be.
                max_geometric_r = 0.4 * link_len
                
                # If calculated radius is way bigger than reasonable for this length
                if radius > max_geometric_r:
                    # Soft clamp: take the geometric limit, but respect Min Radius
                    radius = max(MIN_RADIUS, max_geometric_r)

            # --- Final Fallback ---
            if radius is None:
                # Check for virtual link again via visual/collision presence
                if link.find("visual") is None and link.find("collision") is None:
                    radius = 0.0
                else:
                    radius = DEFAULT_RADIUS
            
            radii[name] = radius

        RADIUS_SCALE = 1 
        for k in radii:
            radii[k] *= RADIUS_SCALE

        # Log for debugging
        debug_str = ", ".join([f"{k}:{v:.3f}" for k, v in radii.items()])
        self.logger.info(f"[RobotModel] Estimated Radii (Scaled x{RADIUS_SCALE}): {debug_str}")
        
        return radii

    def _estimate_link_lengths(self, xml_root: ET.Element) -> Dict[str, float]:
        """Helper: Estimate link length based on distance to child joint.
        
        Returns mapping: link_name -> length (distance to child joint origin).
        """
        link_lengths = {}
        
        # Map link -> child joint origin distance
        # URDF Structure: Joint has <parent link="..."> and <origin xyz="...">
        for joint in xml_root.findall(".//joint"):
            parent_elem = joint.find("parent")
            origin_elem = joint.find("origin")
            
            if parent_elem is not None and origin_elem is not None:
                parent_name = parent_elem.attrib.get("link")
                xyz_str = origin_elem.attrib.get("xyz")
                
                if parent_name and xyz_str:
                    try:
                        xyz = [float(x) for x in xyz_str.split()]
                        dist = math.sqrt(sum(x*x for x in xyz))
                        
                        # A link might have multiple children (branching).
                        # We take the max distance as the "length" of the main body.
                        current_len = link_lengths.get(parent_name, 0.0)
                        link_lengths[parent_name] = max(current_len, dist)
                    except ValueError:
                        pass
        return link_lengths

    # ------------------------------------------------------------------
    # Kinematics API (Delegation)
    # ------------------------------------------------------------------

    def fk(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute end-effector pose (position + quaternion) for q."""
        if q.shape[0] != self.num_joints:
            raise ValueError(
                f"FK expected q with shape ({self.num_joints},), "
                f"got {q.shape}."
            )
        return self._kin_backend.fk_pose(q)
    
    def fk_position(self, q: np.ndarray) -> np.ndarray:
        """Compute end-effector position for q."""
        pos, _ = self.fk(q)
        return pos

    def link_poses(self, q: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Compute poses for all links in the kinematic chain."""
        if q.shape[0] != self.num_joints:
            raise ValueError(
                f"link_poses expected q with shape ({self.num_joints},), "
                f"got {q.shape}."
            )
        return self._kin_backend.link_poses(q)

    def jacobian(self, q: np.ndarray) -> np.ndarray | None:
        """Compute the geometric Jacobian at the end-effector."""
        if self._kin_backend is None:
            self.logger.warning(
                "[RobotModel.jacobian] No kinematics backend available."
            )
            return None

        q_arr = np.asarray(q, dtype=np.float32).reshape(-1)
        if q_arr.shape[0] != self.num_joints:
            raise ValueError(
                f"jacobian expected q with shape ({self.num_joints},), "
                f"got {q_arr.shape}."
            )

        return self._kin_backend.jacobian(q_arr)

    # ------------------------------------------------------------------
    # Internal helpers (Parsing)
    # ------------------------------------------------------------------

    def _parse_urdf_joints(self) -> None:
        """Parse actuated joints and limits from the URDF XML."""
        try:
            tree = ET.parse(self.urdf_path)
            root = tree.getroot()
        except Exception as exc:
            self.logger.error(
                "[RobotModel] Failed to parse URDF XML '%s': %s",
                self.urdf_path,
                exc,
            )
            raise

        # Use './/joint' to be robust to nesting / namespaces
        joint_elements = []
        for joint in root.findall(".//joint"):
            j_type = joint.attrib.get("type", "")
            if j_type in ("revolute", "prismatic", "continuous"):
                joint_elements.append(joint)

        if not joint_elements:
            raise RuntimeError(
                "[RobotModel] URDF contains no revolute/prismatic/"
                "continuous joints. File: "
                f"{self.urdf_path}"
            )

        self._joint_names.clear()
        self._joint_limits.clear()
        self._link_edges.clear()

        # Default ranges: rotational (rad) and prismatic (m)
        default_rot_lower = -np.pi
        default_rot_upper = np.pi
        default_lin_lower = -1.0
        default_lin_upper = 1.0

        for joint in joint_elements:
            name = joint.attrib.get("name")
            j_type = joint.attrib.get("type", "")

            if not name:
                # Skip nameless joints (should technically not happen in valid URDF)
                continue
            
            # 1. Edges for visualization/collision
            parent_elem = joint.find("parent")
            child_elem = joint.find("child")
            if parent_elem is not None and child_elem is not None:
                parent_link = parent_elem.attrib.get("link")
                child_link = child_elem.attrib.get("link")
                if parent_link and child_link:
                    self._link_edges.append((parent_link, child_link))

            # 2. Limits
            limit_elem = joint.find("limit")
            lower, upper = None, None

            if limit_elem is not None:
                lower_str = limit_elem.attrib.get("lower")
                upper_str = limit_elem.attrib.get("upper")

                if lower_str is not None and upper_str is not None:
                    try:
                        lower = float(lower_str)
                        upper = float(upper_str)
                    except ValueError:
                        pass
            
            # Fallback if limits missing or invalid
            if lower is None or upper is None:
                if j_type == "prismatic":
                    lower, upper = default_lin_lower, default_lin_upper
                    self.logger.warning(
                        "[RobotModel] Joint '%s' (prismatic) has incomplete "
                        "limits; using default [%f, %f].",
                        name, lower, upper
                    )
                else:
                    lower, upper = default_rot_lower, default_rot_upper
                    # continuous joints naturally fall here or have huge limits
                    if j_type != "continuous":
                        self.logger.warning(
                            "[RobotModel] Joint '%s' (%s) has incomplete "
                            "limits; using default [%f, %f].",
                            name, j_type, lower, upper
                        )

            self._joint_names.append(name)
            self._joint_limits.append((lower, upper))

    # ------------------------------------------------------------------
    # Visual Assets Parsing (The "Ground Truth" Fix)
    # ------------------------------------------------------------------


    def get_link_visual_data(self, mesh_dir: Path = None) -> Dict[str, Tuple[Path, np.ndarray, np.ndarray]]:
        """
        Parse URDF to extract mesh paths, scaling factors, AND origin transforms.

        Policy:
        1) Prefer <visual> mesh for visualization
        2) Fallback to <collision> mesh if visual mesh missing
        3) If neither exists, skip the link

        Returns:
            Dict[link_name, (mesh_path, scale_array, origin_matrix)]
        """
        mapping: Dict[str, Tuple[Path, np.ndarray, np.ndarray]] = {}

        urdf_dir = self.urdf_path.parent
        # Keep your original assumption, but we’ll add robust fallbacks.
        package_root = urdf_dir.parent

        def _parse_origin(elem: ET.Element) -> np.ndarray:
            T = np.eye(4)
            origin = elem.find("origin")
            if origin is None:
                return T
            xyz_str = origin.attrib.get("xyz", "0 0 0")
            rpy_str = origin.attrib.get("rpy", "0 0 0")
            try:
                xyz = np.fromstring(xyz_str, sep=" ")
                rpy = np.fromstring(rpy_str, sep=" ")
                if xyz.shape == (3,):
                    T[:3, 3] = xyz
                if rpy.shape == (3,):
                    T[:3, :3] = R.from_euler("xyz", rpy).as_matrix()
            except Exception:
                pass
            return T

        def _parse_scale(mesh_elem: ET.Element) -> np.ndarray:
            scale_str = mesh_elem.attrib.get("scale", "1.0 1.0 1.0")
            try:
                s = np.fromstring(scale_str, sep=" ")
                if s.shape != (3,):
                    return np.array([1.0, 1.0, 1.0])
                return s
            except Exception:
                return np.array([1.0, 1.0, 1.0])

        def _resolve_mesh_path(raw_path: str) -> Optional[Path]:
            if not raw_path:
                return None

            # 1) package://pkg_name/rel_path
            if raw_path.startswith("package://"):
                path_no_prefix = raw_path[len("package://"):]
                parts = path_no_prefix.split("/", 1)
                if len(parts) == 2:
                    pkg_name, rel_path = parts[0], parts[1]

                    # Candidate A (your original behavior): package_root/<rel_path>
                    cand_a = package_root / rel_path

                    # Candidate B: package_root/<pkg_name>/<rel_path>
                    # (covers layouts where URDF isn't directly under pkg root)
                    cand_b = package_root / pkg_name / rel_path

                    # Candidate C: relative to URDF directory (sometimes URDF uses package:// but paths are local)
                    cand_c = urdf_dir / rel_path

                    for cand in (cand_a, cand_b, cand_c):
                        if cand.exists():
                            return cand.resolve()

                    # Final fallback: mesh_dir by basename
                    if mesh_dir:
                        cand_d = Path(mesh_dir) / Path(rel_path).name
                        if cand_d.exists():
                            return cand_d.resolve()

                return None

            # 2) Relative path in URDF
            p = Path(raw_path)
            if not p.is_absolute():
                # Candidate A (original): relative to URDF directory
                cand = (urdf_dir / p)
                if cand.exists():
                    return cand.resolve()

                # Candidate B (NEW): relative to package root (common ROS layout)
                # e.g., <pkg_root>/urdf/*.urdf and <pkg_root>/meshes/...
                cand_pkg = (package_root / p)
                if cand_pkg.exists():
                    return cand_pkg.resolve()

                # Candidate C (NEW): one level up from URDF directory
                # covers layouts where package_root is urdf_dir.parent
                cand_parent = (urdf_dir.parent / p)
                if cand_parent.exists():
                    return cand_parent.resolve()

                # Final fallback (original): mesh_dir by basename
                if mesh_dir:
                    cand2 = Path(mesh_dir) / p.name
                    if cand2.exists():
                        return cand2.resolve()
                return None

            # 3) Absolute path
            if p.exists():
                return p.resolve()
            return None
            

        def _extract_mesh_from_node(node: ET.Element) -> Optional[Tuple[Path, np.ndarray, np.ndarray]]:
            """
            node: <visual> or <collision>
            returns (path, scale, T_origin) or None
            """
            geo = node.find("geometry")
            if geo is None:
                return None
            mesh = geo.find("mesh")
            if mesh is None:
                return None
            raw_path = mesh.attrib.get("filename")
            if not raw_path:
                return None
            final_path = _resolve_mesh_path(raw_path)
            if final_path is None:
                return None
            scale_arr = _parse_scale(mesh)
            T = _parse_origin(node)
            return (final_path, scale_arr, T)

        try:
            tree = ET.parse(self.urdf_path)
            root = tree.getroot()
        except Exception as exc:
            self.logger.error(f"[RobotModel] Failed to parse URDF: {exc}")
            return {}

        for link in root.findall(".//link"):
            name = link.attrib.get("name")
            if not name:
                continue

            # Policy: visual first, collision fallback
            picked = None
            visual_node = link.find("visual")
            if visual_node is not None:
                picked = _extract_mesh_from_node(visual_node)

            if picked is None:
                coll_node = link.find("collision")
                if coll_node is not None:
                    picked = _extract_mesh_from_node(coll_node)

            if picked is None:
                # Don’t spam warnings: only warn if link declares a mesh but we can’t resolve it
                # (optional) keep it simple for now:
                continue

            mapping[name] = picked

        self.logger.info(f"[RobotModel] Parsed {len(mapping)} mesh assets (prefer visual, fallback collision).")
        return mapping
