# mcfp/sim/robot_model.py

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

from mcfp.sim.fk_ik import create_kinematics_backend, KinematicsBackend

import numpy as np


class RobotModel:
    """Lightweight robot model built from a URDF file.

    This class is responsible for:
    - Parsing the URDF and discovering actuated joints.
    - Exposing joint metadata (names, limits, count) in a backend-agnostic way.
    - Providing a stable kinematics API (fk / link_poses / jacobian) that can
      later be backed by Pinocchio, PyBullet, MoveIt, etc.

    Parameters
    ----------
    urdf_path:
        Path to the URDF file describing the robot.
    logger:
        Logger instance for reporting status and potential issues.

    Notes
    -----
    This implementation is backend-agnostic. Joint metadata is parsed
    directly from the URDF, while all kinematics queries are delegated
    to a KinematicsBackend instance (currently PyBullet-based).
    Other backends (Pinocchio, MoveIt, etc.) can be plugged in without
    changing the public interface.
    """

    def __init__(
        self,
        urdf_path: Path,
        logger,
        base_link: Optional[str] = None,
        end_effector_link: Optional[str] = None,
    ) -> None:
        
        self.urdf_path = Path(urdf_path).resolve()
        self.logger = logger

        self.base_link: Optional[str] = base_link
        self.end_effector_link: Optional[str] = end_effector_link

        if not self.urdf_path.is_file():
            raise FileNotFoundError(f"URDF file not found: {self.urdf_path}")

        self.logger.info(f"[RobotModel] Loading URDF from: {self.urdf_path}")

        # Joint metadata parsed from URDF
        self._joint_names: List[str] = []
        self._joint_limits: List[Tuple[float, float]] = []

        # store kinematic edges between links parsed from URDF
        self._link_edges: List[Tuple[str, str]] = []

        self._parse_urdf_joints()

        if not self._joint_names:
            self.logger.warning(
                "[RobotModel] No actuated joints found in URDF. "
                "Please check the file or extend the parser."
            )
        else:
            self.logger.info(
                f"[RobotModel] Parsed {len(self._joint_names)} actuated joints "
                f"from URDF."
            )

        # Build kinematics backend using parsed joints
        self._kin_backend: KinematicsBackend = create_kinematics_backend(
            urdf_path=self.urdf_path,
            joint_names=self._joint_names,
            base_link=self.base_link,
            end_effector_link=self.end_effector_link,
            logger=self.logger,
        )

        if self.end_effector_link is None:
            # Backend will fall back to its own default, typically
            # the last link in the kinematic chain.
            self.logger.info(
                "[RobotModel] end_effector_link not provided; "
                "kinematics backend will use its default end-effector."
            )

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
        """Return joint limits as an array of shape (num_joints, 2).

        The first column stores lower bounds, the second column stores
        upper bounds. For continuous or unspecified limits, a default
        range [-pi, pi] is used.
        """
        if not self._joint_limits:
            return np.zeros((0, 2), dtype=float)
        return np.asarray(self._joint_limits, dtype=float)
    
    @property
    def link_edges(self) -> List[Tuple[str, str]]:
        """Return list of (parent_link, child_link) edges from URDF.

        The list follows the order of actuated joints and can be used
        for simple kinematic chain visualisation.
        """
        return list(self._link_edges)


    # ------------------------------------------------------------------
    # Kinematics API stubs
    # ------------------------------------------------------------------

    def fk(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute end-effector pose (position + quaternion) for q.

        Parameters
        ----------
        q:
            Joint configuration as an array of shape (num_joints,).

        Returns
        -------
        position:
            End-effector position as a (3,) NumPy array.
        orientation:
            End-effector orientation as a quaternion (x, y, z, w) with
            shape (4,).
        """
        if q.shape[0] != self.num_joints:
            raise ValueError(
                f"FK expected q with shape ({self.num_joints},), "
                f"got {q.shape}."
            )

        position, orientation = self._kin_backend.fk_pose(q)
        return position, orientation
    
    def fk_position(self, q: np.ndarray) -> np.ndarray:
        """Compute end-effector position for q.

        This is a convenience wrapper around fk(...) that discards
        the orientation component.
        """
        pos, _ = self.fk(q)
        return pos

    def link_poses(self, q: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Compute poses for all links in the kinematic chain.

        Parameters
        ----------
        q:
            Joint configuration as an array of shape (num_joints,).

        Returns
        -------
        poses:
            Mapping from link name to a (position, orientation) tuple.
            Position is a (3,) array and orientation is a quaternion
            (x, y, z, w) with shape (4,).
        """
        if q.shape[0] != self.num_joints:
            raise ValueError(
                f"link_poses expected q with shape ({self.num_joints},), "
                f"got {q.shape}."
            )

        poses = self._kin_backend.link_poses(q)
        return poses


    def jacobian(self, q: np.ndarray) -> np.ndarray | None:
        """Compute the geometric Jacobian at the end-effector.

        This method delegates to the selected kinematics backend.
        The expected shape is (6, num_joints).
        """
        if self._kin_backend is None:
            self.logger.warning(
                "[RobotModel.jacobian] No kinematics backend is available. "
                "Returning None."
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
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_urdf_joints(self) -> None:
        """Parse actuated joints and limits from the URDF XML.

        The parser keeps joints whose type is one of:
        - 'revolute'
        - 'prismatic'
        - 'continuous'

        For each joint:
        - If a <limit> tag is present and both lower/upper are given,
          bounds are read from it.
        - If limits are missing:
            * revolute / continuous: default range [-pi, pi] (rad)
            * prismatic: default range [-1.0, 1.0] (m)
        Any parsing failure raises an exception so that upstream code
        does not silently continue with invalid joint metadata.
        """
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
                raise ValueError(
                    "[RobotModel] Encountered joint without a name in URDF "
                    f"(type='{j_type}'). This is not supported."
                )
            
            parent_elem = joint.find("parent")
            child_elem = joint.find("child")
            if parent_elem is not None and child_elem is not None:
                parent_link = parent_elem.attrib.get("link")
                child_link = child_elem.attrib.get("link")
                if parent_link and child_link:
                    self._link_edges.append((parent_link, child_link))

            limit_elem = joint.find("limit")
            if limit_elem is not None:
                lower_str = limit_elem.attrib.get("lower")
                upper_str = limit_elem.attrib.get("upper")

                if lower_str is not None and upper_str is not None:
                    lower = float(lower_str)
                    upper = float(upper_str)
                else:
                    # Partially specified limits: fall back to defaults
                    if j_type == "prismatic":
                        lower, upper = default_lin_lower, default_lin_upper
                        self.logger.warning(
                            "[RobotModel] Joint '%s' (prismatic) has "
                            "incomplete limits; using default [%f, %f].",
                            name,
                            lower,
                            upper,
                        )
                    else:
                        lower, upper = default_rot_lower, default_rot_upper
                        self.logger.warning(
                            "[RobotModel] Joint '%s' (%s) has incomplete "
                            "limits; using default [%f, %f].",
                            name,
                            j_type,
                            lower,
                            upper,
                        )
            else:
                # No explicit limit → use a default range depending on type
                if j_type == "prismatic":
                    lower, upper = default_lin_lower, default_lin_upper
                    self.logger.warning(
                        "[RobotModel] Joint '%s' (prismatic) has no limits; "
                        "using default [%f, %f].",
                        name,
                        lower,
                        upper,
                    )
                else:
                    lower, upper = default_rot_lower, default_rot_upper

            self._joint_names.append(name)
            self._joint_limits.append((lower, upper))

        self.logger.info(
            "[RobotModel] Parsed %d actuated joints from URDF.",
            len(self._joint_names),
        )

