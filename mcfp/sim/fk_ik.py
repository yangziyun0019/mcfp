# mcfp/sim/fk_ik.py
"""Kinematics backends (FK/IK) for MCFP.

This module provides a thin abstraction layer over concrete kinematics
libraries. For now, a PyBullet-based backend is implemented, which is
sufficient for single-configuration FK of generic articulated robots.

RobotModel should only interact with the abstract KinematicsBackend
interface and never call PyBullet directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Protocol, Tuple, Dict

import numpy as np
import pybullet as p
import pybullet_data


class KinematicsBackend(Protocol):
    """Abstract interface for kinematics backends."""

    def fk_pose(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return end-effector pose (position + quaternion) for q.

        Parameters
        ----------
        q:
            Joint configuration as a 1D array of shape (num_joints,).

        Returns
        -------
        position:
            End-effector position as a (3,) array.
        orientation:
            End-effector orientation as a quaternion (x, y, z, w).
        """
        ...

    def fk_position(self, q: np.ndarray) -> np.ndarray:
        """Return end-effector position for configuration q.

        This convenience method may simply call fk_pose and discard
        the orientation part.
        """
        ...

    def link_poses(self, q: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Return poses of all links for configuration q.

        Parameters
        ----------
        q:
            Joint configuration as a 1D array of shape (num_joints,).

        Returns
        -------
        poses:
            Mapping from link name to a (position, orientation) tuple,
            where position is a (3,) array and orientation is a
            quaternion (x, y, z, w) with shape (4,).
        """
        ...
    
    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """Return the geometric Jacobian at the end-effector.

        Parameters
        ----------
        q:
            Joint configuration as a 1D array of shape (num_joints,).

        Returns
        -------
        jacobian:
            Geometric Jacobian as a NumPy array of shape
            (6, num_joints), mapping joint velocities to spatial
            velocity (linear + angular) of the end-effector in the
            world / base frame.
        """
        ...


class PyBulletKinematics:
    """PyBullet-based implementation of KinematicsBackend.

    This backend loads the URDF into a DIRECT PyBullet client and exposes
    FK for the end-effector. It is designed to work with arbitrary-DOF
    manipulators as long as joint names in the URDF match those parsed
    by RobotModel.

    Parameters
    ----------
    urdf_path:
        Path to the robot URDF file.
    joint_names:
        List of actuated joint names. The order defines the index
        convention for configuration vectors q.
    base_link:
        Name of the base link. Currently not used explicitly and PyBullet
        will rely on the root of the URDF. The argument is kept for
        future extensions.
    end_effector_link:
        Name of the end-effector link. If None, the last link in the
        kinematic chain is used as a fallback.
    logger:
        Logger instance for reporting backend status.
    """

    def __init__(
        self,
        urdf_path: Path,
        joint_names: List[str],
        base_link: Optional[str],
        end_effector_link: Optional[str],
        logger,
    ) -> None:
        self._logger = logger
        self._urdf_path = Path(urdf_path).resolve()
        self._joint_names = list(joint_names)
        self._base_link = base_link
        self._ee_link_name = end_effector_link

        if not self._urdf_path.is_file():
            raise FileNotFoundError(f"URDF file not found: {self._urdf_path}")

        # Initialize a DIRECT client so no GUI is opened.
        self._client_id = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Load robot model.
        self._robot_id = p.loadURDF(
            str(self._urdf_path),
            useFixedBase=True,
            physicsClientId=self._client_id,
        )

        self._logger.info(
            "[PyBulletKinematics] URDF loaded into PyBullet "
            f"(client_id={self._client_id}, robot_id={self._robot_id})."
        )

        # Build joint and link index mappings.
        self._joint_indices = self._build_joint_index_map(self._joint_names)
        self._ee_link_index = self._resolve_end_effector_link(self._ee_link_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fk_pose(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute end-effector pose (position + quaternion) for q."""
        if q.ndim != 1:
            raise ValueError(f"Expected 1D q vector, got shape {q.shape}.")

        if q.shape[0] != len(self._joint_indices):
            raise ValueError(
                "Length of q does not match number of actuated joints: "
                f"{q.shape[0]} vs {len(self._joint_indices)}"
            )

        # Reset joint states according to q
        for q_val, joint_idx in zip(q, self._joint_indices):
            p.resetJointState(
                self._robot_id,
                joint_idx,
                float(q_val),
                physicsClientId=self._client_id,
            )

        # Get link state of the end-effector
        state = p.getLinkState(
            self._robot_id,
            self._ee_link_index,
            computeForwardKinematics=True,
            physicsClientId=self._client_id,
        )
        # worldLinkFramePosition, worldLinkFrameOrientation (x, y, z, w)
        position = np.asarray(state[4], dtype=float)
        orientation = np.asarray(state[5], dtype=float)
        return position, orientation

    def fk_position(self, q: np.ndarray) -> np.ndarray:
        """Compute end-effector position for configuration q."""
        pos, _ = self.fk_pose(q)
        return pos
    
    def link_poses(self, q: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Compute poses for all links in the kinematic chain.

        The returned dictionary maps each link name to its pose in the
        world / base frame for configuration q.
        """
        if q.ndim != 1:
            raise ValueError(f"Expected 1D q vector, got shape {q.shape}.")

        if q.shape[0] != len(self._joint_indices):
            raise ValueError(
                "Length of q does not match number of actuated joints: "
                f"{q.shape[0]} vs {len(self._joint_indices)}"
            )

        # Set joint states according to q
        for q_val, joint_idx in zip(q, self._joint_indices):
            p.resetJointState(
                self._robot_id,
                joint_idx,
                float(q_val),
                physicsClientId=self._client_id,
            )

        poses: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        num_joints = p.getNumJoints(
            self._robot_id,
            physicsClientId=self._client_id,
        )

        # Collect pose for each link attached to a joint
        for idx in range(num_joints):
            info = p.getJointInfo(
                self._robot_id,
                idx,
                physicsClientId=self._client_id,
            )
            link_name = info[12].decode("utf-8")

            state = p.getLinkState(
                self._robot_id,
                idx,
                computeForwardKinematics=True,
                physicsClientId=self._client_id,
            )
            position = np.asarray(state[4], dtype=float)
            orientation = np.asarray(state[5], dtype=float)

            poses[link_name] = (position, orientation)

        # Optionally record the base link pose as identity if a name is given
        if self._base_link is not None and self._base_link not in poses:
            poses[self._base_link] = (
                np.zeros(3, dtype=float),
                np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
            )

        return poses
    
    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """Compute the geometric Jacobian at the end-effector for q."""
        if q.ndim != 1:
            raise ValueError(f"Expected 1D q vector, got shape {q.shape}.")

        if q.shape[0] != len(self._joint_indices):
            raise ValueError(
                "Length of q does not match number of actuated joints: "
                f"{q.shape[0]} vs {len(self._joint_indices)}"
            )

        # Update joint states to match q
        for q_val, joint_idx in zip(q, self._joint_indices):
            p.resetJointState(
                self._robot_id,
                joint_idx,
                float(q_val),
                physicsClientId=self._client_id,
            )

        # PyBullet expects one position per DoF. For typical manipulators
        # this equals the number of actuated joints.
        q_list = [float(v) for v in q]
        zero = [0.0] * len(q_list)

        # Local position is the origin of the end-effector link frame.
        local_pos = [0.0, 0.0, 0.0]

        j_lin, j_ang = p.calculateJacobian(
            self._robot_id,
            self._ee_link_index,
            local_pos,
            q_list,
            zero,
            zero,
            physicsClientId=self._client_id,
        )

        j_lin_arr = np.asarray(j_lin, dtype=float)  # shape (3, num_joints)
        j_ang_arr = np.asarray(j_ang, dtype=float)  # shape (3, num_joints)

        jacobian = np.vstack([j_lin_arr, j_ang_arr])  # shape (6, num_joints)
        return jacobian



    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_joint_index_map(self, joint_names: List[str]) -> List[int]:
        """Map joint names to PyBullet joint indices in a fixed order."""
        name_to_index = {}
        num_joints = p.getNumJoints(
            self._robot_id,
            physicsClientId=self._client_id,
        )

        for idx in range(num_joints):
            info = p.getJointInfo(
                self._robot_id,
                idx,
                physicsClientId=self._client_id,
            )
            # info[1] is joint name (bytes)
            joint_name = info[1].decode("utf-8")
            name_to_index[joint_name] = idx

        indices: List[int] = []
        for name in joint_names:
            if name not in name_to_index:
                raise KeyError(
                    f"Joint name '{name}' not found in PyBullet model."
                )
            indices.append(name_to_index[name])

        self._logger.info(
            "[PyBulletKinematics] Joint name to index mapping built for "
            f"{len(indices)} actuated joints."
        )
        return indices

    def _resolve_end_effector_link(self, ee_link_name: Optional[str]) -> int:
        """Resolve end-effector link index from its name.

        Default behavior:
        - If ee_link_name is None, use the last link in the chain.
        - If ee_link_name is provided but not found, log a warning and
          still fall back to the last link.

        This makes large-scale batch processing robust while allowing
        explicit overrides via configuration.
        """
        num_joints = p.getNumJoints(
            self._robot_id,
            physicsClientId=self._client_id,
        )

        if num_joints == 0:
            raise RuntimeError(
                "[PyBulletKinematics] Robot has no links in PyBullet model."
            )

        # Index of the last link in the chain
        fallback_idx = num_joints - 1

        # 1) No name provided: directly use the last link
        if ee_link_name is None:
            self._logger.info(
                "[PyBulletKinematics] end_effector_link not provided; "
                f"using last link with index {fallback_idx} as end-effector."
            )
            return fallback_idx

        # 2) Name provided: try to resolve it; if fail, fall back
        available_names = []
        for idx in range(num_joints):
            info = p.getJointInfo(
                self._robot_id,
                idx,
                physicsClientId=self._client_id,
            )
            link_name = info[12].decode("utf-8")
            available_names.append(link_name)
            if link_name == ee_link_name:
                self._logger.info(
                    "[PyBulletKinematics] Resolved end-effector link '%s' "
                    "to index %d.",
                    ee_link_name,
                    idx,
                )
                return idx

        # Not found: warn and fall back to last link
        self._logger.warning(
            "[PyBulletKinematics] End-effector link '%s' not found in "
            "PyBullet model. Available links: %s. Falling back to last "
            "link with index %d.",
            ee_link_name,
            available_names,
            fallback_idx,
        )
        return fallback_idx


    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def __del__(self) -> None:
        """Disconnect PyBullet client when the backend is garbage-collected."""
        try:
            if self._client_id is not None:
                p.disconnect(self._client_id)
        except Exception:
            # Avoid raising in destructor
            pass


def create_kinematics_backend(
    urdf_path: Path,
    joint_names: List[str],
    base_link: Optional[str],
    end_effector_link: Optional[str],
    logger,
) -> KinematicsBackend:
    """Factory to create a kinematics backend for RobotModel.

    For now this always returns a PyBulletKinematics instance. If you add
    other backends (Pinocchio, etc.), the selection logic can be extended
    here without changing RobotModel.
    """
    backend = PyBulletKinematics(
        urdf_path=urdf_path,
        joint_names=joint_names,
        base_link=base_link,
        end_effector_link=end_effector_link,
        logger=logger,
    )
    return backend
