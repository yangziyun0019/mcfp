# mcfp/sim/fk_ik.py
"""Kinematics backends (FK/IK) for MCFP.

Now supports two backends:
1. PyBulletKinematics: Legacy wrapper around PyBullet (requires URDF).
2. SpecKinematics: Pure NumPy implementation (requires Morphology JSON).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Protocol, Tuple, Dict, Any

import numpy as np
from scipy.spatial.transform import Rotation as R

try:
    import pybullet as p
    import pybullet_data
    _HAS_PYBULLET = True
except ImportError:
    _HAS_PYBULLET = False


class KinematicsBackend(Protocol):
    """Abstract interface for kinematics backends."""

    def fk_pose(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return end-effector pose (position + quaternion) for q."""
        ...

    def fk_position(self, q: np.ndarray) -> np.ndarray:
        """Return end-effector position for configuration q."""
        ...

    def link_poses(self, q: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Return poses of all links for configuration q."""
        ...
    
    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """Return the geometric Jacobian at the end-effector."""
        ...


class PyBulletKinematics:
    """PyBullet-based implementation of KinematicsBackend.

    This backend loads the URDF into a DIRECT PyBullet client and exposes
    FK for the end-effector. It handles both fixed and floating base robots
    (though MCFP primarily assumes fixed base).
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
        self._target_ee_name = end_effector_link

        if not self._urdf_path.is_file():
            raise FileNotFoundError(f"URDF file not found: {self._urdf_path}")

        # Initialize a DIRECT client
        self._client_id = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Load robot model
        self._robot_id = p.loadURDF(
            str(self._urdf_path),
            basePosition=[0, 0, 0],
            baseOrientation=[0, 0, 0, 1],
            useFixedBase=True,
            physicsClientId=self._client_id,
        )

        # --- 1. Build Full Link/Joint Map (Including Base) ---
        self._link_name_to_index: Dict[str, int] = {}
        self._joint_name_to_index: Dict[str, int] = {}

        # A. Handle Base Link (Index -1)
        # PyBullet returns (base_link_name, robot_name) in bytes
        body_info = p.getBodyInfo(self._robot_id, physicsClientId=self._client_id)
        base_link_name = body_info[0].decode("utf-8")
        self._link_name_to_index[base_link_name] = -1
        
        self._logger.info(
            f"[PyBulletKinematics] Detected Base Link: '{base_link_name}' (Index -1)"
        )

        # B. Handle Other Links (Index 0 to N-1)
        num_joints = p.getNumJoints(self._robot_id, physicsClientId=self._client_id)
        for idx in range(num_joints):
            info = p.getJointInfo(self._robot_id, idx, physicsClientId=self._client_id)
            # info[1]: joint name, info[12]: link name
            j_name = info[1].decode("utf-8")
            l_name = info[12].decode("utf-8")
            
            self._joint_name_to_index[j_name] = idx
            self._link_name_to_index[l_name] = idx

        # --- 2. Map Actuated Joints ---
        self._actuated_indices: List[int] = []
        for name in self._joint_names:
            if name in self._joint_name_to_index:
                self._actuated_indices.append(self._joint_name_to_index[name])
            else:
                self._logger.warning(
                    f"[PyBulletKinematics] Joint '{name}' not found in URDF!"
                )

        # --- 3. Resolve End-Effector ---
        self._ee_link_index = self._resolve_ee_index(end_effector_link, num_joints)

        self._logger.info(
            f"[PyBulletKinematics] Backend ready. "
            f"Base: '{base_link_name}', EE Index: {self._ee_link_index}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _set_joint_states(self, q: np.ndarray):
        """Helper to set robot configuration."""
        if q.shape[0] != len(self._actuated_indices):
             raise ValueError(f"q dim {q.shape[0]} != actuated joints {len(self._actuated_indices)}")
        
        for i, val in zip(self._actuated_indices, q):
            p.resetJointState(self._robot_id, i, float(val), physicsClientId=self._client_id)

    def fk_pose(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self._set_joint_states(q)
        
        # If EE is base (rare, but possible), handle separately
        if self._ee_link_index == -1:
            pos, orn = p.getBasePositionAndOrientation(self._robot_id, physicsClientId=self._client_id)
        else:
            state = p.getLinkState(
                self._robot_id, self._ee_link_index, 
                computeForwardKinematics=True, physicsClientId=self._client_id
            )
            pos, orn = state[4], state[5]
            
        return np.array(pos), np.array(orn)

    def fk_position(self, q: np.ndarray) -> np.ndarray:
        pos, _ = self.fk_pose(q)
        return pos
    
    def link_poses(self, q: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Return poses for ALL known links (Base + Children)."""
        self._set_joint_states(q)
        poses = {}
        
        # Iterate over our pre-built map, not just range(num_joints)
        for link_name, link_idx in self._link_name_to_index.items():
            if link_idx == -1:
                # Special handling for Base
                pos, orn = p.getBasePositionAndOrientation(
                    self._robot_id, physicsClientId=self._client_id
                )
            else:
                # Standard handling for Links
                state = p.getLinkState(
                    self._robot_id, link_idx, 
                    computeForwardKinematics=True, physicsClientId=self._client_id
                )
                pos, orn = state[4], state[5]
            
            poses[link_name] = (np.array(pos), np.array(orn))
            
        return poses
    
    def jacobian(self, q: np.ndarray) -> np.ndarray:
        self._set_joint_states(q)
        q_list = [float(v) for v in q]
        # Zero velocities
        zeros = [0.0] * len(q_list)
        
        # Linear + Angular Jacobian
        # Note: input q must match ALL movable joints in PyBullet. 
        # For this simplified version we assume q covers the relevant chain.
        jac_t, jac_r = p.calculateJacobian(
            self._robot_id, self._ee_link_index, [0,0,0], 
            q_list, zeros, zeros, physicsClientId=self._client_id
        )
        
        return np.vstack([jac_t, jac_r])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_ee_index(self, name: Optional[str], num_joints: int) -> int:
        # Fallback to last link index
        fallback = num_joints - 1
        
        if name is None:
            self._logger.info(f"EE not provided, using index {fallback}")
            return fallback
            
        if name in self._link_name_to_index:
            return self._link_name_to_index[name]
        
        self._logger.warning(f"EE link '{name}' not found. Using fallback {fallback}")
        return fallback

    def __del__(self) -> None:
        try:
            if hasattr(self, '_client_id'):
                p.disconnect(self._client_id)
        except:
            pass

class SpecKinematics:
    """Pure NumPy implementation using Morphology JSON Spec."""

    def __init__(self, spec: Dict[str, Any], joint_names: List[str], base_link: str, end_effector_link: str):
        self.joint_names = joint_names
        self.base_link = base_link
        self.ee_link = end_effector_link
        
        # Parse Link/Joint structure
        self.links = {} # Stores kinematic parameters
        
        # Base
        self.links[base_link] = {
            "parent": None, "T_static": np.eye(4), 
            "axis": np.zeros(3), "type": "fixed", "q_idx": -1
        }
        
        # Build Lookup
        joint_list = spec['chain']['joints']
        
        for j in joint_list:
            child = j['child_link']
            parent = j['parent_link']
            
            # Static Transform
            T = np.eye(4)
            T[:3, 3] = j['origin_xyz']
            # origin_rpy is extrinsic xyz
            T[:3, :3] = R.from_euler('xyz', j['origin_rpy']).as_matrix()
            
            # Axis
            axis = np.array(j['axis'], dtype=float)
            
            # Actuation index
            q_idx = -1
            if j['name'] in self.joint_names:
                q_idx = self.joint_names.index(j['name'])
            
            self.links[child] = {
                "parent": parent,
                "T_static": T,
                "axis": axis,
                "type": j['type'],
                "q_idx": q_idx
            }

        # Topological Sort
        self._ordered_links = self._sort_links()

    def _sort_links(self):
        ordered = [self.base_link]
        pending = [k for k in self.links if k != self.base_link]
        while pending:
            moved = False
            for name in pending[:]:
                parent = self.links[name]['parent']
                if parent in ordered:
                    ordered.append(name)
                    pending.remove(name)
                    moved = True
            if not moved and pending: break
        return ordered

    def _forward_pass(self, q: np.ndarray) -> Dict[str, np.ndarray]:
        transforms = {self.base_link: np.eye(4)}
        for name in self._ordered_links:
            if name == self.base_link: continue
            data = self.links[name]
            parent_T = transforms[data['parent']]
            T_static = data['T_static']
            T_joint = np.eye(4)
            if data['q_idx'] >= 0:
                val = q[data['q_idx']]
                if data['type'] in ['revolute', 'continuous']:
                    rot = R.from_rotvec(data['axis'] * val).as_matrix()
                    T_joint[:3, :3] = rot
                elif data['type'] == 'prismatic':
                    T_joint[:3, 3] = data['axis'] * val
            transforms[name] = parent_T @ T_static @ T_joint
        return transforms

    def fk_pose(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Ts = self._forward_pass(q)
        T_ee = Ts.get(self.ee_link, list(Ts.values())[-1])
        pos = T_ee[:3, 3]
        quat = R.from_matrix(T_ee[:3, :3]).as_quat() # (x, y, z, w)
        return pos, quat

    def fk_position(self, q: np.ndarray) -> np.ndarray:
        return self.fk_pose(q)[0]

    def link_poses(self, q: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        Ts = self._forward_pass(q)
        res = {}
        for name, T in Ts.items():
            res[name] = (T[:3, 3], R.from_matrix(T[:3, :3]).as_quat())
        return res

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        Ts = self._forward_pass(q)
        T_ee = Ts.get(self.ee_link, list(Ts.values())[-1])
        p_e = T_ee[:3, 3]
        J = np.zeros((6, len(self.joint_names)))
        for name in self._ordered_links:
            if name == self.base_link: continue
            data = self.links[name]
            idx = data['q_idx']
            if idx < 0: continue
            T_parent = Ts[data['parent']]
            T_pre_joint = T_parent @ data['T_static']
            z_w = T_pre_joint[:3, :3] @ data['axis']
            p_j = T_pre_joint[:3, 3]
            if data['type'] in ['revolute', 'continuous']:
                J[:3, idx] = np.cross(z_w, p_e - p_j)
                J[3:, idx] = z_w
            elif data['type'] == 'prismatic':
                J[:3, idx] = z_w
        return J

def create_kinematics_backend(urdf_path, joint_names, base_link, end_effector_link, logger):
    return PyBulletKinematics(urdf_path, joint_names, base_link, end_effector_link, logger)