from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional


class RobotModel:
    """Lightweight robot model built from a URDF file.

    Parameters
    ----------
    urdf_path:
        Path to the URDF file describing the robot.
    logger:
        Logger instance for reporting status and potential issues.

    Notes
    -----
    For the first working episode, this class only validates the URDF path
    and exposes placeholder joint metadata. Real URDF parsing and kinematics
    will be implemented later.
    """

    def __init__(self, urdf_path: Path, logger) -> None:
        self.urdf_path: Path = Path(urdf_path).resolve()
        self.logger = logger

        if not self.urdf_path.is_file():
            raise FileNotFoundError(f"URDF file not found: {self.urdf_path}")

        self.logger.info(f"[RobotModel] Loading URDF from: {self.urdf_path}")

        # TODO: 在这里解析 URDF，自动获取关节名称和限位
        # 当前为了跑通流程，先假定 6 关节、[-3.14, 3.14] 升级后删除。
        self._joint_names: List[str] = [f"joint_{i}" for i in range(6)]
        self._joint_limits: List[Tuple[float, float]] = [(-3.14, 3.14)] * 6

        if not self._joint_names:
            self.logger.warning(
                "[RobotModel] Joint metadata is empty. Please implement URDF parsing."
            )
        else:
            self.logger.info(
                f"[RobotModel] Placeholder joint metadata: {len(self._joint_names)} joints."
            )

    @property
    def num_joints(self) -> int:
        """Return the number of actuated joints."""
        return len(self._joint_names)

    @property
    def joint_names(self) -> List[str]:
        """Return the list of joint names."""
        return list(self._joint_names)

    @property
    def joint_limits(self) -> List[Tuple[float, float]]:
        """Return joint limits as (lower, upper) tuples."""
        return list(self._joint_limits)

    # ------------------------------------------------------------------
    # Kinematics API stubs
    # ------------------------------------------------------------------

    def fk(self, q) -> Optional[object]:
        """Compute forward kinematics for the end-effector.

        Parameters
        ----------
        q:
            Joint configuration (iterable of floats).

        Returns
        -------
        pose:
            Placeholder object representing the end-effector pose.
        """
        # TODO: 实现末端正运动学，返回统一的 SE3 位姿类型
        self.logger.warning(
            "[RobotModel.fk] Called FK stub. Kinematics not implemented yet."
        )
        return None

    def link_poses(self, q) -> dict:
        """Compute poses for all links in the kinematic chain.

        Parameters
        ----------
        q:
            Joint configuration.

        Returns
        -------
        poses:
            Mapping from link name to pose object. Currently empty.
        """
        # TODO: 实现所有 link 的位姿计算，用于自碰撞与可视化
        self.logger.warning(
            "[RobotModel.link_poses] Called link_poses stub. "
            "Kinematics not implemented yet."
        )
        return {}

    def jacobian(self, q) -> Optional[object]:
        """Compute the geometric Jacobian at the end-effector.

        Parameters
        ----------
        q:
            Joint configuration.

        Returns
        -------
        jacobian:
            Placeholder object representing the Jacobian matrix.
        """
        # TODO: 实现 Jacobian 计算，用于操控性指标
        self.logger.warning(
            "[RobotModel.jacobian] Called jacobian stub. "
            "Kinematics not implemented yet."
        )
        return None
