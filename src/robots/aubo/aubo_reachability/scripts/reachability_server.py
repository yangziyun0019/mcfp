#!/usr/bin/env python3
from typing import Dict, List, Optional, Tuple

from concurrent.futures import TimeoutError

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import DisplayRobotState, MoveItErrorCodes, RobotState
from moveit_msgs.srv import GetPositionFK, GetPositionIK, GetStateValidity
from sensor_msgs.msg import JointState

try:
    from urdf_parser_py.urdf import URDF
except ImportError:  # pragma: no cover
    URDF = None

from aubo_reachability.srv import CheckJointReachability, CheckPoseReachability


class ReachabilityServer(Node):
    def __init__(self) -> None:
        super().__init__("aubo_reachability_server")

        self._cb_group = ReentrantCallbackGroup()

        self.declare_parameter("group_name", "manipulator")
        self.declare_parameter(
            "default_joint_names",
            [
                "shoulder_joint",
                "foreArm_joint",
                "upperArm_joint",
                "wrist1_joint",
                "wrist2_joint",
                "wrist3_joint",
            ],
        )
        self.declare_parameter("fk_link_name", "ee_link")
        self.declare_parameter("ik_link_name", "ee_link")
        self.declare_parameter("compute_fk_service", "/compute_fk")
        self.declare_parameter("compute_ik_service", "/compute_ik")
        self.declare_parameter("get_state_validity_service", "/get_state_validity")
        self.declare_parameter("service_timeout_sec", 2.0)
        self.declare_parameter("joint_limit_margin", 1e-6)
        self.declare_parameter("publish_debug", True)
        self.declare_parameter("robot_description", "")

        self._group_name = self.get_parameter("group_name").value
        self._default_joint_names = list(
            self.get_parameter("default_joint_names").value
        )
        self._fk_link_name = self.get_parameter("fk_link_name").value
        self._ik_link_name = self.get_parameter("ik_link_name").value
        self._timeout_sec = float(self.get_parameter("service_timeout_sec").value)
        self._limit_margin = float(self.get_parameter("joint_limit_margin").value)
        self._publish_debug = bool(self.get_parameter("publish_debug").value)

        self._joint_limits = self._load_joint_limits()

        self._compute_fk_client = self.create_client(
            GetPositionFK,
            self.get_parameter("compute_fk_service").value,
            callback_group=self._cb_group,
        )
        self._compute_ik_client = self.create_client(
            GetPositionIK,
            self.get_parameter("compute_ik_service").value,
            callback_group=self._cb_group,
        )
        self._state_validity_client = self.create_client(
            GetStateValidity,
            self.get_parameter("get_state_validity_service").value,
            callback_group=self._cb_group,
        )

        self._wait_for_services()

        if self._publish_debug:
            self._display_pub = self.create_publisher(
                DisplayRobotState, "display_robot_state", 10
            )
            self._pose_pub = self.create_publisher(PoseStamped, "requested_pose", 10)
        else:
            self._display_pub = None
            self._pose_pub = None

        self.create_service(
            CheckJointReachability,
            "check_joint_reachability",
            self._handle_joint_request,
            callback_group=self._cb_group,
        )
        self.create_service(
            CheckPoseReachability,
            "check_pose_reachability",
            self._handle_pose_request,
            callback_group=self._cb_group,
        )

    def _wait_for_services(self) -> None:
        service_list = [
            (self._compute_fk_client, "compute_fk"),
            (self._compute_ik_client, "compute_ik"),
            (self._state_validity_client, "get_state_validity"),
        ]
        for client, name in service_list:
            while not client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f"Waiting for {name} service...")

    def _load_joint_limits(self) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        if URDF is None:
            self.get_logger().error("urdf_parser_py not available; joint limits disabled")
            return {}

        robot_description = self.get_parameter("robot_description").value
        if not robot_description:
            self.get_logger().error("robot_description parameter is empty")
            return {}

        try:
            urdf = URDF.from_xml_string(robot_description)
        except Exception as exc:  # pragma: no cover
            self.get_logger().error(f"Failed to parse URDF: {exc}")
            return {}

        limits: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        for joint in urdf.joints:
            if joint.type in ("fixed", "floating", "planar"):
                continue
            if joint.limit is None:
                continue
            lower = joint.limit.lower
            upper = joint.limit.upper
            limits[joint.name] = (lower, upper)
        return limits

    def _check_joint_limits(
        self, joint_names: List[str], joint_positions: List[float]
    ) -> Tuple[bool, str]:
        if not self._joint_limits:
            return False, "joint limits not loaded"

        for name, position in zip(joint_names, joint_positions):
            if name not in self._joint_limits:
                continue
            lower, upper = self._joint_limits[name]
            if lower is not None and position < (lower - self._limit_margin):
                return False, f"{name} below lower limit: {position} < {lower}"
            if upper is not None and position > (upper + self._limit_margin):
                return False, f"{name} above upper limit: {position} > {upper}"
        return True, "ok"

    def _call_service(self, client, request, timeout_sec: float):
        future = client.call_async(request)
        try:
            return future.result(timeout=timeout_sec)
        except TimeoutError:
            return None
        except Exception:  # pragma: no cover
            return None

    def _publish_display_state(self, robot_state: RobotState) -> None:
        if not self._display_pub:
            return
        msg = DisplayRobotState()
        msg.state = robot_state
        self._display_pub.publish(msg)

    def _handle_joint_request(
        self, request: CheckJointReachability.Request, response: CheckJointReachability.Response
    ) -> CheckJointReachability.Response:
        group_name = request.group_name or self._group_name
        joint_names = list(request.joint_names) or list(self._default_joint_names)
        joint_positions = list(request.joint_positions)

        if not joint_names:
            response.message = "joint_names is empty"
            return response
        if len(joint_names) != len(joint_positions):
            response.message = "joint_names and joint_positions size mismatch"
            return response

        within_limits, limit_msg = self._check_joint_limits(joint_names, joint_positions)
        response.within_limits = within_limits

        robot_state = RobotState()
        robot_state.joint_state = JointState()
        robot_state.joint_state.name = joint_names
        robot_state.joint_state.position = joint_positions

        self._publish_display_state(robot_state)

        fk_link_name = request.fk_link_name or self._fk_link_name
        fk_req = GetPositionFK.Request()
        fk_req.fk_link_names = [fk_link_name]
        fk_req.robot_state = robot_state
        fk_resp = self._call_service(self._compute_fk_client, fk_req, self._timeout_sec)

        if fk_resp is None:
            response.fk_success = False
            response.message = "compute_fk service timeout"
        else:
            response.fk_success = (
                fk_resp.error_code.val == MoveItErrorCodes.SUCCESS
            )
            if response.fk_success and fk_resp.pose_stamped:
                response.fk_pose = fk_resp.pose_stamped[0]

        state_req = GetStateValidity.Request()
        state_req.robot_state = robot_state
        state_req.group_name = group_name
        state_resp = self._call_service(
            self._state_validity_client, state_req, self._timeout_sec
        )

        if state_resp is None:
            response.self_collision_free = False
            if response.message:
                response.message += "; get_state_validity timeout"
            else:
                response.message = "get_state_validity timeout"
        else:
            response.self_collision_free = bool(state_resp.valid)

        response.reachable = (
            response.within_limits
            and response.fk_success
            and response.self_collision_free
        )

        if response.message:
            response.message += f"; limit_check={limit_msg}"
        else:
            response.message = f"limit_check={limit_msg}"

        return response

    def _handle_pose_request(
        self, request: CheckPoseReachability.Request, response: CheckPoseReachability.Response
    ) -> CheckPoseReachability.Response:
        group_name = request.group_name or self._group_name
        ik_link_name = request.ik_link_name or self._ik_link_name

        pose = request.target_pose
        if not pose.header.frame_id:
            pose.header.frame_id = "world"

        if self._pose_pub:
            self._pose_pub.publish(pose)

        ik_req = GetPositionIK.Request()
        ik_req.ik_request.group_name = group_name
        ik_req.ik_request.pose_stamped = pose
        ik_req.ik_request.ik_link_name = ik_link_name
        ik_req.ik_request.avoid_collisions = False

        ik_resp = self._call_service(self._compute_ik_client, ik_req, self._timeout_sec)
        if ik_resp is None:
            response.ik_success = False
            response.message = "compute_ik service timeout"
            response.reachable = False
            return response

        response.ik_success = ik_resp.error_code.val == MoveItErrorCodes.SUCCESS
        if not response.ik_success:
            response.reachable = False
            response.message = f"compute_ik failed: {ik_resp.error_code.val}"
            return response

        solution_state = ik_resp.solution
        if solution_state.joint_state.name and solution_state.joint_state.position:
            response.solution_joint_names = list(solution_state.joint_state.name)
            response.solution_joint_positions = list(solution_state.joint_state.position)

        self._publish_display_state(solution_state)

        state_req = GetStateValidity.Request()
        state_req.robot_state = solution_state
        state_req.group_name = group_name
        state_resp = self._call_service(
            self._state_validity_client, state_req, self._timeout_sec
        )

        if state_resp is None:
            response.self_collision_free = False
            response.message = "get_state_validity timeout"
        else:
            response.self_collision_free = bool(state_resp.valid)

        response.reachable = response.ik_success and response.self_collision_free
        return response


def main() -> None:
    rclpy.init()
    node = ReachabilityServer()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
