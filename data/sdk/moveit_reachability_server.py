from __future__ import annotations

import os
import threading
from typing import Dict, List, Optional, Tuple

import rclpy
from fastapi import FastAPI, HTTPException
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import MoveItErrorCodes, RobotState
from moveit_msgs.srv import GetPositionFK, GetPositionIK, GetStateValidity
from pydantic import BaseModel, Field
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
import yaml

try:
    from urdf_parser_py.urdf import URDF
except Exception:
    URDF = None


DEFAULT_GROUP_NAME = os.getenv("MOVEIT_GROUP", "manipulator")
DEFAULT_IK_LINK = os.getenv("MOVEIT_IK_LINK", "ee_link")
DEFAULT_FK_LINK = os.getenv("MOVEIT_FK_LINK", "ee_link")
DEFAULT_FRAME_ID = os.getenv("MOVEIT_BASE_FRAME", "base_link")
DEFAULT_JOINT_NAMES = os.getenv(
    "MOVEIT_JOINT_NAMES",
    "joint1,joint2,joint3,joint4,joint5,joint6",
).split(",")
JOINT_LIMITS_YAML = os.getenv("MOVEIT_JOINT_LIMITS_YAML", "")
ROBOT_DESCRIPTION_XML = os.getenv("ROBOT_DESCRIPTION_XML", "")
SERVICE_TIMEOUT_SEC = float(os.getenv("MOVEIT_SERVICE_TIMEOUT", "3.0"))


class JointQuery(BaseModel):
    joint_positions: List[float] = Field(..., min_items=1)
    joint_names: Optional[List[str]] = None
    fk_link: Optional[str] = None
    frame_id: Optional[str] = None
    group_name: Optional[str] = None


class PoseQuery(BaseModel):
    position: Dict[str, float]
    orientation: Dict[str, float]
    group_name: Optional[str] = None
    ik_link: Optional[str] = None
    frame_id: Optional[str] = None
    seed_joint_positions: Optional[List[float]] = None
    seed_joint_names: Optional[List[str]] = None
    avoid_collisions: Optional[bool] = False


def _load_joint_limits_yaml(path: str) -> Dict[str, Tuple[float, float]]:
    if not path or not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    limits_root = data.get("joint_limits", data)
    limits: Dict[str, Tuple[float, float]] = {}
    for name, cfg in limits_root.items():
        if not isinstance(cfg, dict):
            continue
        if not cfg.get("has_position_limits", True):
            continue
        if "min_position" in cfg and "max_position" in cfg:
            limits[name] = (float(cfg["min_position"]), float(cfg["max_position"]))
    return limits


def _load_joint_limits_urdf(xml_text: str) -> Dict[str, Tuple[float, float]]:
    if URDF is None or not xml_text:
        return {}
    model = URDF.from_xml_string(xml_text)
    limits: Dict[str, Tuple[float, float]] = {}
    for joint in model.joints:
        if joint.type == "fixed" or joint.limit is None:
            continue
        if joint.limit.lower is None or joint.limit.upper is None:
            continue
        limits[joint.name] = (float(joint.limit.lower), float(joint.limit.upper))
    return limits


class MoveItBridge:
    def __init__(self) -> None:
        rclpy.init(args=None)
        self._node = rclpy.create_node("moveit_reachability_bridge")
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._lock = threading.Lock()

        self._ik_client = self._node.create_client(GetPositionIK, "/compute_ik")
        self._fk_client = self._node.create_client(GetPositionFK, "/compute_fk")
        self._valid_client = self._node.create_client(GetStateValidity, "/check_state_validity")

        self._joint_limits = _load_joint_limits_yaml(JOINT_LIMITS_YAML)

        if not self._joint_limits:
            xml_text = ""
            if ROBOT_DESCRIPTION_XML:
                if os.path.isfile(ROBOT_DESCRIPTION_XML):
                    with open(ROBOT_DESCRIPTION_XML, "r", encoding="utf-8") as f:
                        xml_text = f.read()
                else:
                    xml_text = ROBOT_DESCRIPTION_XML
            if xml_text:
                self._joint_limits = _load_joint_limits_urdf(xml_text)

    def shutdown(self) -> None:
        self._executor.remove_node(self._node)
        self._node.destroy_node()
        rclpy.shutdown()

    def _call(self, client, req, timeout_sec: float) -> Tuple[Optional[object], Optional[str]]:
        if not client.wait_for_service(timeout_sec=timeout_sec):
            return None, "service_not_ready"
        future = client.call_async(req)
        rclpy.spin_until_future_complete(
            self._node, future, executor=self._executor, timeout_sec=timeout_sec
        )
        if not future.done():
            return None, "timeout"
        return future.result(), None

    def check_joint_limits(
        self, joint_names: List[str], joint_positions: List[float]
    ) -> Tuple[Optional[bool], Dict[str, Dict[str, float]]]:
        if not self._joint_limits:
            return None, {}
        details: Dict[str, Dict[str, float]] = {}
        ok = True
        for name, pos in zip(joint_names, joint_positions):
            if name not in self._joint_limits:
                continue
            lo, hi = self._joint_limits[name]
            within = bool(lo <= pos <= hi)
            details[name] = {
                "value": float(pos),
                "min": float(lo),
                "max": float(hi),
                "within": 1.0 if within else 0.0,
            }
            if not within:
                ok = False
        return ok, details

    def compute_fk(
        self,
        joint_names: List[str],
        joint_positions: List[float],
        fk_link: str,
        frame_id: str,
    ) -> Tuple[Optional[Dict[str, float]], int, Optional[str]]:
        req = GetPositionFK.Request()
        req.fk_link_names = [fk_link]
        req.robot_state = RobotState()
        req.robot_state.joint_state = JointState()
        req.robot_state.joint_state.name = joint_names
        req.robot_state.joint_state.position = joint_positions
        req.header.frame_id = frame_id
        resp, err = self._call(self._fk_client, req, SERVICE_TIMEOUT_SEC)
        if err is not None or resp is None:
            return None, MoveItErrorCodes.FAILURE, err
        if not resp.pose_stamped:
            return None, resp.error_code.val, "empty_pose"
        pose = resp.pose_stamped[0].pose
        pose_dict = {
            "position": {"x": pose.position.x, "y": pose.position.y, "z": pose.position.z},
            "orientation": {
                "x": pose.orientation.x,
                "y": pose.orientation.y,
                "z": pose.orientation.z,
                "w": pose.orientation.w,
            },
        }
        return pose_dict, resp.error_code.val, None

    def compute_ik(
        self,
        pose: PoseStamped,
        joint_names: List[str],
        seed_positions: List[float],
        group_name: str,
        ik_link: str,
        avoid_collisions: bool,
    ) -> Tuple[Optional[JointState], int, Optional[str]]:
        req = GetPositionIK.Request()
        req.ik_request.group_name = group_name
        req.ik_request.ik_link_name = ik_link
        req.ik_request.pose_stamped = pose
        req.ik_request.avoid_collisions = bool(avoid_collisions)
        req.ik_request.robot_state = RobotState()
        req.ik_request.robot_state.joint_state = JointState()
        req.ik_request.robot_state.joint_state.name = joint_names
        req.ik_request.robot_state.joint_state.position = seed_positions
        req.ik_request.timeout.sec = 0
        req.ik_request.timeout.nanosec = int(0.2 * 1e9)
        resp, err = self._call(self._ik_client, req, SERVICE_TIMEOUT_SEC)
        if err is not None or resp is None:
            return None, MoveItErrorCodes.FAILURE, err
        if resp.solution is None:
            return None, resp.error_code.val, "empty_solution"
        return resp.solution.joint_state, resp.error_code.val, None

    def check_state_validity(
        self,
        joint_names: List[str],
        joint_positions: List[float],
        group_name: str,
    ) -> Tuple[Optional[bool], Optional[str]]:
        req = GetStateValidity.Request()
        req.group_name = group_name
        req.robot_state = RobotState()
        req.robot_state.joint_state = JointState()
        req.robot_state.joint_state.name = joint_names
        req.robot_state.joint_state.position = joint_positions
        resp, err = self._call(self._valid_client, req, SERVICE_TIMEOUT_SEC)
        if err is not None or resp is None:
            return None, err
        return bool(resp.valid), None


bridge = MoveItBridge()
app = FastAPI(title="MoveIt Reachability Bridge", version="0.1.0")


@app.get("/healthz")
def healthz() -> Dict[str, object]:
    return {
        "ik_ready": bridge._ik_client.service_is_ready(),
        "fk_ready": bridge._fk_client.service_is_ready(),
        "validity_ready": bridge._valid_client.service_is_ready(),
        "joint_limits_loaded": bool(bridge._joint_limits),
    }


@app.post("/query/joint")
def query_joint(data: JointQuery) -> Dict[str, object]:
    joint_names = data.joint_names or DEFAULT_JOINT_NAMES
    if len(joint_names) != len(data.joint_positions):
        raise HTTPException(status_code=400, detail="joint_names length mismatch")

    fk_link = data.fk_link or DEFAULT_FK_LINK
    frame_id = data.frame_id or DEFAULT_FRAME_ID
    group_name = data.group_name or DEFAULT_GROUP_NAME

    with bridge._lock:
        within_limits, limits_detail = bridge.check_joint_limits(
            joint_names, data.joint_positions
        )
        fk_pose, fk_code, fk_err = bridge.compute_fk(
            joint_names, data.joint_positions, fk_link, frame_id
        )
        collision_free, valid_err = bridge.check_state_validity(
            joint_names, data.joint_positions, group_name
        )

    fk_success = fk_code == MoveItErrorCodes.SUCCESS
    reachable = bool(
        (within_limits is not False)
        and fk_success
        and (collision_free is True)
    )

    return {
        "within_limits": within_limits,
        "limits_detail": limits_detail,
        "fk_success": fk_success,
        "fk_error_code": int(fk_code),
        "fk_error": fk_err,
        "fk_pose": fk_pose,
        "collision_free": collision_free,
        "collision_error": valid_err,
        "reachable": reachable,
    }


@app.post("/query/pose")
def query_pose(data: PoseQuery) -> Dict[str, object]:
    group_name = data.group_name or DEFAULT_GROUP_NAME
    ik_link = data.ik_link or DEFAULT_IK_LINK
    frame_id = data.frame_id or DEFAULT_FRAME_ID

    seed_names = data.seed_joint_names or DEFAULT_JOINT_NAMES
    if data.seed_joint_positions is None:
        seed_positions = [0.0] * len(seed_names)
    else:
        seed_positions = data.seed_joint_positions

    if len(seed_names) != len(seed_positions):
        raise HTTPException(status_code=400, detail="seed_joint_names length mismatch")

    pose = PoseStamped()
    pose.header.frame_id = frame_id
    pose.pose.position.x = float(data.position.get("x", 0.0))
    pose.pose.position.y = float(data.position.get("y", 0.0))
    pose.pose.position.z = float(data.position.get("z", 0.0))
    pose.pose.orientation.x = float(data.orientation.get("x", 0.0))
    pose.pose.orientation.y = float(data.orientation.get("y", 0.0))
    pose.pose.orientation.z = float(data.orientation.get("z", 0.0))
    pose.pose.orientation.w = float(data.orientation.get("w", 1.0))

    with bridge._lock:
        ik_state, ik_code, ik_err = bridge.compute_ik(
            pose=pose,
            joint_names=seed_names,
            seed_positions=seed_positions,
            group_name=group_name,
            ik_link=ik_link,
            avoid_collisions=bool(data.avoid_collisions),
        )

    ik_success = ik_code == MoveItErrorCodes.SUCCESS
    collision_free = None
    collision_err = None
    within_limits = None
    limits_detail: Dict[str, Dict[str, float]] = {}

    if ik_success and ik_state is not None:
        joint_names = list(ik_state.name)
        joint_positions = list(ik_state.position)
        with bridge._lock:
            within_limits, limits_detail = bridge.check_joint_limits(
                joint_names, joint_positions
            )
            collision_free, collision_err = bridge.check_state_validity(
                joint_names, joint_positions, group_name
            )

    reachable = bool(ik_success and (collision_free is True))
    return {
        "ik_success": ik_success,
        "ik_error_code": int(ik_code),
        "ik_error": ik_err,
        "ik_solution": None
        if ik_state is None
        else {"joint_names": list(ik_state.name), "joint_positions": list(ik_state.position)},
        "within_limits": within_limits,
        "limits_detail": limits_detail,
        "collision_free": collision_free,
        "collision_error": collision_err,
        "reachable": reachable,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
