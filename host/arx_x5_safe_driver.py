"""Script: arx_x5_safe_driver.py
Purpose: Map teleoperation control frames to ARX X5 joint and gripper targets with safety-oriented limits.
Usage: Imported by host.arx_x5_teleop or other host-side teleoperation entry points.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any, Optional

from host.control_types import ControlFrame


_REPO_ROOT = Path(__file__).resolve().parents[1]
_ARX_SDK_ROOT = _REPO_ROOT / "ARX_X5-main" / "py" / "arx_x5_python"


# Edit mappings here first.
DRIVER_CONFIG: dict[str, Any] = {
    # Safety:
    # False keeps the driver in diagnostics-only mode.
    "enable_actuation": False,
    "status_print_hz": 2.0,
    # ARX Python SDK / CAN:
    "sdk_root": str(_ARX_SDK_ROOT),
    "can_port": "can1",
    "arm_type": 0,
    "control_dt_s": 0.05,
    "check_feedback_on_startup": True,
    # Mapping from host control frame to ARX 6 joints:
    # [joint1, joint2, joint3, joint4, joint5, joint6]
    "joint_home_deg": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "joint_input_scale": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "joint_input_sign": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "joint_input_offset_deg": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "joint_limit_min_deg": [-180.0, -180.0, -180.0, -180.0, -180.0, -180.0],
    "joint_limit_max_deg": [180.0, 180.0, 180.0, 180.0, 180.0, 180.0],
    # SDK command units for future actuation:
    # "deg" keeps values as degrees, "rad" converts to radians before send.
    "sdk_joint_unit": "rad",
    # Gripper mapping for future actuation:
    "gripper_open_value": 0.0,
    "gripper_closed_value": -0.2,
    "gripper_invert": False,
}


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _format_list(values: list[float], precision: int = 2) -> str:
    return "[" + ", ".join(f"{v:.{precision}f}" for v in values) + "]"


def _read_can_state(can_port: str) -> str:
    sysfs_path = Path("/sys/class/net") / can_port
    if not sysfs_path.exists():
        return "missing"
    operstate_path = sysfs_path / "operstate"
    try:
        return operstate_path.read_text(encoding="utf-8").strip()
    except OSError:
        return "unknown"


def _load_single_arm_class():
    sdk_root = Path(str(DRIVER_CONFIG["sdk_root"])).resolve()
    if not sdk_root.exists():
        raise RuntimeError(f"ARX SDK root not found: {sdk_root}")

    pybind_modules = list(sdk_root.rglob("arx_x5_python*.so"))
    if not pybind_modules:
        raise RuntimeError(
            "ARX Python SDK pybind module not found. "
            "Run: cd ARX_X5-main/py/arx_x5_python && ./build.sh"
        )

    sdk_root_str = str(sdk_root)
    if sdk_root_str not in sys.path:
        sys.path.insert(0, sdk_root_str)

    try:
        from bimanual import SingleArm  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Failed to import ARX Python SDK. "
            "Try: cd ARX_X5-main/py/arx_x5_python && source ./setup.sh"
        ) from exc

    return SingleArm


def _frame_to_joint_targets_deg(frame: ControlFrame) -> list[float]:
    source_deg = [
        frame.ch1_deg,
        frame.ch2_deg,
        frame.ch3_deg,
        frame.roll_deg,
        frame.pitch_deg,
        frame.yaw_deg,
    ]

    home_deg = [float(v) for v in DRIVER_CONFIG["joint_home_deg"]]
    scale = [float(v) for v in DRIVER_CONFIG["joint_input_scale"]]
    sign = [float(v) for v in DRIVER_CONFIG["joint_input_sign"]]
    offset_deg = [float(v) for v in DRIVER_CONFIG["joint_input_offset_deg"]]
    min_deg = [float(v) for v in DRIVER_CONFIG["joint_limit_min_deg"]]
    max_deg = [float(v) for v in DRIVER_CONFIG["joint_limit_max_deg"]]

    targets_deg: list[float] = []
    for i, value_deg in enumerate(source_deg):
        mapped_deg = home_deg[i] + sign[i] * scale[i] * value_deg + offset_deg[i]
        targets_deg.append(_clip(mapped_deg, min_deg[i], max_deg[i]))
    return targets_deg


def _joint_targets_to_sdk_units(targets_deg: list[float]) -> list[float]:
    if str(DRIVER_CONFIG["sdk_joint_unit"]).lower() == "deg":
        return list(targets_deg)
    return [math.radians(v) for v in targets_deg]


def _frame_to_gripper_target(frame: ControlFrame) -> float:
    grip_active = 1 if frame.grip else 0
    if bool(DRIVER_CONFIG["gripper_invert"]):
        grip_active = 1 - grip_active
    if grip_active:
        return float(DRIVER_CONFIG["gripper_closed_value"])
    return float(DRIVER_CONFIG["gripper_open_value"])


class RobotDriver:
    def __init__(self) -> None:
        self._min_interval = 1.0 / max(float(DRIVER_CONFIG["status_print_hz"]), 0.1)
        self._last_print = 0.0
        self._frame_count = 0
        self._arm: Optional[Any] = None
        self._arm_connected = False
        self._connect_error = ""
        self._can_state = _read_can_state(str(DRIVER_CONFIG["can_port"]))
        self._startup_feedback: Optional[list[float]] = None
        self._init_arm()

    def _init_arm(self) -> None:
        can_port = str(DRIVER_CONFIG["can_port"])
        if self._can_state == "missing":
            self._connect_error = (
                f"CAN interface '{can_port}' not found. "
                "Bring up the CAN device before running the teleop host."
            )
            print(f"[arx_x5] {self._connect_error}")
            return

        try:
            single_arm_cls = _load_single_arm_class()
            arm_config = {
                "can_port": can_port,
                "type": int(DRIVER_CONFIG["arm_type"]),
                "dt": float(DRIVER_CONFIG["control_dt_s"]),
            }
            self._arm = single_arm_cls(arm_config)
            if bool(DRIVER_CONFIG["check_feedback_on_startup"]):
                feedback = self._arm.get_joint_positions()
                if feedback is not None:
                    self._startup_feedback = [float(v) for v in feedback]
            self._arm_connected = True
            print(
                f"[arx_x5] SDK ready. can={can_port} state={self._can_state} "
                f"actuation={'ON' if DRIVER_CONFIG['enable_actuation'] else 'OFF'}"
            )
            if self._startup_feedback:
                print(
                    "[arx_x5] startup joint feedback "
                    f"{_format_list(self._startup_feedback, precision=3)}"
                )
        except Exception as exc:
            self._connect_error = str(exc)
            print(f"[arx_x5] SDK init failed: {self._connect_error}")

    def _print_status(
        self,
        frame: ControlFrame,
        joint_targets_deg: list[float],
        joint_targets_sdk: list[float],
        gripper_target: float,
    ) -> None:
        now = time.monotonic()
        if now - self._last_print < self._min_interval:
            return
        self._last_print = now

        frame_age_ms = (now - frame.t_host) * 1000.0
        arm_state = "connected" if self._arm_connected else "not_connected"
        unit_name = str(DRIVER_CONFIG["sdk_joint_unit"]).lower()
        print(
            f"[arx_x5] arm={arm_state} can={DRIVER_CONFIG['can_port']} "
            f"can_state={self._can_state} frame_count={self._frame_count} "
            f"frame_age_ms={frame_age_ms:.1f}"
        )
        print(
            "[arx_x5] mapped_deg="
            f"{_format_list(joint_targets_deg)} "
            f"mapped_{unit_name}={_format_list(joint_targets_sdk, precision=3)} "
            f"gripper={gripper_target:.3f}"
        )
        if self._connect_error:
            print(f"[arx_x5] connect_error={self._connect_error}")
        if not bool(DRIVER_CONFIG["enable_actuation"]):
            print("[arx_x5] dry-run only. Motion commands are blocked.")

    def send(self, frame: ControlFrame) -> None:
        self._frame_count += 1
        joint_targets_deg = _frame_to_joint_targets_deg(frame)
        joint_targets_sdk = _joint_targets_to_sdk_units(joint_targets_deg)
        gripper_target = _frame_to_gripper_target(frame)

        self._print_status(frame, joint_targets_deg, joint_targets_sdk, gripper_target)

        if not bool(DRIVER_CONFIG["enable_actuation"]):
            return
        if not self._arm_connected or self._arm is None:
            return

        self._arm.set_joint_positions(joint_targets_sdk)
        self._arm.set_catch_pos(gripper_target)

    def close(self) -> None:
        self._arm = None


def create_driver() -> RobotDriver:
    return RobotDriver()
