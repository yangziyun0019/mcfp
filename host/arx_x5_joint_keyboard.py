"""Script: arx_x5_joint_keyboard.py
Purpose: Drive the ARX X5 with keyboard commands for joint-by-joint testing and SDK verification.
Usage: /usr/bin/python3 host/arx_x5_joint_keyboard.py
"""

from __future__ import annotations

import ctypes
import importlib.util
import importlib.machinery
import math
import os
import select
import shlex
import sys
import termios
import time
import tty
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_SDK_ROOT = _REPO_ROOT / "ARX_X5-main" / "py" / "arx_x5_python"
_ROS_HUMBLE_LIB = Path("/opt/ros/humble/lib")
_ROS_HUMBLE_ROOT = Path("/opt/ros/humble")
_ROS_HUMBLE_ARCH_LIB = _ROS_HUMBLE_LIB / "x86_64-linux-gnu"
_TTY_OUTPUT = open("/dev/tty", "w", buffering=1)


# Edit these values first.
CONFIG: dict[str, Any] = {
    "sdk_root": str(_DEFAULT_SDK_ROOT),
    "can_port": "can1",
    "arm_type": 0,
    "control_dt_s": 0.05,
    "poll_hz": 50.0,
    "require_enable_key": True,
    "protect_on_exit": True,
    # The SDK examples suggest joint APIs use radians internally.
    "sdk_command_unit": "rad",
    "sdk_feedback_unit": "rad",
    "joint_step_deg": 1.0,
    "joint_step_deg_min": 0.1,
    "joint_step_deg_max": 10.0,
    "gripper_step": 0.2,
    "gripper_open_value": 0.0,
    "gripper_close_value": -0.2,
    "joint_limit_min_deg": [-180.0, -180.0, -180.0, -180.0, -180.0, -180.0],
    "joint_limit_max_deg": [180.0, 180.0, 180.0, 180.0, 180.0, 180.0],
    "home_hold_seconds": 1.0,
    "home_repeat_gap_seconds": 0.35,
}


def _find_first(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.rglob(pattern))
    return matches[0] if matches else None


def _prepend_env_path(name: str, path: Path) -> None:
    value = str(path)
    current = os.environ.get(name, "")
    parts = [p for p in current.split(":") if p]
    if value not in parts:
        os.environ[name] = value if not current else f"{value}:{current}"


def _prepend_sys_path(path: Path) -> None:
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)


def _load_shared_library(path: Path) -> None:
    if not path.exists():
        return
    mode = getattr(ctypes, "RTLD_GLOBAL", 0)
    ctypes.CDLL(str(path), mode=mode)


def _sdk_setup_hint(sdk_root: Path) -> str:
    return (
        "ARX Python SDK build artifacts not found.\n"
        f"Expected under: {sdk_root / 'bimanual' / 'api'}\n"
        "Run:\n"
        f"  cd {sdk_root}\n"
        "  ./build.sh\n"
        "Then run this script again."
    )


def _prepare_ros_environment() -> None:
    ros_root = _ROS_HUMBLE_ROOT
    ros_lib = _ROS_HUMBLE_LIB
    ros_python = ros_lib / "python3.10" / "site-packages"

    if ros_root.exists():
        os.environ.setdefault("ROS_DISTRO", "humble")
        os.environ.setdefault("ROS_VERSION", "2")
        os.environ.setdefault("ROS_PYTHON_VERSION", "3")
        os.environ.setdefault("AMENT_PREFIX_PATH", str(ros_root))
        os.environ.setdefault("COLCON_PREFIX_PATH", str(ros_root))
        os.environ.setdefault("CMAKE_PREFIX_PATH", str(ros_root))
        os.environ.setdefault("AMENT_CURRENT_PREFIX", str(ros_root))

    for path in [ros_lib, _ROS_HUMBLE_ARCH_LIB, ros_python]:
        _prepend_env_path("LD_LIBRARY_PATH", path)
        _prepend_env_path("PYTHONPATH", path)
        _prepend_sys_path(path)


def _ros_env_ready() -> bool:
    ament = os.environ.get("AMENT_PREFIX_PATH", "")
    ld_library = os.environ.get("LD_LIBRARY_PATH", "")
    return (
        str(_ROS_HUMBLE_ROOT) in ament.split(":")
        and str(_ROS_HUMBLE_LIB) in ld_library.split(":")
        and str(_ROS_HUMBLE_ARCH_LIB) in ld_library.split(":")
    )


def _maybe_reexec_with_ros_env() -> None:
    if os.environ.get("ARX_ROS_ENV_READY") == "1":
        return
    if not _ROS_HUMBLE_ROOT.exists():
        return
    if _ros_env_ready():
        os.environ["ARX_ROS_ENV_READY"] = "1"
        return

    setup_bash = _ROS_HUMBLE_ROOT / "setup.bash"
    if not setup_bash.exists():
        return

    argv = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    quoted_argv = " ".join(shlex.quote(arg) for arg in argv)
    cmd = (
        f"source {shlex.quote(str(setup_bash))} >/dev/null 2>&1 && "
        "export ARX_ROS_ENV_READY=1 && "
        f"exec {quoted_argv}"
    )
    os.execvpe("/bin/bash", ["bash", "-lc", cmd], os.environ.copy())


def _prepare_sdk_environment(sdk_root: Path) -> Path:
    _prepare_ros_environment()

    api_root = sdk_root / "bimanual" / "api"
    if not api_root.exists():
        raise RuntimeError(_sdk_setup_hint(sdk_root))

    compatible_modules = []
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        compatible_modules.extend(api_root.rglob(f"arx_x5_python*{suffix}"))
    pybind_module = sorted(compatible_modules)[0] if compatible_modules else None
    if pybind_module is None:
        any_module = _find_first(api_root, "arx_x5_python*.so")
        if any_module is not None:
            raise RuntimeError(
                "ARX SDK was built for a different Python ABI.\n"
                f"Current interpreter: {sys.executable} ({sys.version.split()[0]})\n"
                f"Found module: {any_module.name}\n"
                "Run this script with /usr/bin/python3, or rebuild the SDK with the same interpreter."
            )
        raise RuntimeError(_sdk_setup_hint(sdk_root))

    solver_module = _find_first(api_root, "kinematic_solver*.so")

    candidate_libs = [
        _ROS_HUMBLE_LIB / "librcutils.so",
        _ROS_HUMBLE_LIB / "liburdf.so",
        _ROS_HUMBLE_LIB / "libkdl_parser.so",
        _ROS_HUMBLE_ARCH_LIB / "liburdfdom_model.so.3.0",
        _ROS_HUMBLE_ARCH_LIB / "liburdfdom_model_state.so.3.0",
        _ROS_HUMBLE_ARCH_LIB / "liburdfdom_world.so.3.0",
        _ROS_HUMBLE_ARCH_LIB / "liburdfdom_sensor.so.3.0",
        _find_first(api_root, "libarx_x5_src.so"),
        _find_first(api_root, "libx5_kinematic_solver.so"),
        sdk_root / "bimanual" / "lib" / "arx_x5_src" / "libarx_x5_src.so",
        sdk_root / "bimanual" / "lib" / "libx5_kinematic_solver.so",
    ]

    for path in [pybind_module.parent, api_root, sdk_root]:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    if solver_module is not None:
        solver_parent = str(solver_module.parent)
        if solver_parent not in sys.path:
            sys.path.insert(0, solver_parent)

    for path in [
        pybind_module.parent,
        api_root,
        api_root / "arx_x5_src",
        Path("/usr/local/lib"),
        _ROS_HUMBLE_LIB,
        _ROS_HUMBLE_ARCH_LIB,
    ]:
        _prepend_env_path("LD_LIBRARY_PATH", path)

    for path in candidate_libs:
        if path is not None and path.exists():
            _load_shared_library(path)

    return pybind_module


def _load_single_arm_class(sdk_root: Path):
    _prepare_sdk_environment(sdk_root)

    module_path = sdk_root / "bimanual" / "script" / "single_arm.py"
    if not module_path.exists():
        raise RuntimeError(f"SingleArm script not found: {module_path}")

    spec = importlib.util.spec_from_file_location("arx_single_arm_script", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load SDK module spec from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    single_arm_cls = module.SingleArm

    if not hasattr(single_arm_cls, "cleanup"):
        setattr(single_arm_cls, "cleanup", lambda self: None)

    setattr(single_arm_cls, "__del__", lambda self: None)

    return single_arm_cls


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _read_can_state(can_port: str) -> str:
    sysfs_path = Path("/sys/class/net") / can_port
    if not sysfs_path.exists():
        return "missing"
    try:
        return (sysfs_path / "operstate").read_text(encoding="utf-8").strip()
    except OSError:
        return "unknown"


def _to_deg(values: list[float]) -> list[float]:
    if str(CONFIG["sdk_feedback_unit"]).lower() == "deg":
        return list(values)
    return [math.degrees(v) for v in values]


def _to_command_units(values_deg: list[float]) -> list[float]:
    if str(CONFIG["sdk_command_unit"]).lower() == "deg":
        return list(values_deg)
    return [math.radians(v) for v in values_deg]


def _format_values(values: list[float], precision: int = 2) -> str:
    return "[" + ", ".join(f"{value:.{precision}f}" for value in values) + "]"


def _tty_print(text: str) -> None:
    _TTY_OUTPUT.write(text + "\n")
    _TTY_OUTPUT.flush()


class _RawTerminal:
    def __enter__(self) -> "_RawTerminal":
        self._fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)


class _MuteNativeOutput:
    def __enter__(self) -> "_MuteNativeOutput":
        self._devnull_fd = os.open(os.devnull, os.O_WRONLY)
        self._saved_stdout_fd = os.dup(1)
        self._saved_stderr_fd = os.dup(2)
        os.dup2(self._devnull_fd, 1)
        os.dup2(self._devnull_fd, 2)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        os.dup2(self._saved_stdout_fd, 1)
        os.dup2(self._saved_stderr_fd, 2)
        os.close(self._saved_stdout_fd)
        os.close(self._saved_stderr_fd)
        os.close(self._devnull_fd)


class ArxJointKeyboardApp:
    def __init__(self) -> None:
        self.sdk_root = Path(str(CONFIG["sdk_root"])).resolve()
        self.can_port = str(CONFIG["can_port"])
        self.can_state = _read_can_state(self.can_port)
        if self.can_state == "missing":
            raise RuntimeError(
                f"CAN interface '{self.can_port}' not found. "
                "Bring it up before running this script."
            )

        single_arm_cls = _load_single_arm_class(self.sdk_root)
        arm_config = {
            "can_port": self.can_port,
            "type": int(CONFIG["arm_type"]),
            "dt": float(CONFIG["control_dt_s"]),
        }
        self.arm = single_arm_cls(arm_config)
        self.last_status = "SDK connected. Waiting for feedback."
        self.last_error = ""
        self.motion_enabled = not bool(CONFIG["require_enable_key"])
        self.selected_joint = 0
        self.step_deg = float(CONFIG["joint_step_deg"])
        self.gripper_target = float(CONFIG["gripper_open_value"])
        self.feedback_raw: list[float] = []
        self.feedback_deg: list[float] = [0.0] * 6
        self.target_deg: list[float] = [0.0] * 6
        self._refresh_feedback(initial_sync=True)
        self._m_hold_start: float | None = None
        self._m_last_event: float | None = None
        self._m_home_sent = False

    def _refresh_feedback(self, initial_sync: bool = False) -> None:
        raw = self.arm.get_joint_positions()
        if raw is None:
            raise RuntimeError("SDK returned no joint feedback.")
        values = [float(v) for v in raw]
        if len(values) < 6:
            raise RuntimeError(
                f"Expected at least 6 joint feedback values, got {len(values)}: {values}"
            )

        self.feedback_raw = values
        self.feedback_deg = _to_deg(values[:6])
        if initial_sync:
            self.target_deg = list(self.feedback_deg)
            if len(values) >= 7:
                self.gripper_target = float(values[6])
            self.last_status = "Feedback received. Targets synced to current pose."

    def _print_feedback_line(self) -> None:
        motion = "ON" if self.motion_enabled else "OFF"
        _tty_print(
            f"sel=J{self.selected_joint + 1} motion={motion} step={self.step_deg:.2f} "
            f"deg={_format_values(self.feedback_deg)} grip={self.gripper_target:.3f}"
        )

    def _send_joint_target(self) -> None:
        joint_min = [float(v) for v in CONFIG["joint_limit_min_deg"]]
        joint_max = [float(v) for v in CONFIG["joint_limit_max_deg"]]
        self.target_deg = [
            _clip(value, joint_min[i], joint_max[i]) for i, value in enumerate(self.target_deg)
        ]
        command = _to_command_units(self.target_deg)
        self.arm.set_joint_positions(command)
        self.last_status = (
            f"Sent joints {self.selected_joint + 1} target {self.target_deg[self.selected_joint]:.2f} deg"
        )
        time.sleep(float(CONFIG["control_dt_s"]))
        self._refresh_feedback()
        self._print_feedback_line()

    def _send_gripper_target(self) -> None:
        self.arm.set_catch_pos(float(self.gripper_target))
        self.last_status = f"Sent gripper target {self.gripper_target:.3f}"
        time.sleep(float(CONFIG["control_dt_s"]))
        self._refresh_feedback()
        self._print_feedback_line()

    def _move_selected_joint(self, delta_deg: float) -> None:
        self.target_deg[self.selected_joint] += delta_deg
        if self.motion_enabled:
            self._send_joint_target()
        else:
            self._print_feedback_line()

    def _move_gripper(self, delta: float) -> None:
        self.gripper_target += delta
        if self.motion_enabled:
            self._send_gripper_target()
        else:
            self._print_feedback_line()

    def _sync_target_to_feedback(self) -> None:
        self.target_deg = list(self.feedback_deg)
        if len(self.feedback_raw) >= 7:
            self.gripper_target = float(self.feedback_raw[6])
        self._print_feedback_line()

    def _set_step(self, scale: float) -> None:
        self.step_deg = _clip(
            self.step_deg * scale,
            float(CONFIG["joint_step_deg_min"]),
            float(CONFIG["joint_step_deg_max"]),
        )
        self._print_feedback_line()

    def _toggle_motion(self) -> None:
        self.motion_enabled = not self.motion_enabled
        self._print_feedback_line()

    def _call_mode(self, fn_name: str) -> None:
        fn = getattr(self.arm, fn_name)
        fn()
        time.sleep(float(CONFIG["control_dt_s"]))
        self._refresh_feedback()
        self._print_feedback_line()

    def _reset_home_hold(self) -> None:
        self._m_hold_start = None
        self._m_last_event = None
        self._m_home_sent = False

    def _handle_m_hold(self, now: float) -> None:
        if self._m_hold_start is None:
            self._m_hold_start = now
            self._m_last_event = now
            self._m_home_sent = False
            return
        gap_s = float(CONFIG["home_repeat_gap_seconds"])
        if self._m_last_event is None or now - self._m_last_event > gap_s:
            self._m_hold_start = now
            self._m_home_sent = False
        self._m_last_event = now
        if not self._m_home_sent and now - self._m_hold_start >= float(CONFIG["home_hold_seconds"]):
            self._m_home_sent = True
            self._call_mode("go_home")

    def run(self) -> None:
        poll_timeout = 1.0 / max(float(CONFIG["poll_hz"]), 1.0)
        self._print_feedback_line()
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], poll_timeout)
            now = time.monotonic()
            if not ready:
                if self._m_last_event is not None and now - self._m_last_event > float(
                    CONFIG["home_repeat_gap_seconds"]
                ):
                    self._reset_home_hold()
                continue
            key = sys.stdin.read(1)
            if not key:
                continue
            if key == "m":
                self._handle_m_hold(now)
                continue
            self._reset_home_hold()
            if key == "q":
                break
            if "1" <= key <= "6":
                self.selected_joint = ord(key) - ord("1")
                self._print_feedback_line()
                continue
            if key in ("j", "["):
                self._move_selected_joint(-self.step_deg)
                continue
            if key in ("k", "]"):
                self._move_selected_joint(self.step_deg)
                continue
            if key == "o":
                self._move_gripper(abs(float(CONFIG["gripper_step"])))
                continue
            if key == "c":
                self._move_gripper(-abs(float(CONFIG["gripper_step"])))
                continue
            if key == "z":
                self._set_step(0.5)
                continue
            if key == "x":
                self._set_step(2.0)
                continue
            if key == "r":
                self._sync_target_to_feedback()
                continue
            if key == "e":
                self._toggle_motion()
                continue
            if key == "g":
                self._call_mode("gravity_compensation")
                continue
            if key == "p":
                self._call_mode("protect_mode")
                continue
            if key == "h":
                self._call_mode("go_home")
                continue

    def close(self) -> None:
        if bool(CONFIG["protect_on_exit"]):
            try:
                self.arm.protect_mode()
            except Exception:
                pass


def main() -> None:
    _maybe_reexec_with_ros_env()
    try:
        app = ArxJointKeyboardApp()
    except Exception as exc:
        _tty_print(f"[arx_x5] Startup failed: {exc}")
        raise SystemExit(1) from exc

    try:
        with _MuteNativeOutput(), _RawTerminal():
            app.run()
    finally:
        app.close()


if __name__ == "__main__":
    main()
