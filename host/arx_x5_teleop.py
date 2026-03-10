"""Script: arx_x5_teleop.py
Purpose: Launch the main ARX X5 teleoperation loop using potentiometer, button, and IMU inputs with direct mapping.
Usage: python -m host.arx_x5_teleop
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.machinery
import importlib.util
import math
import multiprocessing as mp
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
_SYSTEM_PYTHON = Path("/usr/bin/python3")
_SDK_PYTHON_VERSION = (3, 10)
_ROS_HUMBLE_ROOT = Path("/opt/ros/humble")
_ROS_HUMBLE_LIB = _ROS_HUMBLE_ROOT / "lib"
_ROS_HUMBLE_ARCH_LIB = _ROS_HUMBLE_LIB / "x86_64-linux-gnu"
try:
    _TTY_OUTPUT = open("/dev/tty", "w", buffering=1)
except OSError:
    _TTY_OUTPUT = sys.stdout


def _prepend_env_path(name: str, path: Path, env: dict[str, str] | None = None) -> None:
    target_env = os.environ if env is None else env
    value = str(path)
    current = target_env.get(name, "")
    parts = [p for p in current.split(":") if p]
    if value not in parts:
        target_env[name] = value if not current else f"{value}:{current}"


def _prepend_sys_path(path: Path) -> None:
    value = str(path)
    if path.exists() and value not in sys.path:
        sys.path.insert(0, value)


def _prepare_host_python_paths() -> None:
    for path in sorted(Path.home().glob(".local/lib/python*/site-packages")):
        _prepend_env_path("PYTHONPATH", path)
        _prepend_sys_path(path)


def _sdk_python_compatible() -> bool:
    return sys.version_info[:2] == _SDK_PYTHON_VERSION


def _maybe_reexec_with_system_python() -> None:
    if os.environ.get("ARX_HOST_SYSTEM_PY_READY") == "1":
        return
    if _sdk_python_compatible():
        os.environ["ARX_HOST_SYSTEM_PY_READY"] = "1"
        return

    preferred_python = Path(os.environ.get("ARX_TELEOP_PYTHON", str(_SYSTEM_PYTHON)))
    if not preferred_python.exists():
        return

    env = os.environ.copy()
    env["ARX_HOST_SYSTEM_PY_READY"] = "1"
    _prepend_env_path("PYTHONPATH", _REPO_ROOT, env)
    for path in sorted(Path.home().glob(".local/lib/python*/site-packages")):
        _prepend_env_path("PYTHONPATH", path, env)
    os.execvpe(
        str(preferred_python),
        [str(preferred_python), "-m", "host.arx_x5_teleop", *sys.argv[1:]],
        env,
    )


_prepare_host_python_paths()
_maybe_reexec_with_system_python()

from host.control_types import ControlFrame
from host.teleop_input import DEFAULT_INPUT_CONFIG, TeleopInputRuntime, add_input_args


RUN_CONFIG: dict[str, Any] = {
    **DEFAULT_INPUT_CONFIG,
    "mode": "full_teleop",
    "loop_rate_hz": 20.0,
    "allow_imu_zero_fallback": False,
    # ARX SDK / CAN
    "sdk_root": str(_DEFAULT_SDK_ROOT),
    "can_port": "can1",
    "arm_type": 0,
    "control_dt_s": 0.05,
    "sdk_command_unit": "rad",
    "sdk_feedback_unit": "rad",
    "protect_on_exit": True,
    # Startup / teleop flow
    "test_mode": False,
    "home_joint_deg": [90.0, 15.0, 15.0, 0.0, 0.0, 0.0],
    "n_home_joint_deg": [90.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "home_gripper_value": 0.0,
    "target_print_epsilon_deg": 0.1,
    "lower_scale_deg_per_deg": 0.8,
    "imu_scale_deg_per_deg": 0.8,
    "input_jump_threshold_deg": 15.0,
    "enable_joint_target_filter": True,
    "joint_target_filter_min_cutoff_hz": 2.0,
    "joint_target_filter_beta": 0.05,
    "joint_target_filter_d_cutoff_hz": 1.0,
    "joint_target_deadband_deg": 0.6,
    "home_settle_seconds": 1.0,
    "m_hold_seconds": 1.0,
    "m_repeat_gap_seconds": 0.35,
    "n_hold_seconds": 1.0,
    "n_repeat_gap_seconds": 0.35,
    "gripper_rate_units_per_s": 1.5,
    "gripper_min_value": -1.0,
    "gripper_max_value": 5.0,
    "gripper_open_input": 1,
    "gripper_close_input": 0,
    "gripper_input_debounce_s": 0.05,
    "gripper_open_direction": 1.0,
    # Conservative software speed limit for teleop and scripted return moves.
    "enable_joint_rate_limit": True,
    "joint_rate_limit_deg_per_s": [30.0, 10.0, 10.0, 45.0, 45.0, 45.0],
    "joint_limit_min_deg": [-180.0, -180.0, -180.0, -180.0, -180.0, -180.0],
    "joint_limit_max_deg": [180.0, 180.0, 180.0, 180.0, 180.0, 180.0],
    # Mapping in robot joint order [J1, J2, J3, J4, J5, J6].
    # mode=fixed:
    #   joint_deg = fixed_deg
    # mode=absolute:
    #   joint_deg = joint_ref_deg + scale * (source_value - source_ref_value)
    # mode=incremental:
    #   joint_deg = session_ref_joint_deg + scale * unwrapped(source_value - session_ref_source_value)
    "joint_mapping": [
        {"mode": "fixed", "fixed_deg": 90.0},
        {
            "mode": "absolute",
            "source": "ch1_deg",
            "source_ref_value": 300.0,
            "joint_ref_deg": 15.0,
            "scale": -1.0,
            "use_lower_scale": True,
        },
        {
            "mode": "absolute",
            "source": "ch2_deg",
            "source_ref_value": 300.0,
            "joint_ref_deg": 15.0,
            "scale": -1.0,
            "use_lower_scale": True,
        },
        {"mode": "absolute", "source": "roll_deg", "source_ref_value": 0.0, "joint_ref_deg": 0.0, "scale": 1.0, "use_imu_scale": True},
        {"mode": "incremental", "source": "yaw_deg", "joint_ref_deg": 0.0, "scale": -1.0, "use_imu_scale": True},
        {"mode": "absolute", "source": "pitch_deg", "source_ref_value": 0.0, "joint_ref_deg": 0.0, "scale": -1.0, "use_imu_scale": True},
    ],
}


def _find_first(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.rglob(pattern))
    return matches[0] if matches else None


def _load_shared_library(path: Path) -> None:
    if not path.exists():
        return
    mode = getattr(ctypes, "RTLD_GLOBAL", 0)
    ctypes.CDLL(str(path), mode=mode)


def _prepare_ros_environment() -> None:
    if _ROS_HUMBLE_ROOT.exists():
        os.environ.setdefault("ROS_DISTRO", "humble")
        os.environ.setdefault("ROS_VERSION", "2")
        os.environ.setdefault("ROS_PYTHON_VERSION", "3")
        os.environ.setdefault("AMENT_PREFIX_PATH", str(_ROS_HUMBLE_ROOT))
        os.environ.setdefault("COLCON_PREFIX_PATH", str(_ROS_HUMBLE_ROOT))
        os.environ.setdefault("CMAKE_PREFIX_PATH", str(_ROS_HUMBLE_ROOT))
        os.environ.setdefault("AMENT_CURRENT_PREFIX", str(_ROS_HUMBLE_ROOT))

    ros_python = _ROS_HUMBLE_LIB / "python3.10" / "site-packages"
    for path in [_ROS_HUMBLE_LIB, _ROS_HUMBLE_ARCH_LIB, ros_python]:
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

    quoted_args = " ".join(shlex.quote(arg) for arg in sys.argv[1:])
    cmd = (
        f"source {shlex.quote(str(setup_bash))} >/dev/null 2>&1 && "
        "export ARX_ROS_ENV_READY=1 && "
        f"exec {shlex.quote(sys.executable)} -m host.arx_x5_teleop {quoted_args}"
    )
    os.execvpe("/bin/bash", ["bash", "-lc", cmd], os.environ.copy())


def _sdk_setup_hint(sdk_root: Path) -> str:
    return (
        "ARX Python SDK build artifacts not found.\n"
        f"Expected under: {sdk_root / 'bimanual' / 'api'}\n"
        "Run:\n"
        f"  cd {sdk_root}\n"
        "  ./build.sh"
    )


def _require_python_module(name: str, install_hint: str) -> None:
    if importlib.util.find_spec(name) is not None:
        return
    raise RuntimeError(
        f"Missing Python module '{name}' for interpreter {sys.executable} "
        f"({sys.version.split()[0]}).\n"
        f"{install_hint}"
    )


def _prepare_sdk_environment(sdk_root: Path) -> None:
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
                "Rebuild the SDK with the same interpreter or use /usr/bin/python3."
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
        _prepend_sys_path(path)
    if solver_module is not None:
        _prepend_sys_path(solver_module.parent)

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


def _load_single_arm_class(sdk_root: Path):
    _require_python_module(
        "numpy",
        "Recommended setup:\n"
        "  /usr/bin/python3 -m venv .venv-arx\n"
        "  source .venv-arx/bin/activate\n"
        "  python -m pip install numpy pyserial\n"
        "Then run:\n"
        "  ARX_TELEOP_PYTHON=$(pwd)/.venv-arx/bin/python python -m host.arx_x5_teleop",
    )
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
    setattr(single_arm_cls, "__del__", lambda self: None)
    return single_arm_cls


def _read_can_state(can_port: str) -> str:
    sysfs_path = Path("/sys/class/net") / can_port
    if not sysfs_path.exists():
        return "missing"
    try:
        return (sysfs_path / "operstate").read_text(encoding="utf-8").strip()
    except OSError:
        return "unknown"


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _format_list(values: list[float], precision: int = 2) -> str:
    return "[" + ", ".join(f"{value:.{precision}f}" for value in values) + "]"


def _console_print(text: str) -> None:
    _TTY_OUTPUT.write(text + "\n")
    _TTY_OUTPUT.flush()


def _null_print(_: str) -> None:
    return


def _to_deg(values: list[float], unit_name: str) -> list[float]:
    if unit_name.lower() == "deg":
        return list(values)
    return [math.degrees(v) for v in values]


def _to_command_units(values_deg: list[float], unit_name: str) -> list[float]:
    if unit_name.lower() == "deg":
        return list(values_deg)
    return [math.radians(v) for v in values_deg]


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


def _wait_for_enter(prompt: str) -> None:
    _console_print(prompt)
    input()


def _wrap_deg(value: float) -> float:
    while value > 180.0:
        value -= 360.0
    while value < -180.0:
        value += 360.0
    return value


def _apply_rate_limit_deg(
    target_deg: list[float],
    last_deg: list[float] | None,
    dt_s: float,
) -> list[float]:
    if not bool(RUN_CONFIG["enable_joint_rate_limit"]) or last_deg is None:
        return list(target_deg)
    max_rates = [float(v) for v in RUN_CONFIG["joint_rate_limit_deg_per_s"]]
    limited: list[float] = []
    for i, value in enumerate(target_deg):
        max_step = max_rates[i] * dt_s
        limited.append(_clip(value, last_deg[i] - max_step, last_deg[i] + max_step))
    return limited


def _smoothing_factor(dt_s: float, cutoff_hz: float) -> float:
    if dt_s <= 0.0:
        return 1.0
    cutoff_hz = max(cutoff_hz, 1e-6)
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return dt_s / (dt_s + tau)


class _OneEuroScalarFilter:
    def __init__(self, min_cutoff_hz: float, beta: float, d_cutoff_hz: float) -> None:
        self._min_cutoff_hz = float(min_cutoff_hz)
        self._beta = float(beta)
        self._d_cutoff_hz = float(d_cutoff_hz)
        self._x_prev: float | None = None
        self._x_hat_prev: float | None = None
        self._dx_hat_prev = 0.0

    def reset(self, value: float | None = None) -> None:
        self._x_prev = value
        self._x_hat_prev = value
        self._dx_hat_prev = 0.0

    def filter(self, value: float, dt_s: float) -> float:
        if self._x_prev is None or self._x_hat_prev is None or dt_s <= 0.0:
            self.reset(value)
            return float(value)

        dx = (value - self._x_prev) / dt_s
        alpha_d = _smoothing_factor(dt_s, self._d_cutoff_hz)
        dx_hat = alpha_d * dx + (1.0 - alpha_d) * self._dx_hat_prev
        cutoff_hz = self._min_cutoff_hz + self._beta * abs(dx_hat)
        alpha = _smoothing_factor(dt_s, cutoff_hz)
        x_hat = alpha * value + (1.0 - alpha) * self._x_hat_prev

        self._x_prev = float(value)
        self._x_hat_prev = float(x_hat)
        self._dx_hat_prev = float(dx_hat)
        return float(x_hat)


class JointTargetFilter:
    def __init__(self) -> None:
        self._enabled = bool(RUN_CONFIG["enable_joint_target_filter"])
        self._deadband_deg = float(RUN_CONFIG["joint_target_deadband_deg"])
        self._filters = [
            _OneEuroScalarFilter(
                min_cutoff_hz=float(RUN_CONFIG["joint_target_filter_min_cutoff_hz"]),
                beta=float(RUN_CONFIG["joint_target_filter_beta"]),
                d_cutoff_hz=float(RUN_CONFIG["joint_target_filter_d_cutoff_hz"]),
            )
            for _ in range(6)
        ]
        self._last_output: list[float] | None = None

    def reset(self, target_deg: list[float] | None = None) -> None:
        self._last_output = None if target_deg is None else [float(v) for v in target_deg]
        for i, scalar_filter in enumerate(self._filters):
            value = None if target_deg is None else float(target_deg[i])
            scalar_filter.reset(value)

    def filter(self, target_deg: list[float], dt_s: float) -> list[float]:
        targets = [float(v) for v in target_deg]
        if not self._enabled:
            self._last_output = list(targets)
            return list(targets)

        filtered: list[float] = []
        previous_output = self._last_output
        for i, value in enumerate(targets):
            filtered_value = self._filters[i].filter(value, dt_s)
            if previous_output is not None and abs(filtered_value - previous_output[i]) < self._deadband_deg:
                filtered_value = previous_output[i]
            filtered.append(float(filtered_value))
        self._last_output = list(filtered)
        return list(filtered)


class ArxArmSession:
    def __init__(self, args: argparse.Namespace, status_callback: Any | None = None) -> None:
        self.sdk_root = Path(str(args.sdk_root)).resolve()
        self.can_port = str(args.can_port)
        self.arm_type = int(args.arm_type)
        self.control_dt_s = float(RUN_CONFIG["control_dt_s"])
        self.sdk_command_unit = str(RUN_CONFIG["sdk_command_unit"])
        self.sdk_feedback_unit = str(RUN_CONFIG["sdk_feedback_unit"])
        self.test_mode = bool(RUN_CONFIG["test_mode"]) or bool(args.dry_run)
        self._status_output = status_callback or _null_print
        self._arm = None
        self.feedback_raw: list[float] = []
        self.feedback_deg: list[float] = [0.0] * 6
        self.gripper_feedback = float(RUN_CONFIG["home_gripper_value"])

        can_state = _read_can_state(self.can_port)
        if can_state == "missing":
            raise RuntimeError(
                f"CAN interface '{self.can_port}' not found. Bring it up before running this script."
            )

        single_arm_cls = _load_single_arm_class(self.sdk_root)
        arm_config = {
            "can_port": self.can_port,
            "type": self.arm_type,
            "dt": self.control_dt_s,
        }
        with _MuteNativeOutput():
            self._arm = single_arm_cls(arm_config)
        self.refresh_feedback(initial=True)

    def refresh_feedback(self, initial: bool = False) -> None:
        if self._arm is None:
            raise RuntimeError("ARX SDK session is not initialized.")
        with _MuteNativeOutput():
            raw = self._arm.get_joint_positions()
        if raw is None:
            raise RuntimeError("SDK returned no joint feedback.")
        values = [float(v) for v in raw]
        if len(values) < 6:
            raise RuntimeError(f"Expected at least 6 joint feedback values, got {len(values)}: {values}")
        self.feedback_raw = values
        self.feedback_deg = _to_deg(values[:6], self.sdk_feedback_unit)
        if len(values) >= 7:
            self.gripper_feedback = float(values[6])
        if initial:
            self._status_output(f"[arx_x5] feedback_deg={_format_list(self.feedback_deg)}")
            self._status_output(f"[arx_x5] feedback_raw={_format_list(self.feedback_raw, precision=3)}")
            self._status_output(f"[arx_x5] gripper_feedback={self.gripper_feedback:.3f}")

    def move_joint_targets(self, target_deg: list[float], gripper_target: float, reason: str) -> None:
        if self.test_mode:
            self._status_output(
                f"[{reason}] target_deg={_format_list(target_deg)} "
                f"gripper={gripper_target:.3f} (dry-run)"
            )
            return

        if self._arm is None:
            raise RuntimeError("ARX SDK session is not initialized.")
        command = _to_command_units(target_deg, self.sdk_command_unit)
        with _MuteNativeOutput():
            self._arm.set_joint_positions(command)
            self._arm.set_catch_pos(float(gripper_target))
        time.sleep(self.control_dt_s)
        self.refresh_feedback()

    def move_home(self, reason: str) -> None:
        home_joint_deg = [float(v) for v in RUN_CONFIG["home_joint_deg"]]
        home_gripper = float(RUN_CONFIG["home_gripper_value"])
        self.move_joint_targets(home_joint_deg, home_gripper, reason)
        if not self.test_mode:
            time.sleep(float(RUN_CONFIG["home_settle_seconds"]))
            self.refresh_feedback()

    def protect_mode(self) -> None:
        if self.test_mode or self._arm is None:
            return
        try:
            with _MuteNativeOutput():
                self._arm.protect_mode()
        except Exception:
            pass


def _detach_child_terminal() -> None:
    try:
        os.setsid()
    except OSError:
        pass

    devnull_fd = os.open(os.devnull, os.O_RDWR)
    try:
        for fd in (0, 1, 2):
            try:
                os.dup2(devnull_fd, fd)
            except OSError:
                pass
    finally:
        if devnull_fd > 2:
            os.close(devnull_fd)


def _arm_worker_main(conn: Any, worker_args: dict[str, Any]) -> None:
    _detach_child_terminal()
    arm: ArxArmSession | None = None
    try:
        args = argparse.Namespace(**worker_args)
        arm = ArxArmSession(args, status_callback=None)
        conn.send(
            {
                "type": "ready",
                "feedback_deg": list(arm.feedback_deg),
                "feedback_raw": list(arm.feedback_raw),
                "gripper_feedback": float(arm.gripper_feedback),
                "test_mode": bool(arm.test_mode),
            }
        )

        while True:
            msg = conn.recv()
            command = str(msg.get("cmd", ""))
            if command == "shutdown":
                break
            if command == "home":
                arm.move_home(str(msg.get("reason", "home")))
                conn.send({"type": "ack", "cmd": "home"})
                continue
            if command == "send_targets":
                arm.move_joint_targets(
                    [float(v) for v in msg["target_deg"]],
                    float(msg["gripper_target"]),
                    str(msg.get("reason", "teleop")),
                )
                continue
            if command == "protect":
                arm.protect_mode()
                continue
    except EOFError:
        pass
    except Exception as exc:
        try:
            conn.send({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        if arm is not None:
            try:
                arm.protect_mode()
            except Exception:
                pass
        try:
            conn.close()
        except Exception:
            pass


class ArxArmWorker:
    def __init__(self, args: argparse.Namespace) -> None:
        # The input stack (Arduino serial + vendor BLE adapter) is sensitive to
        # inherited process state. Using "fork" here can duplicate imported
        # module state, locks, and file descriptors into the ARX worker before
        # the IMU link is initialized. "spawn" isolates the ARX SDK process and
        # keeps the parent-side serial/BLE path identical to host.run_teleop.
        ctx = mp.get_context("spawn")
        self._parent_conn, child_conn = ctx.Pipe()
        worker_args = {
            "sdk_root": str(args.sdk_root),
            "can_port": str(args.can_port),
            "arm_type": int(args.arm_type),
            "dry_run": bool(args.dry_run),
        }
        self._process = ctx.Process(
            target=_arm_worker_main,
            args=(child_conn, worker_args),
            daemon=True,
        )
        self._process.start()
        child_conn.close()
        self.feedback_deg: list[float] = [0.0] * 6
        self.feedback_raw: list[float] = []
        self.gripper_feedback = float(RUN_CONFIG["home_gripper_value"])
        self.test_mode = bool(RUN_CONFIG["test_mode"]) or bool(args.dry_run)

    def _fail_if_dead(self) -> None:
        if self._process.is_alive():
            return
        raise RuntimeError("ARX worker process exited unexpectedly.")

    def wait_ready(self, timeout_s: float = 10.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self._fail_if_dead()
            if not self._parent_conn.poll(0.1):
                continue
            msg = self._parent_conn.recv()
            if msg.get("type") == "ready":
                self.feedback_deg = [float(v) for v in msg.get("feedback_deg", [])]
                self.feedback_raw = [float(v) for v in msg.get("feedback_raw", [])]
                self.gripper_feedback = float(msg.get("gripper_feedback", self.gripper_feedback))
                self.test_mode = bool(msg.get("test_mode", self.test_mode))
                return
            if msg.get("type") == "error":
                raise RuntimeError(str(msg.get("message", "Unknown ARX worker error.")))
        raise RuntimeError("Timed out waiting for ARX worker readiness.")

    def poll_messages(self) -> None:
        while self._parent_conn.poll():
            msg = self._parent_conn.recv()
            if msg.get("type") == "error":
                raise RuntimeError(str(msg.get("message", "Unknown ARX worker error.")))

    def move_home(self) -> None:
        self._parent_conn.send({"cmd": "home", "reason": "home"})
        while True:
            self._fail_if_dead()
            if not self._parent_conn.poll(0.1):
                continue
            msg = self._parent_conn.recv()
            if msg.get("type") == "ack" and msg.get("cmd") == "home":
                return
            if msg.get("type") == "error":
                raise RuntimeError(str(msg.get("message", "Unknown ARX worker error.")))

    def send_targets(self, target_deg: list[float], gripper_target: float, reason: str) -> None:
        self._parent_conn.send(
            {
                "cmd": "send_targets",
                "target_deg": list(target_deg),
                "gripper_target": float(gripper_target),
                "reason": reason,
            }
        )

    def protect_mode(self) -> None:
        if self._process.is_alive():
            self._parent_conn.send({"cmd": "protect"})

    def close(self) -> None:
        if self._process.is_alive():
            try:
                self._parent_conn.send({"cmd": "shutdown"})
            except Exception:
                pass
            self._process.join(timeout=2.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1.0)


class TeleopMapper:
    def __init__(self) -> None:
        self._mapping = list(RUN_CONFIG["joint_mapping"])
        self._min_deg = [float(v) for v in RUN_CONFIG["joint_limit_min_deg"]]
        self._max_deg = [float(v) for v in RUN_CONFIG["joint_limit_max_deg"]]
        self._jump_threshold_deg = float(RUN_CONFIG["input_jump_threshold_deg"])
        self._lower_scale = float(RUN_CONFIG["lower_scale_deg_per_deg"])
        self._imu_scale = float(RUN_CONFIG["imu_scale_deg_per_deg"])
        self._last_raw_sources: dict[str, float] = {}
        self._session_ref_sources: dict[int, float] = {}
        self._session_ref_joint_deg: dict[int, float] = {}
        self._last_accepted_targets = [float(v) for v in RUN_CONFIG["home_joint_deg"]]

    def reset_for_session(self, start_targets_deg: list[float], frame: ControlFrame | None = None) -> None:
        self._last_accepted_targets = list(start_targets_deg)
        self._last_raw_sources = {}
        self._session_ref_sources = {}
        self._session_ref_joint_deg = {}
        if frame is None:
            return
        for i, item in enumerate(self._mapping):
            mode = str(item.get("mode", "absolute"))
            if mode == "incremental":
                source_name = str(item["source"])
                self._session_ref_sources[i] = float(getattr(frame, source_name))
                self._session_ref_joint_deg[i] = float(start_targets_deg[i])
                self._last_raw_sources[source_name] = float(getattr(frame, source_name))
            elif mode == "absolute":
                source_name = str(item["source"])
                self._last_raw_sources[source_name] = float(getattr(frame, source_name))

    def _scale_for_item(self, item: dict[str, Any]) -> float:
        scale = float(item.get("scale", 1.0))
        if bool(item.get("use_lower_scale", False)):
            scale *= self._lower_scale
        if bool(item.get("use_imu_scale", False)):
            scale *= self._imu_scale
        return scale

    def map_frame(self, frame: ControlFrame) -> tuple[list[float], str | None]:
        for item in self._mapping:
            mode = str(item.get("mode", "absolute"))
            if mode == "fixed":
                continue
            source_name = str(item["source"])
            current_value = float(getattr(frame, source_name))
            previous_value = self._last_raw_sources.get(source_name)
            if previous_value is None:
                self._last_raw_sources[source_name] = current_value
                continue
            delta = _wrap_deg(current_value - previous_value)
            if abs(delta) > self._jump_threshold_deg:
                return (
                    list(self._last_accepted_targets),
                    f"[safety] source jump blocked: {source_name} delta={delta:.2f} deg "
                    f"(threshold={self._jump_threshold_deg:.2f})",
                )

        targets: list[float] = []
        for i, item in enumerate(self._mapping):
            mode = str(item.get("mode", "absolute"))
            if mode == "fixed":
                mapped = float(item["fixed_deg"])
            elif mode == "absolute":
                source_name = str(item["source"])
                source_value = float(getattr(frame, source_name))
                source_ref = float(item["source_ref_value"])
                joint_ref = float(item["joint_ref_deg"])
                mapped = joint_ref + self._scale_for_item(item) * (source_value - source_ref)
            elif mode == "incremental":
                source_name = str(item["source"])
                source_value = float(getattr(frame, source_name))
                source_ref = self._session_ref_sources.get(i, source_value)
                joint_ref = self._session_ref_joint_deg.get(i, float(item.get("joint_ref_deg", 0.0)))
                mapped = joint_ref + self._scale_for_item(item) * _wrap_deg(source_value - source_ref)
            else:
                raise RuntimeError(f"Unknown joint mapping mode: {mode}")
            targets.append(_clip(mapped, self._min_deg[i], self._max_deg[i]))

        for item in self._mapping:
            mode = str(item.get("mode", "absolute"))
            if mode == "fixed":
                continue
            source_name = str(item["source"])
            self._last_raw_sources[source_name] = float(getattr(frame, source_name))

        self._last_accepted_targets = list(targets)
        return list(targets), None


class GripperController:
    def __init__(self, initial_value: float) -> None:
        self._current = float(initial_value)
        self._rate = float(RUN_CONFIG["gripper_rate_units_per_s"])
        self._min_value = float(RUN_CONFIG["gripper_min_value"])
        self._max_value = float(RUN_CONFIG["gripper_max_value"])
        self._open_input = int(RUN_CONFIG["gripper_open_input"])
        self._close_input = int(RUN_CONFIG["gripper_close_input"])
        self._debounce_s = float(RUN_CONFIG["gripper_input_debounce_s"])
        self._open_direction = 1.0 if float(RUN_CONFIG["gripper_open_direction"]) >= 0.0 else -1.0
        self._stable_input = self._close_input
        self._candidate_input = self._stable_input
        self._candidate_since = 0.0

    @property
    def current(self) -> float:
        return self._current

    def reset(self, value: float) -> None:
        self._current = _clip(float(value), self._min_value, self._max_value)
        self._stable_input = self._close_input
        self._candidate_input = self._stable_input
        self._candidate_since = 0.0

    def _filtered_input(self, grip_input: int, now_s: float) -> int:
        raw = int(grip_input)
        if raw == self._stable_input:
            self._candidate_input = raw
            self._candidate_since = now_s
            return self._stable_input
        if raw != self._candidate_input:
            self._candidate_input = raw
            self._candidate_since = now_s
            return self._stable_input
        if now_s - self._candidate_since >= self._debounce_s:
            self._stable_input = raw
        return self._stable_input

    def preview(self, grip_input: int, dt_s: float, now_s: float | None = None) -> float:
        stable_input = self._filtered_input(grip_input, time.monotonic() if now_s is None else now_s)
        if stable_input == self._open_input:
            direction = self._open_direction
        elif stable_input == self._close_input:
            direction = -self._open_direction
        else:
            direction = 0.0
        return _clip(self._current + direction * self._rate * dt_s, self._min_value, self._max_value)

    def update(self, grip_input: int, dt_s: float, now_s: float | None = None) -> float:
        self._current = self.preview(grip_input, dt_s, now_s=now_s)
        return self._current


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ARX X5 teleop script: verify comms, map merged 7D input to joint targets, and optionally send."
    )
    add_input_args(parser, RUN_CONFIG)
    parser.add_argument("--sdk-root", default=str(RUN_CONFIG["sdk_root"]))
    parser.add_argument("--can-port", default=str(RUN_CONFIG["can_port"]))
    parser.add_argument("--arm-type", type=int, default=int(RUN_CONFIG["arm_type"]))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print target joints/gripper without moving the arm.",
    )
    parser.add_argument(
        "--joint-speed-limit-deg-per-s",
        type=float,
        default=None,
        help="Override the software max joint speed limit for all six joints.",
    )
    parser.add_argument(
        "--disable-joint-speed-limit",
        action="store_true",
        help="Disable software joint speed limiting.",
    )
    return parser.parse_args()


def _wait_for_links_ready(runtime: TeleopInputRuntime, loop_hz: float) -> None:
    loop_period = 1.0 / max(loop_hz, 1.0)
    last_wait_print = 0.0
    reported_lower = False
    reported_imu = False

    while True:
        runtime.poll_frame()
        if runtime.lower_ready and not reported_lower:
            _console_print("Arduino stream is active.")
            reported_lower = True
        if runtime.imu_ready and not reported_imu:
            _console_print("IMU stream is active.")
            reported_imu = True
        if runtime.lower_ready and runtime.imu_ready:
            return
        now = time.monotonic()
        if now - last_wait_print >= 1.0:
            _console_print(f"Waiting for: {', '.join(runtime.missing_inputs())}")
            last_wait_print = now
        time.sleep(loop_period)


def _command_changed(
    current_deg: list[float],
    previous_deg: list[float] | None,
    current_gripper: float,
    previous_gripper: float | None,
) -> bool:
    if previous_deg is None or previous_gripper is None:
        return True
    epsilon = float(RUN_CONFIG["target_print_epsilon_deg"])
    if abs(current_gripper - previous_gripper) > 1e-6:
        return True
    return any(abs(a - b) > epsilon for a, b in zip(current_deg, previous_deg))


def _targets_reached(current_deg: list[float], target_deg: list[float], epsilon_deg: float = 1e-3) -> bool:
    return all(abs(a - b) <= epsilon_deg for a, b in zip(current_deg, target_deg))


def _move_arm_with_rate_limit(
    arm: "ArxArmWorker",
    start_deg: list[float],
    target_deg: list[float],
    gripper_target: float,
    reason: str,
    settle_s: float = 0.0,
) -> list[float]:
    dt_s = float(RUN_CONFIG["control_dt_s"])
    current_deg = [float(v) for v in start_deg]
    target_deg = [float(v) for v in target_deg]

    while True:
        next_deg = _apply_rate_limit_deg(target_deg, current_deg, dt_s)
        arm.send_targets(next_deg, gripper_target, reason)
        arm.poll_messages()
        current_deg = list(next_deg)
        if _targets_reached(current_deg, target_deg):
            break
        time.sleep(dt_s)

    if settle_s > 0.0 and not arm.test_mode:
        time.sleep(settle_s)
    return current_deg


def _apply_runtime_overrides(args: argparse.Namespace) -> None:
    if getattr(args, "disable_joint_speed_limit", False):
        RUN_CONFIG["enable_joint_rate_limit"] = False
        return

    limit = getattr(args, "joint_speed_limit_deg_per_s", None)
    if limit is None:
        return
    if limit <= 0.0:
        raise RuntimeError("--joint-speed-limit-deg-per-s must be > 0.")
    RUN_CONFIG["enable_joint_rate_limit"] = True
    RUN_CONFIG["joint_rate_limit_deg_per_s"] = [float(limit)] * 6


def _describe_joint_speed_limit() -> str:
    if not bool(RUN_CONFIG["enable_joint_rate_limit"]):
        return "[arx_x5] joint_speed_limit=disabled"
    limits = [float(v) for v in RUN_CONFIG["joint_rate_limit_deg_per_s"]]
    return f"[arx_x5] joint_speed_limit_deg_per_s={_format_list(limits)}"


def _run_teleop_loop(
    runtime: TeleopInputRuntime,
    arm: ArxArmWorker,
    mapper: TeleopMapper,
    joint_filter: JointTargetFilter,
    gripper: GripperController,
    loop_hz: float,
) -> str:
    loop_period = 1.0 / max(loop_hz, 1.0)
    last_wait_print = 0.0
    previous_printed_deg: list[float] | None = None
    previous_printed_gripper: float | None = None
    previous_command_deg = [float(v) for v in RUN_CONFIG["home_joint_deg"]]
    m_hold_start: float | None = None
    m_last_event: float | None = None
    m_home_sent = False
    n_hold_start: float | None = None
    n_last_event: float | None = None
    n_home_sent = False
    pending_n_home_reset = False

    _console_print(
        "Teleop armed. Press 'q' to quit. Long-hold 'm' to stop teleop and return home. "
        "Long-hold 'n' to return to [90, 0, 0, 0, 0, 0]."
    )

    with _RawTerminal():
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], loop_period)
            now = time.monotonic()
            if ready:
                key = sys.stdin.read(1)
                if key == "q":
                    return "quit"
                if key == "m":
                    if m_hold_start is None:
                        m_hold_start = now
                        m_last_event = now
                        m_home_sent = False
                    else:
                        gap_s = float(RUN_CONFIG["m_repeat_gap_seconds"])
                        if m_last_event is None or now - m_last_event > gap_s:
                            m_hold_start = now
                            m_home_sent = False
                        m_last_event = now
                        if (
                            not m_home_sent
                            and now - m_hold_start >= float(RUN_CONFIG["m_hold_seconds"])
                        ):
                            m_home_sent = True
                            _console_print("[teleop] long-hold m detected. Returning home.")
                            previous_command_deg = _move_arm_with_rate_limit(
                                arm,
                                previous_command_deg,
                                [float(v) for v in RUN_CONFIG["home_joint_deg"]],
                                float(RUN_CONFIG["home_gripper_value"]),
                                "home",
                                settle_s=float(RUN_CONFIG["home_settle_seconds"]),
                            )
                            joint_filter.reset([float(v) for v in RUN_CONFIG["home_joint_deg"]])
                            gripper.reset(float(RUN_CONFIG["home_gripper_value"]))
                            return "stopped"
                elif key == "n":
                    if n_hold_start is None:
                        n_hold_start = now
                        n_last_event = now
                        n_home_sent = False
                    else:
                        gap_s = float(RUN_CONFIG["n_repeat_gap_seconds"])
                        if n_last_event is None or now - n_last_event > gap_s:
                            n_hold_start = now
                            n_home_sent = False
                        n_last_event = now
                        if (
                            not n_home_sent
                            and now - n_hold_start >= float(RUN_CONFIG["n_hold_seconds"])
                        ):
                            n_home_sent = True
                            target_deg = [float(v) for v in RUN_CONFIG["n_home_joint_deg"]]
                            _console_print(
                                f"[teleop] long-hold n detected. Returning to {_format_list(target_deg)}."
                            )
                            previous_command_deg = _move_arm_with_rate_limit(
                                arm,
                                previous_command_deg,
                                target_deg,
                                gripper.current,
                                "n-home",
                            )
                            joint_filter.reset(target_deg)
                            previous_printed_deg = None
                            previous_printed_gripper = None
                            pending_n_home_reset = True
                else:
                    m_hold_start = None
                    m_last_event = None
                    m_home_sent = False
                    n_hold_start = None
                    n_last_event = None
                    n_home_sent = False
            else:
                if m_last_event is not None and now - m_last_event > float(
                    RUN_CONFIG["m_repeat_gap_seconds"]
                ):
                    m_hold_start = None
                    m_last_event = None
                    m_home_sent = False
                if n_last_event is not None and now - n_last_event > float(
                    RUN_CONFIG["n_repeat_gap_seconds"]
                ):
                    n_hold_start = None
                    n_last_event = None
                    n_home_sent = False

            frame = runtime.poll_frame()
            if frame is None:
                if now - last_wait_print >= 1.0:
                    _console_print(f"[teleop] waiting for: {', '.join(runtime.missing_inputs())}")
                    last_wait_print = now
                continue

            if pending_n_home_reset:
                mapper.reset_for_session([float(v) for v in RUN_CONFIG["n_home_joint_deg"]], frame)
                pending_n_home_reset = False
                continue

            target_deg, safety_message = mapper.map_frame(frame)
            if safety_message is not None:
                _console_print(safety_message)
            target_deg = joint_filter.filter(target_deg, loop_period)
            target_deg = _apply_rate_limit_deg(target_deg, previous_command_deg, loop_period)
            gripper_target = gripper.update(int(frame.grip), loop_period, now_s=now)

            if arm.test_mode:
                if _command_changed(
                    target_deg,
                    previous_printed_deg,
                    gripper_target,
                    previous_printed_gripper,
                ):
                    _console_print(
                        f"[teleop] target_deg={_format_list(target_deg)} "
                        f"gripper={gripper_target:.3f}"
                    )
                    previous_printed_deg = list(target_deg)
                    previous_printed_gripper = gripper_target
            else:
                arm.send_targets(target_deg, gripper_target, "teleop")
                arm.poll_messages()

            previous_command_deg = list(target_deg)


def main() -> None:
    _maybe_reexec_with_ros_env()
    args = parse_args()
    _apply_runtime_overrides(args)
    runtime = TeleopInputRuntime(args)
    arm: ArxArmWorker | None = None
    mapper = TeleopMapper()
    joint_filter = JointTargetFilter()
    gripper: GripperController | None = None

    try:
        for line in runtime.describe_setup():
            _console_print(line)
        arm = ArxArmWorker(args)
        arm.wait_ready()
        _console_print(
            f"[arx_x5] sdk_root={Path(str(args.sdk_root)).resolve()} "
            f"can_port={args.can_port} mode={'dry-run' if arm.test_mode else 'live'}"
        )
        _console_print(_describe_joint_speed_limit())
        _console_print(f"[arx_x5] feedback_deg={_format_list(arm.feedback_deg)}")
        _console_print(f"[arx_x5] feedback_raw={_format_list(arm.feedback_raw, precision=3)}")
        _console_print(f"[arx_x5] gripper_feedback={arm.gripper_feedback:.3f}")
        gripper = GripperController(arm.gripper_feedback)

        _wait_for_links_ready(runtime, args.rate_hz)
        _wait_for_enter("All links ready. Press Enter to continue to home pose.")
        _move_arm_with_rate_limit(
            arm,
            list(arm.feedback_deg),
            [float(v) for v in RUN_CONFIG["home_joint_deg"]],
            float(RUN_CONFIG["home_gripper_value"]),
            "home",
            settle_s=float(RUN_CONFIG["home_settle_seconds"]),
        )
        joint_filter.reset([float(v) for v in RUN_CONFIG["home_joint_deg"]])
        gripper.reset(float(RUN_CONFIG["home_gripper_value"]))

        while True:
            _wait_for_enter("Home pose prepared. Press Enter to start teleoperation.")
            start_frame = None
            while start_frame is None:
                start_frame = runtime.poll_frame()
                if start_frame is None:
                    time.sleep(1.0 / max(args.rate_hz, 1.0))
            mapper.reset_for_session([float(v) for v in RUN_CONFIG["home_joint_deg"]], start_frame)
            joint_filter.reset([float(v) for v in RUN_CONFIG["home_joint_deg"]])
            preview_target_deg, preview_safety_message = mapper.map_frame(start_frame)
            preview_target_deg = joint_filter.filter(preview_target_deg, 1.0 / max(args.rate_hz, 1.0))
            preview_gripper_target = gripper.preview(
                int(start_frame.grip),
                1.0 / max(args.rate_hz, 1.0),
                now_s=time.monotonic(),
            )
            if preview_safety_message is not None:
                _console_print(preview_safety_message)
            _console_print(
                f"[preview] target_deg={_format_list(preview_target_deg)} "
                f"gripper={preview_gripper_target:.3f}"
            )
            _wait_for_enter("Preview printed. Press Enter to begin live teleoperation.")
            result = _run_teleop_loop(runtime, arm, mapper, joint_filter, gripper, args.rate_hz)
            if result == "quit":
                break
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        _console_print(f"[arx_x5] runtime failed: {exc}")
        raise
    finally:
        runtime.close()
        if arm is not None:
            if bool(RUN_CONFIG["protect_on_exit"]):
                arm.protect_mode()
            arm.close()


if __name__ == "__main__":
    main()
