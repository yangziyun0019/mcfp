"""Script: teleop_input.py
Purpose: Merge Arduino lower-arm input and IMU orientation input into a shared teleoperation control frame.
Usage: Imported by host.arx_x5_teleop and host.run_teleop.
"""

from __future__ import annotations

import argparse
import time
from typing import Any, Optional

from host.control_types import ControlFrame, ImuInput, LowerInput
from host.imu_sources import MockImuSource, VendorImuSource
from host.serial_reader import SerialLowerReader, autodetect_serial_port


DEFAULT_INPUT_CONFIG: dict[str, Any] = {
    # "imu_only": test IMU stream only
    # "full_teleop": merge Arduino + IMU into 7D control output
    "mode": "full_teleop",
    # 20 Hz matches the current WT adapter stream rate.
    "loop_rate_hz": 20.0,
    # Print rate used in imu_only mode.
    "imu_print_hz": 20.0,
    # Arduino Nano serial input:
    # A0/A1/A2 -> first 3 joints, A3 -> grip button
    "arduino_port": "",
    "arduino_baud": 115200,
    # IMU source:
    # roll/pitch/yaw from vendor BLE adapter + WT901 sensor
    "imu_source": "vendor",  # "vendor" or "mock"
    "imu_module": "host.imu_vendor_adapter",
    # Vendor IMU adapter options
    "imu_adapter_port": "/dev/ttyACM1",
    "imu_device_prefix": "WT",
    "imu_device_index": 0,
    "imu_scan_timeout_s": 12.0,
    "imu_adapter_baud": 115200,
    "imu_read_config_on_connect": False,
    # "keep", "bind", "unbind"
    "imu_binding_mode": "keep",
    "imu_set_rate_on_connect": "keep",
    # If IMU is still pending, keep producing 7D vectors with zeroed RPY.
    "allow_imu_zero_fallback": True,
}


def build_frame(lower: LowerInput, imu: ImuInput) -> ControlFrame:
    return ControlFrame(
        ch1_deg=lower.pot0_deg,
        ch2_deg=lower.pot1_deg,
        ch3_deg=lower.pot2_deg,
        roll_deg=imu.roll_deg,
        pitch_deg=imu.pitch_deg,
        yaw_deg=imu.yaw_deg,
        grip=lower.grip,
        t_host=time.monotonic(),
    )


def zero_imu_input() -> ImuInput:
    return ImuInput(
        roll_deg=0.0,
        pitch_deg=0.0,
        yaw_deg=0.0,
        t_host=time.monotonic(),
    )


def add_input_args(parser: argparse.ArgumentParser, defaults: dict[str, Any]) -> None:
    parser.add_argument(
        "--mode",
        choices=["imu_only", "full_teleop"],
        default=str(defaults["mode"]),
        help="Run mode.",
    )
    parser.add_argument(
        "--serial-port",
        default=str(defaults["arduino_port"]),
        help="Arduino serial port, example: /dev/ttyUSB0",
    )
    parser.add_argument("--baud", type=int, default=int(defaults["arduino_baud"]))
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=float(defaults["loop_rate_hz"]),
        help="Main loop rate.",
    )
    parser.add_argument(
        "--imu-print-hz",
        type=float,
        default=float(defaults["imu_print_hz"]),
        help="Print rate limit in IMU-only mode.",
    )
    parser.add_argument(
        "--imu-source",
        choices=["mock", "vendor"],
        default=str(defaults["imu_source"]),
        help="IMU data source.",
    )
    parser.add_argument(
        "--imu-module",
        default=str(defaults["imu_module"]),
        help="Python module path providing read_euler_deg().",
    )
    parser.add_argument(
        "--imu-adapter-port",
        default=str(defaults["imu_adapter_port"]),
        help="Vendor adapter serial port (example: /dev/ttyUSB1).",
    )
    parser.add_argument(
        "--imu-device-prefix",
        default=str(defaults["imu_device_prefix"]),
        help="Use first scanned device whose name starts with this prefix.",
    )
    parser.add_argument(
        "--imu-device-index",
        type=int,
        default=int(defaults["imu_device_index"]),
        help="Index in filtered IMU device list.",
    )
    parser.add_argument(
        "--imu-scan-timeout",
        type=float,
        default=float(defaults["imu_scan_timeout_s"]),
        help="Vendor SDK scan timeout in seconds.",
    )
    parser.add_argument(
        "--imu-serial-baud",
        type=int,
        default=int(defaults["imu_adapter_baud"]),
        help="Vendor adapter serial baud.",
    )
    parser.add_argument(
        "--imu-read-config",
        action="store_true",
        default=bool(defaults["imu_read_config_on_connect"]),
        help="Read adapter config with AT+READ before scan/connect.",
    )
    parser.add_argument(
        "--imu-binding-mode",
        choices=["keep", "bind", "unbind"],
        default=str(defaults["imu_binding_mode"]),
        help="Optional AT+BINDING mode before scan/connect.",
    )
    parser.add_argument(
        "--imu-set-rate-on-connect",
        choices=["keep", "1hz", "10hz"],
        default=str(defaults["imu_set_rate_on_connect"]),
        help="Optional output-rate command after connect.",
    )
    parser.add_argument(
        "--allow-imu-zero-fallback",
        action="store_true",
        default=bool(defaults["allow_imu_zero_fallback"]),
        help="Keep producing 7D output with zeroed IMU values while IMU is offline.",
    )


def build_imu_source(args: argparse.Namespace) -> Any:
    if args.imu_source == "mock":
        return MockImuSource()
    imu_options = {
        "adapter_port": args.imu_adapter_port,
        "device_prefix": args.imu_device_prefix,
        "device_index": args.imu_device_index,
        "scan_timeout": args.imu_scan_timeout,
        "serial_baud": args.imu_serial_baud,
        "read_config_on_connect": args.imu_read_config,
        "binding_mode": args.imu_binding_mode,
        "set_rate_on_connect": args.imu_set_rate_on_connect,
    }
    return VendorImuSource(args.imu_module, imu_options)


class TeleopInputRuntime:
    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args
        self._imu_source = build_imu_source(args)
        self._latest_lower: Optional[LowerInput] = None
        self._latest_imu: Optional[ImuInput] = None
        self._using_imu_fallback = False
        self.serial_port: str | None = None
        self.lower_reader: SerialLowerReader | None = None

        if args.mode != "imu_only":
            exclude_ports = [args.imu_adapter_port] if args.imu_adapter_port else []
            self.serial_port = args.serial_port or autodetect_serial_port(exclude_ports=exclude_ports)
            if not self.serial_port:
                raise RuntimeError(
                    "No Arduino serial port found. Set DEFAULT_INPUT_CONFIG['arduino_port'] "
                    "or --serial-port manually."
                )
            self.lower_reader = SerialLowerReader(self.serial_port, args.baud)

    @property
    def imu_source(self) -> Any:
        return self._imu_source

    @property
    def lower_ready(self) -> bool:
        return self._latest_lower is not None

    @property
    def imu_ready(self) -> bool:
        return self._latest_imu is not None

    @property
    def using_imu_fallback(self) -> bool:
        return self._using_imu_fallback

    def describe_setup(self) -> list[str]:
        return [
            f"Serial: {self.serial_port or 'disabled'} @ {self._args.baud}",
            f"IMU source: {self._args.imu_source}",
            f"IMU adapter: {self._args.imu_adapter_port or 'auto'}",
            "Input mapping:",
            "  Arduino A0/A1/A2 -> joint1/joint2/joint3",
            "  Arduino A3 -> grip",
            "  WT901 BLE -> roll/pitch/yaw",
            "Control order: [joint1, joint2, joint3, roll, pitch, yaw, grip]",
        ]

    def poll_imu_only(self) -> Optional[ImuInput]:
        sample = self._imu_source.read_latest()
        if sample is not None:
            self._latest_imu = sample
        return sample

    def poll_frame(self) -> Optional[ControlFrame]:
        if self.lower_reader is not None:
            polled_lower = self.lower_reader.poll_latest()
            if polled_lower is not None:
                self._latest_lower = polled_lower

        polled_imu = self._imu_source.read_latest()
        if polled_imu is not None:
            self._latest_imu = polled_imu

        imu_for_frame: Optional[ImuInput] = self._latest_imu
        self._using_imu_fallback = False
        if imu_for_frame is None and self._args.allow_imu_zero_fallback:
            imu_for_frame = zero_imu_input()
            self._using_imu_fallback = True

        if self._latest_lower is None or imu_for_frame is None:
            return None
        return build_frame(self._latest_lower, imu_for_frame)

    def missing_inputs(self) -> list[str]:
        missing: list[str] = []
        if self._latest_lower is None:
            missing.append("Arduino serial")
        if self._latest_imu is None and not self._using_imu_fallback:
            missing.append("IMU")
        return missing

    def close(self) -> None:
        if self.lower_reader is not None:
            self.lower_reader.close()
        if hasattr(self._imu_source, "close"):
            self._imu_source.close()


def run_imu_only_loop(runtime: TeleopInputRuntime, loop_hz: float, print_hz: float) -> None:
    loop_period = 1.0 / max(loop_hz, 1.0)
    min_print_interval = 1.0 / max(print_hz, 0.1)
    last_print = 0.0
    printed_waiting = False
    print("IMU-only mode. Press Ctrl+C to stop.")

    while True:
        sample = runtime.poll_imu_only()
        now = time.monotonic()
        if sample is None:
            if not printed_waiting:
                print("Waiting IMU samples...")
                printed_waiting = True
        else:
            printed_waiting = False
            if now - last_print >= min_print_interval:
                print(
                    f"[imu] roll={sample.roll_deg:8.3f} "
                    f"pitch={sample.pitch_deg:8.3f} yaw={sample.yaw_deg:8.3f}"
                )
                last_print = now
        time.sleep(loop_period)
