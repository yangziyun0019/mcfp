"""Script: drivers.py
Purpose: Provide driver-loading helpers and a print-only fallback driver for the host teleoperation pipeline.
Usage: Imported by host.run_teleop and related host entry points.
"""

from __future__ import annotations

import importlib.util
import os
import time
from types import ModuleType
from typing import Any

from host.control_types import ControlFrame


class PrintDriver:
    def __init__(self, print_hz: float = 10.0) -> None:
        self._last_print = 0.0
        self._min_interval = 1.0 / max(print_hz, 0.1)

    def send(self, frame: ControlFrame) -> None:
        now = time.monotonic()
        if now - self._last_print < self._min_interval:
            return
        self._last_print = now
        values = ", ".join(f"{v:.2f}" for v in frame.as_list())
        print(f"[control7] {values}")

    def close(self) -> None:
        return


def _load_module_from_file(script_path: str) -> ModuleType:
    if not os.path.isfile(script_path):
        raise FileNotFoundError(f"Robot driver script not found: {script_path}")
    module_name = f"robot_driver_{int(time.time() * 1000)}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load robot driver spec: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_driver(script_path: str | None, print_hz: float = 20.0) -> Any:
    if not script_path:
        return PrintDriver(print_hz=print_hz)

    module = _load_module_from_file(script_path)
    if hasattr(module, "create_driver"):
        driver = module.create_driver()
    elif hasattr(module, "RobotDriver"):
        driver = module.RobotDriver()
    else:
        raise RuntimeError(
            "Driver script must provide create_driver() or RobotDriver class."
        )

    if not hasattr(driver, "send"):
        raise RuntimeError("Loaded driver must implement send(frame).")
    return driver
