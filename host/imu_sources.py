"""Script: imu_sources.py
Purpose: Provide IMU source implementations used by the host teleoperation input layer.
Usage: Imported by host.teleop_input.
"""

from __future__ import annotations

import importlib
import math
import time
from typing import Any, Optional

from host.control_types import ImuInput


class MockImuSource:
    def __init__(self) -> None:
        self._t0 = time.monotonic()

    def read_latest(self) -> ImuInput:
        t = time.monotonic() - self._t0
        return ImuInput(
            roll_deg=15.0 * math.sin(0.6 * t),
            pitch_deg=10.0 * math.sin(0.9 * t),
            yaw_deg=45.0 * math.sin(0.3 * t),
            t_host=time.monotonic(),
        )


class VendorImuSource:
    def __init__(
        self,
        module_name: str = "host.imu_vendor_adapter",
        module_options: Optional[dict[str, Any]] = None,
    ) -> None:
        self._adapter = importlib.import_module(module_name)
        self._last_error_print_t = 0.0
        if hasattr(self._adapter, "configure"):
            self._adapter.configure(**(module_options or {}))

    def read_latest(self) -> Optional[ImuInput]:
        # Adapter function should return:
        # (roll_deg, pitch_deg, yaw_deg) or None if no new sample.
        try:
            values = self._adapter.read_euler_deg()
        except RuntimeError as exc:
            now = time.monotonic()
            if now - self._last_error_print_t >= 1.0:
                print(f"IMU pending: {exc}")
                self._last_error_print_t = now
            if hasattr(self._adapter, "close"):
                self._adapter.close()
            return None
        if values is None:
            return None
        roll_deg, pitch_deg, yaw_deg = values
        return ImuInput(
            roll_deg=float(roll_deg),
            pitch_deg=float(pitch_deg),
            yaw_deg=float(yaw_deg),
            t_host=time.monotonic(),
        )

    def close(self) -> None:
        if hasattr(self._adapter, "close"):
            self._adapter.close()
