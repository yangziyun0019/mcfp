"""Script: imu_vendor_adapter.py
Purpose: Adapt the vendor IMU serial interface into a reusable host-side roll, pitch, and yaw reader.
Usage: Imported by host.imu_sources.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional, Tuple


_REPO_ROOT = Path(__file__).resolve().parents[1]
_WIT_ROOT = _REPO_ROOT / "BWT901BLE5.0_python_serial_port"

_config: dict[str, object] = {
    "adapter_port": "/dev/ttyACM1",
    "serial_baud": 115200,
    "device_prefix": "WT",
    "device_index": 0,
    "scan_timeout": 12.0,
    "read_config_on_connect": False,
    "binding_mode": "keep",
    "set_rate_on_connect": "keep",
}

_client_lock = threading.Lock()
_client: Optional["_VendorDirectClient"] = None


def _load_vendor_adapter_class():
    root_str = str(_WIT_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    try:
        from sdk.adapter import WT901Adapter  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Failed to import vendor WT901Adapter from {_WIT_ROOT}") from exc
    return WT901Adapter


class _VendorDirectClient:
    def __init__(
        self,
        adapter_port: str,
        serial_baud: int,
        device_prefix: str,
        device_index: int,
        scan_timeout: float,
        read_config_on_connect: bool,
        binding_mode: str,
        set_rate_on_connect: str,
    ) -> None:
        self._adapter_port = adapter_port
        self._serial_baud = serial_baud
        self._device_prefix = device_prefix.strip()
        self._device_index = max(device_index, 0)
        self._scan_timeout = max(scan_timeout, 3.0)
        self._read_config_on_connect = bool(read_config_on_connect)
        self._binding_mode = str(binding_mode)
        self._set_rate_on_connect = str(set_rate_on_connect)
        self._adapter = None
        self._latest: Optional[Tuple[float, float, float]] = None
        self._lock = threading.Lock()
        self._started = False

    def _on_angle(self, data: dict[str, object]) -> None:
        angle = data.get("angle")
        if not isinstance(angle, tuple) and not isinstance(angle, list):
            return
        if len(angle) != 3:
            return
        roll, pitch, yaw = angle
        with self._lock:
            self._latest = (float(roll), float(pitch), float(yaw))

    def _noop(self, _data: object) -> None:
        return

    def _filter_devices(self, devices: list[dict[str, object]]) -> list[dict[str, object]]:
        prefix = self._device_prefix.upper()
        if not prefix:
            return devices
        filtered = [
            device
            for device in devices
            if str(device.get("name", "")).upper().startswith(prefix)
        ]
        return filtered or devices

    def _wait_for_first_angle(
        self,
        adapter: Any,
        wait_s: float,
        allow_rate_command: bool,
    ) -> bool:
        phase1_deadline = time.monotonic() + max(wait_s, 3.5)
        while time.monotonic() < phase1_deadline:
                with self._lock:
                    if self._latest is not None:
                        return True
                time.sleep(0.05)

        # Only after the spontaneous angle stream has had a fair chance to appear
        # do we send optional rate commands. The vendor manual notes that extra
        # reads too early can interfere with the initial BLE data stream.

        if allow_rate_command:
            try:
                if self._set_rate_on_connect == "10hz":
                    adapter.set_rate_10hz()
                elif self._set_rate_on_connect == "1hz":
                    adapter.set_rate_1hz()
            except Exception:
                pass

            phase2_deadline = time.monotonic() + 3.0
            while time.monotonic() < phase2_deadline:
                with self._lock:
                    if self._latest is not None:
                        return True
                time.sleep(0.05)
        return False

    def _start_once(self, *, force_binding: str | None = None) -> Any:
        WT901Adapter = _load_vendor_adapter_class()
        adapter = WT901Adapter(self._adapter_port, baudrate=self._serial_baud, debug=False)
        if not adapter.open():
            raise RuntimeError(f"Failed to open IMU adapter serial port: {self._adapter_port}")

        if self._read_config_on_connect:
            try:
                adapter.read_config()
            except Exception:
                pass

        binding_mode = self._binding_mode if force_binding is None else force_binding
        if binding_mode == "bind":
            try:
                adapter.set_binding(True)
                time.sleep(0.2)
            except Exception:
                pass
        elif binding_mode == "unbind":
            try:
                adapter.set_binding(False)
                time.sleep(0.2)
            except Exception:
                pass

        adapter.parser.register_callback("acc_gyro_angle", self._on_angle)
        adapter.parser.register_callback("magnetic", self._noop)
        adapter.parser.register_callback("temperature", self._noop)
        adapter.parser.register_callback("battery", self._noop)
        adapter.parser.register_callback("quaternion", self._noop)

        devices = adapter.scan_devices(timeout=self._scan_timeout)
        if not devices:
            adapter.close()
            raise RuntimeError(
                "Vendor adapter scan returned no devices. "
                "Check power, occupancy by phone, and adapter port."
            )

        filtered = self._filter_devices(devices)
        if self._device_index >= len(filtered):
            adapter.close()
            raise RuntimeError(
                f"Configured IMU device index {self._device_index} out of range for devices: {filtered}"
            )

        selected = filtered[self._device_index]
        print(
            f"IMU adapter: {self._adapter_port} @ {self._serial_baud}, "
            f"selected device: {selected.get('name', '<unknown>')} {selected.get('address', '')}"
        )
        # Reuse the vendor's proven connection path verbatim. This is the only
        # sequence the user's hardware has already validated end-to-end.
        if not adapter.connect_device(int(selected["index"])):
            adapter.close()
            raise RuntimeError(f"Failed to connect IMU device: {selected}")

        return adapter

    def start(self) -> None:
        if self._started:
            return

        with self._lock:
            self._latest = None

        adapter = self._start_once()
        got_angle = self._wait_for_first_angle(
            adapter,
            wait_s=max(self._scan_timeout, 8.0),
            allow_rate_command=self._set_rate_on_connect in {"1hz", "10hz"},
        )

        if not got_angle and self._binding_mode == "keep":
            diagnostics = adapter.get_rx_diagnostics()
            adapter.close()
            with self._lock:
                self._latest = None
            print(
                "IMU angle stream did not appear after plain connect. "
                "Retrying once with adapter binding enabled..."
            )
            adapter = self._start_once(force_binding="bind")
            got_angle = self._wait_for_first_angle(
                adapter,
                wait_s=max(self._scan_timeout, 8.0),
                allow_rate_command=self._set_rate_on_connect in {"1hz", "10hz"},
            )
            if not got_angle:
                retry_diagnostics = adapter.get_rx_diagnostics()
                adapter.close()
                raise RuntimeError(
                    "IMU connected, but no angle samples were received. "
                    f"First diagnostics: {diagnostics}; Retry diagnostics: {retry_diagnostics}"
                )

        self._adapter = adapter
        self._started = True
        with self._lock:
            if self._latest is None:
                diagnostics = adapter.get_rx_diagnostics()
                adapter.close()
                raise RuntimeError(
                    "IMU connected, but no angle samples were received. "
                    f"Diagnostics: {diagnostics}"
                )

    def read_latest(self) -> Optional[Tuple[float, float, float]]:
        with self._lock:
            return self._latest

    def close(self) -> None:
        if self._adapter is not None:
            self._adapter.close()
            self._adapter = None


def configure(
    adapter_port: str = "/dev/ttyACM1",
    serial_baud: int = 115200,
    device_prefix: str = "WT",
    device_index: int = 0,
    scan_timeout: float = 8.0,
    read_config_on_connect: bool = False,
    binding_mode: str = "keep",
    set_rate_on_connect: str = "keep",
) -> None:
    _config["adapter_port"] = adapter_port
    _config["serial_baud"] = int(serial_baud)
    _config["device_prefix"] = device_prefix
    _config["device_index"] = int(device_index)
    _config["scan_timeout"] = float(scan_timeout)
    _config["read_config_on_connect"] = bool(read_config_on_connect)
    _config["binding_mode"] = str(binding_mode)
    _config["set_rate_on_connect"] = str(set_rate_on_connect)


def _ensure_client() -> _VendorDirectClient:
    global _client
    with _client_lock:
        if _client is None:
            client = _VendorDirectClient(
                adapter_port=str(_config["adapter_port"]),
                serial_baud=int(_config["serial_baud"]),
                device_prefix=str(_config["device_prefix"]),
                device_index=int(_config["device_index"]),
                scan_timeout=float(_config["scan_timeout"]),
                read_config_on_connect=bool(_config["read_config_on_connect"]),
                binding_mode=str(_config["binding_mode"]),
                set_rate_on_connect=str(_config["set_rate_on_connect"]),
            )
            try:
                client.start()
            except Exception:
                client.close()
                raise
            _client = client
        return _client


def read_euler_deg() -> Optional[Tuple[float, float, float]]:
    client = _ensure_client()
    return client.read_latest()


def close() -> None:
    global _client
    with _client_lock:
        if _client is not None:
            _client.close()
            _client = None
