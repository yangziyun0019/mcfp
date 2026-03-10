"""Script: serial_reader.py
Purpose: Read and parse lower-arm serial frames produced by the Arduino-based potentiometer interface.
Usage: Imported by host.teleop_input.
"""

from __future__ import annotations

import glob
import time
from typing import Optional

from host.control_types import LowerInput

try:
    import serial  # type: ignore
except ImportError:  # pragma: no cover
    serial = None


def autodetect_serial_port(exclude_ports: list[str] | None = None) -> Optional[str]:
    excluded = {p for p in (exclude_ports or []) if p}
    candidates: list[str] = []
    patterns = ("/dev/ttyUSB*", "/dev/ttyACM*", "/dev/cu.usb*", "/dev/cu.usbserial*")
    for pattern in patterns:
        candidates.extend(sorted(glob.glob(pattern)))
    for candidate in candidates:
        if candidate not in excluded:
            return candidate
    return None


def parse_lower_line(line: str) -> Optional[LowerInput]:
    # Expected format from firmware:
    # CTRL,<pot0_deg>,<pot1_deg>,<pot2_deg>,<grip>
    parts = line.strip().split(",")
    if len(parts) != 5:
        return None
    if parts[0] != "CTRL":
        return None
    try:
        pot0 = float(parts[1])
        pot1 = float(parts[2])
        pot2 = float(parts[3])
        grip = int(parts[4])
    except ValueError:
        return None
    grip = 1 if grip else 0
    return LowerInput(
        pot0_deg=pot0,
        pot1_deg=pot1,
        pot2_deg=pot2,
        grip=grip,
        t_host=time.monotonic(),
    )


class SerialLowerReader:
    def __init__(self, port: str, baud: int) -> None:
        if serial is None:
            raise RuntimeError("pyserial is not installed. Run: pip install pyserial")
        self._serial = serial.Serial(port=port, baudrate=baud, timeout=0.0)
        self._buffer = bytearray()
        # Nano/CH340 often resets when the port opens. Give it a short grace period.
        time.sleep(1.5)
        self._serial.reset_input_buffer()

    def poll_latest(self, max_reads: int = 16) -> Optional[LowerInput]:
        latest: Optional[LowerInput] = None
        for _ in range(max_reads):
            waiting = self._serial.in_waiting
            if waiting <= 0:
                break
            self._buffer.extend(self._serial.read(waiting))

        while True:
            newline = self._buffer.find(b"\n")
            if newline < 0:
                break
            raw = bytes(self._buffer[:newline])
            del self._buffer[: newline + 1]
            line = raw.decode("utf-8", errors="ignore").strip()
            parsed = parse_lower_line(line)
            if parsed is not None:
                latest = parsed
        return latest

    def close(self) -> None:
        if self._serial.is_open:
            self._serial.close()
