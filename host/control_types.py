"""Script: control_types.py
Purpose: Define shared host-side data structures for lower-arm input, IMU input, and merged control frames.
Usage: Imported by host teleoperation modules.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LowerInput:
    pot0_deg: float
    pot1_deg: float
    pot2_deg: float
    grip: int
    t_host: float


@dataclass
class ImuInput:
    roll_deg: float
    pitch_deg: float
    yaw_deg: float
    t_host: float


@dataclass
class ControlFrame:
    ch1_deg: float
    ch2_deg: float
    ch3_deg: float
    roll_deg: float
    pitch_deg: float
    yaw_deg: float
    grip: int
    t_host: float

    def as_list(self) -> list[float]:
        return [
            self.ch1_deg,
            self.ch2_deg,
            self.ch3_deg,
            self.roll_deg,
            self.pitch_deg,
            self.yaw_deg,
            float(self.grip),
        ]
