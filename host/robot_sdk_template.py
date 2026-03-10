"""Script: robot_sdk_template.py
Purpose: Provide a minimal template for integrating another robot SDK into the host teleoperation pipeline.
Usage: Copy and customize when adding a new robot-side driver.
"""

from __future__ import annotations

from host.control_types import ControlFrame


class RobotDriver:
    def __init__(self) -> None:
        # TODO: Initialize your robot SDK client here.
        pass

    def send(self, frame: ControlFrame) -> None:
        # TODO: Map control frame to your robot SDK commands.
        # frame.as_list() order:
        # [pot0_deg, pot1_deg, pot2_deg, roll_deg, pitch_deg, yaw_deg, grip]
        _ = frame

    def close(self) -> None:
        # TODO: Close robot SDK resources here.
        pass


def create_driver() -> RobotDriver:
    return RobotDriver()
