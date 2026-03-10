"""Script: run_teleop.py
Purpose: Run the host input pipeline and print merged teleoperation frames without sending robot commands.
Usage: python -m host.run_teleop
"""

from __future__ import annotations

import argparse
import time
from typing import Any

from host.drivers import PrintDriver
from host.teleop_input import DEFAULT_INPUT_CONFIG, TeleopInputRuntime, add_input_args, run_imu_only_loop


# Fixed script role:
# this file only collects Arduino + IMU inputs, merges them into a 7D vector,
# and prints the merged result to the terminal.
RUN_CONFIG: dict[str, Any] = {
    **DEFAULT_INPUT_CONFIG,
    "mode": "full_teleop",
    "control_print_hz": 20.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fixed 7D input merger: reads Arduino + IMU and prints merged control vectors."
    )
    add_input_args(parser, RUN_CONFIG)
    parser.add_argument(
        "--print-hz",
        type=float,
        default=float(RUN_CONFIG["control_print_hz"]),
        help="Print rate limit for merged 7D vectors.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = TeleopInputRuntime(args)

    if args.mode == "imu_only":
        try:
            run_imu_only_loop(runtime, args.rate_hz, args.imu_print_hz)
        except KeyboardInterrupt:
            pass
        finally:
            runtime.close()
        return

    printer = PrintDriver(print_hz=args.print_hz)
    for line in runtime.describe_setup():
        print(line)

    loop_period = 1.0 / max(args.rate_hz, 1.0)
    last_wait_print = 0.0
    reported_lower_ready = False
    reported_imu_ready = False
    reported_zero_fallback = False

    try:
        while True:
            frame = runtime.poll_frame()

            if runtime.lower_ready and not reported_lower_ready:
                print("Arduino stream is active.")
                reported_lower_ready = True

            if runtime.imu_ready and not reported_imu_ready:
                print("IMU stream is active.")
                reported_imu_ready = True

            if runtime.using_imu_fallback and not reported_zero_fallback:
                print("IMU pending. Using roll/pitch/yaw = 0.0 fallback.")
                reported_zero_fallback = True

            if frame is not None:
                printer.send(frame)
            else:
                now = time.monotonic()
                if now - last_wait_print >= 1.0:
                    print(f"Waiting for: {', '.join(runtime.missing_inputs())}")
                    last_wait_print = now

            time.sleep(loop_period)
    except KeyboardInterrupt:
        pass
    finally:
        runtime.close()
        printer.close()


if __name__ == "__main__":
    main()
