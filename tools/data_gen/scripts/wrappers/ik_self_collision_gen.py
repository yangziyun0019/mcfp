#!/usr/bin/env python3
"""Script: ik_self_collision_gen.py
Purpose: Forward a legacy dataset-generation call to the current dataset_generator_cli entry point.
Usage: python3 tools/data_gen/scripts/wrappers/ik_self_collision_gen.py --config <yaml> --output <dir>
"""

import argparse
import subprocess


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Wrapper for dataset_generator_cli (deprecated entry point)"
    )
    parser.add_argument("--config", required=True, help="Path to robot config YAML")
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory (overrides config output.dir)",
    )
    args = parser.parse_args()

    cmd = [
        "ros2",
        "run",
        "reachability_cli",
        "dataset_generator_cli",
        "--config",
        args.config,
        "--output",
        args.output,
    ]
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
