"""Entry script for FK->IK round-trip tests."""

from __future__ import annotations

import argparse

from mcfp.sim.fk_ik_loop import run_fk_ik_loop
from mcfp.utils.config import load_config
from mcfp.utils.logging import setup_logger


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run FK->IK loop tests.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/fk_ik_loop_franka.yaml",
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = _parse_args()
    cfg = load_config(args.config)

    logger = setup_logger(
        name="mcfp.sim.fk_ik_loop",
        log_dir=cfg.logging.log_dir,
    )
    logger.info(f"Loaded configuration from: {args.config}")

    run_fk_ik_loop(cfg, logger)


if __name__ == "__main__":
    main()
