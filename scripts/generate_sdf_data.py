"""Entry script for SDF data generation from URDF."""
from __future__ import annotations

import argparse

from mcfp.sim.sdf_data_gen import generate_sdf_dataset
from mcfp.utils.config import load_config
from mcfp.utils.logging import setup_logger


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate SDF dataset from a URDF.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data_gen_sdf_ur5.yaml",
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = _parse_args()
    cfg = load_config(args.config)
    logger = setup_logger(name="mcfp.sim.sdf_data_gen", log_dir=cfg.logging.log_dir)
    logger.info(f"Loaded configuration from: {args.config}")
    generate_sdf_dataset(cfg, logger)


if __name__ == "__main__":
    main()
