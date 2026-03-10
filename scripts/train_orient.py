"""CLI entry point for Orientation-SDF training.

This script loads a training config, sets up logging, and launches the orientation trainer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mcfp.train.train_orient import train_orient
from mcfp.utils.config import load_config
from mcfp.utils.logging import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Orientation-SDF model.")
    parser.add_argument("--config", type=str, default="configs/train_orient.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    log_dir = getattr(cfg, "logging", None)
    log_dir = getattr(log_dir, "log_dir", None) if log_dir is not None else None
    logger = setup_logger("mcfp.train_orient", log_dir=log_dir)

    logger.info(f"[train_orient] config={Path(args.config).resolve()}")
    train_orient(cfg, logger)


if __name__ == "__main__":
    main()
