"""CLI entry point for Position-SDF training.

This script loads a training config, sets up logging, and launches the position trainer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mcfp.train.train_pos import train_pos
from mcfp.utils.config import load_config
from mcfp.utils.logging import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Position-SDF model.")
    parser.add_argument("--config", type=str, default="configs/train_pos.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    log_dir = getattr(cfg, "logging", None)
    log_dir = getattr(log_dir, "log_dir", None) if log_dir is not None else None
    logger = setup_logger("mcfp.train_pos", log_dir=log_dir)

    logger.info(f"[train_pos] config={Path(args.config).resolve()}")
    train_pos(cfg, logger)


if __name__ == "__main__":
    main()
