from __future__ import annotations

import argparse
from pathlib import Path

from mcfp.train.sdf_stage1 import train_sdf_stage1
from mcfp.utils.config import load_config
from mcfp.utils.logging import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage-1 SDF model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_sdf_stage1.yaml",
        help="Path to training config YAML.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    log_dir = getattr(cfg, "logging", None)
    log_dir = getattr(log_dir, "log_dir", None) if log_dir is not None else None
    logger = setup_logger(name="mcfp.train_sdf_stage1", log_dir=log_dir)

    logger.info(f"[train_sdf_stage1] config={Path(args.config).resolve()}")
    train_sdf_stage1(cfg, logger, config_path=args.config)


if __name__ == "__main__":
    main()
