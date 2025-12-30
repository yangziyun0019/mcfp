from __future__ import annotations

import argparse

from mcfp.models.pose_query import run_pose_query
from mcfp.utils.config import load_config
from mcfp.utils.logging import setup_logger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger(name="mcfp.scripts.pose_query", log_dir=cfg.logging.log_dir)

    run_pose_query(cfg, logger)


if __name__ == "__main__":
    main()
