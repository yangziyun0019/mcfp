# scripts/generate_capability_single.py
"""Entry script for generating IK pose samples from URDF."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from mcfp.utils.config import load_config
from mcfp.utils.logging import setup_logger
from mcfp.sim.pose_dataset import generate_pose_dataset_ik
from mcfp.data.pose_neighbors import build_pose_deltas


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate IK pose samples for a single robot.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data_gen_pose_ik_franka.yaml",
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for single-robot IK pose dataset generation."""
    args = _parse_args()
    cfg = load_config(args.config)

    # 2. Setup Logger
    logger = setup_logger(
        name="mcfp.sim.data_gen_single",
        log_dir=cfg.logging.log_dir,
    )
    logger.info(f"Loaded configuration from: {args.config}")

    # 3. Resolve Paths
    robot_name = cfg.robot.name
    urdf_root = Path(cfg.data.urdf_root)
    urdf_filename = cfg.robot.urdf_filename
    urdf_path = urdf_root / urdf_filename

    output_root = Path(cfg.data.output_root)
    output_name = getattr(cfg.data, "output_name", "pose_samples.npz")
    output_dir = output_root / robot_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name

    if not urdf_path.exists():
        logger.error(f"URDF file does not exist: {urdf_path}")
        sys.exit(1)

    # 4. Extract Sim Parameters
    base_link = getattr(cfg.sim, "base_link", None)
    end_effector_link = getattr(cfg.sim, "end_effector_link", None)

    logger.info(f"Target Output: {output_path}")

    # 5. Execute Generation
    generate_pose_dataset_ik(
        urdf_path=urdf_path,
        output_path=output_path,
        cfg=cfg,
        base_link=base_link,
        end_effector_link=end_effector_link,
        logger=logger,
    )

    delta_cfg = getattr(cfg, "delta", None)
    delta_enable = bool(getattr(delta_cfg, "enable", False)) if delta_cfg is not None else False
    if delta_enable:
        delta_name = getattr(delta_cfg, "output_name", "pose_samples_with_deltas.npz")
        delta_path = output_dir / delta_name
        build_pose_deltas(
            input_path=output_path,
            output_path=delta_path,
            cfg=cfg,
            logger=logger,
        )

    logger.info("Pose dataset generation finished successfully.")


if __name__ == "__main__":
    main()
