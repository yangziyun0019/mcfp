"""Entry script for generating a capability map for a single robot URDF."""

from __future__ import annotations

from pathlib import Path

from mcfp.utils.config import load_config
from mcfp.utils.logging import setup_logger
from mcfp.sim.grid_builder import generate_capability_for_robot


def main() -> None:
    """Main entry point for single-URDF capability map generation."""
    cfg = load_config("configs/data_gen_single.yaml")

    logger = setup_logger(
        name="mcfp.sim.data_gen_single",
        log_dir=cfg.logging.log_dir,
    )
    logger.info("Starting single-robot capability generation.")

    urdf_root = Path(cfg.data.urdf_root)
    cap_root = Path(cfg.data.capability_root)

    robot_name = cfg.robot.name
    urdf_filename = cfg.robot.urdf_filename

    urdf_path = urdf_root / urdf_filename
    # TODO: 如果后续按 robot_name 分目录，可以改成 urdf_root / robot_name / urdf_filename
    output_dir = cap_root / robot_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "capability_map.npz"

    logger.info(f"URDF path: {urdf_path}")
    logger.info(f"Output capability path: {output_path}")

    generate_capability_for_robot(
        urdf_path=urdf_path,
        output_path=output_path,
        grid_cfg=cfg.grid,
        logger=logger,
    )

    logger.info("Single-robot capability generation finished.")


if __name__ == "__main__":
    main()
