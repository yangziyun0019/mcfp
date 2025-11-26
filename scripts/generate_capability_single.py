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

    base_link = getattr(cfg.sim, "base_link", None)
    end_effector_link = getattr(cfg.sim, "end_effector_link", None)

    urdf_path = urdf_root / urdf_filename
    output_dir = cap_root / robot_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "capability_map.npz"

    logger.info(f"URDF path: {urdf_path}")
    logger.info(f"Output capability path: {output_path}")

    generate_capability_for_robot(
        urdf_path=urdf_path,
        output_path=output_path,
        grid_cfg=cfg.grid,
        base_link=base_link,
        end_effector_link=end_effector_link,
        logger=logger,
    )

    logger.info("Single-robot capability generation finished.")


if __name__ == "__main__":
    main()
