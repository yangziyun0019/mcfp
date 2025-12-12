# scripts/generate_capability_single.py
"""Entry script for generating a capability map for a single robot URDF."""

from __future__ import annotations

from pathlib import Path

from mcfp.utils.config import load_config
from mcfp.utils.logging import setup_logger
from mcfp.sim.grid_builder import generate_capability_for_robot


def main() -> None:
    """Main entry point for single-URDF capability map generation."""
    # Load YAML configuration
    cfg = load_config("configs/data_gen_single_wx200.yaml")
    # cfg = load_config("configs/data_gen_single_rm65b.yaml")
    # cfg = load_config("configs/data_gen_single_openmanipulator.yaml")

    # Logger initialised from config
    logger = setup_logger(
        name="mcfp.sim.data_gen_single",
        log_dir=cfg.logging.log_dir,
    )
    logger.info("Starting single-robot capability generation.")

    # Paths
    urdf_root = Path(cfg.data.urdf_root)
    cap_root = Path(cfg.data.capability_root)

    # Robot-specific settings
    robot_name = cfg.robot.name
    urdf_filename = cfg.robot.urdf_filename

    base_link = getattr(cfg.sim, "base_link", None)
    end_effector_link = getattr(cfg.sim, "end_effector_link", None)

    urdf_path = urdf_root / urdf_filename
    output_dir = cap_root / robot_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "capability_map.npz"

    logger.info("URDF path: %s", urdf_path)
    logger.info("Output capability path: %s", output_path)

    # All sampling / indicator-related hyperparameters are inside cfg.grid and cfg.sim.
    # generate_capability_for_robot (in mcfp.sim.grid_builder) is responsible
    # for reading them and passing them down to indicators.
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
