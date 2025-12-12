# scripts/generate_capability_single.py
"""Entry script for generating a capability map from URDF or Morphology Spec."""

from __future__ import annotations

from pathlib import Path
import sys

from mcfp.utils.config import load_config
from mcfp.utils.logging import setup_logger
from mcfp.sim.grid_builder import generate_capability_for_robot


def main() -> None:
    """Main entry point for single-robot capability generation."""
    
    # 1. Load Configuration
    # You can switch config files here or parse args
    config_path = "configs/data_gen_single_rm65b.yaml"
    cfg = load_config(config_path)

    # 2. Setup Logger
    logger = setup_logger(
        name="mcfp.sim.data_gen_single",
        log_dir=cfg.logging.log_dir,
    )
    logger.info(f"Loaded configuration from: {config_path}")

    # 3. Resolve Paths & Source Mode
    robot_name = cfg.robot.name
    cap_root = Path(cfg.data.capability_root)
    output_dir = cap_root / robot_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "capability_map.npz"

    # Determine source type (urdf vs json) based on config
    source_type = getattr(cfg.robot, "source_type", "urdf").lower()
    
    input_path: Path
    is_spec_mode: bool = False

    if source_type == "json":
        # --- JSON Spec Mode ---
        json_root = Path(cfg.data.json_root)
        spec_filename = cfg.robot.spec_filename
        input_path = json_root / spec_filename
        is_spec_mode = True
        logger.info(f"[Mode] Using JSON Morphology Spec: {input_path}")
        
    elif source_type == "urdf":
        # --- URDF Mode (Legacy) ---
        urdf_root = Path(cfg.data.urdf_root)
        urdf_filename = cfg.robot.urdf_filename
        input_path = urdf_root / urdf_filename
        is_spec_mode = False
        logger.info(f"[Mode] Using URDF Description: {input_path}")
        
    else:
        logger.error(f"Unknown source_type: {source_type}")
        sys.exit(1)

    if not input_path.exists():
        logger.error(f"Input file does not exist: {input_path}")
        sys.exit(1)

    # 4. Extract Sim Parameters
    base_link = getattr(cfg.sim, "base_link", None)
    end_effector_link = getattr(cfg.sim, "end_effector_link", None)

    logger.info(f"Target Output: {output_path}")

    # 5. Execute Generation
    # Note: 'urdf_path' argument in grid_builder should be renamed to 'input_source' 
    # to reflect generic usage, or passed positionally if the signature was updated.
    generate_capability_for_robot(
        input_source=input_path,          # Generic input path (was urdf_path)
        output_path=output_path,
        grid_cfg=cfg.grid,
        base_link=base_link,
        end_effector_link=end_effector_link,
        logger=logger,
        is_morphology_spec=is_spec_mode   # Flag to trigger new logic in RobotModel
    )

    logger.info("Capability generation process finished successfully.")


if __name__ == "__main__":
    main()