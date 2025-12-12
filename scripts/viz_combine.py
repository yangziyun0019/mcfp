"""
Script: Interactive Capability Visualizer
Description: 
    Loads capability map and robot model purely from configuration.
    No hardcoded paths or parameters allowed.
"""

import sys
from pathlib import Path
from omegaconf import OmegaConf

# 1. Resolve Project Root
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent
sys.path.append(str(PROJECT_ROOT))

from mcfp.viz.interactive_3d import InteractiveCapabilityViz
from mcfp.utils.logging import setup_logger

def load_config(path: Path):
    """Simple wrapper for OmegaConf."""
    return OmegaConf.load(path)

def main():
    logger = setup_logger(name="mcfp.scripts.viz")
    
    # 2. Load Configuration (Single Source of Truth)
    # The script only needs to know *which* config to use.

    config_rel_path = "configs/data_gen_single_wx200.yaml" 
    # config_rel_path = "configs/data_gen_single_openmanipulator.yaml"
    # config_rel_path = "configs/data_gen_single_rm65b.yaml"

    config_path = PROJECT_ROOT / config_rel_path
    
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        return

    logger.info(f"Loading configuration: {config_rel_path}")
    cfg = load_config(config_path)

    # 3. Resolve Paths from Config
    # Data path logic: project_root / capability_root / robot_name / filename
    cap_root = PROJECT_ROOT / cfg.data.capability_root
    data_path = cap_root / cfg.robot.name / "capability_map.npz"

    # URDF path logic:
    # YAML urdf_root can be absolute or relative. We handle both.
    urdf_root_raw = Path(cfg.data.urdf_root)
    if urdf_root_raw.is_absolute():
        urdf_root_dir = urdf_root_raw
    else:
        urdf_root_dir = PROJECT_ROOT / urdf_root_raw
        
    urdf_path = urdf_root_dir / cfg.robot.urdf_filename
    
    # Mesh dir inference (Standard ROS layout: urdf/../meshes)
    mesh_dir = urdf_root_dir.parent / "meshes"

    # 4. Topology from Config
    base_link = cfg.sim.base_link
    ee_link = cfg.sim.end_effector_link

    # 5. Validation
    if not data_path.exists():
        logger.error(f"Data missing: {data_path}\nPlease run generation script first.")
        return
    if not urdf_path.exists():
        logger.error(f"URDF missing: {urdf_path}")
        return

    # 6. Launch Visualization
    # Note: mesh_scale is REMOVED. It is now auto-read from URDF in robot_model.py.
    logger.info("Initializing Visualization Studio...")
    
    viz = InteractiveCapabilityViz(
        data_path=str(data_path),
        urdf_path=str(urdf_path),
        mesh_dir=str(mesh_dir),
        downsample_rate=20,     # Visualization specific param
        base_link=base_link,
        end_effector_link=ee_link
    )
    
    viz.run()

if __name__ == "__main__":
    main()