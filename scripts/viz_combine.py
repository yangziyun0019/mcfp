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

# Import visualization tools
# Ensure you have 'trimesh', 'pyrender', 'urdfpy' installed for this to work
from mcfp.viz.interactive_3d import InteractiveCapabilityViz
from mcfp.utils.logging import setup_logger

def load_config(path: Path):
    """Simple wrapper for OmegaConf."""
    return OmegaConf.load(path)

def main():
    logger = setup_logger(name="mcfp.scripts.viz")
    
    # 2. Load Configuration
    # Adjust this path to your current active config file
    # config_rel_path = "configs/data_gen_single_rm65b.yaml" 
    # config_rel_path = "configs/data_gen_single_wx200.yaml" 
    # config_rel_path = "configs/data_gen_single_openmanipulator.yaml" 
    config_rel_path = "configs/data_gen_single_franka.yaml"

    config_path = PROJECT_ROOT / config_rel_path
    
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        return

    logger.info(f"Loading configuration: {config_rel_path}")
    cfg = load_config(config_path)

    # 3. Resolve Data Paths (Capability Map)
    if "data" not in cfg or "capability_root" not in cfg.data:
        logger.error("Config missing 'data.capability_root'.")
        return

    cap_root = PROJECT_ROOT / cfg.data.capability_root
    robot_name = cfg.robot.get("name", "unknown_robot")
    data_path = cap_root /  "capability_map.npz"
    # data_path = cap_root / robot_name / "capability_map.npz"

    # 4. Resolve URDF Paths (Critical for Visualization)
    # Visualization ALWAYS needs URDF for meshes, even if data gen was Spec-based.
    
    urdf_filename = cfg.robot.get("urdf_filename")
    
    if not urdf_filename:
        logger.error(
            "CRITICAL: Config is missing 'robot.urdf_filename'.\n"
            "Visualization requires the URDF to load 3D meshes.\n"
            "-> Please uncomment 'urdf_filename' in your YAML config, "
            "even if 'source_type' is set to 'json'."
        )
        return

    # Handle URDF Root (Check both old and new config styles)
    urdf_root_str = cfg.data.get("urdf_root")
    if not urdf_root_str:
        logger.error("Config missing 'data.urdf_root'. Cannot locate URDF.")
        return

    urdf_root_raw = Path(urdf_root_str)
    if urdf_root_raw.is_absolute():
        urdf_root_dir = urdf_root_raw
    else:
        urdf_root_dir = PROJECT_ROOT / urdf_root_raw
        
    urdf_path = urdf_root_dir / urdf_filename
    
    # Standard ROS layout inference: urdf/../meshes
    # If your folder structure is different, you might need to adjust this.
    mesh_dir = urdf_root_dir.parent / "meshes"

    # 5. Topology from Config
    base_link = cfg.sim.get("base_link", "base_link")
    ee_link = cfg.sim.get("end_effector_link", "tool0")

    # 6. Validation
    if not data_path.exists():
        logger.error(f"Capability Data missing: {data_path}\nPlease run generation script first.")
        return
    if not urdf_path.exists():
        logger.error(f"URDF file missing: {urdf_path}")
        return

    # 7. Launch Visualization
    logger.info(f"Visualizing Data: {data_path}")
    logger.info(f"Using Robot Model: {urdf_path}")
    
    viz = InteractiveCapabilityViz(
        data_path=str(data_path),
        urdf_path=str(urdf_path),
        mesh_dir=str(mesh_dir),
        downsample_rate=20,     # Adjust if visualization is too slow
        base_link=base_link,
        end_effector_link=ee_link
    )
    
    viz.run()

if __name__ == "__main__":
    main()