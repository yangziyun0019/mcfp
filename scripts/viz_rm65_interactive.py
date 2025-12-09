"""
MCFP Visualization Script for RM65 Robot.
Usage: python -m scripts.viz_rm65_interactive
"""
import sys
import os
from pathlib import Path

# 添加项目根目录到 path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from mcfp.viz.interactive_3d import InteractiveCapabilityViz
from mcfp.utils.logging import setup_logger

def main():
    logger = setup_logger(name="mcfp.scripts.viz")
    
    # === 1. 硬编码路径 (基于你提供的信息) ===
    # 请确保路径完全正确，不要有拼写错误
    project_root = Path("D:/code/mcfp")
    
    data_path = project_root / "data/capability/rm_65_b/capability_map.npz"
    
    # 注意：URDF 路径，RobotModel 需要它来计算 FK
    urdf_path = project_root / "data/urdf/rm_65_b_description/urdf/rm_65_b_description.urdf"
    
    # 注意：STL 网格路径，用于渲染
    mesh_dir = project_root / "data/urdf/rm_65_b_description/meshes"

    # === 2. 检查 ===
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return
    if not urdf_path.exists():
        logger.error(f"URDF not found: {urdf_path}")
        return
    if not mesh_dir.exists():
        logger.error(f"Mesh dir not found: {mesh_dir}")
        return

    # === 3. 设置降采样 ===
    # 如果看不到颜色，尝试设为 1 (不降采样) 或 5，确保不是因为点太少被漏掉了
    DOWNSAMPLE_RATE = 20 
    
    logger.info("Initializing Interactive Viz...")
    
    viz = InteractiveCapabilityViz(
        data_path=str(data_path),
        urdf_path=str(urdf_path),
        mesh_dir=str(mesh_dir),
        downsample_rate=DOWNSAMPLE_RATE
    )
    
    viz.run()

if __name__ == "__main__":
    main()