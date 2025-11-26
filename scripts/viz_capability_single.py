# scripts/viz_capability_single.py

from pathlib import Path

from mcfp.utils.config import load_config
from mcfp.viz.capability_viz import plot_capability_with_robot


def main() -> None:
    cfg = load_config("configs/data_gen_single.yaml")

    urdf_root = Path(cfg.data.urdf_root)
    capability_root = Path(cfg.data.capability_root)

    urdf_path = urdf_root / cfg.robot.urdf_filename
    cap_path = capability_root / cfg.robot.name / "capability_map.npz"


    base_link = cfg.sim.base_link
    end_effector_link = cfg.sim.end_effector_link

    plot_capability_with_robot(
        urdf_path=urdf_path,
        cap_path=cap_path,
        base_link=base_link,
        end_effector_link=end_effector_link,
        value_key="u",
        log_dir=Path(cfg.logging.log_dir),
    )


if __name__ == "__main__":
    main()
