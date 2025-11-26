from __future__ import annotations

from pathlib import Path

from mcfp.utils.config import load_config
from mcfp.utils.logging import setup_logger
from mcfp.sim.grid_builder import generate_capability_for_robot
from mcfp.data.io import load_capability_map


def test_single_episode_smoke(tmp_path: Path) -> None:
    """Smoke test: run one episode and check output NPZ exists and loads."""
    cfg = load_config("configs/data_gen_single.yaml")
    logger = setup_logger("mcfp.test.single_episode", cfg.logging.log_dir)

    urdf_root = Path(cfg.data.urdf_root)
    urdf_path = urdf_root / cfg.robot.urdf_filename

    out_path = tmp_path / "capability_map.npz"

    generate_capability_for_robot(
        urdf_path=urdf_path,
        output_path=out_path,
        grid_cfg=cfg.grid,
        logger=logger,
    )

    assert out_path.is_file()
    data = load_capability_map(out_path)
    assert "cell_centers" in data
    assert "u" in data
