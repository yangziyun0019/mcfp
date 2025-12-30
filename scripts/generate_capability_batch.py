"""Entry script for generating IK pose samples in batch."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, List

from mcfp.data.pose_neighbors import build_pose_deltas
from mcfp.sim.pose_dataset import generate_pose_dataset_ik
from mcfp.utils.config import load_config
from mcfp.utils.logging import setup_logger
from mcfp.utils.seed import set_seed


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate IK pose samples in batch.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data_gen_pose_ik_batch.yaml",
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def _get_cfg_value(cfg: Any, key: str) -> Any:
    """Retrieve mandatory value from config object or dict."""
    if isinstance(cfg, dict):
        if key not in cfg:
            raise ValueError(f"[data_gen_batch] Config missing key '{key}'.")
        return cfg[key]
    if not hasattr(cfg, key):
        raise ValueError(f"[data_gen_batch] Config missing attr '{key}'.")
    return getattr(cfg, key)


def _get_cfg_val_default(cfg: Any, key: str, default: Any) -> Any:
    """Retrieve optional value from config object or dict."""
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def main() -> None:
    """Main entry point for batch IK pose dataset generation."""
    args = _parse_args()
    cfg = load_config(args.config)

    logger = setup_logger(
        name="mcfp.sim.data_gen_batch",
        log_dir=cfg.logging.log_dir,
    )
    logger.info(f"Loaded configuration from: {args.config}")

    run_cfg = getattr(cfg, "run", None)
    seed = _get_cfg_val_default(run_cfg, "seed", None)
    if seed is not None:
        deterministic = bool(_get_cfg_val_default(run_cfg, "deterministic", True))
        set_seed(int(seed), deterministic=deterministic)

    robots: List[Any] = list(getattr(cfg, "robots", []))
    if not robots:
        raise ValueError("[data_gen_batch] No robots configured.")

    data_cfg = getattr(cfg, "data", None)
    base_output_root = Path(_get_cfg_value(data_cfg, "output_root")).resolve()
    base_output_name = str(_get_cfg_val_default(data_cfg, "output_name", "pose_samples.npz"))

    delta_cfg = getattr(cfg, "delta", None)
    delta_enable = bool(_get_cfg_val_default(delta_cfg, "enable", False)) if delta_cfg is not None else False
    delta_name = str(_get_cfg_val_default(delta_cfg, "output_name", "pose_samples_with_deltas.npz"))

    for idx, robot_cfg in enumerate(robots, start=1):
        urdf_path = Path(_get_cfg_value(robot_cfg, "urdf_path")).resolve()
        if not urdf_path.is_file():
            logger.error(f"[{idx}/{len(robots)}] URDF not found: {urdf_path}")
            continue

        robot_name = _get_cfg_val_default(robot_cfg, "robot_name", None) or urdf_path.stem
        base_link = _get_cfg_val_default(robot_cfg, "base_link", None)
        ee_link = _get_cfg_val_default(robot_cfg, "ee_link", None)

        output_root = Path(_get_cfg_val_default(robot_cfg, "output_root", base_output_root)).resolve()
        output_name = str(_get_cfg_val_default(robot_cfg, "output_name", base_output_name))
        output_dir = output_root / robot_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_name

        logger.info(
            f"[{idx}/{len(robots)}] Generating samples for {robot_name}\n"
            f"  URDF   : {urdf_path}\n"
            f"  Output : {output_path}\n"
            f"  Base   : {base_link or '(auto)'}\n"
            f"  EE     : {ee_link or '(auto)'}"
        )

        generate_pose_dataset_ik(
            urdf_path=urdf_path,
            output_path=output_path,
            cfg=cfg,
            base_link=base_link,
            end_effector_link=ee_link,
            logger=logger,
        )

        if delta_enable:
            delta_path = output_dir / delta_name
            build_pose_deltas(
                input_path=output_path,
                output_path=delta_path,
                cfg=cfg,
                logger=logger,
            )

    logger.info("Batch generation finished.")


if __name__ == "__main__":
    main()
