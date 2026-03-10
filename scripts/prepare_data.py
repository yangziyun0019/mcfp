"""Prepare processed datasets and morphology specs from raw generated assets.

This CLI entry point drives dataset conversion and robot morphology extraction for the MCFP pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mcfp.data.prepare_dataset import prepare_dataset
from mcfp.data.morph_spec import build_morphology_spec, compute_l_ref_from_urdf
from mcfp.utils.config import load_config
from mcfp.utils.logging import setup_logger


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare training datasets and morphology spec.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/prepare_data.yaml",
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)

    # Inject config path for snapshotting.
    if not hasattr(cfg, "paths"):
        cfg.paths = type("obj", (), {})()
    setattr(cfg.paths, "config_path", args.config)

    repo_root = Path(getattr(cfg.paths, "repo_root", ".")).resolve()
    log_dir = getattr(cfg, "logging", None)
    log_dir = getattr(log_dir, "log_dir", None) if log_dir is not None else None
    logger = setup_logger("mcfp.prepare_data", log_dir=log_dir)

    run_cfg = getattr(cfg, "run", None)
    do_dataset = True if run_cfg is None else bool(getattr(run_cfg, "prepare_dataset", True))
    do_morph = True if run_cfg is None else bool(getattr(run_cfg, "prepare_morphology", True))

    urdf_path = Path(getattr(cfg.paths, "urdf_path", ""))
    if not urdf_path.is_absolute():
        urdf_path = (repo_root / urdf_path).resolve()

    base_link = getattr(cfg.robot, "base_link", None) if hasattr(cfg, "robot") else None
    ee_link = getattr(cfg.robot, "ee_link", None) if hasattr(cfg, "robot") else None

    l_ref = compute_l_ref_from_urdf(urdf_path, base_link=base_link, ee_link=ee_link)
    logger.info(f"[prepare_data] L_ref={l_ref:.6f} m")

    if do_dataset:
        prepare_dataset(cfg, logger, repo_root=repo_root, l_ref=l_ref)

    if do_morph:
        build_morphology_spec(cfg, logger, repo_root=repo_root)


if __name__ == "__main__":
    main()
