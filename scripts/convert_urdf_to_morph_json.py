"""Convert URDFs to morphology JSON specs with workspace AABB."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from mcfp.data.morphology_io import save_morph_json, urdf_to_morph_dict
from mcfp.sim.workspace_bounds import compute_urdf_aabb
from mcfp.utils.config import load_config
from mcfp.utils.logging import setup_logger
from mcfp.utils.seed import set_seed


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Convert URDFs to morphology JSON specs.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/convert_morph_json.yaml",
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def _get_cfg_value(cfg: Any, key: str) -> Any:
    """Retrieve mandatory value from config object or dict."""
    if isinstance(cfg, dict):
        if key not in cfg:
            raise ValueError(f"[convert_urdf_to_morph_json] Config missing key '{key}'.")
        return cfg[key]
    if not hasattr(cfg, key):
        raise ValueError(f"[convert_urdf_to_morph_json] Config missing attr '{key}'.")
    return getattr(cfg, key)


def _get_cfg_val_default(cfg: Any, key: str, default: Any) -> Any:
    """Retrieve optional value from config object or dict."""
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def main() -> None:
    """Generate morphology JSON specs for all robots defined in config."""
    args = _parse_args()
    cfg = load_config(args.config)

    logger = setup_logger(
        name="mcfp.scripts.convert_urdf_to_morph_json",
        log_dir=cfg.logging.log_dir,
    )
    logger.info(f"Loaded configuration from: {args.config}")

    run_cfg = getattr(cfg, "run", None)
    seed = getattr(run_cfg, "seed", None) if run_cfg is not None else None
    if seed is not None:
        deterministic = bool(getattr(run_cfg, "deterministic", True))
        set_seed(int(seed), deterministic=deterministic)
        logger.info(f"Global random seed set to {seed}.")

    aabb_cfg = getattr(cfg, "aabb", None)
    bounds_samples = int(getattr(aabb_cfg, "bounds_samples", 50000)) if aabb_cfg is not None else 50000
    margin = float(getattr(aabb_cfg, "margin", 1.1)) if aabb_cfg is not None else 1.1

    robots: List[Dict[str, Any]] = list(getattr(cfg, "robots", []))
    if not robots:
        raise ValueError("[convert_urdf_to_morph_json] No robots configured.")

    for idx, robot_cfg in enumerate(robots):
        urdf_path = Path(_get_cfg_value(robot_cfg, "urdf_path")).resolve()
        output_dir = Path(_get_cfg_value(robot_cfg, "output_dir")).resolve()

        family = _get_cfg_val_default(robot_cfg, "family", None)
        robot_name = _get_cfg_val_default(robot_cfg, "robot_name", None) or urdf_path.stem
        base_link = _get_cfg_val_default(robot_cfg, "base_link", None)
        ee_link = _get_cfg_val_default(robot_cfg, "ee_link", None)

        logger.info(
            f"[{idx + 1}/{len(robots)}] Converting URDF:\n"
            f"  URDF       : {urdf_path}\n"
            f"  Output dir : {output_dir}\n"
            f"  Family     : {family}\n"
            f"  Robot name : {robot_name}\n"
            f"  Base link  : {base_link or '(auto)'}\n"
            f"  EE link    : {ee_link or '(auto)'}"
        )

        if not urdf_path.is_file():
            logger.error(f"[{idx + 1}/{len(robots)}] URDF not found: {urdf_path}")
            continue

        variant_id = f"{robot_name}_base"
        morph_dict = urdf_to_morph_dict(
            urdf_path=urdf_path,
            robot_name=robot_name,
            family=family,
            source="real",
            base_link=base_link,
            ee_link=ee_link,
            variant_id=variant_id,
        )

        try:
            aabb_min, aabb_max = compute_urdf_aabb(
                urdf_path=urdf_path,
                base_link=base_link,
                end_effector_link=ee_link,
                samples=bounds_samples,
                margin=margin,
                logger=logger,
            )
        except Exception as exc:
            logger.error(f"[{idx + 1}/{len(robots)}] AABB failed: {exc}")
            continue

        morph_dict["workspace"] = {
            "aabb_min": aabb_min.tolist(),
            "aabb_max": aabb_max.tolist(),
            "bounds_samples": int(bounds_samples),
            "margin": float(margin),
        }

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{variant_id}.json"
        save_morph_json(morph_dict, output_path)

        logger.info(
            f"[{idx + 1}/{len(robots)}] Done. Generated "
            f"{output_path}"
        )


if __name__ == "__main__":
    main()
