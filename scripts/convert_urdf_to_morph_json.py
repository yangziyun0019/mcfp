# scripts/generate_morph_specs_local.py
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List

from mcfp.data.morphology_io import generate_variants_from_urdf
from mcfp.utils.logging import setup_logger


# ---------------------------------------------------------------------------
# Hard-coded local configuration.
#
# You can edit this ROBOTS list manually whenever you add a new URDF or
# want to change the number of variants / perturbation ranges.
#
# Paths are relative to the repository root by default. Adjust them to
# match your actual directory layout.
# ---------------------------------------------------------------------------

ROBOTS: List[Dict[str, Any]] = [
    {
        "urdf_path": r"D:\code\mcfp\data\urdf\rm_65_b_description\urdf\rm_65_b_description.urdf",
        "output_dir": r"D:\code\mcfp\data\morph_specs\rm65",

        "family": "rm65",
        "robot_name": "rm65",
        "base_link": "base_link",
        "ee_link": "Link6",
        "num_variants": 0,
        "length_scale_range": (0.8, 1.2),
        "joint_limit_scale_range": (0.6, 1.0),
        "joint_limit_shift_fraction": 0.2,
    },
    # {
    #     "urdf_path": "urdf/wx200.urdf",
    #     "output_dir": "data/morph_specs/wx200",
    #     "family": "wx200",
    #     "robot_name": "wx200",
    #     "base_link": "base_link",
    #     "ee_link": "wrist_link",
    #     "num_variants": 250,
    #     "length_scale_range": (0.8, 1.2),
    #     "joint_limit_scale_range": (0.6, 1.0),
    #     "joint_limit_shift_fraction": 0.2,
    # },
]


# Global seed for all perturbations. Change this if you want a new batch.
GLOBAL_SEED: int = 42


def main() -> None:
    """Generate morphology JSON specs for all robots defined in ROBOTS."""
    logger = setup_logger(name="mcfp.scripts.generate_morph_specs_local")
    random.seed(GLOBAL_SEED)
    logger.info(f"Global random seed set to {GLOBAL_SEED}.")

    for idx, cfg in enumerate(ROBOTS):
        urdf_path = Path(cfg["urdf_path"]).resolve()
        output_dir = Path(cfg["output_dir"]).resolve()

        family = cfg.get("family")
        robot_name = cfg.get("robot_name")
        base_link = cfg.get("base_link")
        ee_link = cfg.get("ee_link")
        num_variants = int(cfg.get("num_variants", 0))

        length_scale_range = cfg.get("length_scale_range", (0.7, 1.3))
        joint_limit_scale_range = cfg.get("joint_limit_scale_range", (0.5, 1.0))
        joint_limit_shift_fraction = float(
            cfg.get("joint_limit_shift_fraction", 0.2)
        )

        logger.info(
            f"[{idx + 1}/{len(ROBOTS)}] Generating morph specs for URDF:\n"
            f"  URDF        : {urdf_path}\n"
            f"  Output dir  : {output_dir}\n"
            f"  Family      : {family}\n"
            f"  Robot name  : {robot_name or urdf_path.stem}\n"
            f"  Base link   : {base_link or '(auto)'}\n"
            f"  EE link     : {ee_link or '(auto)'}\n"
            f"  Variants    : {num_variants}\n"
            f"  Length scale: {length_scale_range}\n"
            f"  Limit scale : {joint_limit_scale_range}\n"
            f"  Shift frac  : {joint_limit_shift_fraction}"
        )

        json_paths = generate_variants_from_urdf(
            urdf_path=urdf_path,
            output_dir=output_dir,
            family=family,
            source="real",
            robot_name=robot_name,
            base_link=base_link,
            ee_link=ee_link,
            num_variants=num_variants,
            length_scale_range=(
                float(length_scale_range[0]),
                float(length_scale_range[1]),
            ),
            joint_limit_scale_range=(
                float(joint_limit_scale_range[0]),
                float(joint_limit_scale_range[1]),
            ),
            joint_limit_shift_fraction=joint_limit_shift_fraction,
        )

        logger.info(
            f"[{idx + 1}/{len(ROBOTS)}] Done. Generated "
            f"{len(json_paths)} JSON files under {output_dir}."
        )


if __name__ == "__main__":
    main()
