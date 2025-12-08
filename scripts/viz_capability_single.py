# scripts/viz_capability_single.py

from __future__ import annotations

from pathlib import Path

import numpy as np

from mcfp.utils.config import load_config
from mcfp.viz.capability_viz import (
    load_capability_map,
    plot_capability_with_robot,
)


def main() -> None:
    """Visualise multiple scalar indicators from a single capability map.

    This script is intentionally simple: it loads one capability_map.npz
    file and then calls the plotting routine multiple times with different
    scalar fields, so that each indicator can be inspected separately.
    """
    cfg = load_config("configs/data_gen_single.yaml")

    urdf_root = Path(cfg.data.urdf_root)
    capability_root = Path(cfg.data.capability_root)

    urdf_path = urdf_root / cfg.robot.urdf_filename
    cap_path = capability_root / cfg.robot.name / "capability_map.npz"

    base_link = cfg.sim.base_link
    end_effector_link = cfg.sim.end_effector_link

    log_dir = Path(cfg.logging.log_dir)

    # Load capability map once so we can derive composite fields (ratios)
    cap_data = load_capability_map(cap_path)

    # 1) Simple scalar fields stored in the capability map
    scalar_keys = [
        "u",
        "g_ws",
        "g_self",
        "g_self_qstar",
        "g_lim",
        "g_man",
        "g_sigma_min",
        "g_iso",
        "g_ws_margin",
        "g_red",
    ]

    for key in scalar_keys:
        if key not in cap_data:
            # Skip keys that are not present for this map
            continue

        print(f"[viz_capability_single] Plotting field: {key}")
        plot_capability_with_robot(
            urdf_path=urdf_path,
            cap_path=cap_path,
            base_link=base_link,
            end_effector_link=end_effector_link,
            value_key=key,
            log_dir=log_dir,
        )

    # 2) Self-collision ratio: compare representative pose vs. average
    #    ratio > 1: q* is safer than the average IK in this cell
    #    ratio < 1: q* is more cramped than the average
    if "g_self" in cap_data and "g_self_qstar" in cap_data:
        g_self = np.asarray(cap_data["g_self"], dtype=np.float32)
        g_self_qstar = np.asarray(cap_data["g_self_qstar"], dtype=np.float32)

        eps = 1e-6
        # Avoid division by zero; values where both are ~0 are not informative anyway
        denom = g_self + eps
        self_ratio = g_self_qstar / denom

        print("[viz_capability_single] Plotting self-collision ratio: "
              "g_self_qstar / (g_self + eps)")
        plot_capability_with_robot(
            urdf_path=urdf_path,
            cap_path=cap_path,
            base_link=base_link,
            end_effector_link=end_effector_link,
            value_key="g_self_ratio",
            log_dir=log_dir,
            custom_values=self_ratio,
        )


if __name__ == "__main__":
    main()
