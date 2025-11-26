from __future__ import annotations

from typing import Literal

import numpy as np
import matplotlib.pyplot as plt

from mcfp.data.io import load_capability_map


def plot_capability_slice(
    path,
    indicator: Literal["u", "g_ws", "g_self", "g_lim", "g_man"] = "u",
) -> None:
    """Plot a simple scatter of capability map (XY projection).

    Parameters
    ----------
    path:
        Path to the NPZ capability map.
    indicator:
        Which indicator to visualize as color.
    """
    data = load_capability_map(path)
    centers = data["cell_centers"]
    vals = data[indicator]

    x = centers[:, 0]
    y = centers[:, 1]

    plt.figure()
    sc = plt.scatter(x, y, c=vals, s=20)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Capability map ({indicator})")
    plt.colorbar(sc, label=indicator)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()
