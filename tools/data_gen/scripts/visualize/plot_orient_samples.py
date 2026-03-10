#!/usr/bin/env python3
"""Script: plot_orient_samples.py
Purpose: Visualize per-anchor orientation samples from the generated orientation dataset.
Usage: python3 tools/data_gen/scripts/visualize/plot_orient_samples.py
"""

import math
import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt

# ---- Config (edit here) ----
H5_PATH = "tools/data_gen/outputs/aubo/aubo_i5/voxel_3mm/dataset_orient.h5"
ANCHOR_INDEX = 0
ANCHOR_LIST = 0  # set [] to use single ANCHOR_INDEX
PLOT_MODE = "sphere"  # axes_grid, logmap_grid, sphere_grid, axes, logmap, sphere
MAX_POINTS = 20000  # downsample for logmap
MAX_POINTS_PER_CLASS = 4000  # per-class downsample for grid modes
AXES_LENGTH = 0.05  # meters
AXES_DRAW_FULL = True  # True: draw x/y/z axes, False: only z-axis
AXES_ALPHA = 0.35
AXES_JITTER = 0.0  # meters
SPHERE_POINT_SIZE = 6
SPHERE_SHOW_MESH = False
USE_PHI_ALPHA = True
SHOW_BOUNDARY_ONLY = False
RANDOM_SEED = 0
# ----------------------------


def quat_normalize(q):
    n = np.linalg.norm(q)
    if n < 1e-12:
        return q
    return q / n


def quat_inv(q):
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def quat_mul(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


def quat_to_rot(q):
    q = quat_normalize(q)
    x, y, z, w = q
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def quat_to_axis_angle(q):
    q = quat_normalize(q)
    if q[3] < 0:
        q = -q
    w = max(-1.0, min(1.0, q[3]))
    angle = 2.0 * math.acos(w)
    s = math.sqrt(max(1.0 - w * w, 0.0))
    if s < 1e-8:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        axis = q[:3] / s
    return axis, angle


def log_map(q):
    q = quat_normalize(q)
    if q[3] < 0:
        q = -q
    v = q[:3]
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * math.atan2(norm, q[3])
    axis = v / norm
    return axis * angle


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_mid = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_mid = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_mid = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_mid - plot_radius, x_mid + plot_radius])
    ax.set_ylim3d([y_mid - plot_radius, y_mid + plot_radius])
    ax.set_zlim3d([z_mid - plot_radius, z_mid + plot_radius])


def _downsample(rng, quat, phi, label, max_points):
    if max_points > 0 and len(quat) > max_points:
        idx = rng.choice(len(quat), size=max_points, replace=False)
        return quat[idx], phi[idx], label[idx]
    return quat, phi, label


def _plot_axes_samples(ax, anchor, quat, phi, color, max_abs, rng):
    if len(quat) == 0:
        ax.set_axis_off()
        return
    jitter = rng.normal(scale=AXES_JITTER, size=(len(quat), 3)) if AXES_JITTER > 0 else 0.0
    for i, q in enumerate(quat):
        R = quat_to_rot(q.astype(np.float64))
        base = anchor + (jitter[i] if AXES_JITTER > 0 else 0.0)
        if USE_PHI_ALPHA:
            alpha = float(np.clip(abs(phi[i]) / max_abs, 0.1, 1.0))
        else:
            alpha = AXES_ALPHA
        if not AXES_DRAW_FULL:
            axes = [R[:, 2]]
        else:
            axes = [R[:, 0], R[:, 1], R[:, 2]]
        for v in axes:
            tip = base + v * AXES_LENGTH
            ax.plot([base[0], tip[0]], [base[1], tip[1]], [base[2], tip[2]],
                    color=(color[0], color[1], color[2], alpha), linewidth=1)


def _plot_logmap_samples(ax, quat, phi, label, q_ref):
    q_ref = quat_normalize(q_ref.astype(np.float64))
    q_ref_inv = quat_inv(q_ref)
    pts = np.zeros((len(quat), 3), dtype=np.float64)
    for i, q in enumerate(quat):
        q_rel = quat_mul(q_ref_inv, q.astype(np.float64))
        pts[i] = log_map(q_rel)
    max_abs = max(float(np.max(np.abs(phi))), 1e-6)
    alpha = np.clip(np.abs(phi) / max_abs, 0.1, 1.0)
    colors = np.zeros((len(phi), 4), dtype=np.float64)
    colors[label == 1] = [0.2, 0.6, 1.0, 1.0]
    colors[label == 0] = [1.0, 0.5, 0.2, 1.0]
    colors[label == 2] = [0.0, 0.0, 0.0, 1.0]
    colors[:, 3] = alpha
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=3, c=colors, linewidths=0)


def _plot_sphere_samples(ax, quat, phi, color, max_abs):
    if len(quat) == 0:
        ax.set_axis_off()
        return
    pts = np.zeros((len(quat), 3), dtype=np.float64)
    for i, q in enumerate(quat):
        axis, _ = quat_to_axis_angle(q.astype(np.float64))
        pts[i] = axis
    if USE_PHI_ALPHA:
        alpha = np.clip(np.abs(phi) / max_abs, 0.1, 1.0)
    else:
        alpha = np.full(len(phi), AXES_ALPHA)
    colors = np.zeros((len(phi), 4), dtype=np.float64)
    colors[:, 0] = color[0]
    colors[:, 1] = color[1]
    colors[:, 2] = color[2]
    colors[:, 3] = alpha
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=SPHERE_POINT_SIZE, c=colors, linewidths=0)


def _plot_sphere_mesh(ax):
    if not SPHERE_SHOW_MESH:
        return
    u = np.linspace(0.0, 2.0 * math.pi, 32)
    v = np.linspace(0.0, math.pi, 16)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color="gray", alpha=0.08, linewidth=0)


def main():
    anchor_list = ANCHOR_LIST if ANCHOR_LIST else [ANCHOR_INDEX]
    with h5py.File(H5_PATH, "r") as f:
        if "/csr/anchor_start" not in f:
            print("Missing /csr/anchor_start in HDF5.")
            return 1
        start = f["/csr/anchor_start"][:]
        anchor_pos_all = f["/anchors/pos"][:] if "/anchors/pos" in f else None

        if PLOT_MODE in ("axes_grid", "logmap_grid", "sphere_grid"):
            rows = len(anchor_list)
            fig = plt.figure(figsize=(12, 4 * rows))
            for r, anchor_id in enumerate(anchor_list):
                if anchor_id < 0 or anchor_id + 1 >= len(start):
                    print(f"Invalid ANCHOR_INDEX={anchor_id}, total anchors={len(start) - 1}")
                    return 1
                s0, s1 = int(start[anchor_id]), int(start[anchor_id + 1])
                if s1 <= s0:
                    print(f"Anchor {anchor_id} has no samples.")
                    continue
                quat = f["/samples/quat"][s0:s1]
                phi = f["/samples/phi"][s0:s1]
                label = f["/samples/label"][s0:s1]
                if SHOW_BOUNDARY_ONLY:
                    mask = label == 2
                    quat = quat[mask]
                    phi = phi[mask]
                    label = label[mask]

                rng = np.random.default_rng(RANDOM_SEED + anchor_id * 31)
                anchor = anchor_pos_all[anchor_id] if anchor_pos_all is not None else np.zeros(3)
                max_abs = max(float(np.max(np.abs(phi))), 1e-6)

                categories = [
                    ("inside", 1, (0.2, 0.6, 1.0)),
                    ("outside", 0, (1.0, 0.5, 0.2)),
                    ("boundary", 2, (0.0, 0.0, 0.0)),
                ]
                for c, (name, lab, color) in enumerate(categories):
                    ax = fig.add_subplot(rows, 3, r * 3 + c + 1, projection="3d")
                    mask = label == lab
                    q_c = quat[mask]
                    p_c = phi[mask]
                    q_c, p_c, _ = _downsample(rng, q_c, p_c, label[mask], MAX_POINTS_PER_CLASS)

                    if PLOT_MODE == "logmap_grid":
                        if np.any(label == 1):
                            q_ref = quat[label == 1][0]
                        else:
                            q_ref = quat[0]
                        _plot_logmap_samples(ax, q_c, p_c, np.full(len(q_c), lab), q_ref)
                        ax.set_xlabel("rx")
                        ax.set_ylabel("ry")
                        ax.set_zlabel("rz")
                    elif PLOT_MODE == "sphere_grid":
                        _plot_sphere_samples(ax, q_c, p_c, color, max_abs)
                        _plot_sphere_mesh(ax)
                        ax.set_xlabel("ax")
                        ax.set_ylabel("ay")
                        ax.set_zlabel("az")
                    else:
                        ax.scatter([anchor[0]], [anchor[1]], [anchor[2]], s=20, c="k")
                        _plot_axes_samples(ax, anchor, q_c, p_c, color, max_abs, rng)
                        ax.set_xlabel("x")
                        ax.set_ylabel("y")
                        ax.set_zlabel("z")
                    ax.set_title(f"anchor {anchor_id} - {name} (n={len(q_c)})")
                    set_axes_equal(ax)

            plt.tight_layout()
            plt.show()
            return 0

        # Single-anchor modes
        anchor_id = ANCHOR_INDEX
        if anchor_id < 0 or anchor_id + 1 >= len(start):
            print(f"Invalid ANCHOR_INDEX={anchor_id}, total anchors={len(start) - 1}")
            return 1
        s0, s1 = int(start[anchor_id]), int(start[anchor_id + 1])
        if s1 <= s0:
            print("This anchor has no samples.")
            return 1
        quat = f["/samples/quat"][s0:s1]
        phi = f["/samples/phi"][s0:s1]
        label = f["/samples/label"][s0:s1]
        anchor = anchor_pos_all[anchor_id] if anchor_pos_all is not None else np.zeros(3)

    rng = np.random.default_rng(RANDOM_SEED)
    if PLOT_MODE == "logmap":
        quat, phi, label = _downsample(rng, quat, phi, label, MAX_POINTS)
    else:
        quat, phi, label = _downsample(rng, quat, phi, label, MAX_POINTS_PER_CLASS)

    if SHOW_BOUNDARY_ONLY:
        mask = label == 2
        quat = quat[mask]
        phi = phi[mask]
        label = label[mask]

    print(
        f"anchor={anchor_id} samples={len(phi)} boundary={(label==2).sum()} "
        f"inside={(label==1).sum()} outside={(label==0).sum()}"
    )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    if PLOT_MODE == "logmap":
        if np.any(label == 1):
            q_ref = quat[label == 1][0]
        else:
            q_ref = quat[0]
        _plot_logmap_samples(ax, quat, phi, label, q_ref)
        ax.set_title("Orientation samples (log map)")
        ax.set_xlabel("rx")
        ax.set_ylabel("ry")
        ax.set_zlabel("rz")
    elif PLOT_MODE == "sphere":
        max_abs = max(float(np.max(np.abs(phi))), 1e-6)
        categories = [
            ("inside", 1, (0.2, 0.6, 1.0)),
            ("outside", 0, (1.0, 0.5, 0.2)),
            ("boundary", 2, (0.0, 0.0, 0.0)),
        ]
        for name, lab, color in categories:
            mask = label == lab
            _plot_sphere_samples(ax, quat[mask], phi[mask], color, max_abs)
        _plot_sphere_mesh(ax)
        ax.set_title("Orientation samples (axis sphere)")
        ax.set_xlabel("ax")
        ax.set_ylabel("ay")
        ax.set_zlabel("az")
    else:
        ax.scatter([anchor[0]], [anchor[1]], [anchor[2]], s=50, c="k")
        max_abs = max(float(np.max(np.abs(phi))), 1e-6)
        categories = [
            ("inside", 1, (0.2, 0.6, 1.0)),
            ("outside", 0, (1.0, 0.5, 0.2)),
            ("boundary", 2, (0.0, 0.0, 0.0)),
        ]
        for name, lab, color in categories:
            mask = label == lab
            _plot_axes_samples(ax, anchor, quat[mask], phi[mask], color, max_abs, rng)
        ax.set_title("Orientation samples (axes at anchor)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    set_axes_equal(ax)
    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
