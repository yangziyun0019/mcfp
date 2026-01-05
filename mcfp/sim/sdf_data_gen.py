# mcfp/sim/sdf_data_gen.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import math

import numpy as np
from scipy.spatial.transform import Rotation as R

from mcfp.sim.robot_model import RobotModel
from mcfp.sim.collision import SelfCollisionChecker, MeshSelfCollisionChecker, HybridSelfCollisionChecker
from mcfp.sim.workspace_bounds import estimate_workspace_bounds, apply_aabb_margin
from mcfp.data.morphology_io import urdf_to_morph_dict
from mcfp.data.io import compute_morph_scale
from mcfp.data.sdf_io import save_npz, save_json
from mcfp.utils import se3
from mcfp.utils.seed import set_seed


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    """Read config value from dict-like or object-like cfg."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _get_path(cfg: Any, key: str, default: Any = None) -> Any:
    """Read nested config value using a dotted key path."""
    cur: Any = cfg
    for part in key.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            if part not in cur:
                return default
            cur = cur[part]
        else:
            if not hasattr(cur, part):
                return default
            cur = getattr(cur, part)
    return cur


def _resolve_path(path: str | Path, repo_root: Path) -> Path:
    """Resolve path relative to repo_root if needed."""
    p = Path(path)
    if not p.is_absolute():
        return (repo_root / p).resolve()
    return p.resolve()


def _sample_uniform_quaternion(rng: np.random.Generator) -> np.ndarray:
    """Sample a uniform quaternion (x, y, z, w)."""
    u1, u2, u3 = rng.random(3)
    s1 = math.sqrt(1.0 - u1)
    s2 = math.sqrt(u1)
    theta1 = 2.0 * math.pi * u2
    theta2 = 2.0 * math.pi * u3
    qx = s1 * math.sin(theta1)
    qy = s1 * math.cos(theta1)
    qz = s2 * math.sin(theta2)
    qw = s2 * math.cos(theta2)
    return np.array([qx, qy, qz, qw], dtype=np.float32)


def _pose_to_z(pose: np.ndarray, l_ref: float) -> np.ndarray:
    """Convert pose [x,y,z,qx,qy,qz,qw] to z=[p/L_ref, log(R)]."""
    pose = np.asarray(pose, dtype=np.float32).reshape(7)
    pos = pose[:3]
    quat = se3.quat_normalize(pose[3:])
    rotvec = R.from_quat(quat).as_rotvec().astype(np.float32)
    z = np.concatenate([pos / float(l_ref), rotvec], axis=0)
    return z.astype(np.float32)


@dataclass(frozen=True)
class OracleTier:
    """Oracle tier configuration."""
    seeds: int
    max_iters: int


@dataclass(frozen=True)
class OracleConfig:
    """Configuration for reachability oracle."""
    pos_tol_m: float
    rot_tol_rad: float
    residual_threshold: float
    tiers: List[OracleTier]


class ReachabilityOracle:
    """IK-based reachability oracle with fixed-budget tiers.

    Args:
        robot: RobotModel instance.
        collision_checker: Collision checker with is_collision_free(q).
        cfg: OracleConfig.
        rng: NumPy random generator for rest-pose sampling.
    """

    def __init__(
        self,
        robot: RobotModel,
        collision_checker: HybridSelfCollisionChecker,
        cfg: OracleConfig,
        rng: np.random.Generator,
    ) -> None:
        self.robot = robot
        self.collision_checker = collision_checker
        self.cfg = cfg
        self.rng = rng

    def is_reachable(self, pose: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """Check reachability for a target pose.

        Args:
            pose: Pose array [x,y,z,qx,qy,qz,qw].

        Returns:
            reachable: True if reachable under the fixed oracle budget.
            q: Joint solution if reachable, else None.
        """
        pose = np.asarray(pose, dtype=np.float32).reshape(7)
        target_pos = pose[:3]
        target_quat = se3.quat_normalize(pose[3:])

        if self.robot.joint_limits.size == 0:
            return False, None

        for tier in self.cfg.tiers:
            for _ in range(int(tier.seeds)):
                rest = _sample_rest_pose(self.rng, self.robot.joint_limits)
                q = self.robot.ik(
                    target_pos=target_pos,
                    target_quat=target_quat,
                    rest_pose=rest,
                    max_iters=int(tier.max_iters),
                    residual_threshold=float(self.cfg.residual_threshold),
                )
                if q is None or not np.all(np.isfinite(q)):
                    continue

                if not _within_limits(q, self.robot.joint_limits):
                    continue

                try:
                    pos_fk, quat_fk = self.robot.fk(q)
                except Exception:
                    continue

                pos_err = float(np.linalg.norm(pos_fk - target_pos))
                rot_err = float(np.linalg.norm(se3.quat_delta_axis_angle(target_quat, quat_fk)))

                if pos_err > self.cfg.pos_tol_m or rot_err > self.cfg.rot_tol_rad:
                    continue

                if not self.collision_checker.is_collision_free(q):
                    continue

                return True, np.asarray(q, dtype=np.float32)

        return False, None


def _sample_rest_pose(rng: np.random.Generator, limits: np.ndarray) -> np.ndarray:
    """Sample a rest pose uniformly within joint limits."""
    low = limits[:, 0]
    span = limits[:, 1] - low
    span = np.where(span > 0.0, span, 0.0)
    return (low + rng.random(limits.shape[0]) * span).astype(np.float32)


def _within_limits(q: np.ndarray, limits: np.ndarray, eps: float = 1e-6) -> bool:
    """Check joint limits for q."""
    q_arr = np.asarray(q, dtype=np.float32).reshape(-1)
    low = limits[:, 0].astype(np.float32)
    high = limits[:, 1].astype(np.float32)
    return bool(np.all(q_arr >= (low - eps)) and np.all(q_arr <= (high + eps)))


def compute_l_ref_from_urdf(
    urdf_path: Path,
    base_link: Optional[str],
    end_effector_link: Optional[str],
) -> float:
    """Compute L_ref as sum of main-chain joint origin lengths."""
    spec = urdf_to_morph_dict(
        urdf_path=urdf_path,
        robot_name=urdf_path.stem,
        family=None,
        source="real",
        base_link=base_link,
        ee_link=end_effector_link,
        variant_id=f"{urdf_path.stem}_base",
    )
    return float(compute_morph_scale(spec))


def _compute_g_lim(q: np.ndarray, joint_limits: np.ndarray) -> float:
    """Compute normalized joint limit margin in [0, 1]."""
    if joint_limits.size == 0:
        return 1.0

    q_arr = np.asarray(q, dtype=np.float32).reshape(-1)
    lower = joint_limits[:, 0].astype(np.float32)
    upper = joint_limits[:, 1].astype(np.float32)
    widths = upper - lower
    widths[widths <= 0.0] = 1e-6

    margins_low = q_arr - lower
    margins_high = upper - q_arr
    inside = (margins_low >= 0.0) & (margins_high >= 0.0)
    margin = np.minimum(margins_low, margins_high)
    margin[~inside] = 0.0

    half_width = widths * 0.5
    norm = np.clip(margin / half_width, 0.0, 1.0)
    return float(norm.mean())


def _compute_g_sigma_min(q: np.ndarray, robot: RobotModel, k_sigma: float) -> float:
    """Compute minimum singular value indicator in [0, 1]."""
    try:
        J = robot.jacobian(q)
    except Exception:
        return 0.0

    if J is None:
        return 0.0

    J = np.asarray(J, dtype=np.float64)
    if J.ndim != 2:
        return 0.0

    col_norms = np.linalg.norm(J, axis=0)
    J = J[:, col_norms > 1e-6]
    if J.size == 0:
        return 0.0

    rows, cols = J.shape
    if cols < 6 and rows == 6:
        J = J[:3, :]

    try:
        s = np.linalg.svd(J, compute_uv=False)
    except np.linalg.LinAlgError:
        return 0.0

    if s.size == 0:
        return 0.0

    sigma_min = float(np.min(s))
    if not np.isfinite(sigma_min) or sigma_min <= 0.0:
        return 0.0

    k_sigma = max(float(k_sigma), 1e-6)
    g = sigma_min / (sigma_min + k_sigma)
    return float(np.clip(g, 0.0, 1.0))


def _fibonacci_sphere(num_points: int) -> np.ndarray:
    """Generate uniformly distributed points on a unit sphere."""
    if num_points <= 0:
        raise ValueError("num_points must be positive")
    points = []
    phi = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(num_points):
        y = 1.0 - (i / float(num_points - 1)) * 2.0
        radius = math.sqrt(max(1.0 - y * y, 0.0))
        theta = phi * i
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        points.append([x, y, z])
    return np.asarray(points, dtype=np.float32)


def _axis_to_dir_bin(axis: np.ndarray, ref_points: np.ndarray) -> int:
    """Assign axis to nearest reference direction bin."""
    axis = np.asarray(axis, dtype=np.float32).reshape(3)
    n = float(np.linalg.norm(axis))
    if n <= 1e-8:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        axis = axis / n
    dots = ref_points @ axis
    return int(np.argmax(dots))


def _compute_voxel_indices(
    pos_norm: np.ndarray,
    aabb_min_norm: np.ndarray,
    aabb_max_norm: np.ndarray,
    voxel_size: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute voxel indices and grid dims for normalized positions."""
    spans = aabb_max_norm - aabb_min_norm
    dims = np.ceil(spans / float(voxel_size)).astype(np.int32)
    dims = np.maximum(dims, 1)

    idx = np.floor((pos_norm - aabb_min_norm) / float(voxel_size)).astype(np.int32)
    idx = np.clip(idx, 0, dims - 1)
    return idx, dims


def select_anchor_indices(
    poses: np.ndarray,
    q_pool: np.ndarray,
    robot: RobotModel,
    l_ref: float,
    aabb_min: np.ndarray,
    aabb_max: np.ndarray,
    voxel_size: float,
    dir_bins: int,
    angle_bins: int,
    anchor_ratio: float,
    anchor_min: int,
    anchor_max: int,
    use_sigma: bool,
    sigma_k: float,
    logger,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select boundary anchors with coverage and bias.

    Args:
        poses: (N,7) pose array.
        q_pool: (N,DOF) joint samples.
        robot: RobotModel instance.
        l_ref: L_ref scale.
        aabb_min, aabb_max: Workspace bounds in meters.
        voxel_size: Voxel size in normalized units.
        dir_bins: Number of axis bins on the sphere.
        angle_bins: Number of angle bins in [0, pi].
        anchor_ratio, anchor_min, anchor_max: Target anchor sizing rules.
        use_sigma: Whether to include singularity bias.
        sigma_k: Sigma normalization scale.
        logger: Logger instance.

    Returns:
        anchor_indices: Selected anchor indices into poses/q_pool.
        bias_score: Bias score for each sample.
        g_lim: Joint limit margin score.
        g_sigma: Singularity score (zeros if disabled).
    """
    n = int(poses.shape[0])
    pos = poses[:, :3].astype(np.float32)
    pos_norm = pos / float(l_ref)

    aabb_min_norm = aabb_min.astype(np.float32) / float(l_ref)
    aabb_max_norm = aabb_max.astype(np.float32) / float(l_ref)

    voxel_idx, _ = _compute_voxel_indices(pos_norm, aabb_min_norm, aabb_max_norm, voxel_size)

    ref_dirs = _fibonacci_sphere(dir_bins)

    bias_score = np.zeros((n,), dtype=np.float32)
    g_lim = np.zeros((n,), dtype=np.float32)
    g_sigma = np.zeros((n,), dtype=np.float32)

    logger.info("[sdf_data_gen] Computing anchor bias scores...")
    for i in range(n):
        q = q_pool[i]
        g_lim[i] = float(_compute_g_lim(q, robot.joint_limits))
        if use_sigma:
            g_sigma[i] = float(_compute_g_sigma_min(q, robot, k_sigma=sigma_k))
            bias_score[i] = float(min(g_lim[i], g_sigma[i]))
        else:
            g_sigma[i] = 1.0
            bias_score[i] = float(g_lim[i])

    best_per_bucket: Dict[Tuple[int, int, int, int, int], int] = {}
    best_bias: Dict[Tuple[int, int, int, int, int], float] = {}

    for i in range(n):
        quat = poses[i, 3:]
        rotvec = R.from_quat(se3.quat_normalize(quat)).as_rotvec()
        angle = float(np.linalg.norm(rotvec))
        axis = rotvec if angle > 1e-8 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
        dir_idx = _axis_to_dir_bin(axis, ref_dirs)
        angle_idx = int(min(angle / math.pi * angle_bins, angle_bins - 1))

        vx, vy, vz = voxel_idx[i]
        key = (int(vx), int(vy), int(vz), int(dir_idx), int(angle_idx))

        b = float(bias_score[i])
        if key not in best_per_bucket or b < best_bias[key]:
            best_per_bucket[key] = i
            best_bias[key] = b

    coverage = np.array(list(best_per_bucket.values()), dtype=np.int32)

    target = int(round(n * float(anchor_ratio)))
    target = max(int(anchor_min), min(int(anchor_max), target))
    target = min(target, n)

    logger.info(
        f"[sdf_data_gen] Anchor coverage={len(coverage)} target={target}"
    )

    if len(coverage) > target:
        cov_bias = bias_score[coverage]
        order = np.argsort(cov_bias)
        anchor_indices = coverage[order[:target]]
        return anchor_indices.astype(np.int32), bias_score, g_lim, g_sigma

    remaining = np.setdiff1d(np.arange(n, dtype=np.int32), coverage, assume_unique=False)
    rem_bias = bias_score[remaining]
    rem_order = np.argsort(rem_bias)
    need = target - len(coverage)
    extra = remaining[rem_order[:need]] if need > 0 else np.array([], dtype=np.int32)

    anchor_indices = np.concatenate([coverage, extra], axis=0)
    return anchor_indices.astype(np.int32), bias_score, g_lim, g_sigma


def sample_positive_pool(
    robot: RobotModel,
    collision_checker: HybridSelfCollisionChecker,
    num_samples: int,
    l_ref: float,
    rng: np.random.Generator,
    max_attempts_factor: float,
    progress_interval: int,
    logger,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample FK positive pool A+ with self-collision filtering.

    Args:
        robot: RobotModel instance.
        collision_checker: Hybrid collision checker.
        num_samples: Target number of samples N+.
        l_ref: L_ref scale.
        rng: NumPy random generator.
        max_attempts_factor: Max attempts as a multiple of N+.
        progress_interval: Logging interval.
        logger: Logger instance.

    Returns:
        q_pool: (N+, DOF) joint configurations.
        poses: (N+, 7) end-effector poses.
        z: (N+, 6) normalized pose coordinates.
    """
    dof = robot.num_joints
    q_pool = np.zeros((num_samples, dof), dtype=np.float32)
    poses = np.zeros((num_samples, 7), dtype=np.float32)
    z_pool = np.zeros((num_samples, 6), dtype=np.float32)

    low = robot.joint_limits[:, 0]
    span = robot.joint_limits[:, 1] - low

    max_attempts = int(max_attempts_factor * float(num_samples))
    attempts = 0
    filled = 0

    logger.info(f"[sdf_data_gen] Sampling A+ with target N+={num_samples}")

    while filled < num_samples and attempts < max_attempts:
        q = low + rng.random(dof) * span
        attempts += 1

        if not collision_checker.is_collision_free(q):
            continue

        try:
            pos, quat = robot.fk(q)
        except Exception:
            continue

        pose = np.concatenate([np.asarray(pos, dtype=np.float32), se3.quat_normalize(quat)], axis=0)

        q_pool[filled] = q.astype(np.float32)
        poses[filled] = pose
        z_pool[filled] = _pose_to_z(pose, l_ref)
        filled += 1

        if progress_interval > 0 and (filled % progress_interval == 0):
            logger.info(
                f"[sdf_data_gen] A+ filled={filled}/{num_samples} attempts={attempts}"
            )

    if filled < num_samples:
        raise RuntimeError(
            f"[sdf_data_gen] A+ sampling failed: filled={filled} attempts={attempts}"
        )

    return q_pool, poses, z_pool


def _sample_directions(
    rng: np.random.Generator,
    k: int,
    axis_fraction: float,
    lam: float,
    axis_mode: str,
) -> np.ndarray:
    """Sample directions on the 6D sphere under the lambda metric."""
    k = int(k)
    axis_fraction = float(axis_fraction)
    axis_mode = str(axis_mode).lower()

    axis_dirs: List[np.ndarray] = []
    for i in range(3):
        v = np.zeros(6, dtype=np.float32)
        v[i] = 1.0
        axis_dirs.append(v.copy())
        axis_dirs.append(-v.copy())
    for i in range(3):
        v = np.zeros(6, dtype=np.float32)
        v[3 + i] = float(lam)
        axis_dirs.append(v.copy())
        axis_dirs.append(-v.copy())

    dirs: List[np.ndarray] = []
    if axis_mode in ("fixed_all", "fixed", "all"):
        if k <= len(axis_dirs):
            idx = rng.choice(len(axis_dirs), size=k, replace=False)
            for i in idx:
                dirs.append(axis_dirs[int(i)])
            return np.asarray(dirs, dtype=np.float32)

        dirs.extend(axis_dirs)
        n_rand = k - len(axis_dirs)
        for _ in range(n_rand):
            u_p = rng.normal(size=3)
            u_xi = rng.normal(size=3) * float(lam)
            norm = math.sqrt(
                float(np.dot(u_p, u_p) + np.dot(u_xi, u_xi) / (lam * lam))
            )
            if norm <= 1e-8:
                norm = 1.0
            u = np.concatenate([u_p, u_xi], axis=0) / float(norm)
            dirs.append(u.astype(np.float32))
        return np.asarray(dirs, dtype=np.float32)

    if axis_mode in ("fraction", "ratio"):
        n_axis = int(round(k * axis_fraction))
        n_axis = max(0, min(k, n_axis))
        if n_axis > 0:
            replace = n_axis > len(axis_dirs)
            idx = rng.choice(len(axis_dirs), size=n_axis, replace=replace)
            for i in idx:
                dirs.append(axis_dirs[int(i)])

        for _ in range(k - n_axis):
            u_p = rng.normal(size=3)
            u_xi = rng.normal(size=3) * float(lam)
            norm = math.sqrt(
                float(np.dot(u_p, u_p) + np.dot(u_xi, u_xi) / (lam * lam))
            )
            if norm <= 1e-8:
                norm = 1.0
            u = np.concatenate([u_p, u_xi], axis=0) / float(norm)
            dirs.append(u.astype(np.float32))

        return np.asarray(dirs, dtype=np.float32)

    raise ValueError(f"[sdf_data_gen] Unknown axis_mode: {axis_mode}")


def _pose_at_t(pose0: np.ndarray, u: np.ndarray, t: float, l_ref: float) -> np.ndarray:
    """Evaluate pose along a ray at parameter t."""
    pose0 = np.asarray(pose0, dtype=np.float32).reshape(7)
    p0 = pose0[:3]
    q0 = se3.quat_normalize(pose0[3:])
    u = np.asarray(u, dtype=np.float32).reshape(6)

    u_p = u[:3]
    u_xi = u[3:]

    pos = p0 + (float(t) * float(l_ref)) * u_p
    delta_axis_angle = float(t) * u_xi
    quat = se3.apply_delta_quat(q0, delta_axis_angle)
    return np.concatenate([pos, quat], axis=0).astype(np.float32)


def generate_boundary_samples(
    anchor_indices: np.ndarray,
    poses: np.ndarray,
    oracle: ReachabilityOracle,
    oracle_final: Optional[ReachabilityOracle],
    l_ref: float,
    k: int,
    lam: float,
    t_max: float,
    eps: float,
    delta: float,
    bracket_steps: int,
    max_bisect_iters: int,
    axis_fraction: float,
    axis_mode: str,
    include_boundary_points: bool,
    rng: np.random.Generator,
    progress_interval: int,
    logger,
) -> Dict[str, np.ndarray]:
    """Generate boundary neighborhood samples with ray shooting.

    Returns:
        Dict of arrays for D_nb.
    """
    num_anchors = int(anchor_indices.shape[0])
    per_dir = 2 + (1 if include_boundary_points else 0)
    max_samples = num_anchors * int(k) * per_dir

    poses_out = np.zeros((max_samples, 7), dtype=np.float32)
    z_out = np.zeros((max_samples, 6), dtype=np.float32)
    y_out = np.zeros((max_samples,), dtype=np.float32)
    anchor_out = np.zeros((max_samples,), dtype=np.int32)
    t_out = np.zeros((max_samples,), dtype=np.float32)
    u_out = np.zeros((max_samples, 6), dtype=np.float32)
    side_out = np.zeros((max_samples,), dtype=np.int8)

    t_step = float(t_max) / float(bracket_steps)
    filled = 0

    for a_i, anchor_idx in enumerate(anchor_indices.tolist(), start=1):
        pose0 = poses[int(anchor_idx)]
        dirs = _sample_directions(
            rng,
            k,
            axis_fraction=axis_fraction,
            lam=lam,
            axis_mode=axis_mode,
        )

        for u in dirs:
            t_low = 0.0
            t_high = None

            for step in range(1, int(bracket_steps) + 1):
                t = float(step) * t_step
                pose_t = _pose_at_t(pose0, u, t, l_ref)
                reachable, _ = oracle.is_reachable(pose_t)
                if not reachable:
                    t_high = t
                    t_low = t - t_step
                    break

            if t_high is None:
                continue

            t_lo = float(t_low)
            t_hi = float(t_high)
            for _ in range(int(max_bisect_iters)):
                if (t_hi - t_lo) <= float(eps):
                    break
                t_mid = 0.5 * (t_lo + t_hi)
                pose_mid = _pose_at_t(pose0, u, t_mid, l_ref)
                reachable, _ = oracle.is_reachable(pose_mid)
                if reachable:
                    t_lo = t_mid
                else:
                    t_hi = t_mid

            t_star = 0.5 * (t_lo + t_hi)
            if t_star <= float(delta) or (t_star + float(delta)) >= float(t_max):
                continue
            pose_in = _pose_at_t(pose0, u, t_star - float(delta), l_ref)
            pose_out = _pose_at_t(pose0, u, t_star + float(delta), l_ref)
            if oracle_final is not None:
                reachable_in, _ = oracle_final.is_reachable(pose_in)
                reachable_out, _ = oracle_final.is_reachable(pose_out)
                if not reachable_in or reachable_out:
                    continue

            for side, pose_s, y_s in (
                (1, pose_in, float(delta)),
                (-1, pose_out, -float(delta)),
            ):
                if filled >= max_samples:
                    raise RuntimeError("[sdf_data_gen] D_nb buffer overflow")
                poses_out[filled] = pose_s
                z_out[filled] = _pose_to_z(pose_s, l_ref)
                y_out[filled] = float(y_s)
                anchor_out[filled] = int(anchor_idx)
                t_out[filled] = float(t_star)
                u_out[filled] = u
                side_out[filled] = int(side)
                filled += 1

            if include_boundary_points:
                pose_b = _pose_at_t(pose0, u, t_star, l_ref)
                if filled >= max_samples:
                    raise RuntimeError("[sdf_data_gen] D_nb buffer overflow")
                poses_out[filled] = pose_b
                z_out[filled] = _pose_to_z(pose_b, l_ref)
                y_out[filled] = 0.0
                anchor_out[filled] = int(anchor_idx)
                t_out[filled] = float(t_star)
                u_out[filled] = u
                side_out[filled] = 0
                filled += 1

        if progress_interval > 0 and (a_i % int(progress_interval) == 0):
            logger.info(
                f"[sdf_data_gen] D_nb anchors={a_i}/{num_anchors} samples={filled}"
            )

    out = {
        "poses": poses_out[:filled],
        "z": z_out[:filled],
        "y": y_out[:filled],
        "anchor_idx": anchor_out[:filled],
        "t_star": t_out[:filled],
        "u": u_out[:filled],
        "side": side_out[:filled],
    }
    return out


def _select_unique_positions(
    poses: np.ndarray,
    l_ref: float,
    aabb_min: np.ndarray,
    aabb_max: np.ndarray,
    voxel_size: float,
) -> np.ndarray:
    """Select one pose index per position voxel."""
    pos = poses[:, :3].astype(np.float32)
    pos_norm = pos / float(l_ref)
    aabb_min_norm = aabb_min.astype(np.float32) / float(l_ref)
    aabb_max_norm = aabb_max.astype(np.float32) / float(l_ref)

    voxel_idx, _ = _compute_voxel_indices(pos_norm, aabb_min_norm, aabb_max_norm, voxel_size)

    seen: Dict[Tuple[int, int, int], int] = {}
    for i in range(poses.shape[0]):
        vx, vy, vz = voxel_idx[i]
        key = (int(vx), int(vy), int(vz))
        if key not in seen:
            seen[key] = i

    return np.asarray(list(seen.values()), dtype=np.int32)


def generate_rot_far_samples(
    poses: np.ndarray,
    l_ref: float,
    aabb_min: np.ndarray,
    aabb_max: np.ndarray,
    voxel_size: float,
    n_pos: int,
    n_rot: int,
    oracle: ReachabilityOracle,
    rng: np.random.Generator,
    progress_interval: int,
    logger,
) -> Dict[str, np.ndarray]:
    """Generate rotation-far negative samples."""
    unique_indices = _select_unique_positions(poses, l_ref, aabb_min, aabb_max, voxel_size)
    if unique_indices.size == 0:
        raise RuntimeError("[sdf_data_gen] No positions available for rot-far sampling")

    n_pos = int(min(n_pos, unique_indices.size))
    sel = rng.choice(unique_indices, size=n_pos, replace=False)

    max_total = int(n_pos) * int(n_rot)
    poses_out = np.zeros((max_total, 7), dtype=np.float32)
    z_out = np.zeros((max_total, 6), dtype=np.float32)
    label_out = np.zeros((max_total,), dtype=np.float32)

    filled = 0
    total = 0

    for i, idx in enumerate(sel.tolist(), start=1):
        p = poses[int(idx), :3]
        for _ in range(int(n_rot)):
            quat = _sample_uniform_quaternion(rng)
            pose = np.concatenate([p, quat], axis=0).astype(np.float32)
            reachable, _ = oracle.is_reachable(pose)
            total += 1
            if reachable:
                continue
            poses_out[filled] = pose
            z_out[filled] = _pose_to_z(pose, l_ref)
            label_out[filled] = 0.0
            filled += 1

        if progress_interval > 0 and (i % int(progress_interval) == 0):
            logger.info(
                f"[sdf_data_gen] D_rot_far pos={i}/{n_pos} kept={filled} total={total}"
            )

    return {
        "poses": poses_out[:filled],
        "z": z_out[:filled],
        "label": label_out[:filled],
    }


def _sample_outside_aabb(
    rng: np.random.Generator,
    aabb_min: np.ndarray,
    aabb_max: np.ndarray,
    expanded_min: np.ndarray,
    expanded_max: np.ndarray,
) -> np.ndarray:
    """Sample a position outside aabb_min/aabb_max within expanded bounds."""
    for _ in range(100):
        pos = rng.uniform(expanded_min, expanded_max).astype(np.float32)
        if np.any(pos < aabb_min) or np.any(pos > aabb_max):
            return pos
    return rng.uniform(expanded_min, expanded_max).astype(np.float32)


def generate_pos_far_samples(
    l_ref: float,
    aabb_min: np.ndarray,
    aabb_max: np.ndarray,
    outside_scale: float,
    num_samples: int,
    use_oracle: bool,
    oracle: Optional[ReachabilityOracle],
    rng: np.random.Generator,
    progress_interval: int,
    logger,
) -> Dict[str, np.ndarray]:
    """Generate position-far negative samples."""
    num_samples = int(num_samples)
    center = 0.5 * (aabb_min + aabb_max)
    half = 0.5 * (aabb_max - aabb_min)
    expanded_min = (center - half * float(outside_scale)).astype(np.float32)
    expanded_max = (center + half * float(outside_scale)).astype(np.float32)

    poses_out = np.zeros((num_samples, 7), dtype=np.float32)
    z_out = np.zeros((num_samples, 6), dtype=np.float32)
    label_out = np.zeros((num_samples,), dtype=np.float32)

    filled = 0
    attempts = 0

    while filled < num_samples:
        pos = _sample_outside_aabb(rng, aabb_min, aabb_max, expanded_min, expanded_max)
        quat = _sample_uniform_quaternion(rng)
        pose = np.concatenate([pos, quat], axis=0).astype(np.float32)

        if use_oracle and oracle is not None:
            reachable, _ = oracle.is_reachable(pose)
            attempts += 1
            if reachable:
                continue

        poses_out[filled] = pose
        z_out[filled] = _pose_to_z(pose, l_ref)
        label_out[filled] = 0.0
        filled += 1

        if progress_interval > 0 and (filled % int(progress_interval) == 0):
            logger.info(
                f"[sdf_data_gen] D_pos_far filled={filled}/{num_samples} attempts={attempts}"
            )

    return {
        "poses": poses_out,
        "z": z_out,
        "label": label_out,
    }


def generate_sdf_dataset(cfg: Any, logger) -> Dict[str, Any]:
    """Generate SDF data according to the design memo.

    Args:
        cfg: Loaded configuration.
        logger: Logger instance.

    Returns:
        meta: Metadata dict summarizing the dataset.
    """
    repo_root = Path(_get_path(cfg, "paths.repo_root", ".")).resolve()
    urdf_path = _resolve_path(_get_path(cfg, "paths.urdf_path"), repo_root)
    output_root = _resolve_path(_get_path(cfg, "paths.output_root"), repo_root)

    robot_cfg = _get(cfg, "robot", None)
    base_link = _get(robot_cfg, "base_link", None)
    end_effector_link = _get(robot_cfg, "end_effector_link", None)

    run_cfg = _get(cfg, "run", None)
    seed = _get(run_cfg, "seed", None)
    deterministic = bool(_get(run_cfg, "deterministic", True))
    if seed is not None:
        set_seed(int(seed), deterministic=deterministic)

    rng = np.random.default_rng(int(seed) if seed is not None else None)

    logger.info(f"[sdf_data_gen] URDF: {urdf_path}")
    robot = RobotModel(
        urdf_path=urdf_path,
        logger=logger,
        base_link=base_link,
        end_effector_link=end_effector_link,
    )

    collision_cfg = _get(cfg, "collision", None)
    collision_mode = str(_get(collision_cfg, "mode", "hybrid")).lower()
    d_check = float(_get(collision_cfg, "d_check", 0.015))
    radius_cfg = {
        "radius_min": _get(collision_cfg, "radius_min", None),
        "radius_max": _get(collision_cfg, "radius_max", None),
        "radius_default": _get(collision_cfg, "radius_default", None),
        "radius_scale": _get(collision_cfg, "radius_scale", None),
    }

    capsule_checker = SelfCollisionChecker.from_robot(
        robot=robot,
        cache_dir=urdf_path.parent,
        logger=logger,
        radius_cfg=radius_cfg,
    )

    if bool(_get(collision_cfg, "auto_calibrate", False)):
        samples = int(_get(collision_cfg, "calibration_samples", 500))
        capsule_checker.find_static_collisions(
            sample_fn=lambda: _sample_rest_pose(rng, robot.joint_limits),
            n_samples=samples,
        )

    mesh_checker = None
    if collision_mode in ("hybrid", "mesh_only"):
        mesh_checker = MeshSelfCollisionChecker(
            urdf_path=urdf_path,
            joint_names=robot.joint_names,
            link_edges=robot.link_edges,
            logger=logger,
            use_self_collision=True,
        )

    collision_checker = HybridSelfCollisionChecker(
        capsule_checker=capsule_checker,
        mesh_checker=mesh_checker,
        d_check=d_check,
        mode=collision_mode,
    )

    lref_cfg = _get(cfg, "lref", None)
    l_ref_override = _get(lref_cfg, "override", None)
    if l_ref_override is None:
        l_ref = compute_l_ref_from_urdf(urdf_path, base_link, end_effector_link)
    else:
        l_ref = float(l_ref_override)

    if l_ref <= 0.0:
        raise ValueError("[sdf_data_gen] L_ref must be positive.")

    aabb_cfg = _get(cfg, "aabb", None)
    bounds_samples = int(_get(aabb_cfg, "bounds_samples", 50000))
    margin = float(_get(aabb_cfg, "margin", 1.1))
    xyz_min, xyz_max = estimate_workspace_bounds(robot=robot, samples=bounds_samples, logger=logger)
    aabb_min, aabb_max = apply_aabb_margin(xyz_min, xyz_max, margin=margin)

    sampling_cfg = _get(cfg, "sampling", None)
    num_positive = int(_get(sampling_cfg, "num_positive", 100000))
    max_attempts_factor = float(_get(sampling_cfg, "max_attempts_factor", 50))
    progress_interval = int(_get(sampling_cfg, "progress_interval", 20000))

    q_pool, poses, z_pool = sample_positive_pool(
        robot=robot,
        collision_checker=collision_checker,
        num_samples=num_positive,
        l_ref=l_ref,
        rng=rng,
        max_attempts_factor=max_attempts_factor,
        progress_interval=progress_interval,
        logger=logger,
    )

    anchor_ratio = float(_get(sampling_cfg, "anchor_ratio", 0.05))
    anchor_min = int(_get(sampling_cfg, "anchor_min", 20000))
    anchor_max = int(_get(sampling_cfg, "anchor_max", 200000))
    voxel_size = float(_get(sampling_cfg, "voxel_size", 0.01))
    dir_bins = int(_get(sampling_cfg, "dir_bins", 32))
    angle_bins = int(_get(sampling_cfg, "angle_bins", 6))

    bias_cfg = _get(sampling_cfg, "bias", None)
    use_sigma = bool(_get(bias_cfg, "use_sigma", True))
    sigma_k = float(_get(bias_cfg, "sigma_k", 0.1))

    anchor_indices, bias_score, g_lim, g_sigma = select_anchor_indices(
        poses=poses,
        q_pool=q_pool,
        robot=robot,
        l_ref=l_ref,
        aabb_min=aabb_min,
        aabb_max=aabb_max,
        voxel_size=voxel_size,
        dir_bins=dir_bins,
        angle_bins=angle_bins,
        anchor_ratio=anchor_ratio,
        anchor_min=anchor_min,
        anchor_max=anchor_max,
        use_sigma=use_sigma,
        sigma_k=sigma_k,
        logger=logger,
    )

    oracle_cfg = _get(cfg, "oracle", None)
    pos_tol_mm = float(_get(oracle_cfg, "pos_tol_mm", 1.0))
    rot_tol_deg = float(_get(oracle_cfg, "rot_tol_deg", 1.0))
    residual_threshold = float(_get(oracle_cfg, "residual_threshold", 1e-5))
    final_check_tier3 = bool(_get(oracle_cfg, "final_check_tier3", False))
    tiers_raw = list(_get(oracle_cfg, "tiers", []))
    if not tiers_raw:
        raise ValueError("[sdf_data_gen] oracle.tiers is required.")

    tiers: List[OracleTier] = []
    for t in tiers_raw:
        seeds = _get(t, "seeds", None)
        max_iters = _get(t, "max_iters", None)
        if seeds is None or max_iters is None:
            raise ValueError("[sdf_data_gen] oracle.tiers entries must define seeds and max_iters.")
        tiers.append(OracleTier(seeds=int(seeds), max_iters=int(max_iters)))
    oracle = ReachabilityOracle(
        robot=robot,
        collision_checker=collision_checker,
        cfg=OracleConfig(
            pos_tol_m=pos_tol_mm * 1e-3,
            rot_tol_rad=rot_tol_deg * math.pi / 180.0,
            residual_threshold=residual_threshold,
            tiers=tiers,
        ),
        rng=rng,
    )
    oracle_final = None
    if final_check_tier3 and tiers:
        oracle_final = ReachabilityOracle(
            robot=robot,
            collision_checker=collision_checker,
            cfg=OracleConfig(
                pos_tol_m=pos_tol_mm * 1e-3,
                rot_tol_rad=rot_tol_deg * math.pi / 180.0,
                residual_threshold=residual_threshold,
                tiers=[tiers[-1]],
            ),
            rng=rng,
        )

    ray_cfg = _get(sampling_cfg, "ray", None)
    k = int(_get(ray_cfg, "K", 32))
    lam = float(_get(ray_cfg, "lambda", 0.15))
    t_max = float(_get(ray_cfg, "t_max", 0.30))
    eps = float(_get(ray_cfg, "eps", 1e-3))
    delta = float(_get(ray_cfg, "delta", 5e-3))
    bracket_steps = int(_get(ray_cfg, "bracket_steps", 20))
    max_bisect_iters = int(_get(ray_cfg, "max_bisect_iters", 30))
    axis_fraction = float(_get(ray_cfg, "axis_fraction", 0.25))
    axis_mode = str(_get(ray_cfg, "axis_mode", "fixed_all"))
    include_boundary_points = bool(_get(ray_cfg, "include_boundary_points", False))
    nb_progress = int(_get(ray_cfg, "progress_interval", 200))

    logger.info("[sdf_data_gen] Generating boundary neighborhood samples...")
    d_nb = generate_boundary_samples(
        anchor_indices=anchor_indices,
        poses=poses,
        oracle=oracle,
        oracle_final=oracle_final,
        l_ref=l_ref,
        k=k,
        lam=lam,
        t_max=t_max,
        eps=eps,
        delta=delta,
        bracket_steps=bracket_steps,
        max_bisect_iters=max_bisect_iters,
        axis_fraction=axis_fraction,
        axis_mode=axis_mode,
        include_boundary_points=include_boundary_points,
        rng=rng,
        progress_interval=nb_progress,
        logger=logger,
    )

    rot_cfg = _get(sampling_cfg, "rot_far", None)
    n_pos = int(_get(rot_cfg, "N_p", 20000))
    n_rot = int(_get(rot_cfg, "N_R", 128))
    rot_progress = int(_get(rot_cfg, "progress_interval", 2000))

    logger.info("[sdf_data_gen] Generating rot-far negatives...")
    d_rot_far = generate_rot_far_samples(
        poses=poses,
        l_ref=l_ref,
        aabb_min=aabb_min,
        aabb_max=aabb_max,
        voxel_size=voxel_size,
        n_pos=n_pos,
        n_rot=n_rot,
        oracle=oracle_final if oracle_final is not None else oracle,
        rng=rng,
        progress_interval=rot_progress,
        logger=logger,
    )

    pos_cfg = _get(sampling_cfg, "pos_far", None)
    n_pos_far = int(_get(pos_cfg, "N", 2000000))
    pos_use_oracle = bool(_get(pos_cfg, "use_oracle", False))
    pos_progress = int(_get(pos_cfg, "progress_interval", 200000))
    outside_scale = float(_get(aabb_cfg, "outside_scale", 1.5))

    logger.info("[sdf_data_gen] Generating pos-far negatives...")
    d_pos_far = generate_pos_far_samples(
        l_ref=l_ref,
        aabb_min=aabb_min,
        aabb_max=aabb_max,
        outside_scale=outside_scale,
        num_samples=n_pos_far,
        use_oracle=pos_use_oracle,
        oracle=(oracle_final if oracle_final is not None else oracle) if pos_use_oracle else None,
        rng=rng,
        progress_interval=pos_progress,
        logger=logger,
    )

    output_root.mkdir(parents=True, exist_ok=True)

    save_npz(
        output_root / "positive_pool.npz",
        {
            "q": q_pool,
            "poses": poses,
            "z": z_pool,
        },
    )

    save_npz(
        output_root / "anchors.npz",
        {
            "indices": anchor_indices,
            "bias_score": bias_score,
            "g_lim": g_lim,
            "g_sigma": g_sigma,
        },
    )

    save_npz(output_root / "boundary_samples.npz", d_nb)
    save_npz(output_root / "rot_far_samples.npz", d_rot_far)
    save_npz(output_root / "pos_far_samples.npz", d_pos_far)

    meta = {
        "urdf_path": str(urdf_path),
        "base_link": base_link,
        "end_effector_link": end_effector_link,
        "l_ref": float(l_ref),
        "aabb_min": aabb_min.tolist(),
        "aabb_max": aabb_max.tolist(),
        "num_positive": int(q_pool.shape[0]),
        "num_anchors": int(anchor_indices.shape[0]),
        "num_nb": int(d_nb["poses"].shape[0]),
        "num_rot_far": int(d_rot_far["poses"].shape[0]),
        "num_pos_far": int(d_pos_far["poses"].shape[0]),
        "lambda": float(lam),
        "eps": float(eps),
        "delta": float(delta),
        "t_max": float(t_max),
        "k_dirs": int(k),
        "distance_unit": "normalized_length",
        "seed": None if seed is None else int(seed),
        "collision_mode": collision_mode,
        "collision_d_check_m": float(d_check),
    }

    save_json(output_root / "meta.json", meta)
    logger.info(f"[sdf_data_gen] Done. Output: {output_root}")
    return meta
