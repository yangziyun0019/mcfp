# mcfp/sim/collision.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

from mcfp.sim.robot_model import RobotModel


class SelfCollisionChecker:
    """Self-collision checker for a single robot model.

    This class computes a coarse self-collision clearance proxy for a given
    configuration by checking distances between approximated link segments.
    It incorporates link radii (capsule approximation) to provide a surface-to-surface
    distance metric.
    """

    def __init__(
        self,
        robot: RobotModel,
        pairs: List[Tuple[str, str]],
        logger,
    ) -> None:
        """
        Parameters
        ----------
        robot:
            RobotModel instance that owns kinematics and geometry.
        pairs:
            List of link-name pairs to check for self-collision. When empty,
            a global heuristic over all non-adjacent link segments is used.
        logger:
            Logger instance for status messages.
        """
        self.robot = robot
        self.pairs = pairs
        self.logger = logger

        # Store URDF kinematic edges (parent_link, child_link) as segment
        # templates used in the distance proxy.
        self._link_edges: List[Tuple[str, str]] = list(robot.link_edges)
        
        # Pre-fetch link radii for capsule approximation
        self._link_radii: Dict[str, float] = robot.get_link_radii()

        # Ignored Pairs (Blacklist) - Populated by auto-calibration
        self._ignore_pairs: set[Tuple[str, str]] = set()

        # Internal flags to avoid spamming warnings in large-scale sampling.
        self._warned_no_pairs = False
        self._warned_missing_links = False
        
        # Debug flag
        self._has_printed_debug = False

    # ----------------------------------------------------------------------
    # Construction helpers
    # ----------------------------------------------------------------------

    @classmethod
    def from_robot(
        cls,
        robot: RobotModel,
        cache_dir: Path,
        logger,
    ) -> "SelfCollisionChecker":
        """Create a SelfCollisionChecker for the given robot.

        If a cached JSON file with self-collision pairs exists in cache_dir,
        it will be loaded. Otherwise, an empty list is used and a global
        segment-based heuristic is applied.
        """
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        json_path = cache_dir / "self_collision_pairs.json"

        if json_path.is_file():
            pairs = cls._load_pairs(json_path, logger)
        else:
            logger.warning(
                "[SelfCollisionChecker] No cached self-collision pairs found. "
                "Using empty list and falling back to a global segment-based "
                "distance heuristic."
            )
            pairs = []

        return cls(robot=robot, pairs=pairs, logger=logger)

    @staticmethod
    def _load_pairs(json_path: Path, logger) -> List[Tuple[str, str]]:
        """Load self-collision pairs from a JSON file."""
        logger.info(f"[SelfCollisionChecker] Loading pairs from {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_pairs = data.get("pairs", [])
        pairs: List[Tuple[str, str]] = []
        for item in raw_pairs:
            if isinstance(item, list) and len(item) == 2:
                pairs.append((str(item[0]), str(item[1])))
            else:
                logger.warning(
                    f"[SelfCollisionChecker] Invalid pair entry in JSON: {item}"
                )

        logger.info(
            f"[SelfCollisionChecker] Loaded {len(pairs)} self-collision pairs."
        )
        return pairs
    
    # ----------------------------------------------------------------------
    # Auto-Calibration API
    # ----------------------------------------------------------------------

    def find_static_collisions(
        self, 
        sample_fn, 
        n_samples: int = 500,
        threshold: float = 0.95
    ) -> List[Tuple[str, str]]:
        """Identify link pairs that collide in almost all configurations.

        These are typically structural overlaps (e.g. Link1 vs Link3 at the shoulder)
        due to capsule approximation. We should ignore them.
        """
        self.logger.info(f"[SelfCollisionChecker] Auto-calibrating ACM with {n_samples} samples...")
        
        # Counter: {(link_a, link_b): collision_count}
        collision_counts: Dict[Tuple[str, str], int] = {}
        
        for _ in range(n_samples):
            q = sample_fn()
            
            # 1. Get Poses
            try:
                poses = self.robot.link_poses(q)
            except Exception:
                continue
                
            # 2. Build Segments
            segments, segment_links, _ = self._build_segments(poses)
            num_segments = len(segments)
            
            # 3. Check All Pairs (Global Heuristic logic)
            for i in range(num_segments):
                link_i = segment_links[i][1]
                r_i = self._link_radii.get(link_i, 0.02)
                p0_i, p1_i = segments[i]
                
                for j in range(i + 1, num_segments):
                    links_i = segment_links[i]
                    links_j = segment_links[j]
                    
                    # Skip adjacent
                    if (links_i[0] in links_j or links_i[1] in links_j):
                        continue
                        
                    link_j = segment_links[j][1]
                    
                    # Sort pair name for consistent dictionary keys
                    pair_key = tuple(sorted((link_i, link_j)))
                    
                    r_j = self._link_radii.get(link_j, 0.02)
                    q0_j, q1_j = segments[j]
                    
                    d_center = self._segment_segment_distance(p0_i, p1_i, q0_j, q1_j)
                    d_surf = d_center - (r_i + r_j)
                    
                    if d_surf <= 0:
                        collision_counts[pair_key] = collision_counts.get(pair_key, 0) + 1

        # 4. Filter Static Collisions
        ignored = []
        for pair, count in collision_counts.items():
            rate = count / n_samples
            if rate > threshold:
                ignored.append(pair)
                self.logger.warning(
                    f"[SelfCollisionChecker] Found STATIC collision: {pair} "
                    f"(Rate: {rate:.1%}). Adding to Ignore List."
                )
        
        # Update internal ignore list
        for p in ignored:
            self._ignore_pairs.add(p)
            
        return ignored

    # ----------------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------------

    def _build_segments(
        self,
        poses: Dict[str, Tuple[np.ndarray, np.ndarray]],
    ) -> Tuple[
        List[Tuple[np.ndarray, np.ndarray]],
        List[Tuple[str, str]],
        Dict[str, List[int]],
    ]:
        """Build line segments from URDF link edges and current link poses."""
        segments: List[Tuple[np.ndarray, np.ndarray]] = []
        segment_links: List[Tuple[str, str]] = []
        link_to_segments: Dict[str, List[int]] = {}

        for parent, child in self._link_edges:
            if parent not in poses or child not in poses:
                continue

            p_parent = poses[parent][0]
            p_child = poses[child][0]

            idx = len(segments)
            segments.append((p_parent, p_child))
            segment_links.append((parent, child))

            if parent not in link_to_segments:
                link_to_segments[parent] = []
            if child not in link_to_segments:
                link_to_segments[child] = []
            link_to_segments[parent].append(idx)
            link_to_segments[child].append(idx)

        return segments, segment_links, link_to_segments

    @staticmethod
    def _segment_segment_distance(
        p0: np.ndarray,
        p1: np.ndarray,
        q0: np.ndarray,
        q1: np.ndarray,
    ) -> float:
        """Compute Euclidean distance between two line segments in 3D."""
        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        q0 = np.asarray(q0, dtype=float)
        q1 = np.asarray(q1, dtype=float)

        u = p1 - p0
        v = q1 - q0
        w0 = p0 - q0

        a = float(np.dot(u, u))
        b = float(np.dot(u, v))
        c = float(np.dot(v, v))
        d = float(np.dot(u, w0))
        e = float(np.dot(v, w0))

        denom = a * c - b * b

        # Default parameters
        s = 0.0
        t = 0.0

        if denom > 1e-12:
            # General case: two non-parallel segments
            s = (b * e - c * d) / denom
            t = (a * e - b * d) / denom
        else:
            # Nearly parallel: clamp one parameter, solve for the other
            s = 0.0
            if c > 1e-12:
                t = e / c
            else:
                t = 0.0

        s = float(np.clip(s, 0.0, 1.0))
        t = float(np.clip(t, 0.0, 1.0))

        closest_p = p0 + s * u
        closest_q = q0 + t * v

        return float(np.linalg.norm(closest_p - closest_q))

    # ----------------------------------------------------------------------
    # Distance query API
    # ----------------------------------------------------------------------

    def min_distance(self, q) -> float:
        """Compute approximate self-collision clearance (Surface-to-Surface).

        Links are approximated by straight segments (capsules) between
        neighbouring link frames.

        d_surface = d_centerline - (radius_a + radius_b)

        Returns
        -------
        dist:
            Positive float representing approximate clearance.
            Returns 0.0 if collision occurs.
        """
        q_arr = np.asarray(q, dtype=float).reshape(-1)

        # Query poses for all links from the robot model.
        try:
            poses: Dict[str, Tuple[np.ndarray, np.ndarray]] = self.robot.link_poses(
                q_arr
            )
        except Exception:
            if not self._warned_missing_links:
                self.logger.warning(
                    "[SelfCollisionChecker.min_distance] Failed to query "
                    "link poses; returning a large placeholder distance."
                )
                self._warned_missing_links = True
            return float("nan")

        segments, segment_links, link_to_segments = self._build_segments(poses)

        # Fallback if no segments found
        if not segments:
            # (Fallback logic for point-based distance if pairs exist)
            if self.pairs:
                min_dist = np.inf
                for link_a, link_b in self.pairs:
                    if link_a not in poses or link_b not in poses:
                        continue
                    
                    pos_a, _ = poses[link_a]
                    pos_b, _ = poses[link_b]
                    
                    r_a = self._link_radii.get(link_a, 0.02)
                    r_b = self._link_radii.get(link_b, 0.02)

                    d_center = float(np.linalg.norm(pos_a - pos_b))
                    d_surf = d_center - (r_a + r_b)
                    
                    if d_surf < min_dist:
                        min_dist = d_surf

                if not np.isfinite(min_dist):
                    return float("nan")
                return float(max(0.0, min_dist))

            if not self._warned_no_pairs:
                self.logger.warning(
                    "[SelfCollisionChecker.min_distance] No segments constructed "
                    "returning safe distance."
                )
                self._warned_no_pairs = True
            return float("nan")

        # Variables for Debugging the First Collision
        collision_detected = False
        collision_pair_names = ("", "")
        debug_vals = (0.0, 0.0) # d_center, radii_sum

        # Case 1: explicit link pairs
        if self.pairs:
            min_dist = np.inf
            missing_links = False

            for link_a, link_b in self.pairs:
                if link_a not in poses or link_b not in poses:
                    missing_links = True
                    continue
                
                # Get radii
                r_a = self._link_radii.get(link_a, 0.02)
                r_b = self._link_radii.get(link_b, 0.02)

                seg_indices_a = link_to_segments.get(link_a, [])
                seg_indices_b = link_to_segments.get(link_b, [])

                # Point-to-Point fallback if no segments
                if not seg_indices_a or not seg_indices_b:
                    pos_a, _ = poses[link_a]
                    pos_b, _ = poses[link_b]
                    d_center = float(np.linalg.norm(pos_a - pos_b))
                    d_surf = d_center - (r_a + r_b)
                    
                    if d_surf < min_dist:
                        min_dist = d_surf
                        if d_surf <= 0:
                            collision_detected = True
                            collision_pair_names = (link_a, link_b)
                            debug_vals = (d_center, r_a + r_b)
                    continue

                for idx_a in seg_indices_a:
                    p0, p1 = segments[idx_a]
                    for idx_b in seg_indices_b:
                        q0, q1 = segments[idx_b]
                        
                        d_center = self._segment_segment_distance(p0, p1, q0, q1)
                        d_surf = d_center - (r_a + r_b)
                        
                        if d_surf < min_dist:
                            min_dist = d_surf
                            if d_surf <= 0:
                                collision_detected = True
                                collision_pair_names = (link_a, link_b)
                                debug_vals = (d_center, r_a + r_b)

            if missing_links and not self._warned_missing_links:
                self.logger.warning(
                    "[SelfCollisionChecker] Some pair links missing in poses."
                )
                self._warned_missing_links = True

            if not np.isfinite(min_dist):
                return float("nan")
            
            final_dist = float(max(0.0, min_dist))
            
            # DIAGNOSIS: Print first collision
            if final_dist == 0.0 and collision_detected and not self._has_printed_debug:
                d_c, r_sum = debug_vals
                l_a, l_b = collision_pair_names
                self.logger.warning(
                    f"[SelfCollisionChecker] FIRST COLLISION DETECTED (Explicit Pair): "
                    f"'{l_a}' vs '{l_b}'. "
                    f"CenterDist={d_c:.4f} < RadiiSum={r_sum:.4f}. "
                    "Suppressing further logs."
                )
                self._has_printed_debug = True

            return final_dist

        # Case 2: global heuristic (All non-adjacent)
        if not self._warned_no_pairs:
            self.logger.info(
                "[SelfCollisionChecker] No explicit pairs; using global heuristic."
            )
            self._warned_no_pairs = True

        num_segments = len(segments)
        min_dist = np.inf

        for i in range(num_segments):
            # For heuristic, we associate radius with the child link of the segment
            link_i_name = segment_links[i][1] 
            r_i = self._link_radii.get(link_i_name, 0.02)
            p0_i, p1_i = segments[i]
            
            for j in range(i + 1, num_segments):
                links_i = segment_links[i]
                links_j = segment_links[j]

                # Skip adjacent
                if (
                    links_i[0] == links_j[0]
                    or links_i[0] == links_j[1]
                    or links_i[1] == links_j[0]
                    or links_i[1] == links_j[1]
                ):
                    continue

                link_i_name = segment_links[i][1]
                link_j_name = segment_links[j][1]
                if (link_i_name, link_j_name) in self._ignore_pairs or \
                   (link_j_name, link_i_name) in self._ignore_pairs:
                    continue

                link_j_name = segment_links[j][1]
                r_j = self._link_radii.get(link_j_name, 0.02)
                q0_j, q1_j = segments[j]

                d_center = self._segment_segment_distance(p0_i, p1_i, q0_j, q1_j)
                d_surf = d_center - (r_i + r_j)
                
                if d_surf < min_dist:
                    min_dist = d_surf
                    if d_surf <= 0 and not self._has_printed_debug:
                        collision_detected = True
                        collision_pair_names = (link_i_name, link_j_name)
                        debug_vals = (d_center, r_i + r_j)

        if not np.isfinite(min_dist):
            return float("nan")
            
        final_dist = float(max(0.0, min_dist))

        # DIAGNOSIS: Print first collision
        if final_dist == 0.0 and collision_detected and not self._has_printed_debug:
            d_c, r_sum = debug_vals
            l_a, l_b = collision_pair_names
            self.logger.warning(
                f"[SelfCollisionChecker] FIRST COLLISION DETECTED (Global): "
                f"'{l_a}' vs '{l_b}'. "
                f"CenterDist={d_c:.4f} < RadiiSum={r_sum:.4f}. "
                "Suppressing further logs."
            )
            self._has_printed_debug = True

        return final_dist