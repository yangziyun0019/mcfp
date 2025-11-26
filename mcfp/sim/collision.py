from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from mcfp.sim.robot_model import RobotModel


class SelfCollisionChecker:
    """Self-collision checker for a single robot model.

    This class computes a simple self-collision distance proxy for a given
    configuration by checking distances between selected link pairs.
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
            List of link-name pairs to check for self-collision.
        logger:
            Logger instance for status messages.
        """
        self.robot = robot
        self.pairs = pairs
        self.logger = logger

        # Internal flags to avoid spamming warnings in large-scale sampling.
        self._warned_no_pairs = False
        self._warned_missing_links = False

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
        it will be loaded. Otherwise, an empty placeholder list is used.
        """
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        json_path = cache_dir / "self_collision_pairs.json"

        if json_path.is_file():
            pairs = cls._load_pairs(json_path, logger)
        else:
            logger.warning(
                "[SelfCollisionChecker] No cached self-collision pairs found. "
                "Using empty list for now. Automatic generation will be added later."
            )
            # TODO: automatically generate self-collision pairs and save them.
            pairs = []

        return cls(robot=robot, pairs=pairs, logger=logger)

    @staticmethod
    def _load_pairs(json_path: Path, logger) -> List[Tuple[str, str]]:
        """Load self-collision pairs from a JSON file.

        The JSON file is expected to contain a top-level key "pairs" whose
        value is a list of [link_a, link_b] entries.
        """
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
    # Distance query API
    # ----------------------------------------------------------------------

    def min_distance(self, q) -> float:
        """Compute the minimum self-collision distance for configuration q.

        Current implementation approximates each link by a single point
        (its frame origin) and returns the minimum Euclidean distance
        between all configured link pairs.

        Parameters
        ----------
        q:
            Joint configuration as any array-like object of length num_joints.

        Returns
        -------
        min_dist:
            Minimum distance between any pair of links in self.pairs.
            If no valid pairs are available, a large constant is returned.
        """
        if not self.pairs:
            if not self._warned_no_pairs:
                self.logger.warning(
                    "[SelfCollisionChecker.min_distance] No self-collision "
                    "pairs configured; returning a large placeholder distance. "
                    "You can add pairs to 'self_collision_pairs.json' under "
                    "the URDF directory to enable geometric checking."
                )
                self._warned_no_pairs = True
            return 1e3

        q_arr = np.asarray(q, dtype=float)

        # Query poses for all links from the robot model.
        poses: Dict[str, Tuple[np.ndarray, np.ndarray]] = self.robot.link_poses(q_arr)

        min_dist = np.inf
        missing = False

        for link_a, link_b in self.pairs:
            if link_a not in poses or link_b not in poses:
                missing = True
                continue

            pos_a, _ = poses[link_a]
            pos_b, _ = poses[link_b]

            d = float(np.linalg.norm(pos_a - pos_b))
            if d < min_dist:
                min_dist = d

        if missing and not self._warned_missing_links:
            self.logger.warning(
                "[SelfCollisionChecker.min_distance] Some link names from "
                "self-collision pairs were not found in robot.link_poses(..). "
                "Please check 'self_collision_pairs.json' and URDF link names."
            )
            self._warned_missing_links = True

        if not np.isfinite(min_dist):
            # All pairs were invalid; fall back to a safe large distance.
            return 1e3

        return min_dist
