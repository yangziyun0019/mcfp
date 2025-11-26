from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

from mcfp.sim.robot_model import RobotModel


class SelfCollisionChecker:
    """Self-collision checker for a single robot model.

    Parameters
    ----------
    robot:
        RobotModel instance that owns kinematics and geometry.
    pairs:
        List of link-name pairs to check for self-collision.
    logger:
        Logger instance for status messages.
    """

    def __init__(
        self,
        robot: RobotModel,
        pairs: List[Tuple[str, str]],
        logger,
    ) -> None:
        self.robot = robot
        self.pairs = pairs
        self.logger = logger

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

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

        Parameters
        ----------
        robot:
            RobotModel instance.
        cache_dir:
            Directory where self-collision metadata is stored.
        logger:
            Logger instance.

        Returns
        -------
        checker:
            A SelfCollisionChecker instance.
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
            # TODO: 未来在这里实现随机采样生成自碰对，并保存到 JSON
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

    # ------------------------------------------------------------------
    # Distance query API (stub)
    # ------------------------------------------------------------------

    def min_distance(self, q) -> float:
        """Compute the minimum self-collision distance for configuration q.

        Parameters
        ----------
        q:
            Joint configuration.

        Returns
        -------
        min_dist:
            Minimum distance between any pair of links in self.pairs.
            Currently returns a large constant as a placeholder.
        """
        if not self.pairs:
            self.logger.warning(
                "[SelfCollisionChecker.min_distance] No self-collision pairs. "
                "Returning a large placeholder distance."
            )
            return 1e3

        # TODO: 使用 robot.link_poses(q) 和碰撞几何计算真实的最小距离
        self.logger.warning(
            "[SelfCollisionChecker.min_distance] Collision distance computation "
            "not implemented yet. Returning placeholder value."
        )
        return 1e3
