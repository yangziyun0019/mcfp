from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List

from omegaconf import OmegaConf

from mcfp.data.io import read_jsonl
from mcfp.utils.logging import setup_logger


def main() -> None:
    cfg = OmegaConf.load("configs/build_splits.yaml")
    logger = setup_logger(name="mcfp.scripts.build_splits", log_dir=cfg.logging.log_dir)

    manifest_path = Path(cfg.data.manifest_path).resolve()
    out_dir = Path(cfg.output.split_dir).resolve()

    test_ratio = float(cfg.split.test_ratio)
    val_ratio = float(cfg.split.val_ratio)
    seed = int(cfg.split.seed)

    records = read_jsonl(manifest_path)

    # Group by DOF for stratified splitting.
    by_dof: Dict[int, List[str]] = {}
    for r in records:
        dof = int(r.get("dof", -1))
        vid = str(r["variant_id"])
        by_dof.setdefault(dof, []).append(vid)

    rng = random.Random(seed)

    train_ids: List[str] = []
    val_ids: List[str] = []
    test_ids: List[str] = []

    for dof, vids in by_dof.items():
        vids_sorted = sorted(vids)
        rng.shuffle(vids_sorted)

        n = len(vids_sorted)
        n_test = int(round(n * test_ratio))
        n_val = int(round(n * val_ratio))

        test_part = vids_sorted[:n_test]
        val_part = vids_sorted[n_test : n_test + n_val]
        train_part = vids_sorted[n_test + n_val :]

        train_ids.extend(train_part)
        val_ids.extend(val_part)
        test_ids.extend(test_part)

        logger.info(
            f"[build_splits] dof={dof} n={n} "
            f"train={len(train_part)} val={len(val_part)} test={len(test_part)}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stage1_train.txt").write_text("\n".join(sorted(train_ids)) + "\n", encoding="utf-8")
    (out_dir / "stage1_val.txt").write_text("\n".join(sorted(val_ids)) + "\n", encoding="utf-8")
    (out_dir / "stage1_test.txt").write_text("\n".join(sorted(test_ids)) + "\n", encoding="utf-8")

    logger.info(f"[build_splits] Wrote splits to {out_dir}")


if __name__ == "__main__":
    main()
