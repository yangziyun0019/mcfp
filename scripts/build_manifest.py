from __future__ import annotations

from pathlib import Path

from mcfp.data.io import build_pose_manifest_records, write_jsonl
from mcfp.utils.config import load_config
from mcfp.utils.logging import setup_logger


def main() -> None:
    cfg = load_config("configs/build_manifest.yaml")
    logger = setup_logger(name="mcfp.scripts.build_manifest", log_dir=cfg.logging.log_dir)

    repo_root = Path(cfg.data.repo_root).resolve()
    morph_specs_root = Path(cfg.data.morph_specs_root).resolve()
    pose_samples_root = Path(cfg.data.pose_samples_root).resolve()
    pose_filename = str(cfg.data.pose_filename)
    manifest_path = Path(cfg.output.manifest_path).resolve()

    logger.info(f"[build_manifest] repo_root={repo_root}")
    logger.info(f"[build_manifest] morph_specs_root={morph_specs_root}")
    logger.info(f"[build_manifest] pose_samples_root={pose_samples_root}")
    logger.info(f"[build_manifest] manifest_path={manifest_path}")

    records = build_pose_manifest_records(
        repo_root=repo_root,
        morph_specs_root=morph_specs_root,
        pose_samples_root=pose_samples_root,
        pose_filename=pose_filename,
        logger=logger,
    )

    write_jsonl(manifest_path, records)
    logger.info(f"[build_manifest] Wrote {len(records)} records to {manifest_path}")


if __name__ == "__main__":
    main()
