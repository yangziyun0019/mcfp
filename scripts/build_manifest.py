from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from mcfp.data.io import build_manifest_records, write_jsonl
from mcfp.utils.logging import setup_logger


def main() -> None:
    cfg = OmegaConf.load("configs/build_manifest.yaml")
    logger = setup_logger(name="mcfp.scripts.build_manifest", log_dir=cfg.logging.log_dir)

    repo_root = Path(cfg.data.repo_root).resolve()
    morph_specs_root = Path(cfg.data.morph_specs_root).resolve()
    capability_root = Path(cfg.data.capability_root).resolve()
    manifest_path = Path(cfg.output.manifest_path).resolve()

    logger.info(f"[build_manifest] repo_root={repo_root}")
    logger.info(f"[build_manifest] morph_specs_root={morph_specs_root}")
    logger.info(f"[build_manifest] capability_root={capability_root}")
    logger.info(f"[build_manifest] manifest_path={manifest_path}")

    records = build_manifest_records(
        repo_root=repo_root,
        morph_specs_root=morph_specs_root,
        capability_root=capability_root,
        required_cap_keys=list(cfg.data.required_cap_keys),
        grid_round_decimals=int(cfg.data.grid_round_decimals),
        logger=logger,
    )

    write_jsonl(manifest_path, records)
    logger.info(f"[build_manifest] Wrote {len(records)} records to {manifest_path}")


if __name__ == "__main__":
    main()
