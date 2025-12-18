# scripts/generate_capability_family.py
"""Entry script for generating capability maps for a morphology family (batch mode)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcfp.sim.grid_builder import generate_capability_for_robot
from mcfp.utils.config import load_config
from mcfp.utils.logging import setup_logger


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate capability maps for a morphology family.")
    parser.add_argument(
        "--config",
        type=str,
        # default="configs/data_gen_family_rm65b.yaml",
        default="configs/data_gen_family_openmanipulator.yaml",
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def _list_spec_files(
    json_root: Path,
    include_glob: str,
    exclude_filenames: Optional[List[str]] = None,
) -> List[Path]:
    """List JSON spec files under the family directory."""
    exclude_set = set(exclude_filenames or [])
    files = sorted([p for p in json_root.glob(include_glob) if p.is_file() and p.name not in exclude_set])
    return files


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    """Write JSONL records to disk atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def main() -> None:
    """Main entry point for family batch capability generation."""
    args = _parse_args()
    cfg = load_config(args.config)

    logger = setup_logger(
        name="mcfp.sim.data_gen_family",
        log_dir=cfg.logging.log_dir,
    )
    logger.info(f"Loaded configuration from: {args.config}")

    # Validate source mode
    source_type = getattr(cfg.robot, "source_type", "urdf").lower()
    if source_type != "json":
        logger.error(f"Batch family script requires robot.source_type='json', got: {source_type}")
        sys.exit(1)

    json_root = Path(cfg.data.json_root)
    if not json_root.exists():
        logger.error(f"JSON root does not exist: {json_root}")
        sys.exit(1)

    family_name = cfg.batch.family_name
    include_glob = getattr(cfg.batch, "include_glob", "*.json")
    exclude_filenames = list(getattr(cfg.batch, "exclude_filenames", []))

    spec_files = _list_spec_files(
        json_root=json_root,
        include_glob=include_glob,
        exclude_filenames=exclude_filenames,
    )
    if not spec_files:
        logger.warning(f"No spec files found under: {json_root} (include_glob={include_glob})")
        return

    cap_root = Path(cfg.data.capability_root)
    output_subdir = getattr(cfg.batch, "output_subdir", "morph_specs")
    out_family_root = cap_root / output_subdir / family_name
    out_family_root.mkdir(parents=True, exist_ok=True)

    base_link = getattr(cfg.sim, "base_link", None)
    end_effector_link = getattr(cfg.sim, "end_effector_link", None)

    skip_existing = bool(getattr(cfg.batch, "skip_existing", True))
    fail_fast = bool(getattr(cfg.batch, "fail_fast", False))
    write_manifest = bool(getattr(cfg.batch, "write_manifest", True))

    logger.info(f"Family name: {family_name}")
    logger.info(f"Input JSON root: {json_root}")
    logger.info(f"Output root: {out_family_root}")
    logger.info(f"Total specs: {len(spec_files)} | skip_existing={skip_existing} | fail_fast={fail_fast}")

    records: List[Dict[str, Any]] = []
    n_ok = 0
    n_skip = 0
    n_fail = 0
    t0_all = time.time()

    manifest_path = out_family_root / "manifest.jsonl"
    summary_path = out_family_root / "summary.json"

    for idx, spec_path in enumerate(spec_files, start=1):
        spec_stem = spec_path.stem

        # Output layout: .../<family>/<spec_stem>/capability_map.npz
        out_dir = out_family_root / spec_stem
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "capability_map.npz"

        if skip_existing and out_path.exists():
            n_skip += 1
            logger.info(f"[{idx}/{len(spec_files)}][SKIP] {spec_path.name} -> {out_path}")
            records.append(
                {
                    "family": family_name,
                    "spec_file": str(spec_path.as_posix()),
                    "output_file": str(out_path.as_posix()),
                    "status": "skipped",
                    "reason": "output_exists",
                    "elapsed_sec": 0.0,
                }
            )
            continue

        logger.info(f"[{idx}/{len(spec_files)}][RUN ] {spec_path.name} -> {out_path}")
        t0 = time.time()

        try:
            generate_capability_for_robot(
                input_source=spec_path,
                output_path=out_path,
                grid_cfg=cfg.grid,
                base_link=base_link,
                end_effector_link=end_effector_link,
                logger=logger,
                is_morphology_spec=True,
            )

            elapsed = time.time() - t0
            n_ok += 1
            logger.info(f"[{idx}/{len(spec_files)}][DONE] {spec_path.name} | {elapsed:.2f}s")
            records.append(
                {
                    "family": family_name,
                    "spec_file": str(spec_path.as_posix()),
                    "output_file": str(out_path.as_posix()),
                    "status": "ok",
                    "elapsed_sec": elapsed,
                }
            )

        except Exception as e:
            elapsed = time.time() - t0
            n_fail += 1
            logger.exception(f"[{idx}/{len(spec_files)}][FAIL] {spec_path.name} | {elapsed:.2f}s | {e}")
            records.append(
                {
                    "family": family_name,
                    "spec_file": str(spec_path.as_posix()),
                    "output_file": str(out_path.as_posix()),
                    "status": "failed",
                    "error": repr(e),
                    "elapsed_sec": elapsed,
                }
            )
            if fail_fast:
                break

        if write_manifest and (idx % 10 == 0 or idx == len(spec_files)):
            _write_jsonl(manifest_path, records)

    total_elapsed = time.time() - t0_all
    summary = {
        "family": family_name,
        "total_specs": len(spec_files),
        "ok": n_ok,
        "skipped": n_skip,
        "failed": n_fail,
        "elapsed_sec": total_elapsed,
        "manifest": str(manifest_path.as_posix()) if write_manifest else None,
    }

    if write_manifest:
        _write_jsonl(manifest_path, records)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(f"Summary: {summary}")


if __name__ == "__main__":
    main()
