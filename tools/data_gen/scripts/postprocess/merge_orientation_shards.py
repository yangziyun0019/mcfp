#!/usr/bin/env python3
"""Merge orientation-dataset HDF5 shards into one complete dataset."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import yaml


SAMPLE_DATASETS = ("quat", "phi", "label", "method", "joint")
ANCHOR_REFERENCE_DATASETS = ("voxel_id", "pos", "s_v", "c_v", "g_v", "n_seed")


@dataclass
class ShardInfo:
    path: Path
    start_anchor: int
    end_anchor: int
    total_samples: int
    counts: np.ndarray


def decode_h5_string(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray):
        if value.shape == ():
            item = value[()]
        elif value.size == 1:
            item = value.reshape(-1)[0]
        else:
            return "\n".join(decode_h5_string(item) for item in value.reshape(-1))
        if isinstance(item, bytes):
            return item.decode()
        return str(item)
    return str(value)


def load_start_anchor(meta_group: h5py.Group) -> int:
    if "config_yaml" not in meta_group:
        raise RuntimeError("Missing /meta/config_yaml; cannot infer shard start_anchor")
    config_text = decode_h5_string(meta_group["config_yaml"][()])
    cfg = yaml.safe_load(config_text) or {}
    debug = cfg.get("debug", {})
    return int(debug.get("start_anchor", 0))


def copy_group(src: h5py.Group, dst: h5py.Group) -> None:
    for name in src:
        src.copy(name, dst, name=name)


def create_sample_dataset(
    group: h5py.Group, name: str, src: h5py.Dataset, total_rows: int, chunk_rows: int
) -> h5py.Dataset:
    shape = src.shape
    if len(shape) == 1:
        chunks = (min(chunk_rows, max(total_rows, 1)),)
        return group.create_dataset(name, shape=(total_rows,), dtype=src.dtype, chunks=chunks)
    if len(shape) == 2:
        chunks = (min(chunk_rows, max(total_rows, 1)), shape[1])
        return group.create_dataset(name, shape=(total_rows, shape[1]), dtype=src.dtype, chunks=chunks)
    raise RuntimeError(f"Unsupported sample rank for {name}: {shape}")


def validate_shards(paths: List[Path]) -> tuple[List[ShardInfo], Dict[str, np.ndarray], np.ndarray, np.ndarray, int]:
    shard_infos: List[ShardInfo] = []
    anchor_reference: Dict[str, np.ndarray] = {}
    selected_union: np.ndarray | None = None
    full_reachable_union: np.ndarray | None = None
    total_anchors: int | None = None

    for path in paths:
        with h5py.File(path, "r") as f:
            anchor_start = f["/csr/anchor_start"][:]
            if anchor_start.ndim != 1 or len(anchor_start) < 2:
                raise RuntimeError(f"{path}: invalid /csr/anchor_start shape {anchor_start.shape}")
            counts = np.diff(anchor_start).astype(np.uint64, copy=False)
            start_anchor = load_start_anchor(f["/meta"])
            end_anchor = len(anchor_start) - 2
            total_samples = int(anchor_start[-1])

            if start_anchor < 0 or start_anchor > end_anchor + 1:
                raise RuntimeError(
                    f"{path}: invalid start_anchor={start_anchor} for anchor_start_len={len(anchor_start)}"
                )
            if start_anchor > 0 and np.any(counts[:start_anchor] != 0):
                raise RuntimeError(f"{path}: nonzero counts before start_anchor={start_anchor}")
            if int(counts[start_anchor : end_anchor + 1].sum()) != total_samples:
                raise RuntimeError(f"{path}: local counts do not sum to total_samples")

            for name in SAMPLE_DATASETS:
                dset = f[f"/samples/{name}"]
                if dset.shape[0] != total_samples:
                    raise RuntimeError(
                        f"{path}: /samples/{name} first dimension {dset.shape[0]} != total_samples {total_samples}"
                    )

            voxel_id = f["/anchors/voxel_id"][:]
            selected = f["/anchors/selected"][:]
            if "/anchors/full_reachable" in f:
                full_reachable = f["/anchors/full_reachable"][:].astype(np.uint8, copy=False)
            else:
                full_reachable = np.zeros_like(selected, dtype=np.uint8)
            if total_anchors is None:
                total_anchors = len(voxel_id)
                for name in ANCHOR_REFERENCE_DATASETS:
                    anchor_reference[name] = f[f"/anchors/{name}"][:]
            else:
                if len(voxel_id) != total_anchors:
                    raise RuntimeError(f"{path}: anchor count mismatch")
                for name in ANCHOR_REFERENCE_DATASETS:
                    if not np.array_equal(anchor_reference[name], f[f"/anchors/{name}"][:]):
                        raise RuntimeError(f"{path}: /anchors/{name} differs from the first shard")

            if selected_union is None:
                selected_union = selected.astype(np.uint8, copy=True)
            else:
                if len(selected) != len(selected_union):
                    raise RuntimeError(f"{path}: /anchors/selected length mismatch")
                selected_union = np.maximum(selected_union, selected.astype(np.uint8, copy=False))

            if full_reachable_union is None:
                full_reachable_union = full_reachable.astype(np.uint8, copy=True)
            else:
                if len(full_reachable) != len(full_reachable_union):
                    raise RuntimeError(f"{path}: /anchors/full_reachable length mismatch")
                full_reachable_union = np.maximum(full_reachable_union, full_reachable)

            shard_infos.append(
                ShardInfo(
                    path=path,
                    start_anchor=start_anchor,
                    end_anchor=end_anchor,
                    total_samples=total_samples,
                    counts=counts,
                )
            )

    assert selected_union is not None
    assert full_reachable_union is not None
    assert total_anchors is not None

    shard_infos.sort(key=lambda item: item.start_anchor)
    expected = 0
    for info in shard_infos:
        if info.start_anchor != expected:
            raise RuntimeError(
                f"Shard coverage gap/overlap: expected next start {expected}, got {info.start_anchor} ({info.path})"
            )
        expected = info.end_anchor + 1
    if expected != total_anchors:
        raise RuntimeError(f"Shard coverage incomplete: covered up to {expected - 1}, total anchors {total_anchors}")

    return shard_infos, anchor_reference, selected_union, full_reachable_union, total_anchors


def merge_shards(output_path: Path, shard_paths: List[Path], chunk_rows: int) -> None:
    infos, anchor_reference, selected_union, full_reachable_union, total_anchors = validate_shards(shard_paths)

    full_counts = np.zeros(total_anchors, dtype=np.uint64)
    total_samples = 0
    for info in infos:
        full_counts[info.start_anchor : info.end_anchor + 1] = info.counts[info.start_anchor : info.end_anchor + 1]
        total_samples += info.total_samples

    if int(full_counts.sum()) != total_samples:
        raise RuntimeError("Full anchor counts do not sum to merged total_samples")

    anchor_start = np.zeros(total_anchors + 1, dtype=np.uint64)
    anchor_start[1:] = np.cumsum(full_counts, dtype=np.uint64)

    if output_path.exists():
        raise RuntimeError(f"Output already exists: {output_path}")

    with h5py.File(output_path, "w") as out:
        anchors_group = out.create_group("anchors")
        csr_group = out.create_group("csr")
        samples_group = out.create_group("samples")

        with h5py.File(infos[0].path, "r") as src0:
            copy_group(src0["/meta"], out.create_group("meta"))
            meta_group = out["/meta"]
            for name in SAMPLE_DATASETS:
                create_sample_dataset(samples_group, name, src0[f"/samples/{name}"], total_samples, chunk_rows)

        for name, data in anchor_reference.items():
            anchors_group.create_dataset(name, data=data)
        anchors_group.create_dataset("selected", data=selected_union.astype(np.uint8, copy=False))
        anchors_group.create_dataset("full_reachable", data=full_reachable_union.astype(np.uint8, copy=False))

        csr_group.create_dataset("anchor_start", data=anchor_start)
        sample_index = csr_group.create_dataset(
            "sample_index",
            shape=(total_samples,),
            dtype=np.uint64,
            chunks=(min(chunk_rows, max(total_samples, 1)),),
        )

        write_offset = 0
        for info in infos:
            with h5py.File(info.path, "r") as src:
                for name in SAMPLE_DATASETS:
                    src_dset = src[f"/samples/{name}"]
                    dst_dset = samples_group[name]
                    for start in range(0, info.total_samples, chunk_rows):
                        end = min(start + chunk_rows, info.total_samples)
                        dst_dset[write_offset + start : write_offset + end] = src_dset[start:end]

                for start in range(0, info.total_samples, chunk_rows):
                    end = min(start + chunk_rows, info.total_samples)
                    sample_index[write_offset + start : write_offset + end] = np.arange(
                        write_offset + start, write_offset + end, dtype=np.uint64
                    )

            write_offset += info.total_samples
            out.flush()

        if write_offset != total_samples:
            raise RuntimeError(f"Wrote {write_offset} samples, expected {total_samples}")

        merged_from = "\n".join(str(path) for path in shard_paths)
        dtype = h5py.string_dtype(encoding="utf-8")
        out["/meta"].create_dataset("merged_from", data=merged_from, dtype=dtype)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge orientation HDF5 shards into a complete dataset.")
    parser.add_argument("--output", required=True, help="Path to the merged HDF5 output.")
    parser.add_argument("--chunk-rows", type=int, default=100000, help="Row chunk size for streaming copy.")
    parser.add_argument("shards", nargs="+", help="Input shard HDF5 files in any order.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)
    shard_paths = [Path(item) for item in args.shards]
    merge_shards(output_path, shard_paths, args.chunk_rows)
    print(f"Merged {len(shard_paths)} shards into {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
