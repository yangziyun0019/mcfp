#!/usr/bin/env python3
"""Summarize a prepared Position-SDF dataset and its bucket structure.

This script prints grid-level and index-level statistics for the processed position dataset.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import argparse

import numpy as np

# Edit this config block directly. CLI has no options.
CONFIG = {
    # 数据集路径
    "path": "data/aubo_i5_3mm/dataset_pos.h5",
    # 统计时按 x 方向分块读取，避免一次性占用内存
    # 数值越小越省内存，但更慢
    "chunk_x": 4,
    # 是否统计网格统计（标签/符号/分桶）
    "grid_stats": True,
    # 是否统计 bucket 占用情况（使用 /index，快）
    "bucket_stats": True,
    # 多进程统计（0=关闭）
    "workers": 0,
}


def _print_stats(name: str, arr: np.ndarray) -> None:
    if arr.size == 0:
        print(f"{name}: empty")
        return
    print(
        f"{name}: min={float(np.min(arr)):.6f} max={float(np.max(arr)):.6f} "
        f"mean={float(np.mean(arr)):.6f} std={float(np.std(arr)):.6f}"
    )

def _parse_args() -> SimpleNamespace:
    parser = argparse.ArgumentParser(description="Stats for dataset_pos.h5 (new bucket format).")
    parser.add_argument("--path", type=str, default=CONFIG["path"])
    parser.add_argument("--chunk-x", type=int, default=CONFIG["chunk_x"])
    parser.add_argument("--grid-stats", action="store_true", default=CONFIG["grid_stats"])
    parser.add_argument("--no-grid-stats", action="store_true", default=False)
    parser.add_argument("--bucket-stats", action="store_true", default=CONFIG["bucket_stats"])
    parser.add_argument("--no-bucket-stats", action="store_true", default=False)
    parser.add_argument("--workers", type=int, default=CONFIG["workers"])
    args = parser.parse_args()
    if args.no_grid_stats:
        args.grid_stats = False
    if args.no_bucket_stats:
        args.bucket_stats = False
    return SimpleNamespace(**vars(args))


def _decode_b_s(bucket_ids: np.ndarray, bx: int, by: int, bz: int) -> np.ndarray:
    k = (bucket_ids // 2).astype(np.int64)
    k //= int(bz)
    k //= int(by)
    b_s = (k // int(bx)) + 1
    return b_s.astype(np.int64)


def _grid_worker(path: str, x0: int, x1: int) -> tuple:
    import h5py  # local import for multiprocessing
    import numpy as np

    with h5py.File(path, "r") as h5:
        label = np.asarray(h5["/grid/label"][x0:x1, :, :], dtype=np.uint8)
        sdf = np.asarray(h5["/grid/sdf"][x0:x1, :, :], dtype=np.float32)
        b_s = np.asarray(h5["/grid/b_s"][x0:x1, :, :], dtype=np.uint8)

    label_flat = label.reshape(-1)
    sdf_flat = sdf.reshape(-1)
    b_s_flat = b_s.reshape(-1)

    label_counts = np.bincount(label_flat, minlength=3).astype(np.uint64)
    nb_mask = (label_flat == 1) | (label_flat == 2)
    sign_counts = np.zeros((2,), dtype=np.uint64)
    b_s_counts = None
    b_s_sign = None
    if np.any(nb_mask):
        sgn = (sdf_flat[nb_mask] < 0).astype(np.uint8)
        sign_counts = np.bincount(sgn, minlength=2).astype(np.uint64)
        b_s_nb = b_s_flat[nb_mask].astype(np.int64)
        max_s = int(np.max(b_s_nb)) if b_s_nb.size > 0 else 0
        b_s_counts = np.bincount(b_s_nb, minlength=max_s + 1).astype(np.uint64)
        b_s_sign = np.zeros((max_s + 1, 2), dtype=np.uint64)
        for s in range(1, max_s + 1):
            mask_s = b_s_nb == s
            if np.any(mask_s):
                c = np.bincount(sgn[mask_s], minlength=2).astype(np.uint64)
                b_s_sign[s] = c

    return label_counts, sign_counts, b_s_counts, b_s_sign


def main() -> int:
    args = _parse_args()
    path = Path(args.path)

    try:
        import h5py
    except ImportError:
        print("Missing dependency: h5py")
        print("Install with: python -m pip install h5py")
        return 1

    if not path.exists():
        print(f"dataset_pos.h5 not found: {path}")
        return 1

    chunk_x = max(1, int(args.chunk_x))
    workers = int(args.workers)
    if workers < 0:
        workers = 0

    with h5py.File(path, "r") as h5:
        origin = np.asarray(h5["/grid/origin"], dtype=np.float64).reshape(3)
        dims = np.asarray(h5["/grid/dims"], dtype=np.int64).reshape(3)
        voxel = float(np.asarray(h5["/grid/voxel_size"], dtype=np.float64).reshape(-1)[0])
        num_s = int(np.asarray(h5["/grid/bucket_s_num"], dtype=np.int64).reshape(-1)[0]) if "/grid/bucket_s_num" in h5 else 1
        if "/grid/bucket_spatial_dims" in h5:
            bucket_dims = np.asarray(h5["/grid/bucket_spatial_dims"], dtype=np.int64).reshape(3)
            bx, by, bz = [int(x) for x in bucket_dims]
        else:
            bx = by = bz = 1
        print(f"bucket_s: {num_s} bucket_xyz: ({bx}, {by}, {bz})")

        nx, ny, nz = [int(x) for x in dims]
        total_voxels = int(nx * ny * nz)
        print(f"grid dims: {dims.tolist()}, voxel_size: {voxel:.6f} m, total: {total_voxels}")
        if args.grid_stats:
            label_counts = np.zeros((3,), dtype=np.uint64)
            sign_counts = np.zeros((2,), dtype=np.uint64)
            b_s_counts = np.zeros((num_s + 1,), dtype=np.uint64)
            b_s_sign = np.zeros((num_s + 1, 2), dtype=np.uint64)

            if workers > 0:
                from concurrent.futures import ProcessPoolExecutor, as_completed

                ranges = [(x0, min(nx, x0 + chunk_x)) for x0 in range(0, nx, chunk_x)]
                with ProcessPoolExecutor(max_workers=workers) as ex:
                    futures = [ex.submit(_grid_worker, str(path), x0, x1) for x0, x1 in ranges]
                    for fut in as_completed(futures):
                        lc, sc, bs_c, bs_s = fut.result()
                        label_counts[: lc.shape[0]] += lc
                        sign_counts[: sc.shape[0]] += sc
                        if bs_c is not None:
                            if bs_c.shape[0] > b_s_counts.shape[0]:
                                b_s_counts = np.pad(b_s_counts, (0, bs_c.shape[0] - b_s_counts.shape[0]))
                                b_s_sign = np.pad(b_s_sign, ((0, bs_c.shape[0] - b_s_sign.shape[0]), (0, 0)))
                            b_s_counts[: bs_c.shape[0]] += bs_c
                        if bs_s is not None:
                            if bs_s.shape[0] > b_s_sign.shape[0]:
                                b_s_sign = np.pad(b_s_sign, ((0, bs_s.shape[0] - b_s_sign.shape[0]), (0, 0)))
                            b_s_sign[: bs_s.shape[0], :] += bs_s
            else:
                label_ds = h5["/grid/label"]
                sdf_ds = h5["/grid/sdf"]
                b_s_ds = h5["/grid/b_s"]
                for x0 in range(0, nx, chunk_x):
                    x1 = min(nx, x0 + chunk_x)
                    sl = np.s_[x0:x1, :, :]
                    label = np.asarray(label_ds[sl], dtype=np.uint8)
                    sdf = np.asarray(sdf_ds[sl], dtype=np.float32)
                    b_s = np.asarray(b_s_ds[sl], dtype=np.uint8)

                    label_flat = label.reshape(-1)
                    sdf_flat = sdf.reshape(-1)
                    b_s_flat = b_s.reshape(-1).astype(np.int64)

                    label_counts += np.bincount(label_flat, minlength=3).astype(np.uint64)
                    nb_mask = (label_flat == 1) | (label_flat == 2)
                    if np.any(nb_mask):
                        sgn = (sdf_flat[nb_mask] < 0).astype(np.uint8)
                        sign_counts += np.bincount(sgn, minlength=2).astype(np.uint64)
                        b_s_nb = b_s_flat[nb_mask]
                        b_s_counts += np.bincount(b_s_nb, minlength=num_s + 1).astype(np.uint64)
                        for s in range(1, num_s + 1):
                            mask_s = b_s_nb == s
                            if np.any(mask_s):
                                c = np.bincount(sgn[mask_s], minlength=2).astype(np.uint64)
                                b_s_sign[s] += c

            print(f"label counts: boundary={label_counts[0]} inside={label_counts[1]} outside={label_counts[2]}")
            nb_total = max(int(label_counts[1] + label_counts[2]), 1)
            print(f"sign ratio (non-boundary): pos={sign_counts[0] / nb_total:.4f} neg={sign_counts[1] / nb_total:.4f}")
            for s in range(1, num_s + 1):
                total_s = int(b_s_counts[s]) if s < b_s_counts.shape[0] else 0
                if total_s == 0:
                    continue
                pos_s = int(b_s_sign[s, 0]) if s < b_s_sign.shape[0] else 0
                neg_s = int(b_s_sign[s, 1]) if s < b_s_sign.shape[0] else 0
                print(f"b_s {s}: total={total_s} pos={pos_s} neg={neg_s}")

        if args.bucket_stats and "/index" in h5:
            bucket_spatial = int(bx * by * bz)
            nb_count = np.zeros((num_s * bucket_spatial * 2,), dtype=np.uint64)
            bd_count = np.zeros((bucket_spatial,), dtype=np.uint64)
            for split_id in (0, 1, 2):
                grp = h5[f"/index/split{split_id}"]
                nb_start = np.asarray(grp["nb_start"], dtype=np.uint64)
                bd_start = np.asarray(grp["bd_start"], dtype=np.uint64)
                nb_count += (nb_start[1:] - nb_start[:-1]).astype(np.uint64)
                bd_count += (bd_start[1:] - bd_start[:-1]).astype(np.uint64)

            nb_nonzero = int(np.sum(nb_count > 0))
            nb_avg = float(np.mean(nb_count[nb_count > 0])) if nb_nonzero > 0 else 0.0
            bd_nonzero = int(np.sum(bd_count > 0))
            bd_avg = float(np.mean(bd_count[bd_count > 0])) if bd_nonzero > 0 else 0.0
            print(f"nb bucket occupancy: nonempty={nb_nonzero}/{nb_count.size} avg_count={nb_avg:.2f}")
            print(f"bd bucket occupancy: nonempty={bd_nonzero}/{bd_count.size} avg_count={bd_avg:.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
