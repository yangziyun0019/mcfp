#!/usr/bin/env python3
"""Script: fix_orient_method3_phi.py
Purpose: Recompute or repair method-3 phi values inside an orientation dataset file.
Usage: python3 tools/data_gen/scripts/postprocess/fix_orient_method3_phi.py
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json
import time

import numpy as np

# =========================
# CONFIG (edit here)
# =========================
CONFIG = {
    # input orientation dataset (raw or prepared)
    "orient_h5": "tools/data_gen/outputs/aubo/aubo_i5/voxel_3mm/dataset_orient.h5",
    # optional output path (if None -> in-place update)
    "output_h5": None,
    # anchor range
    "anchor_start": 2000,
    "anchor_end": 2001,
    "anchor_step": 1,
    # boundary subsample (method=2). set max<=0 to use all boundary points
    "boundary_sample_min": 2000,
    "boundary_sample_max": 0,
    "boundary_chunk": 4096,
    # method3 processing chunk size
    "method3_chunk": 2048,
    # label mode: "auto" / "01" / "+-1"
    "label_mode": "auto",
    # reachable condition when label_mode == "auto"
    "label_reach_values": [1],
    # write new phi back
    "write_phi": True,
    # optional: convert label to +/-1 (WARNING: only if label dtype is signed)
    "convert_label_to_pm1": False,
    "label_chunk": 2_000_000,
    # parallel workers
    "num_workers": 12,
    # logging
    "log_every": 50,
    "progress_every": 10,
    "seed": 42,
}


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    n = np.linalg.norm(q, axis=1, keepdims=True)
    return q / np.clip(n, 1e-9, None)


def _select_boundary(q: np.ndarray, rng: np.random.Generator, min_n: int, max_n: int) -> np.ndarray:
    n = q.shape[0]
    if n == 0:
        return q
    if max_n <= 0 or n <= max_n:
        return q
    # sample max_n
    idx = rng.choice(n, size=max_n, replace=False)
    return q[idx]


def _label_reachable(label: np.ndarray, mode: str, reach_values: list[int]) -> np.ndarray:
    if mode == "01":
        return label == 1
    if mode == "+-1":
        return label > 0
    # auto
    uniq = set(np.unique(label).tolist())
    if uniq <= {0, 1, 2}:
        return np.isin(label, reach_values)
    if uniq <= {-1, 1}:
        return label > 0
    return label > 0


def _max_abs_dot_chunked(q_query: np.ndarray, q_boundary: np.ndarray, b_chunk: int) -> np.ndarray:
    q_query = _normalize_quat(q_query)
    q_boundary = _normalize_quat(q_boundary)
    n_q = q_query.shape[0]
    max_dot = np.full((n_q,), -1.0, dtype=np.float32)
    for start in range(0, q_boundary.shape[0], int(b_chunk)):
        end = min(q_boundary.shape[0], start + int(b_chunk))
        dots = np.abs(q_query @ q_boundary[start:end].T)
        max_dot = np.maximum(max_dot, np.max(dots, axis=1))
    return max_dot


def _compute_phi_method3(
    q_query: np.ndarray,
    q_boundary: np.ndarray,
    label_query: np.ndarray,
    label_mode: str,
    reach_values: list[int],
    b_chunk: int,
) -> np.ndarray:
    max_dot = _max_abs_dot_chunked(q_query, q_boundary, b_chunk=int(b_chunk))
    max_dot = np.clip(max_dot, -1.0, 1.0)
    dist = 2.0 * np.arccos(max_dot)
    reach = _label_reachable(label_query, label_mode, reach_values)
    sign = np.where(reach, 1.0, -1.0)
    return sign * dist


def _convert_labels_inplace(label_ds, chunk: int = 2_000_000) -> None:
    dtype = label_ds.dtype
    if not np.issubdtype(dtype, np.signedinteger):
        print(f"[warn] label dtype is unsigned ({dtype}), skip convert_label_to_pm1")
        return
    n = label_ds.shape[0]
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        label = np.asarray(label_ds[start:end], dtype=np.int8)
        uniq = set(np.unique(label).tolist())
        if uniq <= {0, 1, 2}:
            label = np.where(label == 0, -1, 1).astype(np.int8)
        elif uniq <= {-1, 1}:
            label = label.astype(np.int8)
        else:
            label = np.where(label >= 0, 1, -1).astype(np.int8)
        label_ds[start:end] = label
        if start % (chunk * 10) == 0:
            print(f"[label] {start}/{n}")


def _process_anchor(task: tuple[int, str, dict]) -> tuple[int, np.ndarray | None, np.ndarray | None]:
    anchor_id, path_str, cfg = task
    import h5py
    path = Path(path_str)
    with h5py.File(path, "r") as h5:
        anchor_start = h5["/csr/anchor_start"]
        sample_index = h5["/csr/sample_index"]
        quat_ds = h5["/samples/quat"]
        method_ds = h5["/samples/method"]
        label_ds = h5["/samples/label"]

        start = int(anchor_start[anchor_id])
        end = int(anchor_start[anchor_id + 1])
        if end <= start:
            return anchor_id, None, None
        ids = np.asarray(sample_index[start:end], dtype=np.int64)
        methods = np.asarray(method_ds[ids], dtype=np.uint8)

        mask_bd = methods == 2
        if not np.any(mask_bd):
            return anchor_id, None, None
        bd_ids = ids[mask_bd]
        q_bd = np.asarray(quat_ds[bd_ids], dtype=np.float32)
        rng = np.random.default_rng(int(cfg["seed"]) + anchor_id * 9973)
        q_bd = _select_boundary(q_bd, rng, int(cfg["boundary_sample_min"]), int(cfg["boundary_sample_max"]))
        if q_bd.shape[0] == 0:
            return anchor_id, None, None

        mask_g = methods == 3
        if not np.any(mask_g):
            return anchor_id, None, None
        g_ids = ids[mask_g]

        # compute in chunks to reduce memory
        phi_all = np.empty((g_ids.shape[0],), dtype=np.float32)
        chunk = int(cfg["method3_chunk"])
        for i in range(0, g_ids.shape[0], chunk):
            chunk_ids = g_ids[i:i + chunk]
            q_chunk = np.asarray(quat_ds[chunk_ids], dtype=np.float32)
            label_chunk = np.asarray(label_ds[chunk_ids], dtype=np.int8)
            phi_new = _compute_phi_method3(
                q_chunk,
                q_bd,
                label_chunk,
                cfg["label_mode"],
                list(cfg["label_reach_values"]),
                int(cfg["boundary_chunk"]),
            )
            phi_all[i:i + chunk] = phi_new.astype(np.float32)

        return anchor_id, g_ids, phi_all


def main() -> int:
    cfg = SimpleNamespace(**CONFIG)
    path = Path(cfg.orient_h5)
    if not path.exists():
        print(f"[error] missing: {path}")
        return 1

    if cfg.output_h5:
        import shutil
        out_path = Path(cfg.output_h5)
        if not out_path.exists():
            out_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[copy] {path} -> {out_path}")
            shutil.copy2(path, out_path)
        path = out_path

    import h5py

    rng = np.random.default_rng(int(cfg.seed))
    t0 = time.time()

    with h5py.File(path, "r+") as h5:
        label_ds = h5["/samples/label"]
        if cfg.convert_label_to_pm1:
            _convert_labels_inplace(label_ds, chunk=int(cfg.label_chunk))
        anchors = h5["/anchors/pos"].shape[0]

    a0 = int(cfg.anchor_start)
    a1 = int(cfg.anchor_end) if cfg.anchor_end is not None else anchors
    step = int(cfg.anchor_step)
    anchor_ids = list(range(a0, a1, step))

    task_cfg = {
        "boundary_sample_min": cfg.boundary_sample_min,
        "boundary_sample_max": cfg.boundary_sample_max,
        "boundary_chunk": cfg.boundary_chunk,
        "method3_chunk": cfg.method3_chunk,
        "label_mode": cfg.label_mode,
        "label_reach_values": cfg.label_reach_values,
        "seed": cfg.seed,
    }

    results = []
    done = 0
    total = len(anchor_ids)
    progress_every = max(1, int(cfg.progress_every))
    num_workers = int(cfg.num_workers)
    if num_workers > 1 and len(anchor_ids) > 1:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        tasks = [(a, str(path), task_cfg) for a in anchor_ids]
        with ctx.Pool(processes=num_workers) as pool:
            for out in pool.imap_unordered(_process_anchor, tasks):
                done += 1
                if out is not None:
                    results.append(out)
                if done % progress_every == 0 or done == total:
                    print(f"[progress] {done}/{total} anchors processed")
    else:
        for a in anchor_ids:
            out = _process_anchor((a, str(path), task_cfg))
            done += 1
            if out is not None:
                results.append(out)
            if done % progress_every == 0 or done == total:
                print(f"[progress] {done}/{total} anchors processed")

    if cfg.write_phi and results:
        import h5py
        with h5py.File(path, "r+") as h5:
            phi_ds = h5["/samples/phi"]
            for a, g_ids, phi_new in results:
                if g_ids is None or phi_new is None:
                    continue
                phi_ds[g_ids] = phi_new.astype(np.float32)
                if cfg.log_every and (a - a0) % int(cfg.log_every) == 0:
                    print(f"[anchor {a}] method3={g_ids.shape[0]} elapsed={time.time()-t0:.1f}s")

    updated = sum(1 for _, g, _ in results if g is not None)
    print(f"[done] anchors={len(anchor_ids)} updated={updated} elapsed={time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
