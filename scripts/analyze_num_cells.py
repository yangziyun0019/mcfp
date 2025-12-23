#!/usr/bin/env python
"""Analyze num_cells distribution per variant.

Usage:
  python scripts/analyze_num_cells.py --config configs/train_stage1.yaml
  python scripts/analyze_num_cells.py --manifest path/to/manifest.jsonl --splits_dir path/to/splits
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import yaml


def read_jsonl(path: Path) -> List[Dict]:
    out: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def read_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    lines: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for s in f:
            s = s.strip()
            if s:
                lines.append(s)
    return lines


def percentile(sorted_vals: List[int], p: float) -> Optional[float]:
    if not sorted_vals:
        return None
    n = len(sorted_vals)
    if p <= 0:
        return float(sorted_vals[0])
    if p >= 1:
        return float(sorted_vals[-1])
    k = int(math.ceil(p * n)) - 1
    k = max(0, min(n - 1, k))
    return float(sorted_vals[k])


def summarize(vals: List[int]) -> Dict[str, Optional[float]]:
    if not vals:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "stdev": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
        }
    n = len(vals)
    vmin = min(vals)
    vmax = max(vals)
    mean = sum(vals) / n
    if n > 1:
        var = sum((v - mean) ** 2 for v in vals) / (n - 1)
        stdev = math.sqrt(var)
    else:
        stdev = 0.0
    s = sorted(vals)
    return {
        "count": n,
        "min": float(vmin),
        "max": float(vmax),
        "mean": float(mean),
        "stdev": float(stdev),
        "p50": percentile(s, 0.50),
        "p90": percentile(s, 0.90),
        "p95": percentile(s, 0.95),
        "p99": percentile(s, 0.99),
    }


def print_summary(title: str, vals: List[int]) -> None:
    s = summarize(vals)
    print(f"\n[{title}]")
    for k in ["count", "min", "max", "mean", "stdev", "p50", "p90", "p95", "p99"]:
        print(f"  {k}: {s[k]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--manifest", type=str, default=None)
    ap.add_argument("--splits_dir", type=str, default=None)
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    if args.config:
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        manifest = Path(cfg["paths"]["manifest"]).resolve()
        splits_dir = Path(cfg["paths"]["splits_dir"]).resolve()
    else:
        if not args.manifest:
            raise SystemExit("--manifest is required when --config is not provided")
        manifest = Path(args.manifest).resolve()
        splits_dir = Path(args.splits_dir).resolve() if args.splits_dir else None

    records = read_jsonl(manifest)
    num_cells_by_id = {str(r["variant_id"]): int(r["num_cells"]) for r in records}

    all_vals = list(num_cells_by_id.values())
    print_summary("all variants", all_vals)

    if splits_dir is not None and splits_dir.exists():
        train_ids = read_lines(splits_dir / "stage1_train.txt")
        val_ids = read_lines(splits_dir / "stage1_val.txt")
        test_ids = read_lines(splits_dir / "stage1_test.txt")

        for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
            vals = [num_cells_by_id[i] for i in ids if i in num_cells_by_id]
            if vals:
                print_summary(f"{name} variants", vals)

    topk = max(1, int(args.topk))
    items = sorted(num_cells_by_id.items(), key=lambda kv: kv[1])

    print(f"\n[smallest {topk}]")
    for vid, n in items[:topk]:
        print(f"  {vid}: {n}")

    print(f"\n[largest {topk}]")
    for vid, n in items[-topk:][::-1]:
        print(f"  {vid}: {n}")


if __name__ == "__main__":
    main()
