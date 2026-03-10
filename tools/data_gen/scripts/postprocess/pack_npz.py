#!/usr/bin/env python3
"""Script: pack_npz.py
Purpose: Pack legacy .npy outputs from one dataset directory into a single .npz archive.
Usage: python3 tools/data_gen/scripts/postprocess/pack_npz.py --input-dir <dir> --output <file>
"""

import argparse
import pathlib
import sys

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="Pack .npy files into a single .npz")
    parser.add_argument("--input-dir", required=True, help="Directory containing .npy files")
    parser.add_argument("--output", required=True, help="Output .npz path")
    parser.add_argument(
        "--compressed",
        action="store_true",
        help="Use np.savez_compressed (slower, smaller)",
    )
    args = parser.parse_args()

    input_dir = pathlib.Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"Input dir not found: {input_dir}", file=sys.stderr)
        return 1

    npy_files = sorted(p for p in input_dir.iterdir() if p.suffix == ".npy")
    if not npy_files:
        print(f"No .npy files found in {input_dir}", file=sys.stderr)
        return 1

    arrays = {}
    for path in npy_files:
        key = path.stem
        arrays[key] = np.load(path)

    if args.compressed:
        np.savez_compressed(args.output, **arrays)
    else:
        np.savez(args.output, **arrays)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
