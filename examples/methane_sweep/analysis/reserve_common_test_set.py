#!/usr/bin/env python3
"""Reserve a reproducible, leakage-free methane test index set.

The default lower bound, 3,080,000, excludes the sequential training and test
ranges for every paper-sized training pool up to and including 3,000,000.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


TOTAL_NUM_SAMPLES = 7_732_488
DEFAULT_LOWER_BOUND = 3_080_000
DEFAULT_TEST_SIZE = 80_000
DEFAULT_SEED = 20260720


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-size", type=int, default=TOTAL_NUM_SAMPLES)
    parser.add_argument("--lower-bound", type=int, default=DEFAULT_LOWER_BOUND)
    parser.add_argument("--test-size", type=int, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("datasets/methane.extxyz"),
        help="Dataset whose zero-based frame indices are being reserved.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/methane_sweep/test_sets"),
    )
    args = parser.parse_args()

    if not 0 <= args.lower_bound < args.total_size:
        raise ValueError("lower-bound must be inside the dataset")
    population_size = args.total_size - args.lower_bound
    if not 0 < args.test_size <= population_size:
        raise ValueError("test-size must fit in the eligible index range")
    if not args.data.is_file():
        raise FileNotFoundError(args.data)

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(
        np.arange(args.lower_bound, args.total_size, dtype=np.int64),
        size=args.test_size,
        replace=False,
    )
    indices.sort()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"methane_common_test_n{args.test_size}_seed{args.seed}"
    index_path = args.output_dir / f"{stem}.npy"
    metadata_path = args.output_dir / f"{stem}.json"
    if index_path.exists() or metadata_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing reservation: {index_path} or {metadata_path}"
        )

    temporary_index = index_path.with_suffix(".npy.tmp")
    with temporary_index.open("wb") as handle:
        np.save(handle, indices, allow_pickle=False)
    temporary_index.replace(index_path)

    index_sha256 = hashlib.sha256(index_path.read_bytes()).hexdigest()
    data_stat = args.data.stat()
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": str(args.data.resolve()),
        "data_source_mtime_ns": data_stat.st_mtime_ns,
        "data_source_size_bytes": data_stat.st_size,
        "eligible_index_range": [args.lower_bound, args.total_size],
        "excluded_index_range": [0, args.lower_bound],
        "exclusion_reason": (
            "Excludes sequential training pools through 3,000,000 frames and "
            "their adjacent 80,000-frame legacy test slice."
        ),
        "index_convention": "zero-based; eligible range is half-open [start, stop)",
        "index_dtype": str(indices.dtype),
        "index_file": index_path.name,
        "index_file_sha256": index_sha256,
        "maximum_index": int(indices[-1]),
        "minimum_index": int(indices[0]),
        "numpy_version": np.__version__,
        "random_generator": type(rng.bit_generator).__name__,
        "sampling": "uniform without replacement, then sorted",
        "seed": args.seed,
        "test_size": args.test_size,
        "total_num_samples": args.total_size,
        "unique_index_count": int(np.unique(indices).size),
    }
    temporary_metadata = metadata_path.with_suffix(".json.tmp")
    with temporary_metadata.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary_metadata.replace(metadata_path)

    print(index_path)
    print(metadata_path)
    print(f"reserved={indices.size} min={indices[0]} max={indices[-1]}")
    print(f"sha256={index_sha256}")


if __name__ == "__main__":
    main()
