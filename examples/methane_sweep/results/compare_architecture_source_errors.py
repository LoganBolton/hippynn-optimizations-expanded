#!/usr/bin/env python3
"""Write seed-aggregated per-source errors for two architectures."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRICS = ("energy_error", "abs_energy_error", "force_mae", "force_rmse")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--architecture-a", default="L3N4")
    parser.add_argument("--architecture-b", default="L3N5")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def files_for_architecture(input_dir: Path, architecture: str) -> list[Path]:
    # The actual filenames use l3_n4; derive the exact glob from L3N4.
    prefix = architecture.lower().replace("n", "_n")
    return sorted(input_dir.glob(f"{prefix}_*_per_point.csv"))


def main() -> int:
    args = parse_args()
    merged: pd.DataFrame | None = None
    for architecture in (args.architecture_a, args.architecture_b):
        paths = files_for_architecture(args.input_dir, architecture)
        if not paths:
            raise SystemExit(f"No per-point files found for {architecture}")
        frames = [
            pd.read_csv(path, usecols=["source_index", *METRICS]) for path in paths
        ]
        frame = pd.concat(frames, ignore_index=True)
        grouped = frame.groupby("source_index")[list(METRICS)].agg(
            ["mean", "std", "max", "count"]
        )
        grouped.columns = [
            f"{architecture.lower()}_{metric}_{stat}"
            for metric, stat in grouped.columns
        ]
        grouped[f"{architecture.lower()}_n_runs"] = len(paths)
        merged = grouped if merged is None else merged.join(grouped, how="inner")

    assert merged is not None
    merged.reset_index().to_csv(args.output, index=False)
    print(f"Wrote {len(merged):,} shared source rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
