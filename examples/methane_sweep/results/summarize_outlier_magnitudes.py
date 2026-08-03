#!/usr/bin/env python3
"""Summarize the size of pooled architecture outliers.

For each architecture and sigma threshold, this writes the number of points
above ``mean + sigma * sample_std`` and the mean error of those points.  The
input is the raw per-point external-test data; no rows are removed.

Energy MAE and RMSE are aggregate metrics, so their per-point diagnostic uses
the absolute signed energy error.  Consequently their outlier rows are
identical, while the aggregate MAE and RMSE themselves remain different.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


RUN_RE = re.compile(r"l(?P<lmax>\d+)[_-]n(?P<nmax>\d+).*?seed(?P<seed>\d+)", re.I)
METRICS = {
    "energy_mae": "abs_energy_error",
    "energy_rmse": "abs_energy_error",
    "force_mae": "force_mae",
    "force_rmse": "force_rmse",
}


def arch_key(architecture: str) -> tuple[int, int, str]:
    match = re.fullmatch(r"L(\d+)N(\d+)", architecture)
    if match:
        return int(match.group(1)), int(match.group(2)), architecture
    return (10**9, 10**9, architecture)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-sigma", type=float, default=2.0)
    parser.add_argument("--max-sigma", type=float, default=10.0)
    parser.add_argument("--step", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.step <= 0 or args.max_sigma < args.min_sigma:
        raise SystemExit("invalid sigma range")

    pooled: dict[str, dict[str, list[np.ndarray]]] = {}
    n_files: dict[str, int] = {}
    for path in sorted(args.input_dir.rglob("*_per_point.csv")):
        match = RUN_RE.search(f"{path.parent.name}_{path.stem}")
        if not match:
            continue
        architecture = f"L{match.group('lmax')}N{match.group('nmax')}"
        frame = pd.read_csv(path, usecols=list(set(METRICS.values())))
        n_files[architecture] = n_files.get(architecture, 0) + 1
        by_metric = pooled.setdefault(architecture, {name: [] for name in METRICS})
        for metric, column in METRICS.items():
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
            values = values[np.isfinite(values)]
            by_metric[metric].append(values)

    sigma_values = np.arange(
        args.min_sigma, args.max_sigma + args.step * 0.1, args.step
    )
    rows: list[dict[str, object]] = []
    for architecture in sorted(pooled, key=arch_key):
        for metric in METRICS:
            values = np.concatenate(pooled[architecture][metric])
            mean = float(np.mean(values))
            sample_std = float(np.std(values, ddof=1))
            for sigma in sigma_values:
                cutoff = mean + float(sigma) * sample_std
                outliers = values[values > cutoff]
                rows.append(
                    {
                        "architecture": architecture,
                        "metric": metric,
                        "n_runs": n_files[architecture],
                        "n_total_points": int(values.size),
                        "sigma": float(sigma),
                        "all_mean": mean,
                        "all_sample_std": sample_std,
                        "cutoff": cutoff,
                        "n_outliers": int(outliers.size),
                        "outlier_fraction": float(outliers.size / values.size),
                        "mean_outlier_error": float(np.mean(outliers))
                        if outliers.size
                        else float("nan"),
                        "outlier_sample_std": float(np.std(outliers, ddof=1))
                        if outliers.size > 1
                        else float("nan"),
                        "mean_excess_over_cutoff": float(np.mean(outliers - cutoff))
                        if outliers.size
                        else float("nan"),
                        "max_outlier_error": float(np.max(outliers))
                        if outliers.size
                        else float("nan"),
                    }
                )

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Wrote {len(rows):,} rows to {output}")
    for architecture in sorted(pooled, key=arch_key):
        print(f"{architecture}: {n_files[architecture]} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
