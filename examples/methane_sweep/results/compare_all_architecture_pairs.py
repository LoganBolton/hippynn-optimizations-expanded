#!/usr/bin/env python3
"""
Run seed-level pairwise statistical comparisons between every model architecture.

Expected input:
    A run-level metrics CSV with one or more condition columns.

The input should contain these columns:
    architecture, seed, metric, <condition columns>

For every architecture pair, metric, and requested condition, this script reports:
    - architecture means and sample standard deviations
    - mean difference: architecture_b - architecture_a
    - exact two-sided permutation-test p-value
    - bootstrap 95% confidence interval for the mean difference
    - Hedges' g effect size
    - Holm-adjusted p-values

The training run/seed is treated as the independent statistical replicate.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd


RESULTS_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare every architecture pair using seed-level statistics."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=RESULTS_DIR / Path(
            "per_point/random_common_80k/"
            "clean_catastrophe_analysis/run_metrics_raw_and_filtered.csv"
        ),
        help="Input run-level metrics CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / Path(
            "per_point/random_common_80k/"
            "clean_catastrophe_analysis/all_pairwise_statistical_tests.csv"
        ),
        help="Output CSV.",
    )
    parser.add_argument(
        "--condition-columns",
        nargs="+",
        default=["raw", "filtered"],
        help="Metric columns to compare (default: raw filtered).",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=100_000,
        help="Number of seed-level bootstrap resamples.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=20260731,
        help="Random seed used for bootstrap confidence intervals.",
    )
    return parser.parse_args()


def exact_permutation_test(a: np.ndarray, b: np.ndarray) -> float:
    """
    Exact two-sided permutation test for mean(b) - mean(a).

    This enumerates every possible allocation of the pooled observations into
    groups of the original sizes. With the current seed counts, this is small.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    pooled = np.concatenate([a, b])
    n_a = len(a)
    n_total = len(pooled)

    observed = float(np.mean(b) - np.mean(a))
    extreme = 0
    total = 0

    for a_indices_tuple in combinations(range(n_total), n_a):
        in_a = np.zeros(n_total, dtype=bool)
        in_a[list(a_indices_tuple)] = True

        perm_a = pooled[in_a]
        perm_b = pooled[~in_a]
        perm_difference = float(np.mean(perm_b) - np.mean(perm_a))

        if abs(perm_difference) >= abs(observed) - 1e-15:
            extreme += 1
        total += 1

    return extreme / total


def bootstrap_mean_difference_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Percentile bootstrap CI for mean(b) - mean(a), resampling seeds."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    a_samples = rng.choice(a, size=(n_bootstrap, len(a)), replace=True)
    b_samples = rng.choice(b, size=(n_bootstrap, len(b)), replace=True)

    differences = b_samples.mean(axis=1) - a_samples.mean(axis=1)
    low, high = np.percentile(differences, [2.5, 97.5])
    return float(low), float(high)


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    """
    Bias-corrected standardized mean difference for b - a.

    Negative values mean architecture_b has lower error than architecture_a.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    n_a = len(a)
    n_b = len(b)

    if n_a < 2 or n_b < 2:
        return float("nan")

    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)

    pooled_variance = (
        (n_a - 1) * var_a + (n_b - 1) * var_b
    ) / (n_a + n_b - 2)

    if pooled_variance <= 0:
        return float("nan")

    cohen_d = (np.mean(b) - np.mean(a)) / np.sqrt(pooled_variance)
    correction = 1.0 - 3.0 / (4.0 * (n_a + n_b) - 9.0)
    return float(correction * cohen_d)


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm family-wise error correction."""
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)

    running_max = 0.0
    m = len(p_values)

    for rank, original_index in enumerate(order):
        candidate = (m - rank) * p_values[original_index]
        running_max = max(running_max, candidate)
        adjusted[original_index] = min(running_max, 1.0)

    return adjusted


def main() -> int:
    args = parse_args()

    if args.bootstrap_samples <= 0:
        raise SystemExit("--bootstrap-samples must be positive")

    frame = pd.read_csv(args.input)

    if len(set(args.condition_columns)) != len(args.condition_columns):
        raise SystemExit("--condition-columns must not contain duplicates")

    required = {"architecture", "seed", "metric", *args.condition_columns}
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(
            f"Missing required columns in {args.input}: {sorted(missing)}"
        )

    architectures = sorted(frame["architecture"].dropna().unique())
    if len(architectures) < 2:
        raise SystemExit("Need at least two architectures to compare.")

    rng = np.random.default_rng(args.random_seed)
    rows: list[dict[str, object]] = []

    for condition in args.condition_columns:
        for metric in sorted(frame["metric"].dropna().unique()):
            metric_frame = frame[frame["metric"] == metric]

            for architecture_a, architecture_b in combinations(architectures, 2):
                values_a = metric_frame.loc[
                    metric_frame["architecture"] == architecture_a, condition
                ].dropna().to_numpy(float)

                values_b = metric_frame.loc[
                    metric_frame["architecture"] == architecture_b, condition
                ].dropna().to_numpy(float)

                if len(values_a) < 2 or len(values_b) < 2:
                    continue

                difference = float(np.mean(values_b) - np.mean(values_a))
                ci_low, ci_high = bootstrap_mean_difference_ci(
                    values_a,
                    values_b,
                    n_bootstrap=args.bootstrap_samples,
                    rng=rng,
                )

                permutation_count = comb(
                    len(values_a) + len(values_b), len(values_a)
                )

                rows.append(
                    {
                        "condition": condition,
                        "metric": metric,
                        "architecture_a": architecture_a,
                        "architecture_b": architecture_b,
                        "n_a": len(values_a),
                        "n_b": len(values_b),
                        "mean_a": float(np.mean(values_a)),
                        "sample_std_a": float(np.std(values_a, ddof=1)),
                        "mean_b": float(np.mean(values_b)),
                        "sample_std_b": float(np.std(values_b, ddof=1)),
                        "difference_b_minus_a": difference,
                        "difference_ci95_low": ci_low,
                        "difference_ci95_high": ci_high,
                        "hedges_g_b_minus_a": hedges_g(values_a, values_b),
                        "permutation_p_value": exact_permutation_test(
                            values_a, values_b
                        ),
                        "exact_permutations": permutation_count,
                        "lower_error_model": (
                            architecture_b
                            if difference < 0
                            else architecture_a
                            if difference > 0
                            else "tie"
                        ),
                    }
                )

    results = pd.DataFrame(rows)

    if results.empty:
        raise SystemExit("No valid pairwise comparisons could be calculated.")

    # Primary correction: all pairwise architecture tests within each
    # condition and metric.
    results["holm_p_within_condition_metric"] = np.nan
    for _, indices in results.groupby(["condition", "metric"]).groups.items():
        indices = list(indices)
        results.loc[indices, "holm_p_within_condition_metric"] = holm_adjust(
            results.loc[indices, "permutation_p_value"].to_numpy(float)
        )

    # More conservative secondary correction: every pair and metric within
    # each requested condition.
    results["holm_p_all_tests_within_condition"] = np.nan
    for _, indices in results.groupby("condition").groups.items():
        indices = list(indices)
        results.loc[indices, "holm_p_all_tests_within_condition"] = holm_adjust(
            results.loc[indices, "permutation_p_value"].to_numpy(float)
        )

    results["significant_within_metric_at_0.05"] = (
        results["holm_p_within_condition_metric"] < 0.05
    )
    results["significant_global_at_0.05"] = (
        results["holm_p_all_tests_within_condition"] < 0.05
    )
    results["ci_excludes_zero"] = (
        (results["difference_ci95_low"] > 0)
        | (results["difference_ci95_high"] < 0)
    )

    results = results.sort_values(
        ["condition", "metric", "architecture_a", "architecture_b"]
    ).reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)

    display_columns = [
        "condition",
        "metric",
        "architecture_a",
        "architecture_b",
        "n_a",
        "n_b",
        "mean_a",
        "mean_b",
        "difference_b_minus_a",
        "difference_ci95_low",
        "difference_ci95_high",
        "permutation_p_value",
        "holm_p_within_condition_metric",
        "lower_error_model",
        "significant_within_metric_at_0.05",
    ]

    print(
        results[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.6g}",
        )
    )
    print(f"\nSaved: {args.output}")
    print(
        "\nInterpretation: difference_b_minus_a < 0 means architecture_b "
        "has lower error; > 0 means architecture_a has lower error."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
