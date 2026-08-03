#!/usr/bin/env python3
"""Compare L3N4 and L3N5 on physically and geometrically defined subsets.

Subset membership is determined only from reference data, never model errors:

* high force: top fraction by maximum target atomic force magnitude;
* high energy: top fraction by target energy;
* planar: bottom fraction by the H-cloud SVD planarity ratio s_min / s_max.

The script reports all available runs and a paired view of seeds shared by both
architectures.  Results are produced both raw and after removing one explicitly
configured source, because a known catastrophic prediction can otherwise hide
the behavior of the rest of a subset.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ARCH_PATTERNS = {"L3N4": "l3_n4_*_per_point.npz", "L3N5": "l3_n5_*_per_point.npz"}
METRICS = {
    "energy_mae": "Energy MAE",
    "energy_rmse": "Energy RMSE",
    "force_mae": "Force MAE",
    "force_rmse": "Force RMSE",
}
SUBSET_LABELS = {
    "all": "All molecules",
    "high_force": "Top 15% target force",
    "high_energy": "Top 15% target energy",
    "planar": "Most-planar 15%",
}
COLORS = {"L3N4": "#4c78a8", "L3N5": "#f58518"}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    default_input = here / "per_point" / "random_common_80k"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=default_input)
    parser.add_argument("--output-dir", type=Path, default=default_input / "l3n4_l3n5_subset_analysis")
    parser.add_argument("--tail-fraction", type=float, default=0.15)
    parser.add_argument("--remove-source", type=int, default=3937730)
    return parser.parse_args()


def seed_from_path(path: Path) -> int:
    match = re.search(r"_seed(-?\d+)_", path.name)
    if not match:
        raise ValueError(f"Cannot parse seed from {path.name}")
    return int(match.group(1))


def hydrogen_planarity_ratio(numbers: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Return s_min/s_max for the four centered H positions in each methane."""
    if numbers.shape[1] != 5 or not np.all(np.sum(numbers == 1, axis=1) == 4):
        raise ValueError("Planarity calculation expects methane (four H atoms per structure)")
    hydrogen_positions = positions[numbers == 1].reshape(len(numbers), 4, 3).astype(np.float64)
    centered = hydrogen_positions - hydrogen_positions.mean(axis=1, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    return np.divide(
        singular_values[:, -1],
        singular_values[:, 0],
        out=np.full(len(numbers), np.nan),
        where=singular_values[:, 0] > 0,
    )


def run_metrics(data: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, float]:
    energy_error = data["energy_error"][mask].astype(np.float64)
    force_error = data["force_error"][mask].astype(np.float64)
    return {
        "energy_mae": float(np.mean(np.abs(energy_error))),
        "energy_rmse": float(np.sqrt(np.mean(np.square(energy_error)))),
        "force_mae": float(np.mean(np.abs(force_error))),
        "force_rmse": float(np.sqrt(np.mean(np.square(force_error)))),
    }


def load_runs(input_dir: Path):
    paths = {arch: sorted(input_dir.glob(pattern)) for arch, pattern in ARCH_PATTERNS.items()}
    for arch, arch_paths in paths.items():
        if not arch_paths:
            raise FileNotFoundError(f"No {arch} files matching {ARCH_PATTERNS[arch]} in {input_dir}")

    reference_path = paths["L3N4"][0]
    with np.load(reference_path) as z:
        reference = {key: z[key].copy() for key in ("source_indices", "numbers", "positions", "true_energy", "true_forces")}

    runs: dict[str, list[tuple[int, Path, dict[str, np.ndarray]]]] = {arch: [] for arch in paths}
    for arch, arch_paths in paths.items():
        seen_seeds: set[int] = set()
        for path in arch_paths:
            seed = seed_from_path(path)
            if seed in seen_seeds:
                raise ValueError(f"Duplicate {arch} seed {seed}: {path}")
            seen_seeds.add(seed)
            with np.load(path) as z:
                if not np.array_equal(z["source_indices"], reference["source_indices"]):
                    raise ValueError(f"Source order differs in {path}")
                if not np.array_equal(z["numbers"], reference["numbers"]):
                    raise ValueError(f"Atomic numbers differ in {path}")
                if not np.allclose(z["true_energy"], reference["true_energy"], rtol=0, atol=1e-6):
                    raise ValueError(f"Energy targets differ in {path}")
                if not np.allclose(z["true_forces"], reference["true_forces"], rtol=0, atol=1e-5):
                    raise ValueError(f"Force targets differ in {path}")
                data = {key: z[key].copy() for key in ("energy_error", "force_error")}
            runs[arch].append((seed, path, data))
    return reference, runs


def make_subsets(reference: dict[str, np.ndarray], fraction: float):
    if not 0 < fraction < 1:
        raise ValueError("--tail-fraction must be between zero and one")
    true_forces = reference["true_forces"].astype(np.float64)
    max_atomic_force = np.linalg.norm(true_forces, axis=2).max(axis=1)
    target_energy = reference["true_energy"].astype(np.float64)
    planarity = hydrogen_planarity_ratio(reference["numbers"], reference["positions"])
    force_cutoff = float(np.quantile(max_atomic_force, 1 - fraction))
    energy_cutoff = float(np.quantile(target_energy, 1 - fraction))
    planarity_cutoff = float(np.quantile(planarity, fraction))
    masks = {
        "all": np.ones(len(target_energy), dtype=bool),
        "high_force": max_atomic_force >= force_cutoff,
        "high_energy": target_energy >= energy_cutoff,
        "planar": planarity <= planarity_cutoff,
    }
    descriptors = pd.DataFrame(
        {
            "source_index": reference["source_indices"],
            "target_energy": target_energy,
            "max_target_atomic_force": max_atomic_force,
            "h_planarity_ratio": planarity,
            **{f"in_{name}": mask for name, mask in masks.items() if name != "all"},
        }
    )
    cutoffs = pd.DataFrame(
        [
            {"subset": "high_force", "selection": ">=", "descriptor": "max_target_atomic_force", "cutoff": force_cutoff},
            {"subset": "high_energy", "selection": ">=", "descriptor": "target_energy", "cutoff": energy_cutoff},
            {"subset": "planar", "selection": "<=", "descriptor": "h_planarity_ratio", "cutoff": planarity_cutoff},
        ]
    )
    return masks, descriptors, cutoffs


def calculate_run_table(runs, masks, source_indices, remove_source: int) -> pd.DataFrame:
    rows = []
    removal_mask = source_indices != remove_source
    for condition, condition_mask in (("raw", np.ones(len(source_indices), bool)), ("single_point_removed", removal_mask)):
        for subset, subset_mask in masks.items():
            mask = condition_mask & subset_mask
            for arch, arch_runs in runs.items():
                for seed, path, data in arch_runs:
                    rows.append(
                        {
                            "condition": condition,
                            "subset": subset,
                            "architecture": arch,
                            "seed": seed,
                            "run_file": path.name,
                            "n_molecules": int(mask.sum()),
                            **run_metrics(data, mask),
                        }
                    )
    return pd.DataFrame(rows)


def summarize_runs(run_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in run_table.groupby(["condition", "subset", "architecture"], sort=False):
        row = dict(zip(("condition", "subset", "architecture"), keys))
        row["n_runs"] = len(group)
        row["n_molecules"] = int(group["n_molecules"].iloc[0])
        for metric in METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def architecture_tests(run_table: pd.DataFrame) -> pd.DataFrame:
    """Simple directional tests: alternative is L3N5 has lower error."""
    rows = []
    for (condition, subset), group in run_table.groupby(["condition", "subset"], sort=False):
        g4 = group[group.architecture == "L3N4"].set_index("seed")
        g5 = group[group.architecture == "L3N5"].set_index("seed")
        shared = sorted(set(g4.index) & set(g5.index))
        for metric in METRICS:
            welch = stats.ttest_ind(g5[metric], g4[metric], equal_var=False, alternative="less")
            paired = stats.ttest_rel(g5.loc[shared, metric], g4.loc[shared, metric], alternative="less")
            rows.append(
                {
                    "condition": condition,
                    "subset": subset,
                    "metric": metric,
                    "l3n5_minus_l3n4_mean": float(g5[metric].mean() - g4[metric].mean()),
                    "welch_one_sided_p": float(welch.pvalue),
                    "n_l3n4": len(g4),
                    "n_l3n5": len(g5),
                    "shared_seeds": ";".join(map(str, shared)),
                    "paired_shared_seed_mean_difference": float((g5.loc[shared, metric] - g4.loc[shared, metric]).mean()),
                    "paired_one_sided_p": float(paired.pvalue),
                }
            )
    return pd.DataFrame(rows)


def structure_win_table(runs, masks, source_indices, remove_source: int) -> pd.DataFrame:
    """Compare each molecule after taking the median prediction error over seeds."""
    per_arch = {}
    for arch, arch_runs in runs.items():
        energy = np.stack([np.abs(data["energy_error"].astype(np.float64)) for _, _, data in arch_runs])
        force = np.stack(
            [np.sqrt(np.mean(np.square(data["force_error"].astype(np.float64)), axis=(1, 2))) for _, _, data in arch_runs]
        )
        per_arch[arch] = {"energy_abs_error": np.median(energy, axis=0), "force_rmse": np.median(force, axis=0)}

    rows = []
    removal_mask = source_indices != remove_source
    for condition, condition_mask in (("raw", np.ones(len(source_indices), bool)), ("single_point_removed", removal_mask)):
        for subset, subset_mask in masks.items():
            mask = condition_mask & subset_mask
            for metric in ("energy_abs_error", "force_rmse"):
                v4 = per_arch["L3N4"][metric][mask]
                v5 = per_arch["L3N5"][metric][mask]
                difference = v5 - v4
                rows.append(
                    {
                        "condition": condition,
                        "subset": subset,
                        "per_molecule_metric": metric,
                        "n_molecules": int(mask.sum()),
                        "l3n5_win_fraction": float(np.mean(v5 < v4)),
                        "tie_fraction": float(np.mean(v5 == v4)),
                        "median_l3n5_minus_l3n4": float(np.median(difference)),
                        "mean_l3n5_minus_l3n4": float(np.mean(difference)),
                    }
                )
    return pd.DataFrame(rows)


def molecule_comparison_table(runs, descriptors: pd.DataFrame) -> pd.DataFrame:
    """Return robust per-molecule errors and architecture differences."""
    architecture_errors = {}
    for arch, arch_runs in runs.items():
        energy = np.stack([np.abs(data["energy_error"].astype(np.float64)) for _, _, data in arch_runs])
        force = np.stack(
            [np.sqrt(np.mean(np.square(data["force_error"].astype(np.float64)), axis=(1, 2))) for _, _, data in arch_runs]
        )
        architecture_errors[arch] = {
            "energy_abs_error": np.median(energy, axis=0),
            "force_rmse": np.median(force, axis=0),
        }

    result = descriptors.copy()
    for metric in ("energy_abs_error", "force_rmse"):
        result[f"l3n4_median_{metric}"] = architecture_errors["L3N4"][metric]
        result[f"l3n5_median_{metric}"] = architecture_errors["L3N5"][metric]
        result[f"l3n5_minus_l3n4_{metric}"] = architecture_errors["L3N5"][metric] - architecture_errors["L3N4"][metric]
    return result


def strongest_difference_table(comparison: pd.DataFrame, masks, remove_source: int, n: int = 25) -> pd.DataFrame:
    rows = []
    keep_source = comparison.source_index.to_numpy() != remove_source
    descriptor_columns = [
        "source_index",
        "target_energy",
        "max_target_atomic_force",
        "h_planarity_ratio",
        "in_high_force",
        "in_high_energy",
        "in_planar",
    ]
    for subset, subset_mask in masks.items():
        eligible = np.flatnonzero(subset_mask & keep_source)
        for metric in ("energy_abs_error", "force_rmse"):
            difference_column = f"l3n5_minus_l3n4_{metric}"
            values = comparison[difference_column].to_numpy(float)
            for direction, ordered in (
                ("largest_l3n5_improvement", eligible[np.argsort(values[eligible])[:n]]),
                ("largest_l3n5_regression", eligible[np.argsort(values[eligible])[-n:][::-1]]),
            ):
                for rank, row_index in enumerate(ordered, start=1):
                    source_row = comparison.iloc[row_index]
                    rows.append(
                        {
                            "subset": subset,
                            "metric": metric,
                            "direction": direction,
                            "rank": rank,
                            **{column: source_row[column] for column in descriptor_columns},
                            "l3n4_median_error": source_row[f"l3n4_median_{metric}"],
                            "l3n5_median_error": source_row[f"l3n5_median_{metric}"],
                            "l3n5_minus_l3n4": source_row[difference_column],
                        }
                    )
    return pd.DataFrame(rows)


def tail_sensitivity_table(comparison: pd.DataFrame, remove_source: int) -> pd.DataFrame:
    """Check whether a narrower or broader tail changes the qualitative result."""
    table = comparison[comparison.source_index != remove_source]
    specs = (
        ("high_force", "max_target_atomic_force", "high"),
        ("high_energy", "target_energy", "high"),
        ("planar", "h_planarity_ratio", "low"),
    )
    rows = []
    for subset, descriptor, tail in specs:
        for fraction in (0.01, 0.025, 0.05, 0.10, 0.15, 0.25):
            quantile = fraction if tail == "low" else 1 - fraction
            cutoff = float(table[descriptor].quantile(quantile))
            mask = table[descriptor] <= cutoff if tail == "low" else table[descriptor] >= cutoff
            for metric in ("energy_abs_error", "force_rmse"):
                difference = table.loc[mask, f"l3n5_minus_l3n4_{metric}"].to_numpy(float)
                rows.append(
                    {
                        "subset": subset,
                        "tail_fraction": fraction,
                        "descriptor": descriptor,
                        "selection": "<=" if tail == "low" else ">=",
                        "cutoff": cutoff,
                        "metric": metric,
                        "n_molecules": len(difference),
                        "l3n5_win_fraction": float(np.mean(difference < 0)),
                        "median_l3n5_minus_l3n4": float(np.median(difference)),
                        "mean_l3n5_minus_l3n4": float(np.mean(difference)),
                    }
                )
    return pd.DataFrame(rows)


def robust_winner_table(runs, comparison: pd.DataFrame, remove_source: int) -> pd.DataFrame:
    """Rank molecules where L3N5 beats L3N4 for every matched seed."""
    shared_seeds = sorted(
        set(seed for seed, _, _ in runs["L3N4"]) & set(seed for seed, _, _ in runs["L3N5"])
    )
    run_lookup = {
        arch: {seed: data for seed, _, data in arch_runs}
        for arch, arch_runs in runs.items()
    }
    rows = []
    for metric in ("energy_abs_error", "force_rmse"):
        differences = []
        for seed in shared_seeds:
            if metric == "energy_abs_error":
                l3n4_error = np.abs(run_lookup["L3N4"][seed]["energy_error"].astype(np.float64))
                l3n5_error = np.abs(run_lookup["L3N5"][seed]["energy_error"].astype(np.float64))
            else:
                l3n4_error = np.sqrt(
                    np.mean(np.square(run_lookup["L3N4"][seed]["force_error"].astype(np.float64)), axis=(1, 2))
                )
                l3n5_error = np.sqrt(
                    np.mean(np.square(run_lookup["L3N5"][seed]["force_error"].astype(np.float64)), axis=(1, 2))
                )
            differences.append(l3n5_error - l3n4_error)
        matched_differences = np.stack(differences)
        matched_win_count = np.sum(matched_differences < 0, axis=0)
        overall_difference = comparison[f"l3n5_minus_l3n4_{metric}"].to_numpy(float)
        eligible = (
            (comparison.source_index.to_numpy() != remove_source)
            & (matched_win_count == len(shared_seeds))
            & (overall_difference < 0)
        )
        ordered = np.flatnonzero(eligible)[np.argsort(overall_difference[eligible])]
        for rank, row_index in enumerate(ordered, start=1):
            source_row = comparison.iloc[row_index]
            l3n4_error = float(source_row[f"l3n4_median_{metric}"])
            l3n5_error = float(source_row[f"l3n5_median_{metric}"])
            rows.append(
                {
                    "metric": metric,
                    "rank": rank,
                    "source_index": int(source_row.source_index),
                    "l3n4_median_error": l3n4_error,
                    "l3n5_median_error": l3n5_error,
                    "absolute_improvement": l3n4_error - l3n5_error,
                    "percent_improvement": 100 * (l3n4_error - l3n5_error) / l3n4_error,
                    "matched_seed_wins": f"{len(shared_seeds)}/{len(shared_seeds)}",
                    "matched_seed_median_difference": float(np.median(matched_differences[:, row_index])),
                    "target_energy": source_row.target_energy,
                    "max_target_atomic_force": source_row.max_target_atomic_force,
                    "h_planarity_ratio": source_row.h_planarity_ratio,
                    "in_high_force": source_row.in_high_force,
                    "in_high_energy": source_row.in_high_energy,
                    "in_planar": source_row.in_planar,
                }
            )
    return pd.DataFrame(rows)


def plot_metrics(run_table: pd.DataFrame, output_path: Path, condition: str, remove_source: int) -> None:
    table = run_table[run_table.condition == condition]
    subset_order = list(SUBSET_LABELS)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    rng = np.random.default_rng(9)
    for ax, (metric, title) in zip(axes.flat, METRICS.items()):
        for arch_index, arch in enumerate(("L3N4", "L3N5")):
            for subset_index, subset in enumerate(subset_order):
                values = table[(table.architecture == arch) & (table.subset == subset)][metric].to_numpy(float)
                x = subset_index + (-0.13 if arch == "L3N4" else 0.13)
                jitter = rng.uniform(-0.035, 0.035, len(values))
                ax.scatter(x + jitter, values, s=25, alpha=0.5, color=COLORS[arch])
                ax.errorbar(
                    x,
                    values.mean(),
                    yerr=values.std(ddof=1),
                    marker="o",
                    markersize=8,
                    capsize=4,
                    linewidth=1.8,
                    color=COLORS[arch],
                    label=arch if subset_index == 0 else None,
                )
        ax.set_title(title)
        ax.set_xticks(range(len(subset_order)), [SUBSET_LABELS[s] for s in subset_order], rotation=15, ha="right")
        ax.set_ylabel(title)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
    condition_title = "raw" if condition == "raw" else f"source {remove_source} removed"
    fig.suptitle(f"L3N4 vs L3N5 by target/geometry subset ({condition_title})\nDots are runs; markers are architecture mean ± sample SD")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=220)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_win_fractions(win_table: pd.DataFrame, output_path: Path, remove_source: int) -> None:
    table = win_table[win_table.condition == "single_point_removed"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(SUBSET_LABELS))
    width = 0.34
    for offset, (metric, label, color) in zip(
        (-width / 2, width / 2),
        (("energy_abs_error", "Energy absolute error", "#54a24b"), ("force_rmse", "Force RMSE", "#e45756")),
    ):
        values = [
            table[(table.subset == subset) & (table.per_molecule_metric == metric)].l3n5_win_fraction.iloc[0]
            for subset in SUBSET_LABELS
        ]
        ax.bar(x + offset, np.asarray(values) * 100, width, label=label, color=color)
    ax.axhline(50, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x, [SUBSET_LABELS[s] for s in SUBSET_LABELS], rotation=15, ha="right")
    ax.set_ylabel("Molecules where L3N5 has lower median-seed error (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"Per-molecule L3N5 win rate (source {remove_source} removed)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference, runs = load_runs(args.input_dir)
    masks, descriptors, cutoffs = make_subsets(reference, args.tail_fraction)
    run_table = calculate_run_table(runs, masks, reference["source_indices"], args.remove_source)
    summary = summarize_runs(run_table)
    tests = architecture_tests(run_table)
    wins = structure_win_table(runs, masks, reference["source_indices"], args.remove_source)
    molecule_comparison = molecule_comparison_table(runs, descriptors)
    strongest_differences = strongest_difference_table(molecule_comparison, masks, args.remove_source)
    tail_sensitivity = tail_sensitivity_table(molecule_comparison, args.remove_source)
    robust_winners = robust_winner_table(runs, molecule_comparison, args.remove_source)

    descriptors.to_csv(args.output_dir / "molecule_subset_descriptors.csv", index=False)
    cutoffs.to_csv(args.output_dir / "subset_cutoffs.csv", index=False)
    run_table.to_csv(args.output_dir / "run_subset_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "architecture_subset_metrics.csv", index=False)
    tests.to_csv(args.output_dir / "l3n5_better_statistical_tests.csv", index=False)
    wins.to_csv(args.output_dir / "per_molecule_win_rates.csv", index=False)
    molecule_comparison.to_csv(args.output_dir / "molecule_level_architecture_comparison.csv", index=False)
    strongest_differences.to_csv(args.output_dir / "strongest_molecule_differences.csv", index=False)
    tail_sensitivity.to_csv(args.output_dir / "tail_fraction_sensitivity.csv", index=False)
    robust_winners.to_csv(args.output_dir / "robust_l3n5_winners.csv", index=False)
    plot_metrics(run_table, args.output_dir / "subset_metrics_raw.png", "raw", args.remove_source)
    plot_metrics(
        run_table,
        args.output_dir / "subset_metrics_single_point_removed.png",
        "single_point_removed",
        args.remove_source,
    )
    plot_win_fractions(wins, args.output_dir / "l3n5_per_molecule_win_rates.png", args.remove_source)

    print(f"Wrote subset analysis to {args.output_dir}")
    print(cutoffs.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
