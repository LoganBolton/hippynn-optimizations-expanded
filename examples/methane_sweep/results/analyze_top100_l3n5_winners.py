#!/usr/bin/env python3
"""Characterize the strongest robust per-structure L3N5 wins over L3N4.

The input ranking must have been produced by analyze_l3n4_l3n5_subsets.py.
For each metric, this script takes the first N structures that L3N5 wins on
for every shared seed, derives methane geometry descriptors, and compares the
group with both the rest of the test set and baseline-error-matched controls.
"""

from __future__ import annotations

import argparse
import bisect
import itertools
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = ("energy_abs_error", "force_rmse")
FEATURE_LABELS = {
    "target_energy": "Target energy",
    "max_target_atomic_force": "Maximum target atomic force",
    "target_force_rms": "Target force RMS",
    "h_planarity_ratio": "H-cloud planarity ratio",
    "ch_mean": "Mean C-H distance",
    "ch_min": "Minimum C-H distance",
    "ch_max": "Maximum C-H distance",
    "ch_range": "C-H distance range",
    "ch_cv": "C-H radial coefficient of variation",
    "hch_min_deg": "Minimum H-C-H angle",
    "hch_max_deg": "Maximum H-C-H angle",
    "hch_mean_deg": "Mean H-C-H angle (C-centered)",
    "hch_angle_rmse_tetra_deg": "H-C-H angular distortion",
    "chh_mean_deg": "Mean C-H-H angle (H-centered)",
    "hhh_mean_deg": "Mean H-H-H angle (H-centered)",
    "hydrogen_center_angle_mean_deg": "Mean of all H-centered angles",
    "all_atom_angle_mean_deg": "Mean of all atom-centered angles",
    "atom_center_mean_angle_std_deg": "Variation among atom-centered mean angles",
    "tetrahedral_asymmetry": "Combined tetrahedral asymmetry",
    "hh_min": "Minimum H-H distance",
    "hh_max": "Maximum H-H distance",
    "h_tetra_volume": "Hydrogen tetrahedron volume",
}


def tetrahedral_asymmetry(ch_vectors: np.ndarray, ch_distances: np.ndarray) -> np.ndarray:
    """Return normalized RMS distance to the closest ideal tetrahedron."""
    normalized = ch_vectors / ch_distances.mean(axis=1)[:, None, None]
    ideal = np.asarray(
        ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)),
        dtype=np.float64,
    ) / np.sqrt(3)
    observed_norm = np.sum(np.square(normalized), axis=(1, 2))
    ideal_norm = float(np.sum(np.square(ideal)))
    best_squared = np.full(len(normalized), np.inf)
    for permutation in itertools.permutations(range(4)):
        permuted_ideal = ideal[list(permutation)]
        covariance = np.einsum("nki,kj->nij", normalized, permuted_ideal)
        singular_values = np.linalg.svd(covariance, compute_uv=False)
        squared = (observed_norm + ideal_norm - 2 * singular_values.sum(axis=1)) / 4
        best_squared = np.minimum(best_squared, squared)
    return np.sqrt(np.maximum(best_squared, 0))


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    default_input = here / "per_point" / "random_common_80k"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=default_input)
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=default_input / "l3n4_l3n5_subset_analysis",
    )
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--remove-source", type=int, default=3937730)
    return parser.parse_args()


def geometry_descriptors(input_dir: Path) -> pd.DataFrame:
    path = sorted(input_dir.glob("l3_n4_*_per_point.npz"))[0]
    with np.load(path) as data:
        source = data["source_indices"].copy()
        numbers = data["numbers"].copy()
        positions = data["positions"].astype(np.float64)
        forces = data["true_forces"].astype(np.float64)

    if not np.all(numbers[:, 0] == 6) or not np.all(numbers[:, 1:] == 1):
        raise ValueError("Expected methane atom order C, H, H, H, H")

    ch_vectors = positions[:, 1:] - positions[:, :1]
    ch_distances = np.linalg.norm(ch_vectors, axis=2)
    ch_units = ch_vectors / ch_distances[:, :, None]
    angle_cosines = np.einsum("nia,nja->nij", ch_units, ch_units)
    upper_i, upper_j = np.triu_indices(4, 1)
    angles = np.degrees(np.arccos(np.clip(angle_cosines[:, upper_i, upper_j], -1, 1)))

    # At each hydrogen center there are six angles involving the other four
    # atoms: three C-H-H angles and three H-H-H angles.  Methane has only one
    # carbon, so no C-C-containing angle exists in this dataset.
    chh_by_center = []
    hhh_by_center = []
    for center_index in range(1, 5):
        center = positions[:, center_index]
        carbon_vector = positions[:, 0] - center
        carbon_unit = carbon_vector / np.linalg.norm(carbon_vector, axis=1, keepdims=True)
        other_hydrogens = [index for index in range(1, 5) if index != center_index]
        hydrogen_vectors = positions[:, other_hydrogens] - center[:, None, :]
        hydrogen_units = hydrogen_vectors / np.linalg.norm(hydrogen_vectors, axis=2, keepdims=True)
        chh_cosines = np.einsum("na,nha->nh", carbon_unit, hydrogen_units)
        chh_by_center.append(np.degrees(np.arccos(np.clip(chh_cosines, -1, 1))))
        h_i, h_j = np.triu_indices(3, 1)
        hhh_cosines = np.einsum("nha,nka->nhk", hydrogen_units, hydrogen_units)[:, h_i, h_j]
        hhh_by_center.append(np.degrees(np.arccos(np.clip(hhh_cosines, -1, 1))))
    hydrogen_center_means = np.stack(
        [np.concatenate((chh, hhh), axis=1).mean(axis=1) for chh, hhh in zip(chh_by_center, hhh_by_center)],
        axis=1,
    )
    chh_angles = np.concatenate(chh_by_center, axis=1)
    hhh_angles = np.concatenate(hhh_by_center, axis=1)
    all_atom_angles = np.concatenate((angles, chh_angles, hhh_angles), axis=1)
    atom_center_means = np.column_stack((angles.mean(axis=1), hydrogen_center_means))
    hh_distances = np.linalg.norm(
        positions[:, 1:, None, :] - positions[:, None, 1:, :], axis=3
    )[:, upper_i, upper_j]
    tetra_volume = np.abs(
        np.linalg.det(
            np.stack(
                (positions[:, 2] - positions[:, 1], positions[:, 3] - positions[:, 1], positions[:, 4] - positions[:, 1]),
                axis=1,
            )
        )
    ) / 6
    force_magnitudes = np.linalg.norm(forces, axis=2)

    return pd.DataFrame(
        {
            "source_index": source,
            "target_force_rms": np.sqrt(np.mean(np.square(forces), axis=(1, 2))),
            "target_force_mean_atom_magnitude": force_magnitudes.mean(axis=1),
            "ch_mean": ch_distances.mean(axis=1),
            "ch_min": ch_distances.min(axis=1),
            "ch_max": ch_distances.max(axis=1),
            "ch_range": np.ptp(ch_distances, axis=1),
            "ch_cv": ch_distances.std(axis=1) / ch_distances.mean(axis=1),
            "hch_min_deg": angles.min(axis=1),
            "hch_max_deg": angles.max(axis=1),
            "hch_mean_deg": angles.mean(axis=1),
            "hch_angle_rmse_tetra_deg": np.sqrt(np.mean(np.square(angles - 109.4712206), axis=1)),
            "chh_mean_deg": chh_angles.mean(axis=1),
            "hhh_mean_deg": hhh_angles.mean(axis=1),
            "hydrogen_center_angle_mean_deg": np.concatenate((chh_angles, hhh_angles), axis=1).mean(axis=1),
            "all_atom_angle_mean_deg": all_atom_angles.mean(axis=1),
            "atom_center_mean_angle_std_deg": atom_center_means.std(axis=1),
            "tetrahedral_asymmetry": tetrahedral_asymmetry(ch_vectors, ch_distances),
            "hh_min": hh_distances.min(axis=1),
            "hh_max": hh_distances.max(axis=1),
            "h_tetra_volume": tetra_volume,
        }
    )


def matched_control_ids(table: pd.DataFrame, selected_ids: set[int], error_column: str) -> list[int]:
    """Greedily select unique controls nearest in log baseline error."""
    candidates = table.loc[~table.source_index.isin(selected_ids), ["source_index", error_column]].sort_values(error_column)
    available = list(zip(candidates[error_column].to_numpy(float), candidates.source_index.to_numpy(int)))
    selected = table[table.source_index.isin(selected_ids)].sort_values(error_column, ascending=False)
    controls = []
    for target in selected[error_column].to_numpy(float):
        values = [item[0] for item in available]
        insertion = bisect.bisect_left(values, target)
        choices = [i for i in (insertion - 1, insertion) if 0 <= i < len(available)]
        best = min(choices, key=lambda i: abs(np.log(max(available[i][0], 1e-15)) - np.log(max(target, 1e-15))))
        controls.append(available.pop(best)[1])
    return controls


def main() -> int:
    args = parse_args()
    winners = pd.read_csv(args.analysis_dir / "robust_l3n5_winners.csv")
    comparison = pd.read_csv(args.analysis_dir / "molecule_level_architecture_comparison.csv")
    geometry = geometry_descriptors(args.input_dir)
    table = comparison.merge(geometry, on="source_index", validate="one_to_one")
    table = table[table.source_index != args.remove_source].copy()

    top_rows = []
    summary_rows = []
    category_rows = []
    membership: dict[str, set[int]] = {}
    for metric in METRICS:
        ranked = winners[winners.metric == metric].nsmallest(args.n, "rank").copy()
        selected_ids = set(ranked.source_index.astype(int))
        membership[metric] = selected_ids
        baseline_column = f"l3n4_median_{metric}"
        control_ids = set(matched_control_ids(table, selected_ids, baseline_column))
        selected_mask = table.source_index.isin(selected_ids)
        control_mask = table.source_index.isin(control_ids)
        rest_mask = ~selected_mask

        selected = table[selected_mask].merge(
            ranked[["source_index", "rank", "absolute_improvement", "percent_improvement", "matched_seed_wins"]],
            on="source_index",
            validate="one_to_one",
        )
        selected.insert(0, "metric", metric)
        top_rows.append(selected.sort_values("rank"))

        for feature in FEATURE_LABELS:
            top_median = float(table.loc[selected_mask, feature].median())
            summary_rows.append(
                {
                    "metric": metric,
                    "descriptor": feature,
                    "descriptor_label": FEATURE_LABELS[feature],
                    "top100_median": top_median,
                    "rest_of_test_median": float(table.loc[rest_mask, feature].median()),
                    "baseline_error_matched_median": float(table.loc[control_mask, feature].median()),
                    "top100_median_percentile_in_test": 100 * float(np.mean(table[feature] <= top_median)),
                }
            )

        for category in ("in_high_force", "in_high_energy", "in_planar"):
            top_rate = float(table.loc[selected_mask, category].mean())
            test_rate = float(table[category].mean())
            category_rows.append(
                {
                    "metric": metric,
                    "category": category,
                    "top100_count": int(table.loc[selected_mask, category].sum()),
                    "top100_fraction": top_rate,
                    "test_fraction": test_rate,
                    "enrichment": top_rate / test_rate,
                    "baseline_error_matched_fraction": float(table.loc[control_mask, category].mean()),
                }
            )

    top_table = pd.concat(top_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    categories = pd.DataFrame(category_rows)
    overlap = sorted(membership["energy_abs_error"] & membership["force_rmse"])
    overlap_table = top_table[top_table.source_index.isin(overlap)].sort_values(["source_index", "metric"])

    top_table.to_csv(args.analysis_dir / "top100_l3n5_winners_characterized.csv", index=False)
    summary.to_csv(args.analysis_dir / "top100_descriptor_comparison.csv", index=False)
    categories.to_csv(args.analysis_dir / "top100_category_enrichment.csv", index=False)
    overlap_table.to_csv(args.analysis_dir / "top100_energy_force_overlap.csv", index=False)

    plotted = [
        "max_target_atomic_force",
        "target_force_rms",
        "h_planarity_ratio",
        "ch_mean",
        "ch_min",
        "ch_range",
        "tetrahedral_asymmetry",
        "h_tetra_volume",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
    for ax, metric, title in zip(axes, METRICS, ("Top 100 energy-error wins", "Top 100 force-RMSE wins")):
        values = summary[(summary.metric == metric)].set_index("descriptor").loc[plotted]
        y = np.arange(len(plotted))
        ax.scatter(values.top100_median_percentile_in_test, y, color="#f58518", s=55)
        ax.axvline(50, color="black", linestyle="--", linewidth=1)
        ax.set_xlim(0, 100)
        ax.set_yticks(y, [FEATURE_LABELS[name] for name in plotted])
        ax.invert_yaxis()
        ax.set_xlabel("Percentile of top-100 median in full test set")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Physical and geometric location of robust L3N5 wins")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.analysis_dir / "top100_descriptor_percentiles.png", dpi=220)
    fig.savefig(args.analysis_dir / "top100_descriptor_percentiles.pdf")
    plt.close(fig)

    # Repeat the same descriptive view for every strict robust winner.  The
    # energy and force sets have different sizes, so N is reported per panel.
    all_winner_rows = []
    all_category_rows = []
    all_winner_counts = {}
    for metric in METRICS:
        winner_ids = set(
            winners.loc[
                (winners.metric == metric) & (winners.source_index != args.remove_source),
                "source_index",
            ].astype(int)
        )
        winner_mask = table.source_index.isin(winner_ids)
        rest_mask = ~winner_mask
        all_winner_counts[metric] = int(winner_mask.sum())
        for feature in FEATURE_LABELS:
            winner_median = float(table.loc[winner_mask, feature].median())
            all_winner_rows.append(
                {
                    "metric": metric,
                    "n_winners": int(winner_mask.sum()),
                    "descriptor": feature,
                    "descriptor_label": FEATURE_LABELS[feature],
                    "winner_median": winner_median,
                    "rest_of_test_median": float(table.loc[rest_mask, feature].median()),
                    "winner_median_percentile_in_test": 100 * float(np.mean(table[feature] <= winner_median)),
                }
            )
        for category in ("in_high_force", "in_high_energy", "in_planar"):
            winner_rate = float(table.loc[winner_mask, category].mean())
            test_rate = float(table[category].mean())
            all_category_rows.append(
                {
                    "metric": metric,
                    "n_winners": int(winner_mask.sum()),
                    "category": category,
                    "winner_count": int(table.loc[winner_mask, category].sum()),
                    "winner_fraction": winner_rate,
                    "test_fraction": test_rate,
                    "enrichment": winner_rate / test_rate,
                }
            )

    all_winner_summary = pd.DataFrame(all_winner_rows)
    all_winner_categories = pd.DataFrame(all_category_rows)
    all_winner_summary.to_csv(args.analysis_dir / "all_robust_winner_descriptor_comparison.csv", index=False)
    all_winner_categories.to_csv(args.analysis_dir / "all_robust_winner_category_enrichment.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
    titles = (
        f"All robust energy-error wins (N={all_winner_counts['energy_abs_error']:,})",
        f"All robust force-RMSE wins (N={all_winner_counts['force_rmse']:,})",
    )
    for ax, metric, title in zip(axes, METRICS, titles):
        values = all_winner_summary[all_winner_summary.metric == metric].set_index("descriptor").loc[plotted]
        y = np.arange(len(plotted))
        ax.scatter(values.winner_median_percentile_in_test, y, color="#f58518", s=55)
        ax.axvline(50, color="black", linestyle="--", linewidth=1)
        ax.set_xlim(0, 100)
        ax.set_yticks(y, [FEATURE_LABELS[name] for name in plotted])
        ax.invert_yaxis()
        ax.set_xlabel("Percentile of winner-group median in full test set")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Physical and geometric location of all robust L3N5 wins")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.analysis_dir / "all_robust_winner_descriptor_percentiles.png", dpi=220)
    fig.savefig(args.analysis_dir / "all_robust_winner_descriptor_percentiles.pdf")
    plt.close(fig)

    # Also show the strongest decile of each robust-winner population.  Use a
    # ceiling so the selected set contains at least ten percent when N is not
    # divisible by ten.
    decile_rows = []
    decile_category_rows = []
    decile_counts = {}
    for metric in METRICS:
        metric_winners = winners.loc[
            (winners.metric == metric) & (winners.source_index != args.remove_source)
        ].sort_values("rank")
        decile_n = int(np.ceil(0.10 * len(metric_winners)))
        decile_ids = set(metric_winners.head(decile_n).source_index.astype(int))
        decile_mask = table.source_index.isin(decile_ids)
        rest_mask = ~decile_mask
        decile_counts[metric] = int(decile_mask.sum())
        for feature in FEATURE_LABELS:
            decile_median = float(table.loc[decile_mask, feature].median())
            decile_rows.append(
                {
                    "metric": metric,
                    "n_winners_in_decile": int(decile_mask.sum()),
                    "descriptor": feature,
                    "descriptor_label": FEATURE_LABELS[feature],
                    "winner_decile_median": decile_median,
                    "rest_of_test_median": float(table.loc[rest_mask, feature].median()),
                    "winner_decile_median_percentile_in_test": 100
                    * float(np.mean(table[feature] <= decile_median)),
                }
            )
        for category in ("in_high_force", "in_high_energy", "in_planar"):
            decile_rate = float(table.loc[decile_mask, category].mean())
            test_rate = float(table[category].mean())
            decile_category_rows.append(
                {
                    "metric": metric,
                    "n_winners_in_decile": int(decile_mask.sum()),
                    "category": category,
                    "decile_count": int(table.loc[decile_mask, category].sum()),
                    "decile_fraction": decile_rate,
                    "test_fraction": test_rate,
                    "enrichment": decile_rate / test_rate,
                }
            )

    decile_summary = pd.DataFrame(decile_rows)
    decile_categories = pd.DataFrame(decile_category_rows)
    decile_summary.to_csv(args.analysis_dir / "top_decile_robust_winner_descriptor_comparison.csv", index=False)
    decile_categories.to_csv(args.analysis_dir / "top_decile_robust_winner_category_enrichment.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
    titles = (
        f"Strongest 10% energy-error wins (N={decile_counts['energy_abs_error']:,})",
        f"Strongest 10% force-RMSE wins (N={decile_counts['force_rmse']:,})",
    )
    for ax, metric, title in zip(axes, METRICS, titles):
        values = decile_summary[decile_summary.metric == metric].set_index("descriptor").loc[plotted]
        y = np.arange(len(plotted))
        ax.scatter(values.winner_decile_median_percentile_in_test, y, color="#f58518", s=55)
        ax.axvline(50, color="black", linestyle="--", linewidth=1)
        ax.set_xlim(0, 100)
        ax.set_yticks(y, [FEATURE_LABELS[name] for name in plotted])
        ax.invert_yaxis()
        ax.set_xlabel("Percentile of winner-decile median in full test set")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Physical and geometric location of the strongest 10% of robust L3N5 wins")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.analysis_dir / "top_decile_robust_winner_descriptor_percentiles.png", dpi=220)
    fig.savefig(args.analysis_dir / "top_decile_robust_winner_descriptor_percentiles.pdf")
    plt.close(fig)

    # Distribution of the improvement magnitudes for all strict robust wins.
    quantile_rows = []
    for metric in METRICS:
        metric_winners = winners.loc[
            (winners.metric == metric) & (winners.source_index != args.remove_source)
        ]
        for value_name in ("absolute_improvement", "percent_improvement"):
            for quantile in (0, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0):
                quantile_rows.append(
                    {
                        "metric": metric,
                        "n_winners": len(metric_winners),
                        "improvement_measure": value_name,
                        "quantile": quantile,
                        "value": float(metric_winners[value_name].quantile(quantile)),
                    }
                )
    improvement_quantiles = pd.DataFrame(quantile_rows)
    improvement_quantiles.to_csv(args.analysis_dir / "robust_winner_improvement_quantiles.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for row, (metric, metric_title) in enumerate(
        (("energy_abs_error", "Energy absolute-error wins"), ("force_rmse", "Force-RMSE wins"))
    ):
        metric_winners = winners.loc[
            (winners.metric == metric) & (winners.source_index != args.remove_source)
        ]
        absolute = metric_winners.absolute_improvement.to_numpy(float)
        relative = metric_winners.percent_improvement.to_numpy(float)
        weights = np.full(len(metric_winners), 100 / len(metric_winners))

        ax = axes[row, 0]
        positive_min = max(float(absolute.min()), 1e-6)
        bins = np.geomspace(positive_min, float(absolute.max()) * 1.001, 46)
        ax.hist(absolute, bins=bins, weights=weights, color="#4c78a8", alpha=0.85)
        ax.set_xscale("log")
        ax.set_ylabel("Robust winners per bin (%)")
        ax.set_xlabel("Absolute error reduction (log scale)")
        ax.set_title(f"{metric_title}: absolute improvement (N={len(metric_winners):,})")
        for quantile, style, label in ((0.50, "-", "median"), (0.90, "--", "90th"), (0.99, ":", "99th")):
            value = float(np.quantile(absolute, quantile))
            ax.axvline(value, color="#d95319", linestyle=style, linewidth=1.6, label=f"{label}: {value:.3g}")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=9)
        ax.text(
            0.98,
            0.95,
            f"maximum: {absolute.max():.3g}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color="#555",
        )

        ax = axes[row, 1]
        ax.hist(relative, bins=np.linspace(0, 100, 41), weights=weights, color="#f58518", alpha=0.85)
        ax.set_xlim(0, 100)
        ax.set_ylabel("Robust winners per bin (%)")
        ax.set_xlabel("Reduction relative to L3N4 error (%)")
        ax.set_title(f"{metric_title}: percent improvement")
        for quantile, style, label in ((0.50, "-", "median"), (0.90, "--", "90th"), (0.99, ":", "99th")):
            value = float(np.quantile(relative, quantile))
            ax.axvline(value, color="#315fa8", linestyle=style, linewidth=1.6, label=f"{label}: {value:.1f}%")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=9)

    fig.suptitle("How large are L3N5's strict robust per-structure improvements?")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(args.analysis_dir / "robust_winner_improvement_distributions.png", dpi=220)
    fig.savefig(args.analysis_dir / "robust_winner_improvement_distributions.pdf")
    plt.close(fig)

    print(f"Wrote top-{args.n} analysis to {args.analysis_dir}")
    print(categories.to_string(index=False))
    print(f"Energy/force top-{args.n} overlap: {len(overlap)} structures: {overlap}")
    print(
        "All strict robust winners: "
        f"energy N={all_winner_counts['energy_abs_error']}, "
        f"force N={all_winner_counts['force_rmse']}"
    )
    print(
        "Strongest robust-winner decile: "
        f"energy N={decile_counts['energy_abs_error']}, "
        f"force N={decile_counts['force_rmse']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
