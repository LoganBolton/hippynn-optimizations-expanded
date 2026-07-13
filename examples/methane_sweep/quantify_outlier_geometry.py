#!/usr/bin/env python3
"""Quantify whether high-error methane outliers have unusual geometry."""

import argparse
import csv
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TETRAHEDRAL_ANGLE = 109.47122063449069
PLOT_TITLES = {
    "cv_ch": "Coefficient of variation: std(C-H distances) / mean(C-H distances)",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug_dir",
        type=Path,
        default=Path("examples/TEST_METHANE_MODEL_l4_n3_d100000_seed42-b256_fresh/validation_debug"),
    )
    parser.add_argument("--epoch", type=int, default=3000)
    parser.add_argument("--source_indices", nargs="+", type=int, default=[32801, 1136, 60963])
    parser.add_argument("--out_dir", type=Path)
    return parser.parse_args()


def angle_degrees(a, b, c):
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return float("nan")
    cosine = np.dot(ba, bc) / denom
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def tetra_volume(a, b, c, d):
    return abs(float(np.linalg.det(np.stack([b - a, c - a, d - a], axis=1)))) / 6.0


def methane_metrics(numbers, positions):
    carbon_candidates = np.where(numbers == 6)[0]
    hydrogen_indices = np.where(numbers == 1)[0]
    if len(carbon_candidates) != 1 or len(hydrogen_indices) != 4:
        raise ValueError("expected exactly one carbon and four hydrogens")

    carbon = int(carbon_candidates[0])
    hydrogens = [int(i) for i in hydrogen_indices]
    cpos = positions[carbon]

    ch_distances = np.array([np.linalg.norm(positions[h] - cpos) for h in hydrogens], dtype=float)

    hh_distances = []
    hch_angles = []
    for i, h1 in enumerate(hydrogens):
        for h2 in hydrogens[i + 1 :]:
            hh_distances.append(float(np.linalg.norm(positions[h1] - positions[h2])))
            hch_angles.append(angle_degrees(positions[h1], cpos, positions[h2]))
    hh_distances = np.array(hh_distances, dtype=float)
    hch_angles = np.array(hch_angles, dtype=float)

    c_hhh_volumes = []
    for i in range(len(hydrogens)):
        for j in range(i + 1, len(hydrogens)):
            for k in range(j + 1, len(hydrogens)):
                c_hhh_volumes.append(tetra_volume(cpos, positions[hydrogens[i]], positions[hydrogens[j]], positions[hydrogens[k]]))
    c_hhh_volumes = np.array(c_hhh_volumes, dtype=float)

    h_centroid = np.mean(positions[hydrogens], axis=0)
    centered = positions[hydrogens] - h_centroid
    _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    planarity_ratio = float(singular_values[-1] / singular_values[0]) if singular_values[0] > 0 else float("nan")

    return {
        "min_ch": float(np.min(ch_distances)),
        "max_ch": float(np.max(ch_distances)),
        "mean_ch": float(np.mean(ch_distances)),
        "range_ch": float(np.max(ch_distances) - np.min(ch_distances)),
        "cv_ch": float(np.std(ch_distances) / np.mean(ch_distances)),
        "min_hh": float(np.min(hh_distances)),
        "max_hh": float(np.max(hh_distances)),
        "range_hh": float(np.max(hh_distances) - np.min(hh_distances)),
        "min_hch": float(np.min(hch_angles)),
        "max_hch": float(np.max(hch_angles)),
        "range_hch": float(np.max(hch_angles) - np.min(hch_angles)),
        "hch_rmse_from_tetrahedral": float(np.sqrt(np.mean((hch_angles - TETRAHEDRAL_ANGLE) ** 2))),
        "min_c_hhh_volume": float(np.min(c_hhh_volumes)),
        "max_c_hhh_volume": float(np.max(c_hhh_volumes)),
        "range_c_hhh_volume": float(np.max(c_hhh_volumes) - np.min(c_hhh_volumes)),
        "h_planarity_ratio": planarity_ratio,
        "carbon_to_h_centroid": float(np.linalg.norm(cpos - h_centroid)),
    }


def load_epoch(debug_dir, epoch):
    path = debug_dir / f"validation_errors_epoch_{epoch}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path)


def percentile_rank(values, value, high_is_unusual=True):
    values = np.asarray(values, dtype=float)
    if high_is_unusual:
        return 100.0 * float(np.mean(values <= value))
    return 100.0 * float(np.mean(values >= value))


def tail_probability(values, cutoff, high_is_unusual=True):
    values = np.asarray(values, dtype=float)
    if high_is_unusual:
        return float(np.mean(values >= cutoff))
    return float(np.mean(values <= cutoff))


def all_tail_probability_without_replacement(n_total, n_tail, n_draws):
    if n_total < n_draws or n_tail < n_draws:
        return 0.0
    probability = 1.0
    for draw in range(n_draws):
        probability *= (n_tail - draw) / (n_total - draw)
    return probability


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_histograms(out_dir, metric_rows, source_indices, metric_specs):
    source_set = set(source_indices)
    for metric_name, high_is_unusual in metric_specs.items():
        values = np.array([row[metric_name] for row in metric_rows], dtype=float)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(values, bins=60, color="#c9c9c9", edgecolor="#666666")
        for row in metric_rows:
            if row["source_index"] in source_set:
                ax.axvline(row[metric_name], linewidth=2.0, label=f"source {row['source_index']}")
        title = PLOT_TITLES.get(metric_name, metric_name)
        ax.set_title(title)
        ax.set_xlabel(title)
        ax.set_ylabel("validation count")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"hist_{metric_name}.png", dpi=200)
        plt.close(fig)


def plot_error_vs_geometry(out_dir, metric_rows, source_indices, metric_names):
    source_set = set(source_indices)
    force_rmse = np.array([row["force_rmse"] for row in metric_rows], dtype=float)
    for metric_name in metric_names:
        values = np.array([row[metric_name] for row in metric_rows], dtype=float)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(values, force_rmse, s=8, alpha=0.25, color="#4c78a8")
        for row in metric_rows:
            if row["source_index"] in source_set:
                ax.scatter(row[metric_name], row["force_rmse"], s=65, color="#d62728", edgecolor="black", zorder=3)
                ax.annotate(str(row["source_index"]), (row[metric_name], row["force_rmse"]), xytext=(5, 5), textcoords="offset points")
        title = PLOT_TITLES.get(metric_name, metric_name)
        ax.set_xlabel(title)
        ax.set_ylabel("per-config force RMSE")
        ax.set_title(f"Force error vs {title}")
        fig.tight_layout()
        fig.savefig(out_dir / f"scatter_force_rmse_vs_{metric_name}.png", dpi=200)
        plt.close(fig)


def main():
    args = parse_args()
    out_dir = args.out_dir or args.debug_dir / "geometry_rarity"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_epoch(args.debug_dir, args.epoch)
    source_indices = data["source_indices"].astype(int)
    force_error = data["force_error"]
    force_rmse = np.sqrt(np.mean(force_error**2, axis=(1, 2)))

    metric_rows = []
    for row_idx, source_index in enumerate(source_indices):
        row = {
            "valid_row": int(row_idx),
            "source_index": int(source_index),
            "force_rmse": float(force_rmse[row_idx]),
            "energy_error": float(data["energy_error"][row_idx]),
        }
        row.update(methane_metrics(data["numbers"][row_idx].astype(int), data["positions"][row_idx].astype(float)))
        metric_rows.append(row)

    metric_specs = {
        "range_ch": True,
        "cv_ch": True,
        "min_ch": False,
        "max_ch": True,
        "min_hh": False,
        "range_hh": True,
        "range_hch": True,
        "hch_rmse_from_tetrahedral": True,
        "min_c_hhh_volume": False,
        "range_c_hhh_volume": True,
        "h_planarity_ratio": False,
        "carbon_to_h_centroid": True,
    }

    metric_names = list(metric_specs)
    all_fieldnames = list(metric_rows[0])
    write_csv(out_dir / "all_validation_geometry_metrics.csv", metric_rows, all_fieldnames)

    outlier_rows = [row for row in metric_rows if row["source_index"] in set(args.source_indices)]
    if len(outlier_rows) != len(args.source_indices):
        found = {row["source_index"] for row in outlier_rows}
        missing = sorted(set(args.source_indices) - found)
        raise ValueError(f"source indices not found in validation set: {missing}")

    rarity_rows = []
    n_total = len(metric_rows)
    for metric_name, high_is_unusual in metric_specs.items():
        values = np.array([row[metric_name] for row in metric_rows], dtype=float)
        outlier_values = [row[metric_name] for row in outlier_rows]
        cutoff = min(outlier_values) if high_is_unusual else max(outlier_values)
        tail_fraction = tail_probability(values, cutoff, high_is_unusual=high_is_unusual)
        n_tail = int(round(tail_fraction * n_total))
        rarity_rows.append(
            {
                "metric": metric_name,
                "unusual_tail": "high" if high_is_unusual else "low",
                "source_values": ";".join(f"{row['source_index']}={row[metric_name]:.8g}" for row in outlier_rows),
                "source_percentiles": ";".join(
                    f"{row['source_index']}={percentile_rank(values, row[metric_name], high_is_unusual):.3f}"
                    for row in outlier_rows
                ),
                "tail_cutoff_covering_both": cutoff,
                "tail_count_covering_both": n_tail,
                "tail_fraction_covering_both": tail_fraction,
                "prob_random_validation_points_all_in_tail": all_tail_probability_without_replacement(
                    n_total,
                    n_tail,
                    len(outlier_rows),
                ),
            }
        )
    write_csv(out_dir / "outlier_geometry_rarity.csv", rarity_rows, list(rarity_rows[0]))

    source_summary_rows = []
    for row in outlier_rows:
        summary = {
            "source_index": row["source_index"],
            "valid_row": row["valid_row"],
            "force_rmse": row["force_rmse"],
            "energy_error": row["energy_error"],
        }
        for metric_name, high_is_unusual in metric_specs.items():
            values = np.array([metric_row[metric_name] for metric_row in metric_rows], dtype=float)
            summary[metric_name] = row[metric_name]
            summary[f"{metric_name}_percentile"] = percentile_rank(values, row[metric_name], high_is_unusual)
        source_summary_rows.append(summary)
    write_csv(out_dir / "outlier_geometry_summary.csv", source_summary_rows, list(source_summary_rows[0]))

    plot_histograms(out_dir, metric_rows, args.source_indices, metric_specs)
    plot_error_vs_geometry(
        out_dir,
        metric_rows,
        args.source_indices,
        list(metric_specs),
    )

    print(f"Wrote geometry rarity analysis to {out_dir}")
    print(f"Start with {out_dir / 'outlier_geometry_summary.csv'}")
    print(f"Then check {out_dir / 'outlier_geometry_rarity.csv'}")


if __name__ == "__main__":
    main()
