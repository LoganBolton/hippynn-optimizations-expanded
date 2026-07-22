#!/usr/bin/env python3
"""Plot side-by-side 100k test and validation boxplots for selected/reference configs."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_TEST_CSV = Path("examples/methane_sweep/results/paper/test_results_selected_100k.csv")
DEFAULT_VALID_CSV = Path("examples/methane_sweep/results/paper/boxplot_best_valid_rmse_all_configs_100k.csv")
DEFAULT_OVERLAY_CSV = Path("examples/methane_sweep/results/paper/paper_methane_plot_overlay_points.csv")
DEFAULT_OUTPUT = Path("examples/methane_sweep/results/paper/boxplot_test_and_validation_selected_100k_side_by_side.png")

COLORS = {
    "l2n4": "#d99ab5",
    "l3n3": "#56b4e9",
    "l3n4": "#862e9c",
    "l3n5": "#2b8a3e",
}

PLOT_ORDER = [
    ("l2n4_ref_test", "Test"),
    ("l3n3_test", "Test"),
    ("l3n3_valid", "Valid"),
    ("l3n4_test", "Test"),
    ("l3n4_valid", "Valid"),
    ("l3n4_ref_test", "Test"),
    ("l3n5_test", "Test"),
    ("l3n5_valid", "Valid"),
]

GROUP_LABELS = [
    ("$\\ell=2, n=4$\n(from paper)", ["l2n4_ref_test"]),
    ("$\\ell=3, n=3$", ["l3n3_test", "l3n3_valid"]),
    ("$\\ell=3, n=4$", ["l3n4_test", "l3n4_valid"]),
    ("$\\ell=3, n=4$\n(from paper)", ["l3n4_ref_test"]),
    ("$\\ell=3, n=5$", ["l3n5_test", "l3n5_valid"]),
]

POSITIONS = {
    "l2n4_ref_test": 1.0,
    "l3n3_test": 1.85,
    "l3n3_valid": 2.15,
    "l3n4_test": 2.95,
    "l3n4_valid": 3.25,
    "l3n4_ref_test": 4.05,
    "l3n5_test": 4.9,
    "l3n5_valid": 5.2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV)
    parser.add_argument("--valid-csv", type=Path, default=DEFAULT_VALID_CSV)
    parser.add_argument("--overlay-csv", type=Path, default=DEFAULT_OVERLAY_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def configure_matplotlib() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "axes.linewidth": 1.1,
            "axes.labelsize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 12,
            "lines.linewidth": 1.5,
        }
    )


def load_test_groups(path: Path) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            group = row["group"]
            value = float(row["test_T-RMSE"])
            if group == "l2n4_ref":
                grouped["l2n4_ref_test"].append(value)
            elif group == "l3n4_ref":
                grouped["l3n4_ref_test"].append(value)
            elif group == "l3n3":
                grouped["l3n3_test"].append(value)
            elif group == "l3n4":
                grouped["l3n4_test"].append(value)
            elif group == "l3n5":
                grouped["l3n5_test"].append(value)
    return grouped


def load_valid_groups(path: Path, overlay_path: Path) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)

    # Use the curated overlay CSV for l3n3/l3n4 so we include the extra A100 100k point.
    if overlay_path.exists():
        with overlay_path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                try:
                    l_max = int(row["hiphop_l_max"])
                    n_max = int(row["hiphop_n_max"])
                    batch_size = int(row["batch_size"])
                    training_size = int(row["training_set_size"])
                    value = float(row["best_valid_T-RMSE"])
                except (TypeError, ValueError, KeyError):
                    continue
                if batch_size != 256 or training_size != 100000:
                    continue
                if (l_max, n_max) == (3, 3):
                    grouped["l3n3_valid"].append(value)
                elif (l_max, n_max) == (3, 4):
                    grouped["l3n4_valid"].append(value)

    # Keep the per-run l3n5 validation values from the dedicated best-valid CSV.
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            metric_type = row.get("metric_type", "")
            if metric_type != "best_valid":
                continue
            config = row["config"]
            value = float(row["best_valid_T-RMSE"])
            if config == "l3n5":
                grouped["l3n5_valid"].append(value)
    return grouped


def color_for_group(group_key: str) -> str:
    if group_key.startswith("l2n4"):
        return COLORS["l2n4"]
    if group_key.startswith("l3n3"):
        return COLORS["l3n3"]
    if group_key.startswith("l3n4"):
        return COLORS["l3n4"]
    if group_key.startswith("l3n5"):
        return COLORS["l3n5"]
    return "#cccccc"


def main() -> None:
    args = parse_args()
    configure_matplotlib()

    grouped = {}
    grouped.update(load_test_groups(args.test_csv))
    grouped.update(load_valid_groups(args.valid_csv, args.overlay_csv))

    groups = [(key, label) for key, label in PLOT_ORDER if key in grouped]
    values = [grouped[key] for key, _ in groups]
    labels = [label for key, label in groups]

    positions = [POSITIONS[key] for key, _ in groups]

    fig, ax = plt.subplots(figsize=(9.0, 6), dpi=180)
    bp = ax.boxplot(
        values,
        positions=positions,
        widths=0.26,
        patch_artist=True,
        showmeans=False,
        showfliers=False,
        boxprops=dict(edgecolor="black", linewidth=1.2),
        whiskerprops=dict(color="black", linewidth=1.2),
        capprops=dict(color="black", linewidth=1.2),
        medianprops=dict(color="darkblue", linewidth=2),
    )

    for patch, (group_key, _) in zip(bp["boxes"], groups):
        patch.set_facecolor(color_for_group(group_key))
        patch.set_alpha(0.72)

    np.random.seed(42)
    for (group_key, _), pos in zip(groups, positions):
        y = grouped[group_key]
        x = np.ones(len(y)) * pos + np.random.normal(0, 0.025, size=len(y))
        ax.scatter(x, y, alpha=0.65, s=45, color="red", edgecolors="darkred", linewidth=0.5, zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlim(min(positions) - 0.45, max(positions) + 0.45)
    ax.set_ylabel("RMSE (kcal/mol)")
    ax.set_title("100k Test and Validation RMSE Distributions")
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    separator_x = [1.5, 2.55, 3.65, 4.45]
    for x in separator_x:
        ax.axvline(x, color="black", linewidth=0.8, alpha=0.10, zorder=0)

    for group_label, group_keys in GROUP_LABELS:
        active_positions = [POSITIONS[key] for key in group_keys if key in grouped]
        if not active_positions:
            continue
        center = sum(active_positions) / len(active_positions)
        ax.text(center, -0.065, group_label, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=11)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.145)
    fig.savefig(args.output, dpi=180, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    print(f"Saved {args.output}")
    for group_key, _ in groups:
        arr = np.array(grouped[group_key], dtype=float)
        std = arr.std(ddof=1) if len(arr) > 1 else 0.0
        print(f"{group_key}: N={len(arr)}, mean={arr.mean():.5f}, std={std:.5f}")


if __name__ == "__main__":
    main()
