#!/usr/bin/env python3
"""Plot side-by-side 100k test and validation MAE boxplots with paper references."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_PAPER_MAE = Path(
    "examples/methane_sweep/results/paper/paper_methane_mae_results.json"
)
DEFAULT_FINAL_TEST = Path(
    "examples/methane_sweep/logs/plots_l3_b256_final_test/metrics.csv"
)
DEFAULT_A100_VALID = Path(
    "examples/methane_sweep/logs/Replicate_paper/metric_plots_l3_b256/metrics.csv"
)
DEFAULT_TITANV_VALID = Path(
    "examples/methane_sweep/logs/Replicate_paper/metric_plots_titanv_l3_b256/metrics.csv"
)
DEFAULT_L3N5_TEST = Path(
    "examples/methane_sweep/logs/plots_l3n5_100k_all8/final_test_metrics.csv"
)
DEFAULT_L3N5_VALID = Path(
    "examples/methane_sweep/logs/plots_l3n5_100k_all8/metrics.csv"
)
DEFAULT_OUTPUT = Path(
    "examples/methane_sweep/results/paper/"
    "boxplot_test_and_validation_selected_100k_mae_side_by_side.png"
)

DATA_SIZE = 100_000
PAPER_SERIES_BY_GROUP = {
    "l2n4_ref_test": "HipHop_l2_n4",
    "l3n4_ref_test": "HipHop",
}
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
    parser.add_argument("--paper-mae", type=Path, default=DEFAULT_PAPER_MAE)
    parser.add_argument("--final-test", type=Path, default=DEFAULT_FINAL_TEST)
    parser.add_argument("--a100-valid", type=Path, default=DEFAULT_A100_VALID)
    parser.add_argument("--titanv-valid", type=Path, default=DEFAULT_TITANV_VALID)
    parser.add_argument("--l3n5-test", type=Path, default=DEFAULT_L3N5_TEST)
    parser.add_argument("--l3n5-valid", type=Path, default=DEFAULT_L3N5_VALID)
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


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


def finite_float(value: str, *, path: Path, column: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"Non-finite {column} in {path}: {value}")
    return parsed


def paper_groups(path: Path) -> dict[str, list[float]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    size_key = str(DATA_SIZE)
    grouped = {}
    for group, series in PAPER_SERIES_BY_GROUP.items():
        try:
            values = [float(value) for value in data[series][size_key]]
        except KeyError as exc:
            raise ValueError(f"Missing {series}[{size_key}] in {path}") from exc
        if not values or not np.isfinite(values).all():
            raise ValueError(f"Empty or non-finite {series}[{size_key}] in {path}")
        grouped[group] = values
    return grouped


def selected_test_groups(final_test_path: Path, l3n5_path: Path) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    rows = read_rows(final_test_path)
    for l_max, n_max, group in [(3, 3, "l3n3_test"), (3, 4, "l3n4_test")]:
        matches = [
            row
            for row in rows
            if row.get("hiphop_l_max") == str(l_max)
            and row.get("hiphop_n_max") == str(n_max)
            and row.get("data_size") == str(DATA_SIZE)
            and row.get("test_T-MAE", "") != ""
        ]
        values = {
            finite_float(row["test_T-MAE"], path=final_test_path, column="test_T-MAE")
            for row in matches
        }
        if len(values) != 1:
            raise ValueError(f"Expected one distinct {group} MAE in {final_test_path}")
        grouped[group].extend(values)

    for row in read_rows(l3n5_path):
        if row.get("data_size") == str(DATA_SIZE) and row.get("test_T-MAE", "") != "":
            grouped["l3n5_test"].append(
                finite_float(row["test_T-MAE"], path=l3n5_path, column="test_T-MAE")
            )
    return grouped


def minimum_valid_mae_by_run(
    path: Path, *, l_max: int, n_max: int
) -> list[float]:
    values_by_run: dict[str, list[float]] = defaultdict(list)
    for row in read_rows(path):
        if (
            row.get("hiphop_l_max") == str(l_max)
            and row.get("hiphop_n_max") == str(n_max)
            and row.get("data_size") == str(DATA_SIZE)
            and row.get("valid_T-MAE", "") != ""
        ):
            values_by_run[row["run"]].append(
                finite_float(row["valid_T-MAE"], path=path, column="valid_T-MAE")
            )
    return [min(values) for values in values_by_run.values()]


def selected_valid_groups(
    a100_path: Path, titanv_path: Path, l3n5_path: Path
) -> dict[str, list[float]]:
    # Match the RMSE figure's curated hardware coverage: A100 + TitanV for
    # l3n3, TitanV for l3n4, and all dedicated l3n5 replicates.
    return {
        "l3n3_valid": minimum_valid_mae_by_run(a100_path, l_max=3, n_max=3)
        + minimum_valid_mae_by_run(titanv_path, l_max=3, n_max=3),
        "l3n4_valid": minimum_valid_mae_by_run(titanv_path, l_max=3, n_max=4),
        "l3n5_valid": minimum_valid_mae_by_run(l3n5_path, l_max=3, n_max=5),
    }


def color_for_group(group: str) -> str:
    return next(
        (color for prefix, color in COLORS.items() if group.startswith(prefix)),
        "#cccccc",
    )


def main() -> None:
    args = parse_args()
    configure_matplotlib()

    grouped = paper_groups(args.paper_mae)
    grouped.update(selected_test_groups(args.final_test, args.l3n5_test))
    grouped.update(
        selected_valid_groups(args.a100_valid, args.titanv_valid, args.l3n5_valid)
    )
    missing = [group for group, _ in PLOT_ORDER if not grouped.get(group)]
    if missing:
        raise ValueError("No MAE observations for: " + ", ".join(missing))

    groups = PLOT_ORDER
    positions = [POSITIONS[group] for group, _ in groups]
    fig, ax = plt.subplots(figsize=(9.0, 6), dpi=180)
    boxplot = ax.boxplot(
        [grouped[group] for group, _ in groups],
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
    for patch, (group, _) in zip(boxplot["boxes"], groups):
        patch.set_facecolor(color_for_group(group))
        patch.set_alpha(0.72)

    rng = np.random.default_rng(42)
    for (group, _), position in zip(groups, positions):
        values = grouped[group]
        x_values = position + rng.normal(0, 0.025, size=len(values))
        ax.scatter(
            x_values,
            values,
            alpha=0.65,
            s=45,
            color="red",
            edgecolors="darkred",
            linewidth=0.5,
            zorder=3,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([label for _, label in groups])
    ax.set_xlim(min(positions) - 0.45, max(positions) + 0.45)
    ax.set_ylabel("MAE (kcal/mol)")
    ax.set_title("100k Test and Validation MAE Distributions")
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    for separator in [1.5, 2.55, 3.65, 4.45]:
        ax.axvline(separator, color="black", linewidth=0.8, alpha=0.10, zorder=0)
    for label, group_keys in GROUP_LABELS:
        center = sum(POSITIONS[group] for group in group_keys) / len(group_keys)
        ax.text(
            center,
            -0.065,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=11,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.145)
    fig.savefig(args.output, dpi=180, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    print(f"Saved {args.output}")
    for group, _ in groups:
        values = np.asarray(grouped[group], dtype=float)
        std = values.std(ddof=1) if len(values) > 1 else 0.0
        print(f"{group}: N={len(values)}, mean={values.mean():.5f}, std={std:.5f}")


if __name__ == "__main__":
    main()
