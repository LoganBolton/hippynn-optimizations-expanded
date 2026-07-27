#!/usr/bin/env python3
"""Plot side-by-side 1M test and validation MAE boxplots for selected/reference configs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_SELECTION = Path("examples/methane_sweep/results/paper/test_and_validation_selected_1m.csv")
DEFAULT_PAPER_MAE = Path("examples/methane_sweep/results/paper/paper_methane_mae_results.json")
DEFAULT_OUTPUT = Path("examples/methane_sweep/results/paper/boxplot_test_and_validation_selected_1m_mae_side_by_side.png")
DATA_SIZE = 1_000_000

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
    ("l3n4_ref_test", "Test"),
    ("l3n5_test", "Test"),
    ("l3n5_valid", "Valid"),
]

GROUP_LABELS = [
    ("$\\ell=2, n=4$\n(from paper)", ["l2n4_ref_test"]),
    ("$\\ell=3, n=4$\n(from paper)", ["l3n4_ref_test"]),
    ("$\\ell=3, n=5$\n(7/8 runs tested)", ["l3n5_test", "l3n5_valid"]),
]

POSITIONS = {
    "l2n4_ref_test": 1.0,
    "l3n4_ref_test": 1.8,
    "l3n5_test": 2.6,
    "l3n5_valid": 2.9,
}

SEPARATORS = [1.4, 2.2]
EXPECTED_GROUPS = [group_key for group_key, _ in PLOT_ORDER]
REQUIRED_SELECTION_COLUMNS = {"group", "config", "split", "data_size", "seed", "source", "run"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--paper-mae", type=Path, default=DEFAULT_PAPER_MAE)
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


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


def require_columns(path: Path, fieldnames: list[str] | None, required: set[str]) -> None:
    if fieldnames is None:
        raise ValueError(f"CSV has no header: {path}")
    missing = sorted(required - set(fieldnames))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {', '.join(missing)}")


def load_selection_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require_columns(path, reader.fieldnames, REQUIRED_SELECTION_COLUMNS)
        rows = list(reader)

    selected_rows = [row for row in rows if row["group"] in EXPECTED_GROUPS]
    groups_in_selection = {row["group"] for row in selected_rows}
    missing_groups = [group_key for group_key in EXPECTED_GROUPS if group_key not in groups_in_selection]
    if missing_groups:
        raise ValueError(
            "Selection CSV is missing expected selected groups: " + ", ".join(missing_groups)
        )

    return selected_rows


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


def finite_float(value: str, *, path: Path, column: str, row_desc: str) -> float:
    if value == "":
        raise ValueError(f"Blank {column} in {path} for {row_desc}")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"Non-finite {column} in {path} for {row_desc}: {value}")
    return parsed


def lookup_test_mae(selection_row: dict[str, str]) -> float:
    source_path = Path(selection_row["source"])
    rows = read_csv_rows(source_path)
    if not rows:
        raise ValueError(f"No rows found in {source_path}")
    require_columns(source_path, list(rows[0].keys()), {"run", "seed", "test_T-MAE"})

    matches = [
        row
        for row in rows
        if row.get("seed") == selection_row["seed"]
        and row.get("data_size", selection_row["data_size"]) == selection_row["data_size"]
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly 1 test_T-MAE match in {source_path} for run={selection_row['run']!r}, "
            f"seed={selection_row['seed']!r}; found {len(matches)}"
        )

    return finite_float(
        matches[0]["test_T-MAE"],
        path=source_path,
        column="test_T-MAE",
        row_desc=f"run={selection_row['run']} seed={selection_row['seed']}",
    )


def lookup_best_valid_mae(selection_row: dict[str, str]) -> float:
    source_path = Path(selection_row["source"])
    rows = read_csv_rows(source_path)
    if not rows:
        raise ValueError(f"No rows found in {source_path}")
    require_columns(source_path, list(rows[0].keys()), {"run", "seed", "epoch", "valid_T-MAE"})

    candidate_rows = [
        row
        for row in rows
        if row.get("seed") == selection_row["seed"]
        and row.get("data_size", selection_row["data_size"]) == selection_row["data_size"]
        and row.get("valid_T-MAE", "") != ""
    ]
    if not candidate_rows:
        raise ValueError(
            f"No valid_T-MAE rows found in {source_path} for run={selection_row['run']!r}, seed={selection_row['seed']!r}"
        )

    best_row = min(candidate_rows, key=lambda row: float(row["valid_T-MAE"]))
    return finite_float(
        best_row["valid_T-MAE"],
        path=source_path,
        column="valid_T-MAE",
        row_desc=f"run={selection_row['run']} seed={selection_row['seed']} epoch={best_row['epoch']}",
    )


def load_paper_reference_mae(path: Path) -> dict[str, list[float]]:
    with path.open("r", encoding="utf-8") as handle:
        paper_data = json.load(handle)

    grouped: dict[str, list[float]] = {}
    size_key = str(DATA_SIZE)
    for group_key, series_name in PAPER_SERIES_BY_GROUP.items():
        try:
            raw_values = paper_data[series_name][size_key]
        except KeyError as exc:
            raise ValueError(
                f"Missing paper MAE series {series_name!r} at data size {size_key} in {path}"
            ) from exc
        values = [float(value) for value in raw_values]
        if not values or not np.isfinite(np.asarray(values, dtype=float)).all():
            raise ValueError(
                f"Paper MAE series {series_name!r} at data size {size_key} is empty or non-finite"
            )
        grouped[group_key] = values
    return grouped


def resolve_grouped_mae(
    selection_rows: list[dict[str, str]],
    paper_mae_path: Path,
) -> tuple[dict[str, list[float]], dict[str, list[dict[str, str]]], dict[str, str]]:
    grouped_mae: dict[str, list[float]] = defaultdict(list)
    missing_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    metric_column_by_group: dict[str, str] = {}

    for group_key, values in load_paper_reference_mae(paper_mae_path).items():
        grouped_mae[group_key] = values
        metric_column_by_group[group_key] = (
            f"{PAPER_SERIES_BY_GROUP[group_key]}[{DATA_SIZE}] from {paper_mae_path}"
        )

    for row in selection_rows:
        group_key = row["group"]
        source_path = Path(row["source"])

        if group_key in PAPER_SERIES_BY_GROUP:
            continue

        if group_key == "l3n5_test":
            grouped_mae[group_key].append(lookup_test_mae(row))
            metric_column_by_group[group_key] = "test_T-MAE"
            continue

        if group_key == "l3n5_valid":
            grouped_mae[group_key].append(lookup_best_valid_mae(row))
            metric_column_by_group[group_key] = "valid_T-MAE (minimum per run)"
            continue

        missing_rows[group_key].append(row)

    return grouped_mae, missing_rows, metric_column_by_group


def annotate_missing_groups(ax: plt.Axes, missing_rows: dict[str, list[dict[str, str]]], grouped_mae: dict[str, list[float]]) -> None:
    missing_plot_groups = [group_key for group_key, _ in PLOT_ORDER if missing_rows.get(group_key)]
    if not missing_plot_groups:
        return

    available_values = [value for values in grouped_mae.values() for value in values]
    if not available_values:
        return

    y_min = min(available_values)
    y_max = max(available_values)
    y_range = max(y_max - y_min, 1e-6)
    note_y = y_max - 0.16 * y_range

    if {"l2n4_ref_test", "l3n4_ref_test"}.issubset(set(missing_plot_groups)):
        ax.text(
            (POSITIONS["l2n4_ref_test"] + POSITIONS["l3n4_ref_test"]) / 2.0,
            note_y,
            "Paper reference MAE unavailable\n(methane_results.json lacks MAE)",
            ha="center",
            va="top",
            fontsize=9,
            color="#555555",
        )
        remaining = [group_key for group_key in missing_plot_groups if group_key not in {"l2n4_ref_test", "l3n4_ref_test"}]
    else:
        remaining = missing_plot_groups

    for group_key in remaining:
        ax.text(
            POSITIONS[group_key],
            note_y,
            "MAE unavailable",
            ha="center",
            va="top",
            fontsize=9,
            color="#555555",
        )


def validate_group_values(grouped_mae: dict[str, list[float]]) -> None:
    for group_key, values in grouped_mae.items():
        if not values:
            raise ValueError(f"Selected MAE group is empty: {group_key}")
        if not np.isfinite(np.asarray(values, dtype=float)).all():
            raise ValueError(f"Selected MAE group contains non-finite values: {group_key}")


def main() -> None:
    args = parse_args()
    configure_matplotlib()

    selection_rows = load_selection_rows(args.selection)
    grouped_mae, missing_rows, metric_column_by_group = resolve_grouped_mae(
        selection_rows, args.paper_mae
    )
    validate_group_values(grouped_mae)

    if "l3n5_test" not in grouped_mae or "l3n5_valid" not in grouped_mae:
        raise ValueError("Expected l3n5 test and validation MAE groups to be available")

    plotted_groups = [(group_key, label) for group_key, label in PLOT_ORDER if group_key in grouped_mae]
    if not plotted_groups:
        raise ValueError("No MAE groups were available to plot")

    plotted_values = [grouped_mae[group_key] for group_key, _ in plotted_groups]
    plotted_positions = [POSITIONS[group_key] for group_key, _ in plotted_groups]

    fig, ax = plt.subplots(figsize=(6.8, 6), dpi=180)
    bp = ax.boxplot(
        plotted_values,
        positions=plotted_positions,
        widths=0.26,
        patch_artist=True,
        showmeans=False,
        showfliers=False,
        boxprops=dict(edgecolor="black", linewidth=1.2),
        whiskerprops=dict(color="black", linewidth=1.2),
        capprops=dict(color="black", linewidth=1.2),
        medianprops=dict(color="darkblue", linewidth=2),
    )

    for patch, (group_key, _) in zip(bp["boxes"], plotted_groups):
        patch.set_facecolor(color_for_group(group_key))
        patch.set_alpha(0.72)

    np.random.seed(42)
    for (group_key, _), position in zip(plotted_groups, plotted_positions):
        values = grouped_mae[group_key]
        x_values = np.ones(len(values)) * position + np.random.normal(0, 0.025, size=len(values))
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

    full_positions = [POSITIONS[group_key] for group_key, _ in PLOT_ORDER]
    full_tick_labels = [label for _, label in PLOT_ORDER]
    ax.set_xticks(full_positions)
    ax.set_xticklabels(full_tick_labels)
    ax.set_xlim(min(full_positions) - 0.4, max(full_positions) + 0.4)
    ax.set_ylabel("MAE (kcal/mol)")
    ax.set_title("1M Test and Validation MAE Distributions")
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    for x_position in SEPARATORS:
        ax.axvline(x_position, color="black", linewidth=0.8, alpha=0.10, zorder=0)

    for group_label, group_keys in GROUP_LABELS:
        center = sum(POSITIONS[group_key] for group_key in group_keys) / len(group_keys)
        ax.text(center, -0.065, group_label, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=11)

    annotate_missing_groups(ax, missing_rows, grouped_mae)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.145)
    fig.savefig(args.output, dpi=180, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    if not args.output.exists():
        raise RuntimeError(f"Expected output image was not created: {args.output}")

    print(f"Saved {args.output}")
    print(f"Selection manifest: {args.selection}")
    print("MAE columns used:")
    for group_key in EXPECTED_GROUPS:
        if group_key in metric_column_by_group:
            print(f"  {group_key}: {metric_column_by_group[group_key]}")
        elif missing_rows.get(group_key):
            print(f"  {group_key}: unavailable (source lacks MAE)")

    print("\nPlotted observation counts:")
    for group_key, _ in PLOT_ORDER:
        if group_key in grouped_mae:
            values = np.asarray(grouped_mae[group_key], dtype=float)
            std = values.std(ddof=1) if len(values) > 1 else 0.0
            print(f"  {group_key}: N={len(values)}, mean={values.mean():.5f}, std={std:.5f}")
        elif missing_rows.get(group_key):
            print(f"  {group_key}: N=0 plotted, {len(missing_rows[group_key])} source rows missing MAE")

    if any(missing_rows.get(group_key) for group_key, _ in PLOT_ORDER):
        print("\nMissing MAE details:")
        for group_key, _ in PLOT_ORDER:
            for row in missing_rows.get(group_key, []):
                print(
                    f"  {group_key}: seed={row['seed']} run={row['run']} source={row['source']}"
                )


if __name__ == "__main__":
    main()
