#!/usr/bin/env python3
"""Recreate the methane single-molecule paper-style error plot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ENERGY_STD_KCAL_MOL = 100.0

SERIES = [
    ("Hipnn", r"HIP-NN ($\ell=0, n=1$)", "#4c4c4c"),
    ("HipnnVec", r"HIP-NN-TS ($\ell=1, n=2$)", "#f2c14e"),
    ("HipnnQuad", r"HIP-NN-TS ($\ell=2, n=2$)", "#56b4e9"),
    ("HipHop_l2_n3", r"HIP-HOP-NN($\ell=2, n=3$)", "#e07a7a"),
    ("HipHop_l2_n4", r"HIP-HOP-NN($\ell=2, n=4$)", "#d99ab5"),
    ("HipHop", r"HIP-HOP-NN($\ell=3, n=4$)", "#25b99a"),
]

TRAINING_RESULT_CSVS = [
    Path("examples/methane_sweep/logs/metric_plots/paper_style_energy_comparison.csv"),
    Path("examples/methane_sweep/logs/metric_plots_l3_b256/paper_style_energy_comparison.csv"),
    Path("examples/methane_sweep/logs/metric_plots_titanv_l3_b256/paper_style_energy_comparison.csv"),
]

METRICS_BY_RESULT_CSV = {
    Path("examples/methane_sweep/logs/metric_plots/paper_style_energy_comparison.csv"): Path(
        "examples/methane_sweep/logs/metric_plots/metrics.csv"
    ),
    Path("examples/methane_sweep/logs/metric_plots_l3_b256/paper_style_energy_comparison.csv"): Path(
        "examples/methane_sweep/logs/metric_plots_l3_b256/metrics.csv"
    ),
    Path("examples/methane_sweep/logs/metric_plots_titanv_l3_b256/paper_style_energy_comparison.csv"): Path(
        "examples/methane_sweep/logs/metric_plots_titanv_l3_b256/metrics.csv"
    ),
}

ACCELERATOR_BY_RESULT_CSV = {
    Path("examples/methane_sweep/logs/metric_plots/paper_style_energy_comparison.csv"): "A100",
    Path("examples/methane_sweep/logs/metric_plots_l3_b256/paper_style_energy_comparison.csv"): "A100",
    Path("examples/methane_sweep/logs/metric_plots_titanv_l3_b256/paper_style_energy_comparison.csv"): "TitanV",
}

BASIS_COLORS = {
    (3, 3): "#0b7285",
    (3, 4): "#862e9c",
    (4, 3): "#9b2226",
    (4, 4): "#f08c00",
}

ACCELERATOR_MARKERS = {
    "A100": "*",
    "TitanV": "X",
}

ACCELERATOR_MARKER_SIZES = {
    "A100": 105,
    "TitanV": 60,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=Path("examples/methane_sweep/results/paper/methane_results.json"),
        type=Path,
        help="JSON file containing methane RMSE replicate results.",
    )
    parser.add_argument(
        "--output",
        default=Path("examples/methane_sweep/results/paper/paper_methane_plot_batch256.png"),
        type=Path,
        help="Output image path for the batch_size=256 overlay.",
    )
    parser.add_argument(
        "--other-output",
        default=Path("examples/methane_sweep/results/paper/paper_methane_plot_batch16k_plus.png"),
        type=Path,
        help="Output image path for non-256 batch-size overlays.",
    )
    parser.add_argument(
        "--training-csv",
        action="append",
        type=Path,
        dest="training_csvs",
        help="Paper-style training-result CSV to overlay. Can be passed more than once.",
    )
    parser.add_argument(
        "--training-summary-output",
        default=Path("examples/methane_sweep/results/paper/paper_methane_plot_overlay_points.csv"),
        type=Path,
        help="CSV recording the selected best overlay points.",
    )
    return parser.parse_args()


def best_training_points(paths: list[Path]) -> list[dict[str, str | int | float]]:
    best: dict[tuple[str, str, int, int, int], dict[str, str | int | float]] = {}
    for path in paths:
        if not path.exists():
            continue
        accelerator = ACCELERATOR_BY_RESULT_CSV.get(path, path.parent.name)
        batch_sizes = batch_sizes_by_run(METRICS_BY_RESULT_CSV.get(path))
        with path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if not row.get("best_valid_T-RMSE"):
                    continue
                l_max = int(row["hiphop_l_max"])
                n_max = int(row["hiphop_n_max"])
                batch_size = batch_sizes.get(row["run"], row.get("batch_size") or "")
                training_size = int(row["training_set_size"])
                rmse = float(row["best_valid_T-RMSE"])
                key = (batch_size, accelerator, l_max, n_max, training_size)
                candidate = {
                    "hiphop_l_max": l_max,
                    "hiphop_n_max": n_max,
                    "batch_size": batch_size,
                    "accelerator": accelerator,
                    "training_set_size": training_size,
                    "best_valid_T-RMSE": rmse,
                    "normalized_energy_RMSE": rmse / ENERGY_STD_KCAL_MOL,
                    "source_csv": str(path),
                    "run": row["run"],
                    "sweep_task_id": row["sweep_task_id"],
                    "logged_epochs": int(row["logged_epochs"]),
                    "best_epoch": int(row["best_epoch"]),
                }
                if key not in best or rmse < float(best[key]["best_valid_T-RMSE"]):
                    best[key] = candidate
    return [
        best[key]
        for key in sorted(best, key=lambda item: (batch_sort_key(item[0]), item[1], item[2], item[3], item[4]))
    ]


def batch_sort_key(batch_size: str) -> int:
    try:
        return int(batch_size)
    except ValueError:
        return 10**12


def batch_sizes_by_run(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    batch_sizes: dict[str, str] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("run") and row.get("batch_size"):
                batch_sizes.setdefault(row["run"], row["batch_size"])
    return batch_sizes


def write_overlay_summary(path: Path, points: list[dict[str, str | int | float]]) -> None:
    if not points:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "hiphop_l_max",
        "hiphop_n_max",
        "batch_size",
        "accelerator",
        "training_set_size",
        "best_valid_T-RMSE",
        "normalized_energy_RMSE",
        "source_csv",
        "run",
        "sweep_task_id",
        "logged_epochs",
        "best_epoch",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(points)


def configure_matplotlib() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "axes.linewidth": 1.1,
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 8,
            "lines.linewidth": 1.5,
        }
    )


def plot_results(
    data: dict[str, dict[str, list[float]]],
    training_points: list[dict[str, str | int | float]],
    output: Path,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    fig, ax = plt.subplots(figsize=(10.8, 3.7), dpi=180)

    for key, label, color in SERIES:
        series = data[key]
        sizes = np.array(sorted(int(size) for size in series), dtype=float)
        values = [np.array(series[str(int(size))], dtype=float) for size in sizes]
        means = np.array([sample.mean() for sample in values]) / ENERGY_STD_KCAL_MOL
        stderr = np.array([sample.std(ddof=1) / np.sqrt(sample.size) for sample in values]) / ENERGY_STD_KCAL_MOL

        ax.errorbar(
            sizes,
            means,
            yerr=stderr,
            color=color,
            marker="_",
            markersize=7,
            markeredgewidth=1.15,
            capsize=3.0,
            elinewidth=1.0,
            linestyle="-",
            label=label,
            alpha=0.95,
        )

    for point in training_points:
        l_max = int(point["hiphop_l_max"])
        n_max = int(point["hiphop_n_max"])
        accelerator = str(point.get("accelerator") or "")
        color = BASIS_COLORS.get((l_max, n_max), "#212529")
        ax.scatter(
            float(point["training_set_size"]),
            float(point["normalized_energy_RMSE"]),
            marker=ACCELERATOR_MARKERS.get(accelerator, "*"),
            s=ACCELERATOR_MARKER_SIZES.get(accelerator, 95),
            color=color,
            edgecolor="black",
            linewidth=0.45,
            zorder=8,
        )

    if training_points:
        handles = []
        labels = []
        for accelerator, l_max, n_max, batch_size in sorted(
            {
                (
                    str(point.get("accelerator") or ""),
                    int(point["hiphop_l_max"]),
                    int(point["hiphop_n_max"]),
                    str(point.get("batch_size") or "mixed"),
                )
                for point in training_points
            }
        ):
            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker=ACCELERATOR_MARKERS.get(accelerator, "*"),
                    linestyle="",
                    color=BASIS_COLORS.get((l_max, n_max), "#212529"),
                    markeredgecolor="black",
                    markeredgewidth=0.45,
                    markersize=8 if accelerator == "TitanV" else 9,
                )
            )
            label = rf"{accelerator}: $\ell={l_max}, n={n_max}$, batch_size={batch_size}"
            if (l_max, n_max) == (4, 3):
                label += " (run ended early)"
            labels.append(label)
    else:
        handles = []
        labels = []

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(7.0e2, 4.2e6)
    ax.set_ylim(2.6e-3, 2.55e-1)
    ax.set_xlabel("Training set size")
    ax.set_ylabel("Energy RMSE/STD")

    ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=5))
    ax.xaxis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=40))
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=4))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=40))

    secondary = ax.secondary_yaxis(
        "right",
        functions=(
            lambda normalized: normalized * ENERGY_STD_KCAL_MOL,
            lambda kcal: kcal / ENERGY_STD_KCAL_MOL,
        ),
    )
    secondary.set_yscale("log")
    secondary.set_ylabel("Energy RMSE (kcal/mol)")
    secondary.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=4))
    secondary.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=40))

    paper_handles, paper_labels = ax.get_legend_handles_labels()
    legend = fig.legend(
        paper_handles + handles,
        paper_labels + labels,
        loc="upper left",
        bbox_to_anchor=(0.66, 0.92),
        frameon=True,
        fancybox=False,
        borderpad=0.45,
        handlelength=2.0,
    )
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_edgecolor("#dddddd")

    fig.subplots_adjust(left=0.085, right=0.55, bottom=0.22, top=0.95)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, pad_inches=0.02)
    plt.close(fig)


def is_batch_256(point: dict[str, str | int | float]) -> bool:
    return str(point.get("batch_size", "")) == "256"


def include_overlay_point(point: dict[str, str | int | float]) -> bool:
    return (int(point["hiphop_l_max"]), int(point["hiphop_n_max"])) != (4, 4)


def main() -> None:
    args = parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    training_csvs = args.training_csvs or TRAINING_RESULT_CSVS
    training_points = best_training_points(training_csvs)
    plotted_points = [point for point in training_points if include_overlay_point(point)]
    batch256_points = [point for point in plotted_points if is_batch_256(point)]
    other_points = [point for point in plotted_points if not is_batch_256(point)]

    configure_matplotlib()
    plot_results(data, batch256_points, args.output)
    plot_results(data, other_points, args.other_output)
    write_overlay_summary(args.training_summary_output, training_points)


if __name__ == "__main__":
    main()
