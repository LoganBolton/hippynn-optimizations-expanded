#!/usr/bin/env python3
"""Compare the l3/b256 methane runs against larger sweep outputs."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


DEFAULT_INPUTS = [
    ("l3_b256", Path("logs/metric_plots_l3_b256")),
    ("sweep_100k", Path("logs/metric_plots_data_size_100000")),
    ("sweep_1m", Path("logs/metric_plots_data_size_1000000")),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        default=Path("examples/methane_sweep"),
        type=Path,
        help="Methane sweep directory. Relative input paths are resolved from here.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("logs/metric_plots_b256_vs_big_runs"),
        type=Path,
        help="Directory for comparison CSVs and plots, relative to --base-dir unless absolute.",
    )
    return parser.parse_args()


def to_float(value: str | int | float | None) -> float | None:
    if value in ("", None):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def family_path(base_dir: Path, relative: Path) -> Path:
    return relative if relative.is_absolute() else base_dir / relative


def load_best_metrics(base_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for family, relative in DEFAULT_INPUTS:
        path = family_path(base_dir, relative) / "best_metric_pareto.csv"
        for row in read_csv(path):
            rows.append(
                {
                    "family": family,
                    "data_size": int(row["data_size"]),
                    "hiphop_l_max": int(row["hiphop_l_max"]),
                    "hiphop_n_max": int(row["hiphop_n_max"]),
                    "force_seed": int(row["force_seed"]),
                    "force_total_params": int(row["force_total_params"]),
                    "force_logged_epochs": int(row["force_logged_epochs"]),
                    "best_force_epoch": int(row["best_force_epoch"]),
                    "best_force_training_time_h": float(row["best_force_training_time_h"]),
                    "best_valid_F-MAE": float(row["best_valid_F-MAE"]),
                    "energy_seed": int(row["energy_seed"]),
                    "energy_total_params": int(row["energy_total_params"]),
                    "energy_logged_epochs": int(row["energy_logged_epochs"]),
                    "best_energy_epoch": int(row["best_energy_epoch"]),
                    "best_energy_training_time_h": float(row["best_energy_training_time_h"]),
                    "best_valid_T-MAE": float(row["best_valid_T-MAE"]),
                    "force_run": row["force_run"],
                    "energy_run": row["energy_run"],
                }
            )
    return rows


def load_metric_rows(base_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for family, relative in DEFAULT_INPUTS:
        path = family_path(base_dir, relative) / "metrics.csv"
        for row in read_csv(path):
            row["family"] = family
            for key in [
                "sweep_task_id",
                "seed",
                "hiphop_l_max",
                "hiphop_n_max",
                "data_size",
                "total_params",
                "epoch",
                "batch_size",
            ]:
                if row.get(key) not in ("", None):
                    row[key] = int(row[key])
            for key in [
                "training_time_s",
                "total_epoch_time_s",
                "valid_F-MAE",
                "valid_T-MAE",
                "best_valid_T-MAE_so_far",
            ]:
                if row.get(key) not in ("", None):
                    row[key] = float(row[key])
            rows.append(row)
    return rows


def model_label(row: dict[str, object]) -> str:
    return f"{row['family']} l={row['hiphop_l_max']} n={row['hiphop_n_max']}"


def dataset_label(data_size: int) -> str:
    if data_size == 100000:
        return "100k"
    if data_size == 1000000:
        return "1M"
    return f"{data_size:,}"


def run_key(row: dict[str, object]) -> str:
    batch_size = "256" if str(row["family"]) == "l3_b256" else "16k"
    return f"batch_size={batch_size}-dataset={dataset_label(int(row['data_size']))}"


def run_key_color(row: dict[str, object]) -> str:
    colors = {
        "batch_size=256-dataset=100k": "#56B4E9",
        "batch_size=256-dataset=1M": "#0072B2",
        "batch_size=16k-dataset=100k": "#E69F00",
        "batch_size=16k-dataset=1M": "#D55E00",
    }
    return colors[run_key(row)]


def run_key_zorder(row: dict[str, object]) -> int:
    return 3 if int(row["data_size"]) == 100000 else 2


def plot_best_metric_comparison(rows: list[dict[str, object]], output: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    markers = {(3, 3): "o", (3, 4): "s", (4, 3): "^", (4, 4): "D"}

    for ax, metric, time_key, ylabel, title in [
        (axes[0], "best_valid_F-MAE", "best_force_training_time_h", "Best validation force MAE", "Force"),
        (axes[1], "best_valid_T-MAE", "best_energy_training_time_h", "Best validation energy MAE", "Energy"),
    ]:
        for row in sorted(rows, key=lambda item: (int(item["data_size"]), str(item["family"]), int(item["hiphop_l_max"]), int(item["hiphop_n_max"]))):
            marker = markers.get((int(row["hiphop_l_max"]), int(row["hiphop_n_max"])), "o")
            ax.scatter(
                float(row[time_key]),
                float(row[metric]),
                s=90,
                marker=marker,
                color=run_key_color(row),
                edgecolor="#202020",
                linewidth=0.6,
                zorder=run_key_zorder(row),
            )
            ax.annotate(
                f"{int(row['data_size']) // 1000}k l{row['hiphop_l_max']}n{row['hiphop_n_max']}",
                (float(row[time_key]), float(row[metric])),
                xytext=(6, 5),
                textcoords="offset points",
                fontsize=8,
            )
        ax.set_xscale("log")
        ax.set_xlabel("Training time to best checkpoint (hours)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}: accuracy vs time")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda value, _: f"{value:g}"))
        ax.grid(True, alpha=0.25)

    legend_rows = [
        {"family": "l3_b256", "data_size": 100000},
        {"family": "l3_b256", "data_size": 1000000},
        {"family": "sweep_100k", "data_size": 100000},
        {"family": "sweep_1m", "data_size": 1000000},
    ]
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=run_key_color(row), label=run_key(row), markersize=8) for row in legend_rows]
    fig.legend(handles=handles, loc="upper center", ncol=3)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_matched_l3n3(rows: list[dict[str, object]], output: Path) -> None:
    import matplotlib.pyplot as plt

    matched = [
        row
        for row in rows
        if int(row["hiphop_l_max"]) == 3
        and int(row["hiphop_n_max"]) == 3
        and str(row["family"]) in {"l3_b256", "sweep_100k", "sweep_1m"}
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    x_labels = []
    colors = []
    force_values = []
    energy_values = []
    force_times = []
    energy_times = []
    for row in sorted(matched, key=lambda item: (int(item["data_size"]), str(item["family"]))):
        x_labels.append(run_key(row))
        colors.append(run_key_color(row))
        force_values.append(float(row["best_valid_F-MAE"]))
        energy_values.append(float(row["best_valid_T-MAE"]))
        force_times.append(float(row["best_force_training_time_h"]))
        energy_times.append(float(row["best_energy_training_time_h"]))

    x = list(range(len(x_labels)))
    bars = axes[0].bar(x, force_values, color=colors)
    for bar, row in zip(bars, sorted(matched, key=lambda item: (int(item["data_size"]), str(item["family"])))):
        bar.set_zorder(run_key_zorder(row))
    axes[0].set_ylabel("Best validation force MAE")
    axes[0].set_title("Matched l=3 n=3 force error")
    bars = axes[1].bar(x, energy_values, color=colors)
    for bar, row in zip(bars, sorted(matched, key=lambda item: (int(item["data_size"]), str(item["family"])))):
        bar.set_zorder(run_key_zorder(row))
    axes[1].set_ylabel("Best validation energy MAE")
    axes[1].set_title("Matched l=3 n=3 energy error")
    for ax, times in [(axes[0], force_times), (axes[1], energy_times)]:
        ax.set_xticks(x, x_labels, rotation=25, ha="right")
        for index, hours in enumerate(times):
            ax.annotate(f"{hours:.1f} h", (index, ax.patches[index].get_height()), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
        ax.grid(True, axis="y", alpha=0.25)

    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_training_curves(rows: list[dict[str, object]], output: Path) -> None:
    import matplotlib.pyplot as plt

    selected = [
        row
        for row in rows
        if int(row.get("hiphop_l_max", -1)) == 3
        and int(row.get("hiphop_n_max", -1)) == 3
        and row.get("valid_F-MAE") not in ("", None)
        and row.get("valid_T-MAE") not in ("", None)
    ]
    grouped: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in selected:
        grouped.setdefault((str(row["family"]), int(row["data_size"])), []).append(row)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), constrained_layout=True)
    for (family, data_size), group_rows in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        group_rows = sorted(group_rows, key=lambda item: int(item["epoch"]))
        label_row = {"family": family, "data_size": data_size}
        label = run_key(label_row)
        epochs = [int(row["epoch"]) for row in group_rows]
        axes[0].plot(
            epochs,
            [float(row["valid_F-MAE"]) for row in group_rows],
            label=label,
            color=run_key_color(label_row),
            linewidth=1.5,
            zorder=run_key_zorder(label_row),
        )
        axes[1].plot(
            epochs,
            [float(row["valid_T-MAE"]) for row in group_rows],
            label=label,
            color=run_key_color(label_row),
            linewidth=1.5,
            zorder=run_key_zorder(label_row),
        )

    axes[0].set_title("l=3 n=3 validation force MAE")
    axes[0].set_ylabel("Force MAE")
    axes[1].set_title("l=3 n=3 validation energy MAE")
    axes[1].set_ylabel("Energy MAE")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_training_curves_by_compute_time(rows: list[dict[str, object]], output: Path) -> None:
    import matplotlib.pyplot as plt

    selected = [
        row
        for row in rows
        if int(row.get("hiphop_l_max", -1)) == 3
        and int(row.get("hiphop_n_max", -1)) == 3
        and row.get("training_time_s") not in ("", None)
        and row.get("valid_F-MAE") not in ("", None)
        and row.get("valid_T-MAE") not in ("", None)
    ]
    grouped: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in selected:
        grouped.setdefault((str(row["family"]), int(row["data_size"])), []).append(row)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), constrained_layout=True)
    for (family, data_size), group_rows in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        group_rows = sorted(group_rows, key=lambda item: int(item["epoch"]))
        label_row = {"family": family, "data_size": data_size}
        label = run_key(label_row)
        compute_time_h = []
        total_time_s = 0.0
        for row in group_rows:
            total_time_s += float(row["training_time_s"])
            compute_time_h.append(total_time_s / 3600)
        axes[0].plot(
            compute_time_h,
            [float(row["valid_F-MAE"]) for row in group_rows],
            label=label,
            color=run_key_color(label_row),
            linewidth=1.5,
            zorder=run_key_zorder(label_row),
        )
        axes[1].plot(
            compute_time_h,
            [float(row["valid_T-MAE"]) for row in group_rows],
            label=label,
            color=run_key_color(label_row),
            linewidth=1.5,
            zorder=run_key_zorder(label_row),
        )

    axes[0].set_title("l=3 n=3 validation force MAE")
    axes[0].set_ylabel("Force MAE")
    axes[1].set_title("l=3 n=3 validation energy MAE")
    axes[1].set_ylabel("Energy MAE")
    for ax in axes:
        ax.set_xlabel("Cumulative training time (hours)")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def matched_l3n3_deltas(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_key = {
        (str(row["family"]), int(row["data_size"])): row
        for row in rows
        if int(row["hiphop_l_max"]) == 3 and int(row["hiphop_n_max"]) == 3
    }
    comparisons: list[dict[str, object]] = []
    for data_size, baseline_family in [(100000, "sweep_100k"), (1000000, "sweep_1m")]:
        b256 = by_key.get(("l3_b256", data_size))
        baseline = by_key.get((baseline_family, data_size))
        if not b256 or not baseline:
            continue
        comparisons.append(
            {
                "data_size": data_size,
                "baseline_family": baseline_family,
                "b256_force_mae": b256["best_valid_F-MAE"],
                "baseline_force_mae": baseline["best_valid_F-MAE"],
                "force_mae_delta": float(b256["best_valid_F-MAE"]) - float(baseline["best_valid_F-MAE"]),
                "force_mae_ratio": float(b256["best_valid_F-MAE"]) / float(baseline["best_valid_F-MAE"]),
                "b256_energy_mae": b256["best_valid_T-MAE"],
                "baseline_energy_mae": baseline["best_valid_T-MAE"],
                "energy_mae_delta": float(b256["best_valid_T-MAE"]) - float(baseline["best_valid_T-MAE"]),
                "energy_mae_ratio": float(b256["best_valid_T-MAE"]) / float(baseline["best_valid_T-MAE"]),
                "b256_force_time_h": b256["best_force_training_time_h"],
                "baseline_force_time_h": baseline["best_force_training_time_h"],
                "force_time_ratio": float(b256["best_force_training_time_h"]) / float(baseline["best_force_training_time_h"]),
                "b256_energy_time_h": b256["best_energy_training_time_h"],
                "baseline_energy_time_h": baseline["best_energy_training_time_h"],
                "energy_time_ratio": float(b256["best_energy_training_time_h"]) / float(baseline["best_energy_training_time_h"]),
            }
        )
    return comparisons


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else base_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    best_rows = load_best_metrics(base_dir)
    metric_rows = load_metric_rows(base_dir)
    comparison_rows = matched_l3n3_deltas(best_rows)

    best_columns = [
        "family",
        "data_size",
        "hiphop_l_max",
        "hiphop_n_max",
        "force_seed",
        "force_total_params",
        "force_logged_epochs",
        "best_force_epoch",
        "best_force_training_time_h",
        "best_valid_F-MAE",
        "energy_seed",
        "energy_total_params",
        "energy_logged_epochs",
        "best_energy_epoch",
        "best_energy_training_time_h",
        "best_valid_T-MAE",
        "force_run",
        "energy_run",
    ]
    comparison_columns = [
        "data_size",
        "baseline_family",
        "b256_force_mae",
        "baseline_force_mae",
        "force_mae_delta",
        "force_mae_ratio",
        "b256_energy_mae",
        "baseline_energy_mae",
        "energy_mae_delta",
        "energy_mae_ratio",
        "b256_force_time_h",
        "baseline_force_time_h",
        "force_time_ratio",
        "b256_energy_time_h",
        "baseline_energy_time_h",
        "energy_time_ratio",
    ]
    write_csv(output_dir / "best_metrics_combined.csv", best_rows, best_columns)
    write_csv(output_dir / "matched_l3n3_b256_vs_baseline.csv", comparison_rows, comparison_columns)
    plot_best_metric_comparison(best_rows, output_dir / "best_metric_scatter.png")
    plot_matched_l3n3(best_rows, output_dir / "matched_l3n3_bar.png")
    plot_training_curves(metric_rows, output_dir / "matched_l3n3_training_curves.png")
    plot_training_curves_by_compute_time(metric_rows, output_dir / "matched_l3n3_training_curves_compute_time.png")


if __name__ == "__main__":
    main()
