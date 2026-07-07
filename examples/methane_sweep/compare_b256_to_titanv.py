#!/usr/bin/env python3
"""Compare A100 l3/b256 methane runs against TitanV l3/b256 runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


INPUTS = [
    ("A100_b256", Path("logs/metric_plots_l3_b256")),
    ("titanv_b256", Path("logs/metric_plots_titanv_l3_b256")),
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
        default=Path("logs/metric_plots_b256_vs_titanv"),
        type=Path,
        help="Directory for comparison CSVs and plots, relative to --base-dir unless absolute.",
    )
    return parser.parse_args()


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


def dataset_label(data_size: int) -> str:
    if data_size == 100000:
        return "100k"
    if data_size == 1000000:
        return "1M"
    return f"{data_size:,}"


def run_key(row: dict[str, object]) -> str:
    return f"{row['family']}-dataset={dataset_label(int(row['data_size']))}"


def run_key_color(row: dict[str, object]) -> str:
    colors = {
        "A100_b256-dataset=100k": "#56B4E9",
        "A100_b256-dataset=1M": "#0072B2",
        "titanv_b256-dataset=100k": "#E69F00",
        "titanv_b256-dataset=1M": "#D55E00",
    }
    return colors[run_key(row)]


def run_key_zorder(row: dict[str, object]) -> int:
    return 3 if int(row["data_size"]) == 100000 else 2


def load_best_metrics(base_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for family, relative in INPUTS:
        for row in read_csv(family_path(base_dir, relative) / "best_metric_pareto.csv"):
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
    int_columns = {"sweep_task_id", "seed", "hiphop_l_max", "hiphop_n_max", "data_size", "total_params", "epoch", "batch_size"}
    float_columns = {"training_time_s", "total_epoch_time_s", "valid_F-MAE", "valid_T-MAE", "best_valid_T-MAE_so_far"}
    for family, relative in INPUTS:
        for row in read_csv(family_path(base_dir, relative) / "metrics.csv"):
            row["family"] = family
            for key in int_columns:
                if row.get(key) not in ("", None):
                    row[key] = int(row[key])
            for key in float_columns:
                if row.get(key) not in ("", None):
                    row[key] = float(row[key])
            rows.append(row)
    return rows


def plot_best_metric_scatter(rows: list[dict[str, object]], output: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    markers = {(3, 3): "o", (3, 4): "s"}
    for ax, metric, time_key, ylabel, title in [
        (axes[0], "best_valid_F-MAE", "best_force_training_time_h", "Best validation force MAE", "Force"),
        (axes[1], "best_valid_T-MAE", "best_energy_training_time_h", "Best validation energy MAE", "Energy"),
    ]:
        for row in sorted(rows, key=lambda item: (int(item["data_size"]), str(item["family"]), int(item["hiphop_n_max"]))):
            ax.scatter(
                float(row[time_key]),
                float(row[metric]),
                s=90,
                marker=markers.get((int(row["hiphop_l_max"]), int(row["hiphop_n_max"])), "o"),
                color=run_key_color(row),
                edgecolor="#202020",
                linewidth=0.6,
                zorder=run_key_zorder(row),
            )
            ax.annotate(
                f"{run_key(row)} l{row['hiphop_l_max']}n{row['hiphop_n_max']}",
                (float(row[time_key]), float(row[metric])),
                xytext=(6, 5),
                textcoords="offset points",
                fontsize=8,
            )
        ax.set_xlabel("Training time to best checkpoint (hours)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}: accuracy vs compute time")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda value, _: f"{value:g}"))
        ax.grid(True, alpha=0.25)

    legend_rows = [
        {"family": "A100_b256", "data_size": 100000},
        {"family": "A100_b256", "data_size": 1000000},
        {"family": "titanv_b256", "data_size": 100000},
        {"family": "titanv_b256", "data_size": 1000000},
    ]
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=run_key_color(row), label=run_key(row), markersize=8) for row in legend_rows]
    fig.legend(handles=handles, loc="upper center", ncol=2)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_matched_l3n3_bar(rows: list[dict[str, object]], output: Path) -> None:
    import matplotlib.pyplot as plt

    matched = [row for row in rows if int(row["hiphop_l_max"]) == 3 and int(row["hiphop_n_max"]) == 3]
    matched = sorted(matched, key=lambda item: (int(item["data_size"]), str(item["family"])))
    labels = [run_key(row) for row in matched]
    colors = [run_key_color(row) for row in matched]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    x = list(range(len(labels)))
    for ax, values, times, title, ylabel in [
        (
            axes[0],
            [float(row["best_valid_F-MAE"]) for row in matched],
            [float(row["best_force_training_time_h"]) for row in matched],
            "Matched l=3 n=3 force error",
            "Best validation force MAE",
        ),
        (
            axes[1],
            [float(row["best_valid_T-MAE"]) for row in matched],
            [float(row["best_energy_training_time_h"]) for row in matched],
            "Matched l=3 n=3 energy error",
            "Best validation energy MAE",
        ),
    ]:
        bars = ax.bar(x, values, color=colors)
        for bar, row in zip(bars, matched):
            bar.set_zorder(run_key_zorder(row))
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        for index, hours in enumerate(times):
            ax.annotate(f"{hours:.1f} h", (index, values[index]), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
        ax.grid(True, axis="y", alpha=0.25)

    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_training_curves(rows: list[dict[str, object]], output: Path, by_compute_time: bool) -> None:
    import matplotlib.pyplot as plt

    selected = [
        row
        for row in rows
        if int(row.get("hiphop_l_max", -1)) == 3
        and int(row.get("hiphop_n_max", -1)) == 3
        and row.get("valid_F-MAE") not in ("", None)
        and row.get("valid_T-MAE") not in ("", None)
        and (not by_compute_time or row.get("training_time_s") not in ("", None))
    ]
    grouped: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in selected:
        grouped.setdefault((str(row["family"]), int(row["data_size"])), []).append(row)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), constrained_layout=True)
    for (family, data_size), group_rows in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        group_rows = sorted(group_rows, key=lambda item: int(item["epoch"]))
        label_row = {"family": family, "data_size": data_size}
        if by_compute_time:
            total_time_s = 0.0
            x_values = []
            for row in group_rows:
                total_time_s += float(row["training_time_s"])
                x_values.append(total_time_s / 3600)
        else:
            x_values = [int(row["epoch"]) for row in group_rows]
        for ax, metric in [(axes[0], "valid_F-MAE"), (axes[1], "valid_T-MAE")]:
            ax.plot(
                x_values,
                [float(row[metric]) for row in group_rows],
                label=run_key(label_row),
                color=run_key_color(label_row),
                linewidth=1.5,
                zorder=run_key_zorder(label_row),
            )

    axes[0].set_title("l=3 n=3 validation force MAE")
    axes[0].set_ylabel("Force MAE")
    axes[1].set_title("l=3 n=3 validation energy MAE")
    axes[1].set_ylabel("Energy MAE")
    for ax in axes:
        ax.set_xlabel("Cumulative training time (hours)" if by_compute_time else "Epoch")
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
    for data_size in [100000, 1000000]:
        regular = by_key.get(("A100_b256", data_size))
        titanv = by_key.get(("titanv_b256", data_size))
        if not regular or not titanv:
            continue
        comparisons.append(
            {
                "data_size": data_size,
                "b256_force_mae": regular["best_valid_F-MAE"],
                "titanv_force_mae": titanv["best_valid_F-MAE"],
                "force_mae_delta_titanv_minus_b256": float(titanv["best_valid_F-MAE"]) - float(regular["best_valid_F-MAE"]),
                "force_mae_ratio_titanv_over_b256": float(titanv["best_valid_F-MAE"]) / float(regular["best_valid_F-MAE"]),
                "b256_energy_mae": regular["best_valid_T-MAE"],
                "titanv_energy_mae": titanv["best_valid_T-MAE"],
                "energy_mae_delta_titanv_minus_b256": float(titanv["best_valid_T-MAE"]) - float(regular["best_valid_T-MAE"]),
                "energy_mae_ratio_titanv_over_b256": float(titanv["best_valid_T-MAE"]) / float(regular["best_valid_T-MAE"]),
                "b256_force_time_h": regular["best_force_training_time_h"],
                "titanv_force_time_h": titanv["best_force_training_time_h"],
                "force_time_ratio_titanv_over_b256": float(titanv["best_force_training_time_h"]) / float(regular["best_force_training_time_h"]),
                "b256_energy_time_h": regular["best_energy_training_time_h"],
                "titanv_energy_time_h": titanv["best_energy_training_time_h"],
                "energy_time_ratio_titanv_over_b256": float(titanv["best_energy_training_time_h"]) / float(regular["best_energy_training_time_h"]),
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
        "b256_force_mae",
        "titanv_force_mae",
        "force_mae_delta_titanv_minus_b256",
        "force_mae_ratio_titanv_over_b256",
        "b256_energy_mae",
        "titanv_energy_mae",
        "energy_mae_delta_titanv_minus_b256",
        "energy_mae_ratio_titanv_over_b256",
        "b256_force_time_h",
        "titanv_force_time_h",
        "force_time_ratio_titanv_over_b256",
        "b256_energy_time_h",
        "titanv_energy_time_h",
        "energy_time_ratio_titanv_over_b256",
    ]
    write_csv(output_dir / "best_metrics_combined.csv", best_rows, best_columns)
    write_csv(output_dir / "matched_l3n3_b256_vs_titanv.csv", comparison_rows, comparison_columns)
    plot_best_metric_scatter(best_rows, output_dir / "best_metric_scatter.png")
    plot_matched_l3n3_bar(best_rows, output_dir / "matched_l3n3_bar.png")
    plot_training_curves(metric_rows, output_dir / "matched_l3n3_training_curves.png", by_compute_time=False)
    plot_training_curves(metric_rows, output_dir / "matched_l3n3_training_curves_compute_time.png", by_compute_time=True)


if __name__ == "__main__":
    main()
