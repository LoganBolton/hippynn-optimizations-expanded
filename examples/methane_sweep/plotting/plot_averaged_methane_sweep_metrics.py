#!/usr/bin/env python3
"""Average methane sweep metrics across runs for each architecture and plot them together."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

from plot_methane_sweep_logs import (
    apply_metric_limits,
    apply_sparse_y_axis,
    metric_title,
    model_plot_style,
    should_use_log_y,
    write_csv,
)


NUMERIC_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?$")
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        action="append",
        required=True,
        type=Path,
        help="Existing plot directory containing a metrics.csv file. May be passed multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where averaged CSVs and plots will be written.",
    )
    parser.add_argument(
        "--exclude-run",
        action="append",
        default=[],
        help="Exact run label to exclude before averaging. May be passed multiple times.",
    )
    return parser.parse_args()


def is_number(value: str) -> bool:
    return bool(NUMERIC_RE.fullmatch(value.strip()))


def read_metrics(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def config_label(row: dict[str, str]) -> str:
    data_size = int(row["data_size"])
    if data_size == 1_000_000:
        size_label = "1M"
    elif data_size == 100_000:
        size_label = "100k"
    else:
        size_label = f"{data_size:,}"
    return f"l={row['hiphop_l_max']} n={row['hiphop_n_max']} d={size_label} avg"


def config_key(row: dict[str, str]) -> tuple[int, int, int]:
    return (int(row["hiphop_l_max"]), int(row["hiphop_n_max"]), int(row["data_size"]))


def average_rows(rows: list[dict[str, str]]) -> list[dict[str, float | int | str]]:
    rows_by_epoch: dict[int, list[dict[str, str]]] = defaultdict(list)
    rows_by_run: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("epoch", "") == "":
            continue
        epoch = int(row["epoch"])
        rows_by_epoch[epoch].append(row)
        run_name = row.get("run", "")
        if run_name:
            rows_by_run[run_name].append(row)

    # Older input CSVs predate this derived field. Construct it here as well
    # so the running-best T-RMSE plot remains available alongside the raw
    # per-epoch average curve.
    for run_rows in rows_by_run.values():
        best_rmse = math.inf
        sorted_rows = sorted(run_rows, key=lambda item: int(item["epoch"]))
        values: list[float | None] = []
        for row in sorted_rows:
            value = row.get("valid_T-RMSE", "")
            if value not in ("", None) and is_number(value):
                candidate = float(value)
                if math.isfinite(candidate):
                    values.append(candidate)
                    continue
            values.append(None)

        for index, (row, candidate) in enumerate(zip(sorted_rows, values)):
            isolated_drop = False
            if candidate is not None and 0 < index < len(values) - 1:
                previous = values[index - 1]
                following = values[index + 1]
                isolated_drop = (
                    previous is not None
                    and following is not None
                    and previous >= candidate * 10.0
                    and following >= candidate * 10.0
                )
            if candidate is not None and not isolated_drop:
                best_rmse = min(best_rmse, candidate)
            if candidate is not None and math.isfinite(best_rmse):
                row["best_valid_T-RMSE_so_far"] = str(best_rmse)

    first = rows[0]
    label = config_label(first)
    averaged_rows: list[dict[str, float | int | str]] = []
    all_run_names = sorted(rows_by_run)
    for epoch in sorted(rows_by_epoch):
        epoch_rows = rows_by_epoch[epoch]
        averaged: dict[str, float | int | str] = {
            "run": label,
            "source_file": "",
            "sweep_task_id": "",
            "seed": "avg",
            "hiphop_l_max": int(first["hiphop_l_max"]),
            "hiphop_n_max": int(first["hiphop_n_max"]),
            "data_size": int(first["data_size"]),
            "epoch": epoch,
            "run_count": len({row.get('run', '') for row in epoch_rows if row.get('run', '')}),
            "total_source_runs": len(all_run_names),
        }

        numeric_columns = sorted(
            {
                key
                for row in epoch_rows
                for key, value in row.items()
                if key not in {"run", "source_file", "sweep_task_id", "seed", "hiphop_l_max", "hiphop_n_max", "data_size", "epoch"}
                and value not in ("", None)
                and is_number(value)
            }
        )

        for column in numeric_columns:
            values = [
                float(row[column])
                for row in epoch_rows
                if row.get(column, "") not in ("", None) and is_number(row[column])
            ]

            finite_values = [value for value in values if math.isfinite(value)]
            if not finite_values:
                continue
            mean_value = sum(finite_values) / len(finite_values)
            std_value = (
                math.sqrt(sum((value - mean_value) ** 2 for value in finite_values) / len(finite_values))
                if len(finite_values) > 1
                else 0.0
            )
            averaged[column] = mean_value
            averaged[f"{column}__std"] = std_value

        averaged_rows.append(averaged)

    return averaged_rows


def summarize_runs(rows: list[dict[str, str]]) -> dict[str, float | int | str]:
    first = rows[0]
    run_names = sorted({row["run"] for row in rows if row.get("run")})
    epochs = sorted({int(row["epoch"]) for row in rows if row.get("epoch", "") != ""})
    return {
        "run": config_label(first),
        "hiphop_l_max": int(first["hiphop_l_max"]),
        "hiphop_n_max": int(first["hiphop_n_max"]),
        "data_size": int(first["data_size"]),
        "source_runs": len(run_names),
        "max_epoch": epochs[-1] if epochs else "",
    }


def metric_columns(columns: list[str]) -> list[str]:
    skip = {
        "run",
        "source_file",
        "sweep_task_id",
        "seed",
        "hiphop_l_max",
        "hiphop_n_max",
        "data_size",
        "total_params",
        "epoch",
        "batch_size",
        "current_max_epochs",
        "run_count",
    }
    return [
        column
        for column in columns
        if column not in skip
        and not column.endswith("__std")
        and any(token in column for token in ("train_", "valid_", "_time_s", "learning_rate", "epochs_since_best"))
    ]


def finite_pairs_with_std(
    run_rows: list[dict[str, float | int | str]], metric: str
) -> tuple[list[int], list[float], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    stds: list[float] = []
    std_key = f"{metric}__std"
    for row in run_rows:
        if metric not in row or row[metric] == "":
            continue
        value = float(row[metric])
        if not math.isfinite(value):
            continue
        std_value = 0.0
        if std_key in row and row[std_key] != "":
            candidate = float(row[std_key])
            if math.isfinite(candidate):
                std_value = candidate
        xs.append(int(row["epoch"]))
        ys.append(value)
        stds.append(std_value)
    return xs, ys, stds


def plot_metric_with_std(
    metric: str,
    runs: dict[str, list[dict[str, float | int | str]]],
    output: Path,
    detailed_y_axis: bool = True,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    plotted = False
    for label, run_rows in sorted(runs.items()):
        xs, ys, stds = finite_pairs_with_std(run_rows, metric)
        if not xs:
            continue
        style = model_plot_style(run_rows[0])
        line = ax.plot(xs, ys, linewidth=1.8, linestyle=style["linestyle"], label=label)[0]
        color = line.get_color()
        lower = [y - s for y, s in zip(ys, stds)]
        upper = [y + s for y, s in zip(ys, stds)]
        ax.fill_between(xs, lower, upper, color=color, alpha=0.18, linewidth=0)
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_title(f"{metric_title(metric)} (mean ± std)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("seconds" if metric.endswith("_time_s") else metric)
    log_scale = should_use_log_y(metric, runs)
    if log_scale:
        ax.set_yscale("log")
        ax.set_ylabel(f"{metric} (log scale)")
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(bottom=max(ymin, 1e-12), top=ymax)
    apply_metric_limits(ax, metric)
    apply_sparse_y_axis(ax, metric, log_scale, detailed=detailed_y_axis)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(output, dpi=180)
    plt.close(fig)



def plot_dashboard_with_std(
    metrics: list[str],
    runs: dict[str, list[dict[str, float | int | str]]],
    output: Path,
    detailed_y_axis: bool = True,
) -> None:
    import matplotlib.pyplot as plt

    ncols = 3
    nrows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(17, 4.2 * nrows), constrained_layout=True)
    flat_axes = list(axes.flat if hasattr(axes, "flat") else [axes])

    for ax, metric in zip(flat_axes, metrics):
        plotted = False
        for label, run_rows in sorted(runs.items()):
            xs, ys, stds = finite_pairs_with_std(run_rows, metric)
            if not xs:
                continue
            style = model_plot_style(run_rows[0])
            line = ax.plot(xs, ys, linewidth=1.2, linestyle=style["linestyle"], label=label)[0]
            color = line.get_color()
            lower = [y - s for y, s in zip(ys, stds)]
            upper = [y + s for y, s in zip(ys, stds)]
            ax.fill_between(xs, lower, upper, color=color, alpha=0.15, linewidth=0)
            plotted = True
        ax.set_title(f"{metric_title(metric)} (mean ± std)", fontsize=10)
        ax.set_xlabel("Epoch")
        log_scale = should_use_log_y(metric, runs)
        if log_scale:
            ax.set_yscale("log")
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(bottom=max(ymin, 1e-12), top=ymax)
        apply_metric_limits(ax, metric)
        apply_sparse_y_axis(ax, metric, log_scale, detailed=detailed_y_axis)
        ax.grid(True, alpha=0.25)
        if not plotted:
            ax.axis("off")

    for ax in flat_axes[len(metrics) :]:
        ax.axis("off")

    handles, labels = flat_axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), fontsize=9)
    fig.savefig(output, dpi=160)
    plt.close(fig)



def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows_by_config: dict[tuple[int, int, int], list[dict[str, str]]] = defaultdict(list)
    summaries: list[dict[str, float | int | str]] = []

    for input_dir in args.input_dir:
        metrics_path = input_dir / "metrics.csv"
        if not metrics_path.exists():
            raise SystemExit(f"Missing metrics.csv in {input_dir}")
        rows = read_metrics(metrics_path)
        if args.exclude_run:
            excluded = set(args.exclude_run)
            rows = [row for row in rows if row.get("run", "") not in excluded]
        if not rows:
            continue
        key = config_key(rows[0])
        rows_by_config[key].extend(rows)
        summaries.append(summarize_runs(rows))

    averaged_runs: dict[str, list[dict[str, float | int | str]]] = {}
    all_rows: list[dict[str, float | int | str]] = []

    for key in sorted(rows_by_config):
        averaged_rows = average_rows(rows_by_config[key])
        if not averaged_rows:
            continue
        label = str(averaged_rows[0]["run"])
        averaged_runs[label] = averaged_rows
        all_rows.extend(averaged_rows)

    if not all_rows:
        raise SystemExit("No rows available to average.")

    metrics_csv = args.output_dir / "metrics.csv"
    columns = write_csv(all_rows, metrics_csv)

    summary_path = args.output_dir / "input_run_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["run", "hiphop_l_max", "hiphop_n_max", "data_size", "source_runs", "max_epoch"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(summaries, key=lambda row: (int(row["hiphop_l_max"]), int(row["hiphop_n_max"]))))

    averaged_summary_path = args.output_dir / "run_summary.csv"
    with averaged_summary_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["run", "hiphop_l_max", "hiphop_n_max", "data_size", "source_runs", "epochs"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for label, run_rows in sorted(averaged_runs.items()):
            first = run_rows[0]
            source_runs = max(int(row.get("run_count", 0)) for row in run_rows)
            writer.writerow(
                {
                    "run": label,
                    "hiphop_l_max": first["hiphop_l_max"],
                    "hiphop_n_max": first["hiphop_n_max"],
                    "data_size": first["data_size"],
                    "source_runs": source_runs,
                    "epochs": len(run_rows),
                }
            )

    metrics = metric_columns(columns)
    metric_dir = args.output_dir / "by_metric"
    metric_dir.mkdir(exist_ok=True)
    for metric in metrics:
        filename = re.sub(r"[^A-Za-z0-9_.-]+", "_", metric) + ".png"
        plot_metric_with_std(metric, averaged_runs, metric_dir / filename)

    dashboard_metrics = [
        metric
        for metric in [
            "valid_T-MAE",
            "valid_T-RMSE",
            "valid_F-MAE",
            "valid_F-RMSE",
            "valid_Loss",
            "valid_Error Loss",
            "train_T-MAE",
            "train_F-MAE",
            "total_epoch_time_s",
        ]
        if metric in metrics
    ]
    if dashboard_metrics:
        plot_dashboard_with_std(dashboard_metrics, averaged_runs, args.output_dir / "summary_dashboard.png")

    print(f"Averaged {len(all_rows)} epoch rows across {len(averaged_runs)} configs.")
    print(f"Wrote {metrics_csv}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {averaged_summary_path}")
    print(f"Wrote {len(metrics)} metric plots to {metric_dir}")
    if dashboard_metrics:
        print(f"Wrote {args.output_dir / 'summary_dashboard.png'}")


if __name__ == "__main__":
    main()
