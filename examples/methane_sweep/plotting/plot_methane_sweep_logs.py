#!/usr/bin/env python3
"""Parse methane sweep training logs and plot metrics across runs."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path


EPOCH_RE = re.compile(r"^Epoch\s+(\d+):")
FLOAT_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
METRIC_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9 -]*?)\s*:")
TOTAL_COUNT_RE = re.compile(r"Total Count:\s*(\d+)")
LR_RE = re.compile(r"Learning rate:\s*(%s)" % FLOAT_RE.pattern)
BATCH_RE = re.compile(r"Batch Size:\s*(\d+)")
REQUESTED_BATCH_RE = re.compile(r"Batch config:\s*requested_train_batch=(\d+)")
TIME_RE = re.compile(r"(Training time|Total epoch time):\s*(%s)\s*s" % FLOAT_RE.pattern)
BEST_RE = re.compile(r"Best T-MAE so far:\s*(%s)" % FLOAT_RE.pattern)
SINCE_BEST_RE = re.compile(r"Epochs since last best:\s*(\d+)")
CURRENT_MAX_RE = re.compile(r"Current max epochs:\s*(\d+)")
FINAL_EVALUATION_MARKERS = (
    "Training phase ended.",
    "Reverting to best model found.",
    "Making plots over training time...",
    "Testing model...",
    "Training complete.",
)
TEST_STATISTICS_RE = re.compile(r"^TEST_SET_STATISTICS_JSON:\s*(\{.*\})$")


METRIC_NAMES = {
    "T-RMSE",
    "T-MAE",
    "T-RSQ",
    "F-RMSE",
    "F-MAE",
    "F-RSQ",
    "Error Loss",
    "L2",
    "Loss",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", default="logs", type=Path, help="Directory containing .out logs.")
    parser.add_argument(
        "--log-pattern",
        default="runs/**/*_methane_*.out",
        help="Glob pattern, relative to --log-dir, for selecting logs to parse.",
    )
    parser.add_argument(
        "--extra-log-pattern",
        action="append",
        default=[],
        help="Additional glob pattern, relative to --log-dir, for selecting logs to parse.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("logs/metric_plots"),
        type=Path,
        help="Directory for metrics CSV and plots.",
    )
    parser.add_argument(
        "--sweep-config",
        default=Path("examples/methane_sweep/configs/methane-l4-n4.yml"),
        type=Path,
        help="Sweep YAML used to map task ids to hyperparameters.",
    )
    parser.add_argument(
        "--data-size",
        type=int,
        help="Only include runs with this training data size.",
    )
    parser.add_argument(
        "--hiphop-l-max",
        type=int,
        help="Only include runs with this HIP-HOP l_max.",
    )
    parser.add_argument(
        "--hiphop-n-max",
        type=int,
        help="Only include runs with this HIP-HOP n_max.",
    )
    parser.add_argument(
        "--pareto-seed",
        type=int,
        help="Only include this seed in the force/energy Pareto plots. Defaults to the best seed for each model/data-size setup.",
    )
    parser.add_argument(
        "--test-statistics-dir",
        default=Path("examples/methane_sweep/data_statistics"),
        type=Path,
        help="Directory containing persisted held-out test target statistics.",
    )
    parser.add_argument("--test-set-size", default=80_000, type=int)
    parser.add_argument(
        "--requested-train-batch-size",
        type=int,
        help="Only include runs whose earliest log requested this starting training batch size.",
    )
    return parser.parse_args()


def simple_yaml_parameters(path: Path) -> dict[str, dict[str, object]]:
    try:
        import yaml

        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle).get("parameters", {})
    except ModuleNotFoundError:
        pass

    parameters: dict[str, dict[str, object]] = {}
    current: str | None = None
    in_parameters = False
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip()
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped == "parameters:":
                in_parameters = True
                continue
            if in_parameters and not line.startswith(" "):
                break
            if not in_parameters:
                continue
            if line.startswith("  ") and not line.startswith("    ") and stripped.endswith(":"):
                current = stripped[:-1]
                parameters[current] = {}
                continue
            if current is not None and line.startswith("    ") and ":" in stripped:
                key, value = stripped.split(":", 1)
                value = value.strip()
                if value.startswith("[") and value.endswith("]"):
                    parameters[current][key] = [parse_scalar(item.strip()) for item in value[1:-1].split(",")]
                else:
                    parameters[current][key] = parse_scalar(value)
    return parameters


def parse_scalar(value: str) -> object:
    if value == "":
        return value
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value.strip("'\"")


def parameter_values(spec: dict[str, object]) -> list[object]:
    if "values" in spec:
        return list(spec["values"])  # type: ignore[arg-type]
    if "value" in spec:
        return [spec["value"]]
    if "min" in spec and "max" in spec:
        step = int(spec.get("step", 1))
        return list(range(int(spec["min"]), int(spec["max"]) + 1, step))
    return []


def format_data_size(value: object) -> str:
    size = int(value)
    if size == 100_000:
        return "100k"
    if size == 1_000_000:
        return "1M"
    if size >= 1_000_000 and size % 1_000_000 == 0:
        return f"{size // 1_000_000}M"
    if size >= 1_000 and size % 1_000 == 0:
        return f"{size // 1_000}k"
    return f"{size:,}"


def task_id_from_path(path: Path) -> int | None:
    match = re.search(
        r"_(\d+)(?:_w\d+)?_methane_(?:sweep|resume_selected|l3_b256|l3n5|l4_titanv)(?:_[A-Za-z0-9-]+)?\.out$",
        path.name,
    )
    return int(match.group(1)) if match else None


def sweep_jobs(path: Path) -> dict[int, dict[str, object]]:
    if not path.exists():
        return {}
    parameters = simple_yaml_parameters(path)
    grid_names = ["seed", "hiphop_l_max", "hiphop_n_max"]
    if "data_size" in parameters:
        grid_names.append("data_size")
    if any(name not in parameters for name in grid_names):
        return {}
    values = [parameter_values(parameters[name]) for name in grid_names]
    jobs = {}
    for task_id, combo in enumerate(itertools.product(*values)):
        jobs[task_id] = dict(zip(grid_names, combo))
    return jobs


def run_label(path: Path, total_count: int | None, task_params: dict[str, object] | None) -> str:
    task_id = task_id_from_path(path)
    task = str(task_id) if task_id is not None else path.stem
    if task_params:
        detail = " ".join(
            [
                f"l={task_params['hiphop_l_max']}",
                f"n={task_params['hiphop_n_max']}",
            ]
        )
        if "data_size" in task_params:
            detail += f" d={format_data_size(task_params['data_size'])}"
        return f"task {task}: {detail}"
    return f"task {task}"


def parse_metric_values(line: str) -> tuple[str, float, float] | None:
    name_match = METRIC_RE.match(line)
    if not name_match:
        return None
    metric = name_match.group(1).strip()
    if metric not in METRIC_NAMES:
        return None
    values = [float(value) for value in FLOAT_RE.findall(line[name_match.end() :])]
    if len(values) < 2:
        return None
    return metric, values[-2], values[-1]


def parse_log(
    path: Path, task_params: dict[str, object] | None
) -> tuple[list[dict[str, float | int | str]], int | None]:
    rows: list[dict[str, float | int | str]] = []
    current: dict[str, float | int | str] | None = None
    total_count: int | None = None
    test_statistics: dict[str, object] = {}
    final_test_metrics: dict[str, float] = {}
    testing_sections = 0
    requested_train_batch_size: int | None = None

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n")

            if match := REQUESTED_BATCH_RE.search(line):
                requested_train_batch_size = int(match.group(1))
                continue

            if match := TEST_STATISTICS_RE.match(line):
                test_statistics = json.loads(match.group(1))
                continue

            if line == "Testing model...":
                testing_sections += 1
                continue

            # The explicit post-training test_model call prints a one-column
            # metric table. The earlier training summary has three columns.
            if testing_sections >= 1:
                name_match = METRIC_RE.match(line)
                if name_match and name_match.group(1).strip() in METRIC_NAMES:
                    values = [float(value) for value in FLOAT_RE.findall(line[name_match.end() :])]
                    if len(values) == 1:
                        final_test_metrics[name_match.group(1).strip()] = values[0]

            if match := TOTAL_COUNT_RE.search(line):
                total_count = int(match.group(1))
                continue

            if match := EPOCH_RE.match(line):
                if current is not None:
                    rows.append(current)
                current = {"epoch": int(match.group(1)), "source_file": path.name}
                continue

            if current is None:
                continue

            if line in FINAL_EVALUATION_MARKERS:
                rows.append(current)
                current = None
                continue

            if match := LR_RE.search(line):
                current["learning_rate"] = float(match.group(1))
                continue
            if match := BATCH_RE.search(line):
                current["batch_size"] = int(match.group(1))
                continue
            if match := TIME_RE.search(line):
                key = "training_time_s" if match.group(1) == "Training time" else "total_epoch_time_s"
                current[key] = float(match.group(2))
                continue
            if match := BEST_RE.search(line):
                current["best_valid_T-MAE_so_far"] = float(match.group(1))
                continue
            if match := SINCE_BEST_RE.search(line):
                current["epochs_since_best"] = int(match.group(1))
                continue
            if match := CURRENT_MAX_RE.search(line):
                current["current_max_epochs"] = int(match.group(1))
                continue

            parsed = parse_metric_values(line)
            if parsed is not None:
                metric, train_value, valid_value = parsed
                current[f"train_{metric}"] = train_value
                current[f"valid_{metric}"] = valid_value

    if current is not None:
        rows.append(current)

    label = run_label(path, total_count, task_params)
    for row in rows:
        row["run"] = label
        task_id = task_id_from_path(path)
        if task_id is not None:
            row["sweep_task_id"] = task_id
        if task_params:
            row.update(task_params)
        if total_count is not None:
            row["total_params"] = total_count
        if requested_train_batch_size is not None:
            row["requested_train_batch_size"] = requested_train_batch_size
        if "energy_std_kcal_per_mol" in test_statistics:
            row["test_energy_std_kcal_per_mol"] = float(test_statistics["energy_std_kcal_per_mol"])
            row["test_statistics_path"] = str(test_statistics.get("path", ""))
        for metric, value in final_test_metrics.items():
            row[f"test_{metric}"] = value

    return rows, total_count


def write_csv(rows: list[dict[str, float | int | str]], path: Path) -> list[str]:
    preferred = [
        "run",
        "source_file",
        "sweep_task_id",
        "seed",
        "hiphop_l_max",
        "hiphop_n_max",
        "data_size",
        "total_params",
        "requested_train_batch_size",
        "epoch",
        "learning_rate",
        "batch_size",
        "training_time_s",
        "total_epoch_time_s",
        "best_valid_T-MAE_so_far",
        "epochs_since_best",
        "current_max_epochs",
    ]
    discovered = sorted({key for row in rows for key in row})
    columns = preferred + [key for key in discovered if key not in preferred]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    return columns


def write_sweep_task_map(
    jobs: dict[int, dict[str, object]],
    runs: dict[str, list[dict[str, float | int | str]]],
    path: Path,
) -> None:
    logged_by_task = {}
    for run_rows in runs.values():
        first = run_rows[0]
        task_id = first.get("sweep_task_id")
        if task_id != "":
            logged_by_task[int(task_id)] = first

    columns = [
        "sweep_task_id",
        "seed",
        "hiphop_l_max",
        "hiphop_n_max",
        "data_size",
        "source_file",
        "total_params",
        "epochs",
        "logged",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for task_id, params in sorted(jobs.items()):
            logged = logged_by_task.get(task_id)
            row = {"sweep_task_id": task_id, **params}
            if logged is not None:
                row.update(
                    {
                        "source_file": logged.get("source_file", ""),
                        "total_params": logged.get("total_params", ""),
                        "epochs": len(runs[str(logged["run"])]),
                        "logged": "yes",
                    }
                )
            else:
                row.update({"source_file": "", "total_params": "", "epochs": "", "logged": "no"})
            writer.writerow(row)


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
    }
    return [
        column
        for column in columns
        if column not in skip and any(isinstance_marker in column for isinstance_marker in ("train_", "valid_", "_time_s", "learning_rate", "epochs_since_best"))
    ]


def deduplicate_epochs(
    runs: dict[str, list[dict[str, float | int | str]]],
) -> tuple[list[dict[str, float | int | str]], dict[str, list[dict[str, float | int | str]]]]:
    deduplicated_runs: dict[str, list[dict[str, float | int | str]]] = {}
    all_rows: list[dict[str, float | int | str]] = []

    for label, run_rows in runs.items():
        rows_by_epoch: dict[int, dict[str, float | int | str]] = {}
        for row in run_rows:
            rows_by_epoch[int(row["epoch"])] = row
        deduplicated_rows = [rows_by_epoch[epoch] for epoch in sorted(rows_by_epoch)]
        deduplicated_runs[label] = deduplicated_rows
        all_rows.extend(deduplicated_rows)

    return all_rows, deduplicated_runs


def add_stored_test_statistics(
    runs: dict[str, list[dict[str, float | int | str]]], statistics_dir: Path, test_set_size: int
) -> None:
    """Add directly calculated test target statistics to old and new logs."""
    for run_rows in runs.values():
        first = run_rows[0]
        if "test_energy_std_kcal_per_mol" in first or "data_size" not in first:
            continue
        path = statistics_dir / f"test_energy_d{int(first['data_size'])}_n{test_set_size}_sequential.json"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            statistics = json.load(handle)
        for row in run_rows:
            row["test_energy_std_kcal_per_mol"] = float(statistics["energy_std_kcal_per_mol"])
            row["test_statistics_path"] = str(path)


def finite_pairs(run_rows: list[dict[str, float | int | str]], metric: str) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    for row in run_rows:
        if metric not in row or row[metric] == "":
            continue
        value = float(row[metric])
        if math.isfinite(value):
            xs.append(int(row["epoch"]))
            ys.append(value)
    return xs, ys


def model_plot_style(values: dict[str, float | int | str]) -> dict[str, str]:
    """Keep model architecture visually distinct in every plot."""
    n_max = str(values.get("hiphop_n_max", ""))
    # if n_max == "3":
    #     return {"linestyle": "-", "marker": "o"}
    # if n_max == "4":
    #     return {"linestyle": "--", "marker": "D"}
    # return {"linestyle": ":", "marker": "s"}
    
    # just plot each the same with a solid line
    return {"linestyle": "-", "marker": "o"}


def should_use_log_y(metric: str, runs: dict[str, list[dict[str, float | int | str]]]) -> bool:
    if metric.endswith("_time_s"):
        return False
    log_metric_tokens = ("RMSE", "MAE", "RSQ", "Loss", "L2")
    if not any(token in metric for token in log_metric_tokens):
        return False
    values: list[float] = []
    for run_rows in runs.values():
        _, ys = finite_pairs(run_rows, metric)
        values.extend(ys)
    values = [value for value in values if value > 0 and math.isfinite(value)]
    if len(values) < 2:
        return False
    return max(values) / min(values) >= 1.5


def apply_sparse_y_axis(ax, metric: str, log_scale: bool) -> None:
    import matplotlib.ticker as ticker

    if log_scale:
        if "RSQ" in metric:
            ax.yaxis.set_major_locator(ticker.FixedLocator([0.98, 0.99, 1.0]))
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda value, _: f"{value:g}"))
        else:
            ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=3))
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda value, _: f"{value:g}"))
        ax.yaxis.set_minor_locator(ticker.NullLocator())
    else:
        if metric.endswith("_time_s"):
            ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
        else:
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
        ax.yaxis.set_minor_locator(ticker.NullLocator())
        if metric.endswith("_time_s"):
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda value, _: f"{value:g}"))


def apply_metric_limits(ax, metric: str) -> None:
    if "RSQ" in metric:
        ax.set_ylim(0.98, 1.001)
    elif metric == "training_time_s":
        ax.set_ylim(bottom=0, top=250)


def metric_title(metric: str) -> str:
    split = ""
    base = metric
    if metric.startswith("train_"):
        split = "Train "
        base = metric.removeprefix("train_")
    elif metric.startswith("valid_"):
        split = "Validation "
        base = metric.removeprefix("valid_")

    replacements = {
        "T-RMSE": "Energy RMSE",
        "T-MAE": "Energy MAE",
        "T-RSQ": "Energy R-squared",
        "F-RMSE": "Force RMSE",
        "F-MAE": "Force MAE",
        "F-RSQ": "Force R-squared",
        "Error Loss": "Error Loss",
        "Loss": "Total Loss",
        "L2": "L2 Regularization",
        "training_time_s": "Training Time",
        "total_epoch_time_s": "Total Epoch Time",
        "learning_rate": "Learning Rate",
        "best_valid_T-MAE_so_far": "Best Validation Energy MAE So Far",
        "epochs_since_best": "Epochs Since Best Validation Energy MAE",
    }
    return split + replacements.get(base, base.replace("_", " "))


def plot_metric(metric: str, runs: dict[str, list[dict[str, float | int | str]]], output: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    plotted = False
    for label, run_rows in sorted(runs.items()):
        xs, ys = finite_pairs(run_rows, metric)
        if not xs:
            continue
        style = model_plot_style(run_rows[0])
        ax.plot(xs, ys, linewidth=1.8, linestyle=style["linestyle"], label=label)
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_title(metric_title(metric))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("seconds" if metric.endswith("_time_s") else metric)
    log_scale = should_use_log_y(metric, runs)
    if log_scale:
        ax.set_yscale("log")
        ax.set_ylabel(f"{metric} (log scale)")
    apply_metric_limits(ax, metric)
    apply_sparse_y_axis(ax, metric, log_scale)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_dashboard(metrics: list[str], runs: dict[str, list[dict[str, float | int | str]]], output: Path) -> None:
    import matplotlib.pyplot as plt

    ncols = 3
    nrows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(17, 4.2 * nrows), constrained_layout=True)
    flat_axes = list(axes.flat if hasattr(axes, "flat") else [axes])

    for ax, metric in zip(flat_axes, metrics):
        for label, run_rows in sorted(runs.items()):
            xs, ys = finite_pairs(run_rows, metric)
            if xs:
                style = model_plot_style(run_rows[0])
                ax.plot(xs, ys, linewidth=1.2, linestyle=style["linestyle"], label=label)
        ax.set_title(metric_title(metric), fontsize=10)
        ax.set_xlabel("Epoch")
        log_scale = should_use_log_y(metric, runs)
        if log_scale:
            ax.set_yscale("log")
        apply_metric_limits(ax, metric)
        apply_sparse_y_axis(ax, metric, log_scale)
        ax.grid(True, alpha=0.25)

    for ax in flat_axes[len(metrics) :]:
        ax.axis("off")

    handles, labels = flat_axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), fontsize=9)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def run_average(rows: list[dict[str, float | int | str]], metric: str) -> float:
    values = [float(row[metric]) for row in rows if metric in row and row[metric] != ""]
    values = [value for value in values if math.isfinite(value)]
    return sum(values) / len(values)


def pareto_frontier(points: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    frontier = []
    best_error = math.inf
    for point in sorted(points, key=lambda item: (float(item["training_time_to_best_s"]), float(item["best_checkpoint_valid_F-MAE"]))):
        error = float(point["best_checkpoint_valid_F-MAE"])
        if error < best_error:
            frontier.append(point)
            best_error = error
    return frontier


def best_checkpoint_row(rows: list[dict[str, float | int | str]]) -> dict[str, float | int | str]:
    candidates = [row for row in rows if row.get("valid_Loss") not in ("", None)]
    if not candidates:
        raise ValueError("run has no valid_Loss values")
    return min(candidates, key=lambda row: float(row["valid_Loss"]))


def best_metric_row(rows: list[dict[str, float | int | str]], metric: str) -> dict[str, float | int | str]:
    candidates = [row for row in rows if row.get(metric) not in ("", None)]
    if not candidates:
        raise ValueError(f"run has no {metric} values")
    return min(candidates, key=lambda row: float(row[metric]))


def cumulative_training_time_to_epoch(rows: list[dict[str, float | int | str]], epoch: int) -> float:
    total = 0.0
    for row in rows:
        if int(row["epoch"]) > epoch:
            break
        if row.get("training_time_s") not in ("", None):
            total += float(row["training_time_s"])
    return total


def dataset_size_title(points: list[dict[str, float | int | str]]) -> str:
    sizes = sorted({int(point["data_size"]) for point in points if point.get("data_size") not in ("", None)})
    if len(sizes) != 1:
        return "Dataset Size=Mixed"
    size = sizes[0]
    return f"Dataset Size={format_data_size(size)}"


def model_group_key(point: dict[str, float | int | str]) -> tuple[str, str, str]:
    return (
        str(point.get("hiphop_l_max", "")),
        str(point.get("hiphop_n_max", "")),
        str(point.get("data_size", "")),
    )


def best_point_per_model(
    points: list[dict[str, float | int | str]], metric: str
) -> list[dict[str, float | int | str]]:
    best_by_group: dict[tuple[str, str, str], dict[str, float | int | str]] = {}
    for point in points:
        key = model_group_key(point)
        if key not in best_by_group or float(point[metric]) < float(best_by_group[key][metric]):
            best_by_group[key] = point
    return list(best_by_group.values())


def plot_force_accuracy_pareto(
    runs: dict[str, list[dict[str, float | int | str]]],
    output: Path,
    csv_output: Path,
    seed_filter: int | None = None,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    points: list[dict[str, float | int | str]] = []
    for label, run_rows in sorted(runs.items()):
        if (
            not run_rows
            or "training_time_s" not in run_rows[0]
            or "valid_F-MAE" not in run_rows[0]
            or "valid_T-MAE" not in run_rows[0]
        ):
            continue
        first = run_rows[0]
        best_row = best_checkpoint_row(run_rows)
        best_epoch = int(best_row["epoch"])
        points.append(
            {
                "run": label,
                "sweep_task_id": first.get("sweep_task_id", ""),
                "seed": first.get("seed", ""),
                "hiphop_l_max": first.get("hiphop_l_max", ""),
                "hiphop_n_max": first.get("hiphop_n_max", ""),
                "data_size": first.get("data_size", ""),
                "total_params": first.get("total_params", ""),
                "epochs": len(run_rows),
                "best_checkpoint_metric": "valid_Loss",
                "best_checkpoint_epoch": best_epoch,
                "training_time_to_best_s": cumulative_training_time_to_epoch(run_rows, best_epoch),
                "training_time_to_best_h": cumulative_training_time_to_epoch(run_rows, best_epoch) / 3600,
                "best_checkpoint_valid_Loss": float(best_row["valid_Loss"]),
                "best_checkpoint_valid_F-MAE": float(best_row["valid_F-MAE"]),
                "best_checkpoint_valid_T-MAE": float(best_row["valid_T-MAE"]),
            }
        )

    selected_seed = seed_filter
    if selected_seed is not None:
        points = [point for point in points if int(point["seed"]) == selected_seed]
    else:
        points = best_point_per_model(points, "best_checkpoint_valid_Loss")

    columns = [
        "run",
        "sweep_task_id",
        "seed",
        "hiphop_l_max",
        "hiphop_n_max",
        "data_size",
        "total_params",
        "epochs",
        "best_checkpoint_metric",
        "best_checkpoint_epoch",
        "training_time_to_best_s",
        "training_time_to_best_h",
        "best_checkpoint_valid_Loss",
        "best_checkpoint_valid_F-MAE",
        "best_checkpoint_valid_T-MAE",
        "on_frontier",
    ]
    frontier = pareto_frontier(points)
    frontier_ids = {point["sweep_task_id"] for point in frontier}
    with csv_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for point in points:
            row = dict(point)
            row["on_frontier"] = "yes" if point["sweep_task_id"] in frontier_ids else "no"
            writer.writerow(row)

    if not points:
        return

    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    energy_ax = ax.twinx()
    energy_ax.set_zorder(ax.get_zorder() + 1)
    energy_ax.patch.set_visible(False)
    rightmost_x = max(float(point["training_time_to_best_h"]) for point in points)
    sorted_points = sorted(points, key=lambda point: float(point["training_time_to_best_s"]))
    x_values = [float(point["training_time_to_best_h"]) for point in sorted_points]
    force_values = [float(point["best_checkpoint_valid_F-MAE"]) for point in sorted_points]
    energy_values = [float(point["best_checkpoint_valid_T-MAE"]) for point in sorted_points]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    force_color = "#202020"
    energy_color = "#6f6f6f"
    force_line = ax.plot(
        x_values,
        force_values,
        linewidth=2.0,
        color=force_color,
        label="Best validation force MAE",
        zorder=1,
    )[0]
    energy_line = energy_ax.plot(
        x_values,
        energy_values,
        linewidth=2.0,
        linestyle=":",
        color=energy_color,
        label="Best validation energy MAE",
        zorder=1,
    )[0]

    for index, point in enumerate(sorted_points):
        x_value = float(point["training_time_to_best_h"])
        force_value = float(point["best_checkpoint_valid_F-MAE"])
        energy_value = float(point["best_checkpoint_valid_T-MAE"])
        color = colors[index % len(colors)]
        marker = model_plot_style(point)["marker"]
        ax.scatter(
            x_value,
            force_value,
            s=95,
            marker=marker,
            color=color,
            zorder=10,
        )
        energy_ax.scatter(
            x_value,
            energy_value,
            s=85,
            marker=marker,
            color=color,
            zorder=10,
        )
        is_rightmost = x_value == rightmost_x
        setup_label = f"l={point['hiphop_l_max']} n={point['hiphop_n_max']}"
        setup_label += f" best_epoch={point['best_checkpoint_epoch']}"
        ax.annotate(
            setup_label,
            (x_value, force_value),
            xytext=(-8, 12) if is_rightmost else (7, 6),
            textcoords="offset points",
            ha="right" if is_rightmost else "left",
            va="bottom",
            fontsize=9,
            zorder=11,
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "none", "alpha": 0.0},
        )

    title_seed = f", Seed {selected_seed}" if selected_seed is not None else ", Best Seed per Model"
    ax.set_title(f"Training Cost vs Best Checkpoint ({dataset_size_title(points)}){title_seed}")
    ax.set_xlabel("Training time (hours)")
    ax.set_ylabel("Force MAE")
    energy_ax.set_ylabel("Energy MAE (kcal/mol)")
    ax.set_xscale("log")
    force_padding = max((max(force_values) - min(force_values)) * 0.35, 0.15)
    energy_padding = max((max(energy_values) - min(energy_values)) * 0.35, 0.15)
    ax.set_ylim(min(force_values) - force_padding, max(force_values) + force_padding)
    energy_ax.set_ylim(min(energy_values) - energy_padding, max(energy_values) + energy_padding)
    ax.tick_params(axis="y", labelcolor=force_color)
    energy_ax.tick_params(axis="y", labelcolor=energy_color)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    energy_ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=4))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda value, _: f"{value:g}"))
    ax.grid(True, alpha=0.25)
    ax.legend(handles=[force_line, energy_line], fontsize=9, loc="upper right")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_best_metric_pareto(
    runs: dict[str, list[dict[str, float | int | str]]],
    output: Path,
    csv_output: Path,
    seed_filter: int | None = None,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    points: list[dict[str, float | int | str]] = []
    for label, run_rows in sorted(runs.items()):
        if (
            not run_rows
            or "training_time_s" not in run_rows[0]
            or "valid_F-MAE" not in run_rows[0]
            or "valid_T-MAE" not in run_rows[0]
        ):
            continue
        first = run_rows[0]
        force_row = best_metric_row(run_rows, "valid_F-MAE")
        energy_row = best_metric_row(run_rows, "valid_T-MAE")
        force_epoch = int(force_row["epoch"])
        energy_epoch = int(energy_row["epoch"])
        points.append(
            {
                "run": label,
                "sweep_task_id": first.get("sweep_task_id", ""),
                "seed": first.get("seed", ""),
                "hiphop_l_max": first.get("hiphop_l_max", ""),
                "hiphop_n_max": first.get("hiphop_n_max", ""),
                "data_size": first.get("data_size", ""),
                "total_params": first.get("total_params", ""),
                "epochs": len(run_rows),
                "best_force_epoch": force_epoch,
                "best_force_training_time_s": cumulative_training_time_to_epoch(run_rows, force_epoch),
                "best_force_training_time_h": cumulative_training_time_to_epoch(run_rows, force_epoch) / 3600,
                "best_valid_F-MAE": float(force_row["valid_F-MAE"]),
                "best_energy_epoch": energy_epoch,
                "best_energy_training_time_s": cumulative_training_time_to_epoch(run_rows, energy_epoch),
                "best_energy_training_time_h": cumulative_training_time_to_epoch(run_rows, energy_epoch) / 3600,
                "best_valid_T-MAE": float(energy_row["valid_T-MAE"]),
            }
        )

    selected_seed = seed_filter
    if selected_seed is not None:
        points = [point for point in points if int(point["seed"]) == selected_seed]
        force_points = points
        energy_points = points
    else:
        force_points = best_point_per_model(points, "best_valid_F-MAE")
        energy_points = best_point_per_model(points, "best_valid_T-MAE")

    columns = [
        "hiphop_l_max",
        "hiphop_n_max",
        "data_size",
        "force_run",
        "force_sweep_task_id",
        "force_seed",
        "force_total_params",
        "force_logged_epochs",
        "best_force_epoch",
        "best_force_training_time_s",
        "best_force_training_time_h",
        "best_valid_F-MAE",
        "energy_run",
        "energy_sweep_task_id",
        "energy_seed",
        "energy_total_params",
        "energy_logged_epochs",
        "best_energy_epoch",
        "best_energy_training_time_s",
        "best_energy_training_time_h",
        "best_valid_T-MAE",
    ]
    force_by_group = {model_group_key(point): point for point in force_points}
    energy_by_group = {model_group_key(point): point for point in energy_points}
    with csv_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for key in sorted(force_by_group):
            force = force_by_group[key]
            energy = energy_by_group[key]
            writer.writerow(
                {
                    "hiphop_l_max": force["hiphop_l_max"],
                    "hiphop_n_max": force["hiphop_n_max"],
                    "data_size": force["data_size"],
                    "force_run": force["run"],
                    "force_sweep_task_id": force["sweep_task_id"],
                    "force_seed": force["seed"],
                    "force_total_params": force["total_params"],
                    "force_logged_epochs": force["epochs"],
                    "best_force_epoch": force["best_force_epoch"],
                    "best_force_training_time_s": force["best_force_training_time_s"],
                    "best_force_training_time_h": force["best_force_training_time_h"],
                    "best_valid_F-MAE": force["best_valid_F-MAE"],
                    "energy_run": energy["run"],
                    "energy_sweep_task_id": energy["sweep_task_id"],
                    "energy_seed": energy["seed"],
                    "energy_total_params": energy["total_params"],
                    "energy_logged_epochs": energy["epochs"],
                    "best_energy_epoch": energy["best_energy_epoch"],
                    "best_energy_training_time_s": energy["best_energy_training_time_s"],
                    "best_energy_training_time_h": energy["best_energy_training_time_h"],
                    "best_valid_T-MAE": energy["best_valid_T-MAE"],
                }
            )

    if not force_points or not energy_points:
        return

    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    energy_ax = ax.twinx()
    energy_ax.set_zorder(ax.get_zorder() + 1)
    energy_ax.patch.set_visible(False)
    sorted_by_force = sorted(force_points, key=lambda point: float(point["best_force_training_time_h"]))
    sorted_by_energy = sorted(energy_points, key=lambda point: float(point["best_energy_training_time_h"]))
    force_x = [float(point["best_force_training_time_h"]) for point in sorted_by_force]
    force_y = [float(point["best_valid_F-MAE"]) for point in sorted_by_force]
    energy_x = [float(point["best_energy_training_time_h"]) for point in sorted_by_energy]
    energy_y = [float(point["best_valid_T-MAE"]) for point in sorted_by_energy]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    force_color = "#202020"
    energy_color = "#6f6f6f"
    force_line = ax.plot(
        force_x,
        force_y,
        linewidth=2.0,
        color=force_color,
        label="Best validation force MAE",
        zorder=1,
    )[0]
    energy_line = energy_ax.plot(
        energy_x,
        energy_y,
        linewidth=2.0,
        linestyle=":",
        color=energy_color,
        label="Best validation energy MAE",
        zorder=1,
    )[0]

    rightmost_force_x = max(force_x)
    rightmost_energy_x = max(energy_x)
    color_by_task = {
        model_group_key(point): colors[index % len(colors)]
        for index, point in enumerate(sorted(force_points, key=model_group_key))
    }
    for point in sorted(points, key=lambda item: float(item["best_force_training_time_h"])):
        if point not in force_points:
            continue
        color = color_by_task[model_group_key(point)]
        marker = model_plot_style(point)["marker"]
        x_value = float(point["best_force_training_time_h"])
        y_value = float(point["best_valid_F-MAE"])
        is_rightmost = x_value == rightmost_force_x
        ax.scatter(x_value, y_value, s=95, marker=marker, color=color, zorder=10)
        ax.annotate(
            f"l={point['hiphop_l_max']} n={point['hiphop_n_max']} F best_epoch={point['best_force_epoch']}",
            (x_value, y_value),
            xytext=(-8, 12) if is_rightmost else (7, 6),
            textcoords="offset points",
            ha="right" if is_rightmost else "left",
            va="bottom",
            fontsize=8.5,
            zorder=11,
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "none", "alpha": 0.0},
        )
    for point in sorted(points, key=lambda item: float(item["best_energy_training_time_h"])):
        if point not in energy_points:
            continue
        color = color_by_task[model_group_key(point)]
        marker = model_plot_style(point)["marker"]
        x_value = float(point["best_energy_training_time_h"])
        y_value = float(point["best_valid_T-MAE"])
        is_rightmost = x_value == rightmost_energy_x
        energy_ax.scatter(x_value, y_value, s=85, marker=marker, color=color, zorder=10)
        energy_ax.annotate(
            f"E best_epoch={point['best_energy_epoch']}",
            (x_value, y_value),
            xytext=(-8, -14) if is_rightmost else (7, -13),
            textcoords="offset points",
            ha="right" if is_rightmost else "left",
            va="top",
            fontsize=8.5,
            zorder=11,
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "none", "alpha": 0.0},
        )

    title_seed = f", Seed {selected_seed}" if selected_seed is not None else ", Best Seed per Model"
    ax.set_title(f"Training Cost vs Best Metric Values ({dataset_size_title(points)}){title_seed}")
    ax.set_xlabel("Training time (hours)")
    ax.set_ylabel("Force MAE")
    energy_ax.set_ylabel("Energy MAE (kcal/mol)")
    ax.set_xscale("log")
    force_padding = max((max(force_y) - min(force_y)) * 0.35, 0.15)
    energy_padding = max((max(energy_y) - min(energy_y)) * 0.35, 0.15)
    ax.set_ylim(min(force_y) - force_padding, max(force_y) + force_padding)
    energy_ax.set_ylim(min(energy_y) - energy_padding, max(energy_y) + energy_padding)
    ax.tick_params(axis="y", labelcolor=force_color)
    energy_ax.tick_params(axis="y", labelcolor=energy_color)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    energy_ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=4))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda value, _: f"{value:g}"))
    ax.grid(True, alpha=0.25)
    ax.legend(handles=[force_line, energy_line], fontsize=9, loc="upper right")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def median(values: list[float]) -> float:
    sorted_values = sorted(values)
    midpoint = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[midpoint]
    return 0.5 * (sorted_values[midpoint - 1] + sorted_values[midpoint])


def plot_paper_style_energy_comparison(
    runs: dict[str, list[dict[str, float | int | str]]],
    output: Path,
    csv_output: Path,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    seed_points = []
    for label, run_rows in sorted(runs.items()):
        if not run_rows:
            continue
        first = run_rows[0]
        final_test_rows = [row for row in run_rows if row.get("test_T-RMSE", "") != ""]
        if not final_test_rows or "test_energy_std_kcal_per_mol" not in first:
            print(f"Skipping {label}: final test RMSE or stored test-set STD is unavailable.")
            continue
        final_test_row = max(final_test_rows, key=lambda row: int(row["epoch"]))
        training_set_size = int(first.get("data_size", 1_000_000))
        selected = min(
            (row for row in run_rows if row.get("valid_T-MAE", "") != ""),
            key=lambda row: float(row["valid_T-MAE"]),
            default=None,
        )
        rmse = float(final_test_row["test_T-RMSE"])
        energy_std = float(first["test_energy_std_kcal_per_mol"])
        seed_points.append(
            {
                "run": label,
                "sweep_task_id": first.get("sweep_task_id", ""),
                "seed": first.get("seed", ""),
                "hiphop_l_max": first.get("hiphop_l_max", ""),
                "hiphop_n_max": first.get("hiphop_n_max", ""),
                "data_size": first.get("data_size", ""),
                "training_set_size": training_set_size,
                "logged_epochs": len(run_rows),
                "max_logged_epoch": int(run_rows[-1]["epoch"]),
                "selected_checkpoint_epoch": selected["epoch"] if selected is not None else "",
                "test_T-RMSE": rmse,
                "test_energy_std_kcal_per_mol": energy_std,
                "test_energy_RMSE_over_STD": rmse / energy_std,
            }
        )

    grouped: dict[tuple[object, object, object], list[dict[str, object]]] = defaultdict(list)
    for point in seed_points:
        grouped[(point["hiphop_l_max"], point["hiphop_n_max"], point["data_size"])].append(point)

    points = []
    for (l_max, n_max, data_size), members in sorted(grouped.items()):
        normalized = [float(member["test_energy_RMSE_over_STD"]) for member in members]
        rmses = [float(member["test_T-RMSE"]) for member in members]
        count = len(members)
        mean_normalized = sum(normalized) / count
        mean_rmse = sum(rmses) / count
        normalized_std = (
            math.sqrt(sum((value - mean_normalized) ** 2 for value in normalized) / (count - 1))
            if count > 1
            else 0.0
        )
        rmse_std = (
            math.sqrt(sum((value - mean_rmse) ** 2 for value in rmses) / (count - 1))
            if count > 1
            else 0.0
        )
        points.append(
            {
                "hiphop_l_max": l_max,
                "hiphop_n_max": n_max,
                "data_size": data_size,
                "training_set_size": int(members[0]["training_set_size"]),
                "completed_seed_count": count,
                "seeds": " ".join(str(member.get("seed", "")) for member in members),
                "test_T-RMSE_mean": mean_rmse,
                "test_T-RMSE_std": rmse_std,
                "test_energy_std_kcal_per_mol": float(members[0]["test_energy_std_kcal_per_mol"]),
                "test_energy_RMSE_over_STD_mean": mean_normalized,
                "test_energy_RMSE_over_STD_std": normalized_std,
            }
        )

    columns = [
        "hiphop_l_max",
        "hiphop_n_max",
        "data_size",
        "training_set_size",
        "completed_seed_count",
        "seeds",
        "test_T-RMSE_mean",
        "test_T-RMSE_std",
        "test_energy_std_kcal_per_mol",
        "test_energy_RMSE_over_STD_mean",
        "test_energy_RMSE_over_STD_std",
    ]
    with csv_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(points)

    if not points:
        return

    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    for point in points:
        x_value = float(point["training_set_size"])
        y_value = float(point["test_energy_RMSE_over_STD_mean"])
        y_error = float(point["test_energy_RMSE_over_STD_std"])
        ax.errorbar(
            x_value,
            y_value,
            yerr=y_error,
            fmt=model_plot_style(point)["marker"],
            markersize=9,
            capsize=5,
            elinewidth=1.5,
            markeredgecolor="black",
            zorder=3,
        )
        seed_label = "seed" if int(point["completed_seed_count"]) == 1 else "seeds"
        setup_label = (
            f"l={point['hiphop_l_max']} n={point['hiphop_n_max']} "
            f"({point['completed_seed_count']} {seed_label})"
        )
        if point.get("data_size") not in ("", None):
            setup_label += f" d={format_data_size(point['data_size'])}"
        ax.annotate(
            setup_label,
            (x_value, y_value),
            xytext=(8, 5),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(8e2, 3e6)
    ax.set_ylim(3e-3, 2.5e-1)
    ax.set_xlabel("Training set size")
    ax.set_ylabel("Energy RMSE / STD")
    ax.set_title("Held-Out Test Energy Error, Mean ± Seed STD")
    ax.grid(True, which="both", alpha=0.25)

    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=4))

    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output_dir / ".matplotlib"))

    log_patterns = [args.log_pattern, *args.extra_log_pattern]
    log_paths = sorted(
        {
            path
            for pattern in log_patterns
            for path in args.log_dir.glob(pattern)
            if re.search(
                r"_\d+(?:_w\d+)?_methane_(?:sweep|resume_selected|l3_b256|l3n5)(?:_[A-Za-z0-9-]+)?\.out$",
                path.name,
            )
        }
    )
    if not log_paths:
        raise SystemExit(f"No array-task methane sweep .out files found in {args.log_dir}")

    runs: dict[str, list[dict[str, float | int | str]]] = defaultdict(list)
    original_requested_batch: dict[str, int] = {}
    jobs = sweep_jobs(args.sweep_config)
    for path in log_paths:
        task_id = task_id_from_path(path)
        rows, _ = parse_log(path, jobs.get(task_id) if task_id is not None else None)
        if not rows:
            print(f"warning: no epochs parsed from {path}")
            continue
        if args.data_size is not None:
            data_size = rows[0].get("data_size")
            if data_size in ("", None) or int(data_size) != args.data_size:
                continue
        if args.hiphop_l_max is not None:
            hiphop_l_max = rows[0].get("hiphop_l_max")
            if hiphop_l_max in ("", None) or int(hiphop_l_max) != args.hiphop_l_max:
                continue
        if args.hiphop_n_max is not None:
            hiphop_n_max = rows[0].get("hiphop_n_max")
            if hiphop_n_max in ("", None) or int(hiphop_n_max) != args.hiphop_n_max:
                continue
        label = str(rows[0]["run"])
        requested_batch = rows[0].get("requested_train_batch_size")
        if requested_batch not in ("", None):
            original_requested_batch.setdefault(label, int(requested_batch))
        runs[label].extend(rows)

    if args.requested_train_batch_size is not None:
        runs = defaultdict(
            list,
            {
                label: run_rows
                for label, run_rows in runs.items()
                if original_requested_batch.get(label) == args.requested_train_batch_size
            },
        )
    for label, run_rows in runs.items():
        for row in run_rows:
            row["original_requested_train_batch_size"] = original_requested_batch.get(label, "")

    all_rows, runs = deduplicate_epochs(runs)
    add_stored_test_statistics(runs, args.test_statistics_dir, args.test_set_size)

    if not all_rows:
        filters = []
        if args.data_size is not None:
            filters.append(f"data_size={args.data_size}")
        if args.hiphop_l_max is not None:
            filters.append(f"hiphop_l_max={args.hiphop_l_max}")
        if args.hiphop_n_max is not None:
            filters.append(f"hiphop_n_max={args.hiphop_n_max}")
        filter_message = f" with {', '.join(filters)}" if filters else ""
        raise SystemExit(f"No epochs parsed from {args.log_dir}{filter_message}")

    csv_path = args.output_dir / "metrics.csv"
    columns = write_csv(all_rows, csv_path)
    run_summary_path = args.output_dir / "run_summary.csv"
    sweep_task_map_path = args.output_dir / "sweep_task_map.csv"
    summary_columns = [
        "run",
        "source_file",
        "sweep_task_id",
        "seed",
        "hiphop_l_max",
        "hiphop_n_max",
        "data_size",
        "total_params",
        "requested_train_batch_size",
        "original_requested_train_batch_size",
        "epochs",
    ]
    with run_summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_columns)
        writer.writeheader()
        for label, run_rows in sorted(runs.items()):
            first = run_rows[0]
            writer.writerow({column: first.get(column, "") for column in summary_columns[:-1]} | {"epochs": len(run_rows)})
    if jobs:
        write_sweep_task_map(jobs, runs, sweep_task_map_path)
    metrics = metric_columns(columns)

    metric_dir = args.output_dir / "by_metric"
    metric_dir.mkdir(exist_ok=True)
    for metric in metrics:
        filename = re.sub(r"[^A-Za-z0-9_.-]+", "_", metric) + ".png"
        plot_metric(metric, runs, metric_dir / filename)

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
    plot_dashboard(dashboard_metrics, runs, args.output_dir / "summary_dashboard.png")
    plot_force_accuracy_pareto(
        runs,
        args.output_dir / "force_mae_pareto.png",
        args.output_dir / "force_mae_pareto.csv",
        seed_filter=args.pareto_seed,
    )
    plot_best_metric_pareto(
        runs,
        args.output_dir / "best_metric_pareto.png",
        args.output_dir / "best_metric_pareto.csv",
        seed_filter=args.pareto_seed,
    )
    plot_paper_style_energy_comparison(
        runs,
        args.output_dir / "paper_style_energy_comparison.png",
        args.output_dir / "paper_style_energy_comparison.csv",
    )

    print(f"Parsed {len(all_rows)} epochs from {len(runs)} runs.")
    print(f"Wrote {csv_path}")
    print(f"Wrote {run_summary_path}")
    if jobs:
        print(f"Wrote {sweep_task_map_path}")
    print(f"Wrote {len(metrics)} metric plots to {metric_dir}")
    print(f"Wrote {args.output_dir / 'summary_dashboard.png'}")
    print(f"Wrote {args.output_dir / 'force_mae_pareto.png'}")
    print(f"Wrote {args.output_dir / 'best_metric_pareto.png'}")
    print(f"Wrote {args.output_dir / 'paper_style_energy_comparison.png'}")


if __name__ == "__main__":
    main()
