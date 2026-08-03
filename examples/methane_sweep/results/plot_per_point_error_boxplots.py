#!/usr/bin/env python3
"""Plot per-structure error distributions from methane external-test CSV files.

The script recursively finds ``*_per_point.csv`` files, makes one box per run
(or per architecture), and can optionally exclude unusually large errors using
an upper one-sided threshold:

    error > mean(error) + N * sample_std(error)

Filtering is performed independently for each run and never changes the source
CSV files. Every excluded row is written to a separate audit CSV.

Examples
--------
Raw energy absolute-error box plots:

    python plot_per_point_error_boxplots.py \
        --input-dir ../results/per_point/random_common_80k \
        --metric energy_abs_error

Remove values more than 3 standard deviations above their run mean:

    python plot_per_point_error_boxplots.py \
        --input-dir ../results/per_point/random_common_80k \
        --metric energy_abs_error \
        --remove-outliers --std-threshold 3

Plot force RMSE on a logarithmic y-axis:

    python plot_per_point_error_boxplots.py \
        --input-dir ../results/per_point/random_common_80k \
        --metric force_rmse --log-y
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "energy_abs_error": (
        "energy_abs_error",
        "energy_absolute_error",
        "abs_energy_error",
        "absolute_energy_error",
        "t_abs_error",
        "t_absolute_error",
    ),
    "force_rmse": (
        "force_rmse",
        "forces_rmse",
        "f_rmse",
        "per_structure_force_rmse",
    ),
    "force_mae": (
        "force_mae",
        "forces_mae",
        "f_mae",
        "per_structure_force_mae",
    ),
    "force_max_abs": (
        "force_max_abs_error",
        "force_max_absolute_error",
        "max_abs_force_error",
        "max_force_component_error",
        "force_max_abs",
    ),
}

SOURCE_INDEX_ALIASES = (
    "source_index",
    "dataset_index",
    "frame_index",
    "index",
)

RUN_RE = re.compile(
    r"l(?P<lmax>\d+)[_-]n(?P<nmax>\d+).*?seed(?P<seed>\d+)",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class RunInfo:
    path: Path
    label: str
    architecture: str
    seed: str
    metric_column: str
    source_index_column: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create box plots from per-point external-test errors, with an "
            "optional one-sided mean + N*standard-deviation outlier filter."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory recursively containing *_per_point.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Default: <input-dir>/boxplots. Source files are "
            "never modified."
        ),
    )
    parser.add_argument(
        "--metric",
        choices=tuple(METRIC_ALIASES),
        default="energy_abs_error",
        help="Known metric to plot. Default: energy_abs_error.",
    )
    parser.add_argument(
        "--metric-column",
        default=None,
        help="Exact CSV column name; overrides --metric and alias detection.",
    )
    parser.add_argument(
        "--glob",
        default="*_per_point.csv",
        help="Recursive input filename pattern. Default: *_per_point.csv.",
    )
    parser.add_argument(
        "--group-by",
        choices=("run", "architecture"),
        default="run",
        help="One box per run or pooled by architecture. Default: run.",
    )
    parser.add_argument(
        "--remove-outliers",
        action="store_true",
        help=(
            "Exclude upper-tail values above run_mean + std_threshold * "
            "run_sample_std before plotting."
        ),
    )
    parser.add_argument(
        "--std-threshold",
        type=float,
        default=3.0,
        help="Number of sample standard deviations for filtering. Default: 3.",
    )
    parser.add_argument(
        "--std-iterations",
        type=int,
        default=1,
        help=(
            "Recompute the threshold after removals this many times. Default: 1. "
            "Use a larger value, such as 10, for iterative clipping."
        ),
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=2,
        help="Minimum finite values required for a run. Default: 2.",
    )
    parser.add_argument(
        "--log-y",
        action="store_true",
        help="Use a logarithmic y-axis. Nonpositive values are omitted.",
    )
    parser.add_argument(
        "--hide-boxplot-fliers",
        action="store_true",
        help=(
            "Hide Matplotlib's 1.5-IQR flier markers without removing those data. "
            "This is separate from --remove-outliers."
        ),
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Custom plot title.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="PNG resolution. Default: 200.",
    )
    parser.add_argument(
        "--max-points-per-box",
        type=int,
        default=0,
        help=(
            "Randomly sample at most this many retained values per box for plotting. "
            "0 keeps all values. Summary statistics still use all retained values."
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=20260731,
        help="Seed used only for --max-points-per-box sampling.",
    )
    return parser.parse_args()


def normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def resolve_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    normalized_to_original = {normalize(column): column for column in columns}
    for candidate in candidates:
        match = normalized_to_original.get(normalize(candidate))
        if match is not None:
            return match
    return None


def read_header(path: Path) -> list[str]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        try:
            return next(reader)
        except StopIteration as exc:
            raise ValueError(f"CSV is empty: {path}") from exc


def infer_run_info(path: Path, args: argparse.Namespace) -> RunInfo:
    columns = read_header(path)

    if args.metric_column is not None:
        metric_column = resolve_column(columns, (args.metric_column,))
        if metric_column is None:
            raise ValueError(
                f"Requested metric column {args.metric_column!r} not found. "
                f"Available columns: {columns}"
            )
    else:
        metric_column = resolve_column(columns, METRIC_ALIASES[args.metric])
        if metric_column is None:
            raise ValueError(
                f"Could not find a column for metric {args.metric!r}. "
                f"Aliases tried: {METRIC_ALIASES[args.metric]}. "
                f"Available columns: {columns}. Use --metric-column if needed."
            )

    source_index_column = resolve_column(columns, SOURCE_INDEX_ALIASES)

    searchable = f"{path.parent.name}_{path.stem}"
    match = RUN_RE.search(searchable)
    if match:
        lmax = match.group("lmax")
        nmax = match.group("nmax")
        seed = match.group("seed")
        architecture = f"L{lmax}N{nmax}"
        label = f"{architecture}\nseed {seed}"
    else:
        architecture = "unknown"
        seed = "unknown"
        label = path.stem.removesuffix("_per_point")

    return RunInfo(
        path=path,
        label=label,
        architecture=architecture,
        seed=seed,
        metric_column=metric_column,
        source_index_column=source_index_column,
    )


def architecture_sort_key(architecture: str) -> tuple[int, int, str]:
    match = re.fullmatch(r"L(\d+)N(\d+)", architecture)
    if match:
        return int(match.group(1)), int(match.group(2)), architecture
    return math.inf, math.inf, architecture


def run_sort_key(run: RunInfo) -> tuple[int, int, int, str]:
    architecture_key = architecture_sort_key(run.architecture)
    try:
        seed_key = int(run.seed)
    except ValueError:
        seed_key = math.inf
    return architecture_key[0], architecture_key[1], seed_key, run.label


def clip_upper_std(
    values: np.ndarray,
    threshold_multiplier: float,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float | int]]]:
    """Return keep mask, exclusion iteration, and iteration statistics."""
    keep = np.ones(values.shape[0], dtype=bool)
    exclusion_iteration = np.zeros(values.shape[0], dtype=np.int32)
    history: list[dict[str, float | int]] = []

    for iteration in range(1, iterations + 1):
        current = values[keep]
        if current.size < 2:
            break

        mean = float(np.mean(current))
        std = float(np.std(current, ddof=1))
        threshold = mean + threshold_multiplier * std

        if not np.isfinite(std) or std == 0.0:
            remove = np.zeros_like(keep)
        else:
            remove = keep & (values > threshold)

        removed_count = int(remove.sum())
        history.append(
            {
                "iteration": iteration,
                "n_before": int(current.size),
                "mean": mean,
                "sample_std": std,
                "threshold": threshold,
                "n_removed": removed_count,
            }
        )

        if removed_count == 0:
            break

        exclusion_iteration[remove] = iteration
        keep[remove] = False

    return keep, exclusion_iteration, history


def safe_float_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "sample_std": float(np.std(values, ddof=1)) if values.size > 1 else float("nan"),
        "median": float(np.median(values)),
        "q1": float(np.quantile(values, 0.25)),
        "q3": float(np.quantile(values, 0.75)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def main() -> int:
    args = parse_args()

    if not args.input_dir.is_dir():
        print(f"Input directory does not exist: {args.input_dir}", file=sys.stderr)
        return 2
    if args.std_threshold <= 0:
        print("--std-threshold must be greater than zero", file=sys.stderr)
        return 2
    if args.std_iterations < 1:
        print("--std-iterations must be at least 1", file=sys.stderr)
        return 2
    if args.max_points_per_box < 0:
        print("--max-points-per-box cannot be negative", file=sys.stderr)
        return 2

    output_dir = args.output_dir or args.input_dir / "boxplots"
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(path for path in args.input_dir.rglob(args.glob) if path.is_file())
    if not paths:
        print(
            f"No files matching {args.glob!r} found below {args.input_dir}",
            file=sys.stderr,
        )
        return 1

    runs: list[RunInfo] = []
    for path in paths:
        try:
            runs.append(infer_run_info(path, args))
        except ValueError as exc:
            print(f"Skipping {path}: {exc}", file=sys.stderr)

    if not runs:
        print("No usable per-point CSV files were found.", file=sys.stderr)
        return 1

    runs.sort(key=run_sort_key)
    rng = np.random.default_rng(args.random_seed)

    retained_by_run: list[tuple[RunInfo, np.ndarray]] = []
    summary_rows: list[dict[str, object]] = []
    excluded_frames: list[pd.DataFrame] = []
    threshold_rows: list[dict[str, object]] = []

    for run in runs:
        usecols = [run.metric_column]
        if run.source_index_column and run.source_index_column not in usecols:
            usecols.append(run.source_index_column)

        frame = pd.read_csv(run.path, usecols=usecols)
        numeric = pd.to_numeric(frame[run.metric_column], errors="coerce").to_numpy(dtype=float)
        finite_mask = np.isfinite(numeric)
        if args.log_y:
            finite_mask &= numeric > 0

        values = numeric[finite_mask]
        source_rows = np.flatnonzero(finite_mask)

        if values.size < args.min_points:
            print(
                f"Skipping {run.path}: only {values.size} usable values found.",
                file=sys.stderr,
            )
            continue

        if args.remove_outliers:
            keep, exclusion_iteration, history = clip_upper_std(
                values,
                threshold_multiplier=args.std_threshold,
                iterations=args.std_iterations,
            )
        else:
            keep = np.ones(values.size, dtype=bool)
            exclusion_iteration = np.zeros(values.size, dtype=np.int32)
            history = []

        retained = values[keep]
        if retained.size < args.min_points:
            print(
                f"Skipping {run.path}: filtering left only {retained.size} values.",
                file=sys.stderr,
            )
            continue

        retained_by_run.append((run, retained))

        raw_stats = safe_float_summary(values)
        retained_stats = safe_float_summary(retained)
        summary_rows.append(
            {
                "run_label": run.label.replace("\n", " "),
                "architecture": run.architecture,
                "seed": run.seed,
                "source_file": str(run.path),
                "metric_column": run.metric_column,
                "n_csv_rows": int(frame.shape[0]),
                "n_usable_raw": int(values.size),
                "n_retained": int(retained.size),
                "n_removed": int((~keep).sum()),
                "removed_fraction": float((~keep).mean()),
                **{f"raw_{key}": value for key, value in raw_stats.items()},
                **{f"retained_{key}": value for key, value in retained_stats.items()},
            }
        )

        for item in history:
            threshold_rows.append(
                {
                    "run_label": run.label.replace("\n", " "),
                    "architecture": run.architecture,
                    "seed": run.seed,
                    "source_file": str(run.path),
                    "metric_column": run.metric_column,
                    "std_threshold": args.std_threshold,
                    **item,
                }
            )

        removed_local = np.flatnonzero(~keep)
        if removed_local.size:
            removed_source_rows = source_rows[removed_local]
            excluded = pd.DataFrame(
                {
                    "run_label": run.label.replace("\n", " "),
                    "architecture": run.architecture,
                    "seed": run.seed,
                    "source_file": str(run.path),
                    "csv_row_zero_based": removed_source_rows,
                    "metric_column": run.metric_column,
                    "metric_value": values[removed_local],
                    "exclusion_iteration": exclusion_iteration[removed_local],
                    "std_threshold": args.std_threshold,
                }
            )
            if run.source_index_column:
                excluded.insert(
                    4,
                    "source_index",
                    frame.iloc[removed_source_rows][run.source_index_column].to_numpy(),
                )
            excluded_frames.append(excluded)

    if not retained_by_run:
        print("No runs remained after loading/filtering.", file=sys.stderr)
        return 1

    if args.group_by == "run":
        labels = [run.label for run, _ in retained_by_run]
        complete_plot_values = [values for _, values in retained_by_run]
    else:
        pooled: dict[str, list[np.ndarray]] = {}
        for run, values in retained_by_run:
            pooled.setdefault(run.architecture, []).append(values)
        labels = sorted(pooled, key=architecture_sort_key)
        complete_plot_values = [np.concatenate(pooled[label]) for label in labels]

    plot_values: list[np.ndarray] = []
    for values in complete_plot_values:
        if args.max_points_per_box and values.size > args.max_points_per_box:
            indices = rng.choice(values.size, size=args.max_points_per_box, replace=False)
            plot_values.append(values[indices])
        else:
            plot_values.append(values)

    metric_name = args.metric_column or args.metric
    filter_suffix = (
        f"filtered_gt_mean_plus_{args.std_threshold:g}std"
        if args.remove_outliers
        else "raw"
    )
    grouping_suffix = f"by_{args.group_by}"
    stem = f"boxplot_{normalize(metric_name)}_{filter_suffix}_{grouping_suffix}"

    figure_width = max(8.0, 0.62 * len(labels) + 2.5)
    fig, ax = plt.subplots(figsize=(figure_width, 6.5))
    ax.boxplot(
        plot_values,
        # ``labels`` was renamed to ``tick_labels`` in Matplotlib 3.9 and
        # removed in newer releases.
        tick_labels=labels,
        showfliers=not args.hide_boxplot_fliers,
        patch_artist=True,
        medianprops={"linewidth": 1.5},
    )
    ax.set_ylabel(metric_name.replace("_", " "))
    ax.set_xlabel("Run" if args.group_by == "run" else "Architecture")
    if args.log_y:
        ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.25)

    if args.title:
        title = args.title
    elif args.remove_outliers:
        title = (
            f"{metric_name.replace('_', ' ').title()} by {args.group_by}\n"
            f"Values above each run mean + {args.std_threshold:g} sample SD removed"
        )
    else:
        title = f"{metric_name.replace('_', ' ').title()} by {args.group_by}"
    ax.set_title(title)

    if args.group_by == "run":
        ax.tick_params(axis="x", labelrotation=45)
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment("right")

    fig.tight_layout()
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    summary_path = output_dir / f"{stem}_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    group_summary_rows: list[dict[str, object]] = []
    for label, values in zip(labels, complete_plot_values, strict=True):
        group_summary_rows.append(
            {
                "group": label,
                "metric": metric_name,
                "n_values": int(values.size),
                **safe_float_summary(values),
            }
        )
    group_summary_path = output_dir / f"{stem}_group_summary.csv"
    group_summary = pd.DataFrame(group_summary_rows)
    group_summary.to_csv(group_summary_path, index=False)

    # Energy MAE and RMSE are aggregate metrics over signed energy errors and
    # must be reported separately.  The per-structure absolute-error boxplot
    # above is useful for diagnostics, but its mean is MAE, not RMSE.
    energy_by_group: dict[str, list[np.ndarray]] = {}
    for run in runs:
        energy_error_column = resolve_column(read_header(run.path), ("energy_error",))
        if energy_error_column is None:
            continue
        energy_frame = pd.read_csv(run.path, usecols=[energy_error_column])
        energy_errors = pd.to_numeric(
            energy_frame[energy_error_column], errors="coerce"
        ).to_numpy(dtype=float)
        energy_errors = energy_errors[np.isfinite(energy_errors)]
        if args.remove_outliers and energy_errors.size >= 2:
            keep, _, _ = clip_upper_std(
                np.abs(energy_errors),
                threshold_multiplier=args.std_threshold,
                iterations=args.std_iterations,
            )
            energy_errors = energy_errors[keep]
        energy_by_group.setdefault(run.architecture, []).append(energy_errors)

    energy_metric_rows: list[dict[str, object]] = []
    for architecture in sorted(energy_by_group, key=architecture_sort_key):
        errors = np.concatenate(energy_by_group[architecture])
        energy_metric_rows.append(
            {
                "architecture": architecture,
                "n_values": int(errors.size),
                "energy_mae": float(np.mean(np.abs(errors))),
                "energy_rmse": float(np.sqrt(np.mean(errors**2))),
                "energy_error_sample_std": float(np.std(errors, ddof=1)),
            }
        )
    energy_metric_path = output_dir / f"{stem}_energy_mae_rmse_summary.csv"
    if energy_metric_rows:
        pd.DataFrame(energy_metric_rows).to_csv(energy_metric_path, index=False)

    excluded_path = output_dir / f"{stem}_excluded_points.csv"
    if excluded_frames:
        pd.concat(excluded_frames, ignore_index=True).to_csv(excluded_path, index=False)
    else:
        pd.DataFrame(
            columns=(
                "run_label",
                "architecture",
                "seed",
                "source_file",
                "source_index",
                "csv_row_zero_based",
                "metric_column",
                "metric_value",
                "exclusion_iteration",
                "std_threshold",
            )
        ).to_csv(excluded_path, index=False)

    thresholds_path = output_dir / f"{stem}_thresholds.csv"
    pd.DataFrame(threshold_rows).to_csv(thresholds_path, index=False)

    total_raw = sum(int(row["n_usable_raw"]) for row in summary_rows)
    total_removed = sum(int(row["n_removed"]) for row in summary_rows)
    print(f"Plotted {len(labels)} boxes from {len(retained_by_run)} runs.")
    print(f"Usable values: {total_raw:,}")
    print(f"Excluded values: {total_removed:,}")
    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")
    print(f"Summary: {summary_path}")
    print(f"Pooled-group summary: {group_summary_path}")
    if energy_metric_rows:
        print(f"Energy MAE/RMSE summary: {energy_metric_path}")
    print(f"Excluded-point audit: {excluded_path}")
    print(f"Threshold history: {thresholds_path}")
    print(f"\n{metric_name} mean +/- sample std by {args.group_by}:")
    for row in group_summary_rows:
        print(
            f"  {row['group']}: {row['mean']:.8g} +/- "
            f"{row['sample_std']:.8g} (n={row['n_values']:,})"
        )
    if energy_metric_rows:
        print(f"\nEnergy MAE and RMSE by architecture:")
        for row in energy_metric_rows:
            print(
                f"  {row['architecture']}: MAE={row['energy_mae']:.8g}, "
                f"RMSE={row['energy_rmse']:.8g} (n={row['n_values']:,})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
