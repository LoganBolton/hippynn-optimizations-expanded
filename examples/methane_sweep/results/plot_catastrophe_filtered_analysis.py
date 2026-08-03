#!/usr/bin/env python3
"""Create a conservative, auditable external-test catastrophe analysis.

This analysis does *not* delete structures or edit the input files. A shared
source-level mask first removes structures that several architectures find
extremely difficult. A metric value for one remaining run/structure pair is
then flagged only when both conditions hold:

1. log1p(error) is more than ``--robust-z-threshold`` scaled MAD above the
   median log-error for that run; and
2. the error is at least ``--consensus-ratio-threshold`` times the median
   error made by the other runs on the same source structure.

The first condition finds an extreme result within a run.  The second protects
structures which are genuinely hard for all models.  Energy MAE and energy MSE
use the same absolute-energy-error flags, but are computed as distinct
aggregate metrics.  Force MAE and force RMSE are likewise reported distinctly.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd


RUN_RE = re.compile(
    r"l(?P<lmax>\d+)[_-]n(?P<nmax>\d+).*?seed(?P<seed>\d+)", re.IGNORECASE
)


@dataclass(frozen=True)
class Run:
    path: Path
    architecture: str
    seed: int

    @property
    def label(self) -> str:
        return f"{self.architecture} s{self.seed}"


DIAGNOSTICS = {
    "energy": "abs_energy_error",
    "force_mae": "force_mae",
    "force_rmse": "force_rmse",
}

REPORT_METRICS = (
    "energy_mae",
    "energy_mse",
    "force_mae",
    "force_rmse",
)

METRIC_LABELS = {
    "energy_mae": "Energy MAE",
    "energy_mse": "Energy MSE",
    "force_mae": "Force MAE",
    "force_rmse": "Force RMSE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot raw and conservatively catastrophe-filtered test metrics."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--robust-z-threshold", type=float, default=10.0)
    parser.add_argument("--consensus-ratio-threshold", type=float, default=100.0)
    parser.add_argument(
        "--shared-force-rmse-threshold",
        type=float,
        default=200.0,
        help="Shared source threshold applied to architecture-mean force RMSE.",
    )
    parser.add_argument(
        "--shared-hard-min-architectures",
        type=int,
        default=2,
        help="Minimum architectures exceeding the shared threshold.",
    )
    parser.add_argument(
        "--shared-only",
        action="store_true",
        help="Apply only the shared hard-source mask; keep all other points.",
    )
    parser.add_argument(
        "--no-shared-hard-filter",
        action="store_true",
        help="Use only the original run-specific catastrophe filter.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def architecture_key(name: str) -> tuple[int, int]:
    match = re.fullmatch(r"L(\d+)N(\d+)", name)
    if match is None:
        return (10**9, 10**9)
    return int(match.group(1)), int(match.group(2))


def discover_runs(input_dir: Path) -> list[Run]:
    runs: list[Run] = []
    for path in sorted(input_dir.glob("*_per_point.csv")):
        match = RUN_RE.search(path.stem)
        if match is None:
            continue
        runs.append(
            Run(
                path=path,
                architecture=f"L{match.group('lmax')}N{match.group('nmax')}",
                seed=int(match.group("seed")),
            )
        )
    runs.sort(key=lambda run: (*architecture_key(run.architecture), run.seed))
    return runs


def scaled_mad(values: np.ndarray) -> float:
    median = float(np.median(values))
    return 1.4826 * float(np.median(np.abs(values - median)))


def leave_one_run_out_median(values: np.ndarray) -> np.ndarray:
    """Return row medians of every column except the current column."""
    medians = np.empty_like(values)
    for column in range(values.shape[1]):
        others = np.delete(values, column, axis=1)
        medians[:, column] = np.nanmedian(others, axis=1)
    return medians


def metric_value(metric: str, frame: pd.DataFrame, keep: np.ndarray) -> float:
    if not np.any(keep):
        return float("nan")
    if metric == "energy_mae":
        return float(np.mean(frame.loc[keep, "abs_energy_error"]))
    if metric == "energy_mse":
        errors = frame.loc[keep, "energy_error"].to_numpy(float)
        return float(np.mean(np.square(errors)))
    if metric == "force_mae":
        return float(np.mean(frame.loc[keep, "force_mae"]))
    if metric == "force_rmse":
        per_structure_rmse = frame.loc[keep, "force_rmse"].to_numpy(float)
        return float(np.sqrt(np.mean(np.square(per_structure_rmse))))
    raise ValueError(metric)


def plot_metric_comparison(
    run_metrics: pd.DataFrame,
    architectures: list[str],
    output_dir: Path,
    dpi: int,
) -> tuple[Path, Path]:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    colors = {"raw": "#3b6fb6", "filtered": "#dc7f2a"}
    rng = np.random.default_rng(20260731)

    for ax, metric in zip(axes.flat, REPORT_METRICS, strict=True):
        subset = run_metrics[run_metrics["metric"] == metric]
        for condition_index, condition in enumerate(("raw", "filtered")):
            means: list[float] = []
            stds: list[float] = []
            for architecture_index, architecture in enumerate(architectures):
                values = subset.loc[
                    subset["architecture"] == architecture, condition
                ].to_numpy(float)
                means.append(float(np.mean(values)))
                stds.append(float(np.std(values, ddof=1)) if values.size > 1 else 0.0)
                jitter = rng.uniform(-0.045, 0.045, values.size)
                offset = -0.11 if condition == "raw" else 0.11
                ax.scatter(
                    architecture_index + offset + jitter,
                    values,
                    s=22,
                    alpha=0.55,
                    color=colors[condition],
                    edgecolors="none",
                    zorder=2,
                )
            x = np.arange(len(architectures), dtype=float)
            offset = -0.11 if condition == "raw" else 0.11
            ax.errorbar(
                x + offset,
                means,
                yerr=stds,
                fmt="o",
                markersize=7,
                capsize=4,
                linewidth=1.5,
                color=colors[condition],
                label=f"{condition.title()} mean ± seed SD",
                zorder=3,
            )
        ax.set_xticks(np.arange(len(architectures)))
        ax.set_xticklabels(architectures)
        ax.set_title(METRIC_LABELS[metric])
        ax.set_yscale("log")
        ax.set_ylabel(METRIC_LABELS[metric] + " (log scale)")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle(
        "Random common 80k external test: raw vs catastrophe-filtered\n"
        "Dots are seeds; markers are architecture mean ± sample SD",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = output_dir / "architecture_metrics_raw_vs_catastrophe_filtered.png"
    pdf = output_dir / "architecture_metrics_raw_vs_catastrophe_filtered.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def plot_filter_counts(
    run_metrics: pd.DataFrame,
    architectures: list[str],
    output_dir: Path,
    dpi: int,
) -> tuple[Path, Path]:
    count_frame = run_metrics.pivot_table(
        index=["architecture", "seed"],
        columns="metric",
        values="n_filtered",
        aggfunc="first",
    ).reset_index()
    fig, ax = plt.subplots(figsize=(11, 6.5))
    width = 0.19
    x = np.arange(len(architectures), dtype=float)
    colors = ("#4c78a8", "#f58518", "#54a24b", "#e45756")
    for metric_index, (metric, color) in enumerate(
        zip(REPORT_METRICS, colors, strict=True)
    ):
        totals = np.array(
            [
                count_frame.loc[
                    count_frame["architecture"] == architecture, metric
                ].sum()
                for architecture in architectures
            ],
            dtype=float,
        )
        positions = x + (metric_index - 1.5) * width
        bars = ax.bar(
            positions,
            totals,
            width=width,
            label=METRIC_LABELS[metric],
            color=color,
        )
        ax.bar_label(bars, fmt="%.0f", fontsize=8, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(architectures)
    ax.set_ylabel("Filtered run–structure predictions")
    ax.set_title("How much the catastrophe filter removes (lower is better)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncols=2)
    fig.tight_layout()
    png = output_dir / "catastrophe_filtered_prediction_counts.png"
    pdf = output_dir / "catastrophe_filtered_prediction_counts.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def plot_metrics_table(
    architecture_summary: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> tuple[Path, Path]:
    """Render the architecture summary as a publication-friendly table."""
    display = architecture_summary.copy()
    display["Metric"] = display["metric"].map(METRIC_LABELS)
    display["Architecture"] = display["architecture"]
    display["Raw mean ± seed SD"] = display.apply(
        lambda row: f"{row.raw_mean:.3f} ± {row.raw_seed_std:.3f}", axis=1
    )
    display["Filtered mean ± seed SD"] = display.apply(
        lambda row: f"{row.filtered_mean:.3f} ± {row.filtered_seed_std:.3f}",
        axis=1,
    )
    display["Removed"] = display["n_predictions_filtered"].astype(int).astype(str)
    display["Seeds"] = display["n_seeds"].astype(int).astype(str)
    display = display[
        [
            "Architecture",
            "Metric",
            "Seeds",
            "Raw mean ± seed SD",
            "Filtered mean ± seed SD",
            "Removed",
        ]
    ]

    fig, ax = plt.subplots(figsize=(13.5, 8.2))
    ax.axis("off")
    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.13, 0.15, 0.08, 0.25, 0.25, 0.10],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.0, 1.55)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#b8b8b8")
        if row == 0:
            cell.set_facecolor("#34495e")
            cell.set_text_props(color="white", weight="bold")
        elif (row - 1) // len(REPORT_METRICS) % 2 == 0:
            cell.set_facecolor("#f2f5f7")
        else:
            cell.set_facecolor("white")
    ax.set_title(
        "Random common 80k external test: raw vs catastrophe-filtered\n"
        "Architecture averages across seeds (± sample standard deviation)",
        fontsize=15,
        pad=18,
    )
    fig.tight_layout()
    png = output_dir / "architecture_metrics_raw_vs_catastrophe_filtered_table.png"
    pdf = output_dir / "architecture_metrics_raw_vs_catastrophe_filtered_table.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def plot_flag_heatmap(
    diagnostic: str,
    raw: pd.DataFrame,
    ratios: pd.DataFrame,
    flags: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> Path | None:
    flagged_sources = flags.index[flags.any(axis=1)]
    if flagged_sources.empty:
        return None
    order = ratios.loc[flagged_sources].max(axis=1).sort_values(ascending=False).index
    displayed = ratios.loc[order].where(flags.loc[order])
    finite = displayed.to_numpy()[np.isfinite(displayed.to_numpy())]
    vmin = max(1.0, float(np.min(finite)))
    vmax = max(vmin * 1.01, float(np.max(finite)))

    fig_width = max(12.0, 0.45 * displayed.shape[1] + 4.0)
    fig_height = max(5.5, 0.32 * displayed.shape[0] + 2.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(
        displayed.to_numpy(),
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
        norm=LogNorm(vmin=vmin, vmax=vmax),
    )
    ax.set_xticks(np.arange(displayed.shape[1]))
    ax.set_xticklabels(displayed.columns, rotation=90, fontsize=8)
    ax.set_yticks(np.arange(displayed.shape[0]))
    ax.set_yticklabels(displayed.index.astype(str), fontsize=8)
    ax.set_xlabel("Model configuration and seed")
    ax.set_ylabel("Source index")
    ax.set_title(
        f"Flagged {diagnostic.replace('_', ' ')} predictions\n"
        "Color = error / median error of other runs; blank = retained"
    )
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("times worse than other-run median")
    fig.tight_layout()
    path = output_dir / f"flagged_predictions_{diagnostic}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> int:
    args = parse_args()
    if args.shared_only and args.no_shared_hard_filter:
        raise SystemExit("--shared-only and --no-shared-hard-filter are incompatible")
    if (
        args.robust_z_threshold <= 0
        or args.consensus_ratio_threshold <= 1
        or args.shared_force_rmse_threshold <= 0
        or args.shared_hard_min_architectures < 1
    ):
        raise SystemExit("Invalid catastrophe/shared-hard-point thresholds")
    runs = discover_runs(args.input_dir)
    if not runs:
        raise SystemExit(f"No per-point run CSVs found in {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    columns = [
        "source_index",
        "energy_error",
        "abs_energy_error",
        "force_mae",
        "force_rmse",
    ]
    frames: dict[str, pd.DataFrame] = {}
    common_index: pd.Index | None = None
    for run in runs:
        frame = pd.read_csv(run.path, usecols=columns).set_index("source_index")
        if frame.index.has_duplicates:
            raise SystemExit(f"Duplicate source indices in {run.path}")
        frames[run.label] = frame
        common_index = frame.index if common_index is None else common_index.intersection(frame.index)
    assert common_index is not None
    common_index = common_index.sort_values()
    if len(common_index) == 0:
        raise SystemExit("Runs have no common source indices")

    # Average seeds within each architecture before deciding whether a source
    # is broadly difficult. This prevents one pathological seed from defining
    # the shared test-set mask. The mask is then applied to every architecture
    # and every reported metric.
    architecture_force_rmse: dict[str, pd.Series] = {}
    for architecture in sorted({run.architecture for run in runs}, key=architecture_key):
        architecture_runs = [run for run in runs if run.architecture == architecture]
        values = pd.DataFrame(
            {
                run.label: frames[run.label].loc[common_index, "force_rmse"]
                for run in architecture_runs
            },
            index=common_index,
        )
        architecture_force_rmse[architecture] = values.mean(axis=1)

    architecture_force_rmse_frame = pd.DataFrame(architecture_force_rmse)
    shared_hard_architecture_count = (
        architecture_force_rmse_frame >= args.shared_force_rmse_threshold
    ).sum(axis=1)
    shared_hard_mask = (
        shared_hard_architecture_count >= args.shared_hard_min_architectures
    )
    if args.no_shared_hard_filter:
        shared_hard_mask = pd.Series(False, index=common_index)
    shared_hard_audit = architecture_force_rmse_frame.copy()
    shared_hard_audit.insert(0, "n_architectures_at_or_above_threshold", shared_hard_architecture_count)
    shared_hard_audit.insert(0, "shared_hard", shared_hard_mask)
    shared_hard_audit.insert(0, "source_index", shared_hard_audit.index)
    shared_hard_audit.to_csv(
        args.output_dir / "shared_hard_point_audit.csv", index=False
    )

    diagnostic_data: dict[str, dict[str, pd.DataFrame]] = {}
    audit_frames: list[pd.DataFrame] = []
    for diagnostic, column in DIAGNOSTICS.items():
        raw = pd.DataFrame(
            {
                run.label: frames[run.label].loc[common_index, column]
                for run in runs
            },
            index=common_index,
        ).astype(float)
        values = raw.to_numpy()
        log_values = np.log1p(np.maximum(values, 0.0))
        robust_z = np.empty_like(log_values)
        for column_index in range(log_values.shape[1]):
            run_log = log_values[:, column_index]
            median = float(np.median(run_log))
            mad = scaled_mad(run_log)
            robust_z[:, column_index] = (
                (run_log - median) / mad if mad > 0 else np.zeros_like(run_log)
            )
        other_median = leave_one_run_out_median(values)
        ratio = np.divide(
            values,
            other_median,
            out=np.full_like(values, np.inf),
            where=other_median > 0,
        )
        flags_array = (
            (robust_z > args.robust_z_threshold)
            & (ratio >= args.consensus_ratio_threshold)
            & np.isfinite(values)
        )
        flags = pd.DataFrame(flags_array, index=common_index, columns=raw.columns)
        ratios = pd.DataFrame(ratio, index=common_index, columns=raw.columns)
        robust_z_frame = pd.DataFrame(
            robust_z, index=common_index, columns=raw.columns
        )
        diagnostic_data[diagnostic] = {
            "raw": raw,
            "flags": flags,
            "ratios": ratios,
            "robust_z": robust_z_frame,
        }

        row_index, column_index = np.nonzero(flags_array)
        if row_index.size:
            audit_frames.append(
                pd.DataFrame(
                    {
                        "diagnostic": diagnostic,
                        "source_index": common_index.to_numpy()[row_index],
                        "run_label": raw.columns.to_numpy()[column_index],
                        "error": values[row_index, column_index],
                        "other_runs_median_error": other_median[row_index, column_index],
                        "ratio_to_other_runs_median": ratio[row_index, column_index],
                        "robust_z_log_error": robust_z[row_index, column_index],
                    }
                )
            )

    run_rows: list[dict[str, object]] = []
    for run in runs:
        frame = frames[run.label].loc[common_index].reset_index()
        finite = np.isfinite(frame["energy_error"].to_numpy(float))
        for metric in REPORT_METRICS:
            diagnostic = "energy" if metric.startswith("energy_") else metric
            flags = diagnostic_data[diagnostic]["flags"][run.label].to_numpy(bool)
            raw_keep = finite.copy()
            if metric.startswith("force_"):
                raw_keep = np.isfinite(frame[metric].to_numpy(float))
            filtered_keep = raw_keep & ~shared_hard_mask.to_numpy()
            if not args.shared_only:
                filtered_keep &= ~flags
            run_rows.append(
                {
                    "architecture": run.architecture,
                    "seed": run.seed,
                    "run_label": run.label,
                    "source_file": str(run.path),
                    "metric": metric,
                    "raw": metric_value(metric, frame, raw_keep),
                    "filtered": metric_value(metric, frame, filtered_keep),
                    "n_total": int(raw_keep.sum()),
                    "n_shared_hard": int(
                        (raw_keep & shared_hard_mask.to_numpy()).sum()
                    ),
                    "n_diagnostic_filtered": int(
                        (raw_keep & flags & ~shared_hard_mask.to_numpy()).sum()
                    ),
                    "n_filtered": int((raw_keep & ~filtered_keep).sum()),
                    "filtered_fraction": float(
                        (raw_keep & ~filtered_keep).sum() / raw_keep.sum()
                    ),
                }
            )
    run_metrics = pd.DataFrame(run_rows)
    run_metrics.to_csv(args.output_dir / "run_metrics_raw_and_filtered.csv", index=False)

    architecture_summary = (
        run_metrics.groupby(["architecture", "metric"], sort=False)
        .agg(
            n_seeds=("seed", "size"),
            raw_mean=("raw", "mean"),
            raw_seed_std=("raw", "std"),
            filtered_mean=("filtered", "mean"),
            filtered_seed_std=("filtered", "std"),
            n_predictions_filtered=("n_filtered", "sum"),
            n_predictions_total=("n_total", "sum"),
        )
        .reset_index()
    )
    architecture_summary["filtered_fraction"] = (
        architecture_summary["n_predictions_filtered"]
        / architecture_summary["n_predictions_total"]
    )
    architecture_summary["raw_mean_plus_minus_std"] = architecture_summary.apply(
        lambda row: f"{row.raw_mean:.3f} ± {row.raw_seed_std:.3f}", axis=1
    )
    architecture_summary["filtered_mean_plus_minus_std"] = architecture_summary.apply(
        lambda row: f"{row.filtered_mean:.3f} ± {row.filtered_seed_std:.3f}", axis=1
    )
    architecture_summary.to_csv(
        args.output_dir / "architecture_metrics_raw_and_filtered.csv", index=False
    )

    if audit_frames:
        audit = pd.concat(audit_frames, ignore_index=True)
        metadata = pd.DataFrame(
            [
                {
                    "run_label": run.label,
                    "architecture": run.architecture,
                    "seed": run.seed,
                    "source_file": str(run.path),
                }
                for run in runs
            ]
        )
        audit = audit.merge(metadata, on="run_label", how="left")
        audit = audit.sort_values(
            ["diagnostic", "ratio_to_other_runs_median"], ascending=[True, False]
        )
    else:
        audit = pd.DataFrame(
            columns=(
                "diagnostic",
                "source_index",
                "run_label",
                "error",
                "other_runs_median_error",
                "ratio_to_other_runs_median",
                "robust_z_log_error",
                "architecture",
                "seed",
                "source_file",
            )
        )
    audit.to_csv(args.output_dir / "filtered_prediction_audit.csv", index=False)

    removal_rows: list[dict[str, object]] = []
    for architecture in sorted({run.architecture for run in runs}, key=architecture_key):
        architecture_runs = [run for run in runs if run.architecture == architecture]
        labels = [run.label for run in architecture_runs]
        architecture_audit = audit[audit["architecture"] == architecture]
        removal_rows.append(
            {
                "architecture": architecture,
                "n_runs": len(architecture_runs),
                "run_structure_predictions_per_diagnostic": len(architecture_runs)
                * len(common_index),
                "energy_predictions_filtered": int(
                    diagnostic_data["energy"]["flags"][labels].to_numpy().sum()
                ),
                "force_mae_predictions_filtered": int(
                    diagnostic_data["force_mae"]["flags"][labels].to_numpy().sum()
                ),
                "force_rmse_predictions_filtered": int(
                    diagnostic_data["force_rmse"]["flags"][labels].to_numpy().sum()
                ),
                "unique_source_structures_in_any_filter": int(
                    architecture_audit["source_index"].nunique()
                ),
                "shared_hard_source_structures_deleted": int(shared_hard_mask.sum()),
                "complete_source_structures_deleted": int(shared_hard_mask.sum()),
            }
        )
    removal_counts = pd.DataFrame(removal_rows)
    removal_counts.loc[len(removal_counts)] = {
        "architecture": "ALL",
        "n_runs": len(runs),
        "run_structure_predictions_per_diagnostic": len(runs) * len(common_index),
        "energy_predictions_filtered": int(
            diagnostic_data["energy"]["flags"].to_numpy().sum()
        ),
        "force_mae_predictions_filtered": int(
            diagnostic_data["force_mae"]["flags"].to_numpy().sum()
        ),
        "force_rmse_predictions_filtered": int(
            diagnostic_data["force_rmse"]["flags"].to_numpy().sum()
        ),
        "unique_source_structures_in_any_filter": int(audit["source_index"].nunique()),
        "shared_hard_source_structures_deleted": int(shared_hard_mask.sum()),
        "complete_source_structures_deleted": int(shared_hard_mask.sum()),
    }
    removal_counts.to_csv(args.output_dir / "filter_removal_counts.csv", index=False)

    architectures = sorted(run_metrics["architecture"].unique(), key=architecture_key)
    metric_png, metric_pdf = plot_metric_comparison(
        run_metrics, architectures, args.output_dir, args.dpi
    )
    table_png, table_pdf = plot_metrics_table(
        architecture_summary, args.output_dir, args.dpi
    )
    count_png, count_pdf = plot_filter_counts(
        run_metrics, architectures, args.output_dir, args.dpi
    )
    heatmap_paths: list[Path] = []
    for diagnostic in DIAGNOSTICS:
        item = diagnostic_data[diagnostic]
        path = plot_flag_heatmap(
            diagnostic,
            item["raw"],
            item["ratios"],
            item["flags"],
            args.output_dir,
            args.dpi,
        )
        if path is not None:
            heatmap_paths.append(path)

    readme = args.output_dir / "README.txt"
    filter_mode = (
        "Only the shared hard-source mask was applied (--shared-only).\n\n"
        if args.shared_only
        else "Only the original run-specific catastrophe mask was applied (--no-shared-hard-filter).\n\n"
        if args.no_shared_hard_filter
        else "The run-specific catastrophe mask was also applied.\n\n"
    )
    readme.write_text(
        "Conservative catastrophe-filtered analysis\n"
        "==========================================\n\n"
        f"Input: {args.input_dir.resolve()}\n"
        f"Runs: {len(runs)}; common structures per run: {len(common_index)}\n\n"
        "A shared hard-source mask is applied to every architecture and metric. "
        "Seeds are averaged within architecture; a source is removed when at "
        f"least {args.shared_hard_min_architectures} architectures have force "
        f"RMSE >= {args.shared_force_rmse_threshold:g}.\n"
        f"Shared hard source structures removed: {int(shared_hard_mask.sum()):,}.\n\n"
        + filter_mode
        + "No input rows were deleted or changed. A run-structure prediction is "
        "excluded from the diagnostic filtered metric only when:\n"
        f"  robust z of log1p(error) > {args.robust_z_threshold:g}, and\n"
        f"  error / median(other runs' error) >= {args.consensus_ratio_threshold:g}.\n\n"
        "Energy MAE = mean(abs(energy_error)).\n"
        "Energy MSE = mean(energy_error squared).\n"
        "Force MAE = mean(per-structure force_mae).\n"
        "Force RMSE = sqrt(mean(per-structure force_rmse squared)).\n\n"
        "Energy MAE and MSE share the energy diagnostic mask. Force MAE and "
        "Force RMSE have their own masks. Raw metrics remain the official "
        "unmodified test-set results.\n",
        encoding="utf-8",
    )

    print(f"Runs: {len(runs)}; common source indices: {len(common_index):,}")
    print(
        "Shared hard source structures removed: "
        f"{int(shared_hard_mask.sum()):,} "
        f"(>= {args.shared_force_rmse_threshold:g} force RMSE in "
        f"at least {args.shared_hard_min_architectures} architectures)"
    )
    print(f"Flagged diagnostic predictions: {len(audit):,}")
    print(f"Metrics figure: {metric_png}")
    print(f"Metrics PDF: {metric_pdf}")
    print(f"Metrics table: {table_png}")
    print(f"Metrics table PDF: {table_pdf}")
    print(f"Counts figure: {count_png}")
    print(f"Counts PDF: {count_pdf}")
    for path in heatmap_paths:
        print(f"Audit heatmap: {path}")
    print(f"Tables and definition: {args.output_dir}")
    print("\nArchitecture summary:")
    print(
        architecture_summary[
            [
                "architecture",
                "metric",
                "raw_mean_plus_minus_std",
                "filtered_mean_plus_minus_std",
                "n_predictions_filtered",
            ]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
