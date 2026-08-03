#!/usr/bin/env python3
"""Plot raw metrics and two filtering methods in one figure."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_DIR = Path(__file__).resolve().parent
DEFAULT_BASE = RESULTS_DIR / "per_point/random_common_80k"
METRICS = ("energy_mae", "energy_mse", "force_mae", "force_rmse")
METRIC_LABELS = {
    "energy_mae": "Energy MAE",
    "energy_mse": "Energy MSE",
    "force_mae": "Force MAE",
    "force_rmse": "Force RMSE",
}
CONDITIONS = (
    ("raw", "Raw", "#3b6fb6"),
    ("previous_filter", "Original catastrophe filter", "#dc7f2a"),
    ("single_point", "Remove source 3937730 only", "#3a9d5d"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare raw, original-filtered, and single-point-filtered metrics."
    )
    parser.add_argument(
        "--previous-filter-input",
        type=Path,
        default=DEFAULT_BASE
        / "clean_catastrophe_analysis_previous_filter/run_metrics_raw_and_filtered.csv",
    )
    parser.add_argument(
        "--single-point-input",
        type=Path,
        default=DEFAULT_BASE
        / "clean_catastrophe_analysis_single_point/run_metrics_raw_and_filtered.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_BASE / "clean_catastrophe_analysis_filter_comparison",
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def architecture_key(name: str) -> tuple[int, int]:
    match = re.fullmatch(r"L(\d+)N(\d+)", name)
    if match is None:
        return (10**9, 10**9)
    return int(match.group(1)), int(match.group(2))


def load_combined(previous_path: Path, single_path: Path) -> pd.DataFrame:
    previous = pd.read_csv(previous_path)
    single = pd.read_csv(single_path)
    keys = ["architecture", "seed", "run_label", "metric"]
    required = set(keys) | {"raw", "filtered"}
    for path, frame in ((previous_path, previous), (single_path, single)):
        missing = required - set(frame.columns)
        if missing:
            raise SystemExit(f"Missing columns in {path}: {sorted(missing)}")
        if frame.duplicated(keys).any():
            raise SystemExit(f"Duplicate architecture/seed/metric rows in {path}")

    merged = previous[keys + ["raw", "filtered"]].merge(
        single[keys + ["raw", "filtered"]],
        on=keys,
        how="outer",
        suffixes=("_previous", "_single"),
        validate="one_to_one",
        indicator=True,
    )
    if not (merged["_merge"] == "both").all():
        raise SystemExit("The two inputs do not contain identical runs and metrics")
    if not np.allclose(
        merged["raw_previous"], merged["raw_single"], equal_nan=True
    ):
        raise SystemExit("Raw values differ between the two input analyses")

    return merged[keys].assign(
        raw=merged["raw_previous"],
        previous_filter=merged["filtered_previous"],
        single_point=merged["filtered_single"],
    )


def plot_comparison(frame: pd.DataFrame, output_dir: Path, dpi: int) -> tuple[Path, Path]:
    architectures = sorted(frame["architecture"].unique(), key=architecture_key)
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.0))
    rng = np.random.default_rng(20260731)
    offsets = (-0.22, 0.0, 0.22)

    for ax, metric in zip(axes.flat, METRICS, strict=True):
        subset = frame[frame["metric"] == metric]
        for condition_index, ((column, label, color), offset) in enumerate(
            zip(CONDITIONS, offsets, strict=True)
        ):
            means: list[float] = []
            stds: list[float] = []
            for architecture_index, architecture in enumerate(architectures):
                values = subset.loc[
                    subset["architecture"] == architecture, column
                ].to_numpy(float)
                means.append(float(np.mean(values)))
                stds.append(float(np.std(values, ddof=1)) if values.size > 1 else 0.0)
                jitter = rng.uniform(-0.035, 0.035, values.size)
                ax.scatter(
                    architecture_index + offset + jitter,
                    values,
                    s=20,
                    alpha=0.45,
                    color=color,
                    edgecolors="none",
                    zorder=2,
                )
            ax.errorbar(
                np.arange(len(architectures), dtype=float) + offset,
                means,
                yerr=stds,
                fmt="o",
                markersize=7,
                capsize=4,
                linewidth=1.5,
                color=color,
                label=label,
                zorder=3,
            )

        ax.set_xticks(np.arange(len(architectures)))
        ax.set_xticklabels(architectures)
        ax.set_title(METRIC_LABELS[metric])
        ax.set_yscale("log")
        raw_l4n4 = subset.loc[
            subset["architecture"] == "L4N4", "raw"
        ].to_numpy(float)
        if raw_l4n4.size:
            raw_l4n4_std = (
                float(np.std(raw_l4n4, ddof=1)) if raw_l4n4.size > 1 else 0.0
            )
            raw_l4n4_bottom = float(np.mean(raw_l4n4)) - raw_l4n4_std
            if raw_l4n4_bottom > 0:
                ax.set_ylim(bottom=raw_l4n4_bottom / 1.05)
        ax.set_ylabel(METRIC_LABELS[metric] + " (log scale)")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle(
        "Random common 80k external test: raw vs two filtering methods\n"
        "Dots are seeds; markers are architecture mean ± sample SD",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = output_dir / "architecture_metrics_raw_vs_two_filter_methods.png"
    pdf = output_dir / "architecture_metrics_raw_vs_two_filter_methods.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main() -> int:
    args = parse_args()
    if args.dpi <= 0:
        raise SystemExit("--dpi must be positive")
    frame = load_combined(args.previous_filter_input, args.single_point_input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "run_metrics_three_conditions.csv", index=False)
    summary = (
        frame.melt(
            id_vars=["architecture", "seed", "run_label", "metric"],
            value_vars=[condition[0] for condition in CONDITIONS],
            var_name="condition",
            value_name="value",
        )
        .groupby(["architecture", "metric", "condition"], sort=False)["value"]
        .agg(n_seeds="size", mean="mean", seed_std="std")
        .reset_index()
    )
    summary.to_csv(args.output_dir / "architecture_metrics_three_conditions.csv", index=False)
    png, pdf = plot_comparison(frame, args.output_dir, args.dpi)
    print(f"Combined rows: {len(frame)}")
    print(f"PNG: {png}")
    print(f"PDF: {pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
