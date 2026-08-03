#!/usr/bin/env python3
"""Plot the worst external-test structures across every model run and seed.

Each run is standardized independently, using
``(error - run_mean) / run_sample_std``.  A structure therefore has the same
meaningful scale across configurations: its number of sample standard
deviations above that particular run's mean error.

The heatmaps retain only values at or above ``--outlier-sigma``.  Exact raw
errors and sigma scores are also written to CSV, both for every outlier and
for every run on the displayed worst structures.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd


RUN_RE = re.compile(
    r"l(?P<lmax>\d+)[_-]n(?P<nmax>\d+).*?seed(?P<seed>\d+)", re.IGNORECASE
)
METRICS = ("abs_energy_error", "force_mae", "force_rmse")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--outlier-sigma",
        type=float,
        default=2.0,
        help="Only show/write points at or above this many SD above run mean.",
    )
    parser.add_argument(
        "--top-sources",
        type=int,
        default=50,
        help="Number of source structures shown in each heatmap.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def run_metadata(path: Path) -> dict[str, object] | None:
    match = RUN_RE.search(f"{path.parent.name}_{path.stem}")
    if match is None:
        return None
    architecture = f"L{match.group('lmax')}N{match.group('nmax')}"
    seed = int(match.group("seed"))
    return {
        "path": path,
        "architecture": architecture,
        "seed": seed,
        "run_label": f"{architecture} s{seed}",
    }


def run_sort_key(run: dict[str, object]) -> tuple[int, int, int]:
    architecture = str(run["architecture"])
    match = re.fullmatch(r"L(\d+)N(\d+)", architecture)
    assert match is not None
    return int(match.group(1)), int(match.group(2)), int(run["seed"])


def plot_heatmaps(
    *,
    metric: str,
    source_indices: pd.Index,
    errors: pd.DataFrame,
    sigmas: pd.DataFrame,
    outlier_sigma: float,
    output_dir: Path,
    dpi: int,
) -> tuple[Path, Path]:
    outlier_mask = sigmas >= outlier_sigma
    displayed_errors = errors.where(outlier_mask)
    displayed_sigmas = sigmas.where(outlier_mask)
    sigma_max = float(np.nanmax(displayed_sigmas.to_numpy()))
    error_min = float(np.nanmin(displayed_errors.to_numpy()))
    error_max = float(np.nanmax(displayed_errors.to_numpy()))

    fig_height = max(8.0, 0.24 * len(source_indices) + 3.8)
    fig_width = max(16.0, 0.48 * len(errors.columns) + 9.0)
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), sharey=True)

    error_image = axes[0].imshow(
        displayed_errors.to_numpy(),
        aspect="auto",
        interpolation="nearest",
        norm=LogNorm(vmin=error_min, vmax=error_max),
        cmap="magma",
    )
    sigma_image = axes[1].imshow(
        displayed_sigmas.to_numpy(),
        aspect="auto",
        interpolation="nearest",
        norm=LogNorm(vmin=outlier_sigma, vmax=max(sigma_max, outlier_sigma * 1.01)),
        cmap="viridis",
    )

    for ax in axes:
        ax.set_xticks(np.arange(len(errors.columns)))
        ax.set_xticklabels(errors.columns, rotation=90, fontsize=8)
        ax.set_yticks(np.arange(len(source_indices)))
        ax.set_yticklabels(source_indices.astype(str), fontsize=8)
        ax.set_xlabel("Model configuration and seed")
    axes[0].set_ylabel("Source index (ranked by worst sigma score)")
    axes[0].set_title("Raw error\n(log scale; blank if below threshold)")
    axes[1].set_title("Standard deviations above that run's mean\n(log scale)")

    error_bar = fig.colorbar(error_image, ax=axes[0], pad=0.02)
    error_bar.set_label(metric.replace("_", " "))
    sigma_bar = fig.colorbar(sigma_image, ax=axes[1], pad=0.02)
    sigma_bar.set_label("sample SD above run mean")
    fig.suptitle(
        f"Worst {len(source_indices)} structures across all external-test runs: "
        f"{metric.replace('_', ' ')} (≥ {outlier_sigma:g} SD)",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    stem = f"all_runs_worst_structures_{metric}_ge_{outlier_sigma:g}std"
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main() -> int:
    args = parse_args()
    if args.outlier_sigma <= 0 or args.top_sources < 1:
        raise SystemExit("--outlier-sigma and --top-sources must be positive")

    runs = [
        metadata
        for path in sorted(args.input_dir.rglob("*_per_point.csv"))
        if (metadata := run_metadata(path)) is not None
    ]
    runs.sort(key=run_sort_key)
    if not runs:
        raise SystemExit("No recognizable per-point CSV files found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []

    for metric in METRICS:
        errors_by_run: dict[str, pd.Series] = {}
        sigmas_by_run: dict[str, pd.Series] = {}
        run_stats: dict[str, dict[str, object]] = {}
        outlier_frames: list[pd.DataFrame] = []

        for run in runs:
            path = Path(run["path"])
            frame = pd.read_csv(path, usecols=["source_index", metric])
            values = pd.to_numeric(frame[metric], errors="coerce").to_numpy(float)
            source_indices = frame["source_index"].to_numpy()
            finite = np.isfinite(values)
            values = values[finite]
            source_indices = source_indices[finite]
            if values.size < 2:
                continue
            mean = float(np.mean(values))
            sample_std = float(np.std(values, ddof=1))
            if sample_std == 0 or not np.isfinite(sample_std):
                continue
            sigmas = (values - mean) / sample_std
            label = str(run["run_label"])
            errors_by_run[label] = pd.Series(values, index=source_indices)
            sigmas_by_run[label] = pd.Series(sigmas, index=source_indices)
            top_index = int(np.argmax(sigmas))
            run_stat = {
                "metric": metric,
                "architecture": run["architecture"],
                "seed": run["seed"],
                "run_label": label,
                "source_file": str(path),
                "n_points": int(values.size),
                "mean": mean,
                "sample_std": sample_std,
                "n_ge_outlier_sigma": int(np.count_nonzero(sigmas >= args.outlier_sigma)),
                "n_ge_3std": int(np.count_nonzero(sigmas >= 3)),
                "n_ge_5std": int(np.count_nonzero(sigmas >= 5)),
                "n_ge_10std": int(np.count_nonzero(sigmas >= 10)),
                "worst_source_index": int(source_indices[top_index]),
                "worst_error": float(values[top_index]),
                "worst_sigma": float(sigmas[top_index]),
            }
            summary_rows.append(run_stat)
            run_stats[label] = run_stat

            keep = sigmas >= args.outlier_sigma
            if np.any(keep):
                outlier_frames.append(
                    pd.DataFrame(
                        {
                            "metric": metric,
                            "source_index": source_indices[keep],
                            "architecture": run["architecture"],
                            "seed": run["seed"],
                            "run_label": label,
                            "source_file": str(path),
                            "error": values[keep],
                            "run_mean": mean,
                            "run_sample_std": sample_std,
                            "std_above_run_mean": sigmas[keep],
                        }
                    )
                )

        errors = pd.DataFrame(errors_by_run)
        sigmas = pd.DataFrame(sigmas_by_run)
        run_labels = [str(run["run_label"]) for run in runs if str(run["run_label"]) in errors]
        errors = errors.reindex(columns=run_labels)
        sigmas = sigmas.reindex(columns=run_labels)
        maximum_sigma = sigmas.max(axis=1)
        source_indices = maximum_sigma.nlargest(args.top_sources).index
        top_errors = errors.loc[source_indices]
        top_sigmas = sigmas.loc[source_indices]

        png_path, pdf_path = plot_heatmaps(
            metric=metric,
            source_indices=source_indices,
            errors=top_errors,
            sigmas=top_sigmas,
            outlier_sigma=args.outlier_sigma,
            output_dir=args.output_dir,
            dpi=args.dpi,
        )

        top_rows: list[dict[str, object]] = []
        for source_index in source_indices:
            for label in run_labels:
                stat = run_stats[label]
                top_rows.append(
                    {
                        "metric": metric,
                        "source_index": int(source_index),
                        "architecture": stat["architecture"],
                        "seed": stat["seed"],
                        "run_label": label,
                        "error": float(top_errors.loc[source_index, label]),
                        "run_mean": stat["mean"],
                        "run_sample_std": stat["sample_std"],
                        "std_above_run_mean": float(top_sigmas.loc[source_index, label]),
                        "is_ge_outlier_sigma": bool(
                            top_sigmas.loc[source_index, label] >= args.outlier_sigma
                        ),
                    }
                )
        pd.DataFrame(top_rows).to_csv(
            args.output_dir / f"all_runs_top_sources_{metric}.csv", index=False
        )
        pd.concat(outlier_frames, ignore_index=True).to_csv(
            args.output_dir / f"all_runs_{metric}_ge_{args.outlier_sigma:g}std.csv",
            index=False,
        )
        print(f"{metric}: {png_path}")
        print(f"{metric}: {pdf_path}")

    pd.DataFrame(summary_rows).to_csv(
        args.output_dir / "all_runs_outlier_summary.csv", index=False
    )
    print(f"Summary: {args.output_dir / 'all_runs_outlier_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
