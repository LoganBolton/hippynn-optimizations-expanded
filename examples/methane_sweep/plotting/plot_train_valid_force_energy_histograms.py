#!/usr/bin/env python3
"""Plot methane train/validation force and energy histograms with outlier markers."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import ase.io
import matplotlib.pyplot as plt
import numpy as np
import torch


ENERGY_MEAN = -25042.327220945674
FORCE_CONVERSION = 51.42208619083232 * 23.060541945329334
ENERGY_CONVERSION = 627.5096080305927

DEFAULT_DEBUG_DIR = Path("examples/TEST_METHANE_MODEL_l4_n3_d100000_seed42-b256_fresh/validation_debug")
DEFAULT_DATASET = Path("datasets/methane.extxyz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug-dir", type=Path, default=DEFAULT_DEBUG_DIR)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--data-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epoch", type=int, default=3000)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--force-component-limit", type=float, default=5000.0)
    parser.add_argument("--force-magnitude-limit", type=float, default=5000.0)
    parser.add_argument("--force-rmse-limit", type=float, default=1000.0)
    parser.add_argument("--refresh-cache", action="store_true")
    return parser.parse_args()


def split_indices(n_items: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    remaining = torch.arange(n_items, dtype=torch.int64)
    gen = torch.Generator().manual_seed(seed)

    test_perm = torch.randperm(len(remaining), generator=gen)
    test = remaining[test_perm[: int(0.1 * n_items)]].sort().values
    mask = ~torch.isin(remaining, test)
    remaining = remaining[mask]

    valid_fraction_of_remaining = 0.1 / (1.0 - 0.1)
    valid_perm = torch.randperm(len(remaining), generator=gen)
    valid = remaining[valid_perm[: int(valid_fraction_of_remaining * len(remaining))]].sort().values
    mask = ~torch.isin(remaining, valid)
    train = remaining[mask].sort().values
    return train.numpy(), valid.numpy(), test.numpy()


def read_training_pool(dataset: Path, data_size: int, cache_path: Path, refresh: bool) -> dict[str, np.ndarray]:
    if cache_path.exists() and not refresh:
        with np.load(cache_path) as data:
            return {key: data[key] for key in data.files}

    energies = np.empty(data_size, dtype=np.float32)
    forces = np.empty((data_size, 5, 3), dtype=np.float32)
    for idx, frame in enumerate(ase.io.iread(dataset, index=f":{data_size}")):
        if idx >= data_size:
            break
        energies[idx] = frame.get_total_energy() * ENERGY_CONVERSION - ENERGY_MEAN
        forces[idx] = frame.get_forces() * FORCE_CONVERSION
        if (idx + 1) % 10_000 == 0:
            print(f"Read {idx + 1}/{data_size} frames from {dataset}", flush=True)

    np.savez_compressed(cache_path, energy=energies, forces=forces)
    return {"energy": energies, "forces": forces}


def load_outliers(debug_dir: Path, epoch: int, top_n: int) -> list[dict[str, str]]:
    path = debug_dir / "spike_analysis" / f"culprits_epoch_{epoch}_vs_2200.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[:top_n]


def force_magnitude(forces: np.ndarray) -> np.ndarray:
    return np.linalg.norm(forces, axis=-1)


def plot_hist(
    train_values: np.ndarray,
    valid_values: np.ndarray,
    outlier_values: list[float],
    labels: list[str],
    title: str,
    xlabel: str,
    output: Path,
    bins: int = 120,
    value_range: tuple[float, float] | None = None,
    log_y: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.8), constrained_layout=True)
    ax.hist(train_values, bins=bins, range=value_range, alpha=0.55, density=True, label=f"train n={train_values.size:,}")
    ax.hist(valid_values, bins=bins, range=value_range, alpha=0.55, density=True, label=f"valid n={valid_values.size:,}")
    colors = ["#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    for idx, (value, label) in enumerate(zip(outlier_values, labels)):
        ax.axvline(value, color=colors[idx % len(colors)], linewidth=2.2, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sorted_values = np.sort(values[np.isfinite(values)])
    y = 100.0 * np.arange(1, sorted_values.size + 1) / sorted_values.size
    return sorted_values, y


def plot_cdf(
    train_values: np.ndarray,
    valid_values: np.ndarray,
    outlier_values: list[float],
    labels: list[str],
    title: str,
    xlabel: str,
    output: Path,
    xlim: tuple[float, float] | None = None,
    tail: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.8), constrained_layout=True)
    train_x, train_y = ecdf(train_values)
    valid_x, valid_y = ecdf(valid_values)
    if tail:
        train_y = 100.0 - train_y
        valid_y = 100.0 - valid_y
        ylabel = "percent above value"
    else:
        ylabel = "percentile: percent <= value"
    ax.plot(train_x, train_y, linewidth=2.0, label=f"train n={train_values.size:,}")
    ax.plot(valid_x, valid_y, linewidth=2.0, label=f"valid n={valid_values.size:,}")
    colors = ["#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    for idx, (value, label) in enumerate(zip(outlier_values, labels)):
        train_pct = percentile(train_values, value)
        valid_pct = percentile(valid_values, value)
        y_value = 100.0 - valid_pct if tail else valid_pct
        short_label = label.split()[0] + " " + label.split()[1]
        ax.axvline(value, color=colors[idx % len(colors)], linewidth=1.7, alpha=0.8, label=short_label)
        ax.scatter([value], [y_value], color=colors[idx % len(colors)], s=50, zorder=4)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if tail:
        ax.set_yscale("log")
        ax.set_ylim(0.01, 100)
    else:
        ax.set_ylim(0, 100.5)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def percentile(values: np.ndarray, x: float) -> float:
    return float(100.0 * np.mean(values <= x))


def tight_upper_limit(*arrays: np.ndarray | list[float], quantile: float = 99.95, pad: float = 1.08) -> float:
    finite_arrays = []
    for array in arrays:
        values = np.asarray(array, dtype=float).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size:
            finite_arrays.append(values)
    if not finite_arrays:
        return 1.0
    combined = np.concatenate(finite_arrays)
    upper = max(float(np.percentile(combined, quantile)), float(np.max(combined)))
    return upper * pad if upper > 0 else 1.0


def main() -> None:
    args = parse_args()
    out_dir = args.debug_dir / "geometry_rarity" / "force_energy_histograms"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / f"training_pool_targets_first_{args.data_size}_seed{args.seed}.npz"

    pool = read_training_pool(args.dataset, args.data_size, cache_path, args.refresh_cache)
    train_idx, valid_idx, test_idx = split_indices(args.data_size, args.seed)

    saved_valid = np.load(args.debug_dir / "validation_targets.npz")
    valid_energy = saved_valid["energy"]
    valid_forces = saved_valid["forces"]
    train_energy = pool["energy"][train_idx]
    train_forces = pool["forces"][train_idx]

    outliers = load_outliers(args.debug_dir, args.epoch, args.top_n)
    outlier_rows = [int(row["valid_row"]) for row in outliers]
    outlier_labels = [f"rank {row['rank']} row {row['valid_row']}" for row in outliers]
    outlier_energy = [float(valid_energy[row]) for row in outlier_rows]
    outlier_max_force = [float(force_magnitude(valid_forces[row]).max()) for row in outlier_rows]
    outlier_abs_component = [float(np.abs(valid_forces[row]).max()) for row in outlier_rows]
    outlier_force_rmse = [float(row["target_force_rmse"]) for row in outliers]

    train_force_mag = force_magnitude(train_forces).reshape(-1)
    valid_force_mag = force_magnitude(valid_forces).reshape(-1)
    train_force_mag_by_config = force_magnitude(train_forces)
    valid_force_mag_by_config = force_magnitude(valid_forces)
    train_config_max_force = train_force_mag_by_config.max(axis=1)
    valid_config_max_force = valid_force_mag_by_config.max(axis=1)
    train_config_mean_force = train_force_mag_by_config.mean(axis=1)
    valid_config_mean_force = valid_force_mag_by_config.mean(axis=1)
    train_config_rms_force = np.sqrt(np.mean(train_force_mag_by_config**2, axis=1))
    valid_config_rms_force = np.sqrt(np.mean(valid_force_mag_by_config**2, axis=1))
    train_abs_force_component = np.abs(train_forces).reshape(-1)
    valid_abs_force_component = np.abs(valid_forces).reshape(-1)
    valid_force_rmse = np.load(args.debug_dir / f"validation_errors_epoch_{args.epoch}.npz")["per_config_force_rmse"]
    force_mag_xlim = (0, tight_upper_limit(train_force_mag, valid_force_mag, outlier_max_force))
    config_max_xlim = (0, tight_upper_limit(train_config_max_force, valid_config_max_force, outlier_max_force))

    plot_hist(
        train_force_mag,
        valid_force_mag,
        outlier_max_force,
        outlier_labels,
        "Target Force Magnitude Distribution",
        "per-atom |F| (kcal/mol/Ang)",
        out_dir / "target_force_magnitude_hist.png",
        value_range=force_mag_xlim,
    )
    plot_hist(
        train_abs_force_component,
        valid_abs_force_component,
        outlier_abs_component,
        outlier_labels,
        "Target Absolute Force Component Distribution",
        "|force component| (kcal/mol/Ang)",
        out_dir / "target_abs_force_component_hist.png",
        value_range=(0, args.force_component_limit),
    )
    plot_hist(
        train_energy,
        valid_energy,
        outlier_energy,
        outlier_labels,
        "Target Energy Distribution",
        "shifted energy (kcal/mol)",
        out_dir / "target_energy_hist.png",
        value_range=(float(np.percentile(pool["energy"], 0.1)), float(np.percentile(pool["energy"], 99.9))),
    )
    plot_hist(
        valid_force_rmse,
        valid_force_rmse,
        outlier_force_rmse,
        outlier_labels,
        f"Validation Per-Configuration Force RMSE at Epoch {args.epoch}",
        "force RMSE (kcal/mol/Ang)",
        out_dir / f"validation_force_rmse_epoch_{args.epoch}_hist.png",
        value_range=(0, args.force_rmse_limit),
    )
    plot_cdf(
        train_force_mag,
        valid_force_mag,
        outlier_max_force,
        outlier_labels,
        "Target Per-Atom Force Magnitude Percentiles",
        "per-atom |F| (kcal/mol/Ang)",
        out_dir / "target_force_magnitude_cdf.png",
        xlim=force_mag_xlim,
    )
    plot_cdf(
        train_force_mag,
        valid_force_mag,
        outlier_max_force,
        outlier_labels,
        "Target Per-Atom Force Magnitude Tail",
        "per-atom |F| (kcal/mol/Ang)",
        out_dir / "target_force_magnitude_tail_cdf.png",
        xlim=force_mag_xlim,
        tail=True,
    )
    plot_cdf(
        train_config_max_force,
        valid_config_max_force,
        outlier_max_force,
        outlier_labels,
        "Per-Configuration Max Force Percentiles",
        "max atom |F| in configuration (kcal/mol/Ang)",
        out_dir / "target_config_max_force_cdf.png",
        xlim=config_max_xlim,
    )
    plot_cdf(
        train_config_max_force,
        valid_config_max_force,
        outlier_max_force,
        outlier_labels,
        "Per-Configuration Max Force Tail",
        "max atom |F| in configuration (kcal/mol/Ang)",
        out_dir / "target_config_max_force_tail_cdf.png",
        xlim=config_max_xlim,
        tail=True,
    )
    outlier_mean_force = [float(force_magnitude(valid_forces[row]).mean()) for row in outlier_rows]
    outlier_rms_force = [float(np.sqrt(np.mean(force_magnitude(valid_forces[row]) ** 2))) for row in outlier_rows]
    config_mean_xlim = (0, tight_upper_limit(train_config_mean_force, valid_config_mean_force, outlier_mean_force))
    config_rms_xlim = (0, tight_upper_limit(train_config_rms_force, valid_config_rms_force, outlier_rms_force))
    plot_hist(
        train_config_mean_force,
        valid_config_mean_force,
        outlier_mean_force,
        outlier_labels,
        "Per-Configuration Mean Force Magnitude Distribution",
        "mean atom |F| in configuration (kcal/mol/Ang)",
        out_dir / "target_config_mean_force_hist.png",
        value_range=config_mean_xlim,
    )
    plot_cdf(
        train_config_mean_force,
        valid_config_mean_force,
        outlier_mean_force,
        outlier_labels,
        "Per-Configuration Mean Force Percentiles",
        "mean atom |F| in configuration (kcal/mol/Ang)",
        out_dir / "target_config_mean_force_cdf.png",
        xlim=config_mean_xlim,
    )
    plot_cdf(
        train_config_mean_force,
        valid_config_mean_force,
        outlier_mean_force,
        outlier_labels,
        "Per-Configuration Mean Force Tail",
        "mean atom |F| in configuration (kcal/mol/Ang)",
        out_dir / "target_config_mean_force_tail_cdf.png",
        xlim=config_mean_xlim,
        tail=True,
    )
    plot_hist(
        train_config_rms_force,
        valid_config_rms_force,
        outlier_rms_force,
        outlier_labels,
        "Per-Configuration RMS Force Magnitude Distribution",
        "RMS atom |F| in configuration (kcal/mol/Ang)",
        out_dir / "target_config_rms_force_hist.png",
        value_range=config_rms_xlim,
    )
    plot_cdf(
        train_config_rms_force,
        valid_config_rms_force,
        outlier_rms_force,
        outlier_labels,
        "Per-Configuration RMS Force Percentiles",
        "RMS atom |F| in configuration (kcal/mol/Ang)",
        out_dir / "target_config_rms_force_cdf.png",
        xlim=config_rms_xlim,
    )
    plot_cdf(
        train_config_rms_force,
        valid_config_rms_force,
        outlier_rms_force,
        outlier_labels,
        "Per-Configuration RMS Force Tail",
        "RMS atom |F| in configuration (kcal/mol/Ang)",
        out_dir / "target_config_rms_force_tail_cdf.png",
        xlim=config_rms_xlim,
        tail=True,
    )
    plot_cdf(
        train_energy,
        valid_energy,
        outlier_energy,
        outlier_labels,
        "Target Energy Percentiles",
        "shifted energy (kcal/mol)",
        out_dir / "target_energy_cdf.png",
    )
    plot_cdf(
        valid_force_rmse,
        valid_force_rmse,
        outlier_force_rmse,
        outlier_labels,
        f"Validation Force RMSE Percentiles at Epoch {args.epoch}",
        "force RMSE (kcal/mol/Ang)",
        out_dir / f"validation_force_rmse_epoch_{args.epoch}_cdf.png",
        xlim=(0, args.force_rmse_limit),
    )

    summary_path = out_dir / "top_outlier_distribution_percentiles.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "rank",
            "valid_row",
            "source_index",
            "energy",
            "energy_train_percentile",
            "energy_valid_percentile",
            "max_force_magnitude",
            "max_force_magnitude_train_percentile",
            "max_force_magnitude_valid_percentile",
            "mean_force_magnitude",
            "mean_force_magnitude_train_percentile",
            "mean_force_magnitude_valid_percentile",
            "rms_force_magnitude",
            "rms_force_magnitude_train_percentile",
            "rms_force_magnitude_valid_percentile",
            "max_abs_force_component",
            "max_abs_force_component_train_percentile",
            "max_abs_force_component_valid_percentile",
            "force_rmse_epoch",
            "force_rmse_valid_percentile",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, energy, max_force, mean_force, rms_force, max_component, force_rmse in zip(
            outliers,
            outlier_energy,
            outlier_max_force,
            outlier_mean_force,
            outlier_rms_force,
            outlier_abs_component,
            outlier_force_rmse,
        ):
            writer.writerow(
                {
                    "rank": row["rank"],
                    "valid_row": row["valid_row"],
                    "source_index": row["source_index"],
                    "energy": energy,
                    "energy_train_percentile": percentile(train_energy, energy),
                    "energy_valid_percentile": percentile(valid_energy, energy),
                    "max_force_magnitude": max_force,
                    "max_force_magnitude_train_percentile": percentile(train_force_mag, max_force),
                    "max_force_magnitude_valid_percentile": percentile(valid_force_mag, max_force),
                    "mean_force_magnitude": mean_force,
                    "mean_force_magnitude_train_percentile": percentile(train_config_mean_force, mean_force),
                    "mean_force_magnitude_valid_percentile": percentile(valid_config_mean_force, mean_force),
                    "rms_force_magnitude": rms_force,
                    "rms_force_magnitude_train_percentile": percentile(train_config_rms_force, rms_force),
                    "rms_force_magnitude_valid_percentile": percentile(valid_config_rms_force, rms_force),
                    "max_abs_force_component": max_component,
                    "max_abs_force_component_train_percentile": percentile(train_abs_force_component, max_component),
                    "max_abs_force_component_valid_percentile": percentile(valid_abs_force_component, max_component),
                    "force_rmse_epoch": force_rmse,
                    "force_rmse_valid_percentile": percentile(valid_force_rmse, force_rmse),
                }
            )

    print(f"train split: {len(train_idx):,}; valid split: {len(valid_idx):,}; test split: {len(test_idx):,}")
    print(f"Wrote plots and summary to {out_dir}")


if __name__ == "__main__":
    main()
