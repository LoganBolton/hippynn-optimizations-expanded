#!/usr/bin/env python3
"""Create box plots for 100k and 1M methane datasets showing RMSE distribution."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# Population standard deviation over all 7,732,488 methane configurations.
ENERGY_STD_KCAL_MOL = 73.8287338435902

# Data files
DATA_FILES = {
    "100k": Path("examples/methane_sweep/logs/plots_l3n5_100k_all8/final_test_metrics.csv"),
    "1M": Path("examples/methane_sweep/logs/plots_l3n5_1m_updated/final_test_metrics.csv"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-100k",
        default=Path("examples/methane_sweep/results/paper/boxplot_rmse_100k.png"),
        type=Path,
        help="Output image path for the 100k dataset box plot.",
    )
    parser.add_argument(
        "--output-1m",
        default=Path("examples/methane_sweep/results/paper/boxplot_rmse_1m.png"),
        type=Path,
        help="Output image path for the 1M dataset box plot.",
    )
    return parser.parse_args()


def load_test_metrics(path: Path) -> list[dict]:
    """Load test metrics from CSV file."""
    if not path.exists():
        print(f"Warning: {path} does not exist")
        return []
    
    results = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            results.append(row)
    return results


def configure_matplotlib() -> None:
    """Configure matplotlib for publication-quality plots."""
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
            "legend.fontsize": 10,
            "lines.linewidth": 1.5,
        }
    )


def create_boxplot(data: list[float], title: str, output: Path, dataset_name: str) -> None:
    """Create a box plot for RMSE values."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=180)
    
    # Convert to numpy array
    data_array = np.array(data)
    
    # Create box plot
    bp = ax.boxplot(
        [data_array],
        widths=0.5,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker='D', markerfacecolor='red', markeredgecolor='red', markersize=8),
        boxprops=dict(facecolor='lightblue', edgecolor='black', linewidth=1.5),
        whiskerprops=dict(color='black', linewidth=1.5),
        capprops=dict(color='black', linewidth=1.5),
        medianprops=dict(color='darkblue', linewidth=2),
        flierprops=dict(marker='o', markerfacecolor='red', markersize=8, alpha=0.5),
    )
    
    # Calculate statistics
    mean_val = np.mean(data_array)
    median_val = np.median(data_array)
    std_val = np.std(data_array, ddof=1)
    min_val = np.min(data_array)
    max_val = np.max(data_array)
    
    # Normalize by energy std
    mean_normalized = mean_val / ENERGY_STD_KCAL_MOL
    std_normalized = std_val / ENERGY_STD_KCAL_MOL
    
    # Set labels
    ax.set_ylabel("Test RMSE (kcal/mol)", fontsize=15)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xticks([1])
    ax.set_xticklabels([f'{dataset_name}\n(N={len(data_array)})'], fontsize=13)
    
    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Add statistics text box
    stats_text = (
        f"Mean: {mean_val:.4f} kcal/mol\n"
        f"Median: {median_val:.4f} kcal/mol\n"
        f"Std: {std_val:.4f} kcal/mol\n"
        f"Min: {min_val:.4f} kcal/mol\n"
        f"Max: {max_val:.4f} kcal/mol\n"
        f"\nNormalized RMSE/STD:\n"
        f"{mean_normalized:.6f} ± {std_normalized:.6f}"
    )
    
    ax.text(
        0.98, 0.97, stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, pad=0.8)
    )
    
    # Save figure
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"Saved box plot to {output}")


def create_combined_boxplot(data_100k: list[float], data_1m: list[float], output: Path) -> None:
    """Create a combined box plot comparing 100k and 1M datasets."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=180)
    
    # Convert to numpy arrays
    data_100k_array = np.array(data_100k)
    data_1m_array = np.array(data_1m)
    
    # Create box plot
    bp = ax.boxplot(
        [data_100k_array, data_1m_array],
        widths=0.5,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker='D', markerfacecolor='red', markeredgecolor='red', markersize=8),
        boxprops=dict(edgecolor='black', linewidth=1.5),
        whiskerprops=dict(color='black', linewidth=1.5),
        capprops=dict(color='black', linewidth=1.5),
        medianprops=dict(color='darkblue', linewidth=2),
        flierprops=dict(marker='o', markerfacecolor='red', markersize=8, alpha=0.5),
    )
    
    # Color the boxes differently
    colors = ['lightblue', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    # Calculate statistics
    stats = []
    for data_array, label in [(data_100k_array, "100k"), (data_1m_array, "1M")]:
        mean_val = np.mean(data_array)
        std_val = np.std(data_array, ddof=1)
        mean_normalized = mean_val / ENERGY_STD_KCAL_MOL
        std_normalized = std_val / ENERGY_STD_KCAL_MOL
        stats.append({
            'label': label,
            'mean': mean_val,
            'std': std_val,
            'mean_norm': mean_normalized,
            'std_norm': std_normalized,
            'n': len(data_array)
        })
    
    # Set labels
    ax.set_ylabel("Test RMSE (kcal/mol)", fontsize=15)
    ax.set_title(r"HIP-HOP-NN($\ell=3, n=5$) Test RMSE Distribution", fontsize=16, fontweight='bold')
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f'100k\n(N={stats[0]["n"]})', f'1M\n(N={stats[1]["n"]})'], fontsize=13)
    
    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Add statistics text box
    stats_text = "Statistics:\n\n"
    for s in stats:
        stats_text += (
            f"{s['label']} dataset:\n"
            f"  Mean: {s['mean']:.4f} kcal/mol\n"
            f"  Std: {s['std']:.4f} kcal/mol\n"
            f"  RMSE/STD: {s['mean_norm']:.6f} ± {s['std_norm']:.6f}\n\n"
        )
    
    ax.text(
        0.98, 0.97, stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, pad=0.8)
    )
    
    # Save figure
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"Saved combined box plot to {output}")


def main() -> None:
    args = parse_args()
    
    # Configure matplotlib
    configure_matplotlib()
    
    # Load data
    data_100k_raw = load_test_metrics(DATA_FILES["100k"])
    data_1m_raw = load_test_metrics(DATA_FILES["1M"])
    
    # Extract RMSE values
    data_100k = [float(row["test_T-RMSE"]) for row in data_100k_raw if "test_T-RMSE" in row]
    data_1m = [float(row["test_T-RMSE"]) for row in data_1m_raw if "test_T-RMSE" in row]
    
    print(f"Loaded {len(data_100k)} data points for 100k dataset")
    print(f"Loaded {len(data_1m)} data points for 1M dataset")
    
    if not data_100k:
        print("Error: No data loaded for 100k dataset")
        return
    
    if not data_1m:
        print("Error: No data loaded for 1M dataset")
        return
    
    # Create individual box plots
    create_boxplot(
        data_100k,
        r"HIP-HOP-NN($\ell=3, n=5$) Test RMSE - 100k Dataset",
        args.output_100k,
        "100k"
    )
    
    create_boxplot(
        data_1m,
        r"HIP-HOP-NN($\ell=3, n=5$) Test RMSE - 1M Dataset",
        args.output_1m,
        "1M"
    )
    
    # Create combined box plot
    combined_output = args.output_100k.parent / "boxplot_rmse_combined.png"
    create_combined_boxplot(data_100k, data_1m, combined_output)


if __name__ == "__main__":
    main()
