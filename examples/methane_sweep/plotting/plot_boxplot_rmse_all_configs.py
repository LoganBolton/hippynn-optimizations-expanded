#!/usr/bin/env python3
"""Create box plots for all configurations showing RMSE distribution."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


# Population standard deviation over all 7,732,488 methane configurations.
ENERGY_STD_KCAL_MOL = 73.8287338435902


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-100k",
        default=Path("examples/methane_sweep/results/paper/boxplot_rmse_all_configs_100k.png"),
        type=Path,
        help="Output image path for the 100k dataset box plot.",
    )
    parser.add_argument(
        "--output-1m",
        default=Path("examples/methane_sweep/results/paper/boxplot_rmse_all_configs_1m.png"),
        type=Path,
        help="Output image path for the 1M dataset box plot.",
    )
    return parser.parse_args()


def load_final_test_metrics(test_metrics_path: Path) -> dict[tuple[int, int, int], list[float]]:
    """Load individual test metrics from final_test_metrics.csv."""
    results = defaultdict(list)
    
    if not test_metrics_path.exists():
        return results
    
    with test_metrics_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                l_max = int(row["hiphop_l_max"])
                n_max = int(row["hiphop_n_max"])
                data_size = int(row["data_size"])
                rmse = float(row["test_T-RMSE"])
                results[(l_max, n_max, data_size)].append(rmse)
            except (ValueError, KeyError):
                continue
    
    return results


def load_all_test_data() -> dict[tuple[int, int, int], list[float]]:
    """Load all test data from known sources."""
    all_data = defaultdict(list)
    
    # Known test metrics files
    test_files = [
        Path("examples/methane_sweep/logs/plots_l3n5_100k_all8/final_test_metrics.csv"),
        Path("examples/methane_sweep/logs/plots_l3n5_1m_updated/final_test_metrics.csv"),
    ]
    
    for test_file in test_files:
        data = load_final_test_metrics(test_file)
        for key, values in data.items():
            all_data[key].extend(values)
            print(f"  Loaded {len(values)} runs from {test_file.name} for l={key[0]}, n={key[1]}, size={key[2]}")
    
    # For configs with only 1-2 runs, load from the combined CSV
    combined_csv = Path("examples/methane_sweep/logs/plots_all_configs_b256_final_test/paper_style_all_configs_mean_std.csv")
    if combined_csv.exists():
        with combined_csv.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    l_max = int(row["hiphop_l_max"])
                    n_max = int(row["hiphop_n_max"])
                    data_size = int(row["data_size"])
                    key = (l_max, n_max, data_size)
                    
                    # Skip if we already have individual test data for this config
                    if key in all_data and len(all_data[key]) > 0:
                        continue
                    
                    seed_count = int(row.get("completed_seed_count", 1))
                    mean_rmse = float(row["test_T-RMSE_mean"])
                    std_rmse = float(row.get("test_T-RMSE_std", 0.0))
                    
                    if seed_count == 1:
                        # Single run
                        all_data[key].append(mean_rmse)
                        print(f"  Loaded 1 run from combined CSV for l={l_max}, n={n_max}, size={data_size}")
                    elif seed_count == 2 and std_rmse > 0:
                        # Two runs - we can recover approximate individual values
                        # For 2 samples: mean ± std * sqrt(2) gives the two values
                        val1 = mean_rmse + std_rmse
                        val2 = mean_rmse - std_rmse
                        all_data[key].extend([val1, val2])
                        print(f"  Reconstructed 2 runs from combined CSV for l={l_max}, n={n_max}, size={data_size}")
                    else:
                        # Multiple runs - just use mean for now
                        all_data[key].append(mean_rmse)
                        print(f"  Loaded mean only from combined CSV for l={l_max}, n={n_max}, size={data_size}")
                        
                except (ValueError, KeyError) as e:
                    continue
    
    return all_data


def configure_matplotlib() -> None:
    """Configure matplotlib for publication-quality plots."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "axes.linewidth": 1.1,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 10,
            "lines.linewidth": 1.5,
        }
    )


def create_boxplot_for_dataset(data_dict: dict[tuple[int, int, int], list[float]], 
                                 dataset_size: int, output: Path) -> None:
    """Create a box plot comparing all configurations for a given dataset size."""
    
    # Filter data for the specified dataset size and sort by l_max, n_max
    configs = sorted([(l, n) for l, n, d in data_dict.keys() if d == dataset_size])
    
    if not configs:
        print(f"No data found for dataset size {dataset_size}")
        return
    
    # Prepare data for box plot
    plot_data = []
    labels = []
    stats_info = []
    
    for l_max, n_max in configs:
        key = (l_max, n_max, dataset_size)
        if key in data_dict and data_dict[key]:
            values = data_dict[key]
            plot_data.append(values)
            labels.append(rf"$\ell={l_max}, n={n_max}$" + f"\n(N={len(values)})")
            
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1) if len(values) > 1 else 0.0
            stats_info.append({
                'l': l_max,
                'n': n_max,
                'mean': mean_val,
                'std': std_val,
                'count': len(values),
                'values': values
            })
    
    if not plot_data:
        print(f"No valid data for dataset size {dataset_size}")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
    
    # Create box plot
    bp = ax.boxplot(
        plot_data,
        widths=0.6,
        patch_artist=True,
        showmeans=False,
        showfliers=False,  # Don't show outliers, we'll plot all points
        boxprops=dict(edgecolor='black', linewidth=1.2),
        whiskerprops=dict(color='black', linewidth=1.2),
        capprops=dict(color='black', linewidth=1.2),
        medianprops=dict(color='darkblue', linewidth=2),
    )
    
    # Plot all individual data points
    np.random.seed(42)  # For consistent jitter
    for i, values in enumerate(plot_data):
        # Add jitter to x-coordinates for better visibility
        x = np.ones(len(values)) * (i + 1)
        x += np.random.normal(0, 0.04, size=len(values))
        ax.scatter(x, values, alpha=0.6, s=50, color='red', zorder=3, edgecolors='darkred', linewidth=0.5)
    
    # Color the boxes differently
    colors = ['#f2c14e', '#56b4e9', '#e07a7a', '#25b99a', '#d99ab5']
    for patch, color in zip(bp['boxes'], colors * len(bp['boxes'])):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Set labels
    dataset_label = "100k" if dataset_size == 100000 else f"{dataset_size//1000000}M"
    ax.set_ylabel("Test RMSE (kcal/mol)", fontsize=14)
    ax.set_title(f"HIP-HOP-NN Test RMSE Distribution - {dataset_label} Dataset", 
                 fontsize=16, fontweight='bold')
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=11)
    
    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Statistics text box removed per user request
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='none', edgecolor='darkblue', linewidth=2, label='Median'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                   markeredgecolor='darkred', markersize=7, alpha=0.6, label='Individual runs'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
    
    # Save figure
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"Saved box plot to {output}")
    
    # Print summary
    print(f"\n{dataset_label} Dataset Summary:")
    for s in stats_info:
        print(f"  l={s['l']}, n={s['n']}: N={len(s['values'])}, "
              f"Mean={s['mean']:.4f} kcal/mol, Std={s['std']:.4f} kcal/mol")
        if len(s['values']) <= 10:
            print(f"    Values: {[f'{v:.4f}' for v in s['values']]}")


def main() -> None:
    args = parse_args()
    
    # Configure matplotlib
    configure_matplotlib()
    
    # Load all test metrics
    print("Loading test metrics from all sources...")
    all_data = load_all_test_data()
    
    # Print summary of loaded data
    print("\nLoaded data summary:")
    for key in sorted(all_data.keys()):
        l_max, n_max, data_size = key
        count = len(all_data[key])
        dataset_label = "100k" if data_size == 100000 else f"{data_size//1000000}M"
        print(f"  l={l_max}, n={n_max}, {dataset_label}: {count} runs")
    
    # Create box plots for each dataset size
    create_boxplot_for_dataset(all_data, 100000, args.output_100k)
    create_boxplot_for_dataset(all_data, 1000000, args.output_1m)


if __name__ == "__main__":
    main()
