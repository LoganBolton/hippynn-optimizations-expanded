#!/usr/bin/env python3
"""Compare per-configuration target-force magnitude with prediction RMSE."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DEBUG_DIR = Path(
    "examples/TEST_METHANE_MODEL_l4_n3_d100000_seed42-b256_fresh/validation_debug"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug-dir", type=Path, default=DEFAULT_DEBUG_DIR)
    parser.add_argument("--epoch", type=int, default=3000)
    parser.add_argument("--highlight", nargs="*", type=int, default=[32801, 1136, 60963])
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = args.debug_dir / f"validation_errors_epoch_{args.epoch}.npz"
    output_path = args.out or (
        args.debug_dir
        / "geometry_rarity"
        / f"scatter_force_rmse_vs_mean_target_force_magnitude_epoch_{args.epoch}.png"
    )

    with np.load(input_path) as data:
        source_indices = data["source_indices"].astype(np.int64)
        mean_target_magnitude = np.linalg.norm(data["forces"], axis=-1).mean(axis=1)
        force_rmse = np.sqrt(np.mean(data["force_error"] ** 2, axis=(1, 2)))

    fig, ax = plt.subplots(figsize=(9, 6.5), constrained_layout=True)
    ax.scatter(
        mean_target_magnitude,
        force_rmse,
        s=13,
        alpha=0.28,
        color="#4c78a8",
        edgecolors="none",
        label=f"validation configurations (n={len(force_rmse):,})",
    )

    colors = ["#d62728", "#ff7f0e", "#9467bd"]
    for rank, source_index in enumerate(args.highlight, start=1):
        matches = np.flatnonzero(source_indices == source_index)
        if len(matches) != 1:
            print(f"Skipping source {source_index}: found {len(matches)} matching rows")
            continue
        row = int(matches[0])
        color = colors[(rank - 1) % len(colors)]
        ax.scatter(
            mean_target_magnitude[row],
            force_rmse[row],
            s=85,
            color=color,
            edgecolor="black",
            linewidth=0.7,
            zorder=3,
            label=f"culprit {rank}: source {source_index}",
        )
        ax.annotate(
            f" {source_index}",
            (mean_target_magnitude[row], force_rmse[row]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mean target force magnitude")
    ax.set_ylabel("Force RMSE")
    ax.set_title("Target force vs force RMSE")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(loc="best", fontsize=9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
