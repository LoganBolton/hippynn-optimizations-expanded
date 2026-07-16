#!/usr/bin/env python3
"""Compare target-force scale and force RMSE across the 100k and 1M models."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
MODEL_100K = ROOT / "examples/TEST_METHANE_MODEL_l4_n3_d100000_seed42-b256_fresh/validation_debug"
MODEL_1M = ROOT / "examples/TEST_METHANE_MODEL_l4_n3_d1000000_seed7-b256_fresh/validation_debug"


def load_metrics(path):
    with np.load(path) as data:
        source = data["source_indices"].astype(np.int64)
        mean_force = np.linalg.norm(data["forces"], axis=-1).mean(axis=1)
        rmse = np.sqrt(np.mean(data["force_error"] ** 2, axis=(1, 2)))
    return source, mean_force, rmse


def highlight(ax, source, mean_force, rmse, source_index, **kwargs):
    matches = np.flatnonzero(source == source_index)
    if len(matches) != 1:
        return
    row = int(matches[0])
    ax.scatter(mean_force[row], rmse[row], **kwargs)


def main():
    source_100k, force_100k, rmse_100k = load_metrics(MODEL_100K / "validation_errors_epoch_3000.npz")
    source_1m, force_1m, rmse_1m = load_metrics(MODEL_1M / "validation_errors_epoch_2400.npz")

    fig, ax = plt.subplots(figsize=(9.5, 6.8), constrained_layout=True)
    ax.scatter(
        force_100k,
        rmse_100k,
        s=13,
        alpha=0.22,
        color="#4c78a8",
        edgecolors="none",
        label="100k model validation",
    )
    ax.scatter(
        force_1m,
        rmse_1m,
        s=10,
        alpha=0.10,
        color="#59a14f",
        edgecolors="none",
        label="1M model validation",
    )

    old_colors = ["#d62728", "#ff7f0e", "#9467bd"]
    for rank, (source_index, color) in enumerate(zip((32801, 1136, 60963), old_colors), start=1):
        highlight(
            ax,
            source_100k,
            force_100k,
            rmse_100k,
            source_index,
            s=90,
            marker="o",
            color=color,
            edgecolor="black",
            linewidth=0.8,
            zorder=4,
            label=f"100k culprit {rank}: {source_index}",
        )

    highlight(
        ax,
        source_1m,
        force_1m,
        rmse_1m,
        307505,
        s=190,
        marker="*",
        color="#111111",
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
        label="1M culprit: 307505",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mean target force magnitude")
    ax.set_ylabel("Force RMSE")
    ax.set_title("Target force vs force RMSE")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(loc="best", fontsize=9)

    output = MODEL_1M / "geometry_rarity/cross_model_target_force_vs_rmse.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
