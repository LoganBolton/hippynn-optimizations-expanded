#!/usr/bin/env python3
"""Analyze which validation configurations drive force RMSE spikes.

This script consumes files written by debug_validation_errors.py. It does not
re-evaluate the model.
"""

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug_dir",
        type=Path,
        default=Path("examples/TEST_METHANE_MODEL_l4_n3_d100000_seed42-b256_fresh/validation_debug"),
    )
    parser.add_argument("--baseline_epoch", type=int, default=2200)
    parser.add_argument("--epochs", nargs="+", type=int, default=[2200, 3000, 4000])
    parser.add_argument("--top_n", type=int, default=100)
    parser.add_argument("--out_dir", type=Path)
    return parser.parse_args()


def load_epoch(debug_dir, epoch):
    path = debug_dir / f"validation_errors_epoch_{epoch}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path)
    force_error = data["force_error"]
    force_sse = np.sum(force_error**2, axis=(1, 2))
    force_rmse = np.sqrt(np.mean(force_error**2, axis=(1, 2)))
    force_mae = np.mean(np.abs(force_error), axis=(1, 2))
    return {
        "epoch": epoch,
        "path": path,
        "source_indices": data["source_indices"].astype(np.int64),
        "valid_rows": np.arange(len(data["source_indices"]), dtype=np.int64),
        "positions": data["positions"],
        "forces": data["forces"],
        "energy": data["energy"],
        "pred_energy": data["pred_energy"],
        "energy_error": data["energy_error"],
        "force_error": force_error,
        "force_sse": force_sse,
        "force_rmse": force_rmse,
        "force_mae": force_mae,
        "max_abs_force_component_error": np.max(np.abs(force_error), axis=(1, 2)),
    }


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_rmse_distributions(out_dir, epochs):
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, max(float(np.max(e["force_rmse"])) for e in epochs), 80)
    for epoch_data in epochs:
        ax.hist(
            epoch_data["force_rmse"],
            bins=bins,
            histtype="step",
            linewidth=1.8,
            label=f"epoch {epoch_data['epoch']}",
        )
    ax.set_xlabel("Per-configuration force RMSE")
    ax.set_ylabel("Validation configurations")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title("Validation Force Error Tail")
    fig.tight_layout()
    fig.savefig(out_dir / "force_rmse_distribution.png", dpi=200)
    plt.close(fig)


def plot_sorted_tail(out_dir, epochs, top_n):
    fig, ax = plt.subplots(figsize=(9, 5))
    for epoch_data in epochs:
        sorted_rmse = np.sort(epoch_data["force_rmse"])[::-1]
        n = min(top_n, len(sorted_rmse))
        ax.plot(np.arange(1, n + 1), sorted_rmse[:n], marker=".", linewidth=1.2, label=f"epoch {epoch_data['epoch']}")
    ax.set_xlabel("Rank within validation set")
    ax.set_ylabel("Per-configuration force RMSE")
    ax.legend()
    ax.set_title(f"Worst {top_n} Validation Configurations")
    fig.tight_layout()
    fig.savefig(out_dir / "worst_force_rmse_by_rank.png", dpi=200)
    plt.close(fig)


def plot_sse_concentration(out_dir, epoch_data):
    order = np.argsort(epoch_data["force_sse"])[::-1]
    cumulative = np.cumsum(epoch_data["force_sse"][order]) / np.sum(epoch_data["force_sse"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(1, len(cumulative) + 1), cumulative, linewidth=1.8)
    ax.set_xscale("log")
    ax.set_xlabel("Top-N validation configurations by force squared error")
    ax.set_ylabel("Fraction of total force SSE")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Error Concentration at Epoch {epoch_data['epoch']}")
    for frac in (0.25, 0.5, 0.75):
        needed = int(np.searchsorted(cumulative, frac) + 1)
        ax.axhline(frac, color="0.75", linewidth=0.8)
        ax.text(needed, frac, f" {needed} configs", va="bottom")
    fig.tight_layout()
    fig.savefig(out_dir / f"epoch_{epoch_data['epoch']}_sse_concentration.png", dpi=200)
    plt.close(fig)


def culprit_rows(baseline, target, top_n):
    if not np.array_equal(baseline["source_indices"], target["source_indices"]):
        raise ValueError("source_indices differ between epochs; cannot compare rows directly")

    delta_sse = target["force_sse"] - baseline["force_sse"]
    order = np.argsort(delta_sse)[::-1]
    positive_delta_total = float(np.sum(delta_sse[delta_sse > 0]))
    rows = []
    cumulative = 0.0
    for rank, row in enumerate(order[:top_n], start=1):
        contribution = float(delta_sse[row])
        cumulative += max(contribution, 0.0)
        rows.append(
            {
                "rank": rank,
                "source_index": int(target["source_indices"][row]),
                "valid_row": int(row),
                "baseline_epoch": baseline["epoch"],
                "target_epoch": target["epoch"],
                "baseline_force_rmse": float(baseline["force_rmse"][row]),
                "target_force_rmse": float(target["force_rmse"][row]),
                "delta_force_rmse": float(target["force_rmse"][row] - baseline["force_rmse"][row]),
                "baseline_force_mae": float(baseline["force_mae"][row]),
                "target_force_mae": float(target["force_mae"][row]),
                "delta_force_sse": contribution,
                "positive_delta_sse_fraction": (
                    cumulative / positive_delta_total if positive_delta_total > 0 else 0.0
                ),
                "target_max_abs_force_component_error": float(target["max_abs_force_component_error"][row]),
                "baseline_energy_error": float(baseline["energy_error"][row]),
                "target_energy_error": float(target["energy_error"][row]),
                "target_energy": float(target["energy"][row]),
                "target_pred_energy": float(target["pred_energy"][row]),
                "min_pair_distance": float(min_pair_distance(target["positions"][row])),
                "max_force_magnitude": float(max_force_magnitude(target["forces"][row])),
            }
        )
    return rows


def min_pair_distance(positions):
    distances = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            distances.append(float(np.linalg.norm(positions[i] - positions[j])))
    return min(distances)


def max_force_magnitude(forces):
    return float(np.max(np.linalg.norm(forces, axis=1)))


def plot_culprit_scatter(out_dir, baseline, target, rows):
    source_to_rank = {row["source_index"]: row["rank"] for row in rows}
    top_mask = np.array([idx in source_to_rank for idx in target["source_indices"]])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(baseline["force_rmse"], target["force_rmse"], s=8, alpha=0.25, label="all validation")
    ax.scatter(
        baseline["force_rmse"][top_mask],
        target["force_rmse"][top_mask],
        s=24,
        alpha=0.9,
        label=f"top {len(rows)} delta-SSE culprits",
    )
    lim = max(float(np.max(baseline["force_rmse"])), float(np.max(target["force_rmse"])))
    ax.plot([0, lim], [0, lim], color="black", linewidth=1, linestyle="--")
    ax.set_xlabel(f"Epoch {baseline['epoch']} per-config force RMSE")
    ax.set_ylabel(f"Epoch {target['epoch']} per-config force RMSE")
    ax.set_title(f"Validation Configurations That Worsened by Epoch {target['epoch']}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"epoch_{target['epoch']}_vs_{baseline['epoch']}_force_rmse_scatter.png", dpi=200)
    plt.close(fig)


def write_summary(out_dir, epochs):
    rows = []
    for epoch_data in epochs:
        force_error = epoch_data["force_error"]
        energy_error = epoch_data["energy_error"]
        order = np.argsort(epoch_data["force_sse"])[::-1]
        total_sse = float(np.sum(epoch_data["force_sse"]))
        row = {
            "epoch": epoch_data["epoch"],
            "force_rmse": float(np.sqrt(np.mean(force_error**2))),
            "force_mae": float(np.mean(np.abs(force_error))),
            "energy_rmse": float(np.sqrt(np.mean(energy_error**2))),
            "energy_mae": float(np.mean(np.abs(energy_error))),
        }
        for n in (1, 5, 10, 25, 50, 100):
            row[f"top_{n}_force_sse_fraction"] = float(np.sum(epoch_data["force_sse"][order[:n]]) / total_sse)
        rows.append(row)
    write_csv(out_dir / "spike_summary.csv", rows, list(rows[0]))


def main():
    args = parse_args()
    out_dir = args.out_dir or args.debug_dir / "spike_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = [load_epoch(args.debug_dir, epoch) for epoch in args.epochs]
    by_epoch = {epoch_data["epoch"]: epoch_data for epoch_data in epochs}
    if args.baseline_epoch not in by_epoch:
        raise ValueError(f"baseline epoch {args.baseline_epoch} is not in --epochs")

    baseline = by_epoch[args.baseline_epoch]
    plot_rmse_distributions(out_dir, epochs)
    plot_sorted_tail(out_dir, epochs, args.top_n)
    write_summary(out_dir, epochs)

    for epoch_data in epochs:
        plot_sse_concentration(out_dir, epoch_data)
        if epoch_data["epoch"] == args.baseline_epoch:
            continue
        rows = culprit_rows(baseline, epoch_data, args.top_n)
        fields = list(rows[0])
        write_csv(out_dir / f"culprits_epoch_{epoch_data['epoch']}_vs_{baseline['epoch']}.csv", rows, fields)
        plot_culprit_scatter(out_dir, baseline, epoch_data, rows)

    print(f"Wrote spike analysis to {out_dir}")
    print(f"Start with {out_dir / 'spike_summary.csv'}")
    for epoch_data in epochs:
        if epoch_data["epoch"] != args.baseline_epoch:
            culprit_name = f"culprits_epoch_{epoch_data['epoch']}_vs_{baseline['epoch']}.csv"
            print(f"Top culprits: {out_dir / culprit_name}")


if __name__ == "__main__":
    main()
