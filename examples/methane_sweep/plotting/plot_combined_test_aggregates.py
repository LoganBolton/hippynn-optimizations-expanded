#!/usr/bin/env python3
"""Combine seed-aggregated methane final-test results into one paper-style plot."""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path


def pooled(groups: list[dict[str, str]], mean_key: str, std_key: str) -> tuple[float, float, int]:
    count = sum(int(group["completed_seed_count"]) for group in groups)
    mean = sum(
        int(group["completed_seed_count"]) * float(group[mean_key]) for group in groups
    ) / count
    if count < 2:
        return mean, 0.0, count
    squared_deviations = sum(
        (int(group["completed_seed_count"]) - 1) * float(group[std_key]) ** 2
        + int(group["completed_seed_count"]) * (float(group[mean_key]) - mean) ** 2
        for group in groups
    )
    return mean, math.sqrt(squared_deviations / (count - 1)), count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--csv-output", required=True, type=Path)
    args = parser.parse_args()

    grouped: dict[tuple[int, int, int], list[dict[str, str]]] = defaultdict(list)
    for path in args.input:
        with path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (int(row["hiphop_l_max"]), int(row["hiphop_n_max"]), int(row["data_size"]))
                row["source_csv"] = str(path)
                grouped[key].append(row)

    points = []
    for (l_max, n_max, data_size), groups in sorted(grouped.items()):
        rmse_mean, rmse_std, count = pooled(groups, "test_T-RMSE_mean", "test_T-RMSE_std")
        normalized_mean, normalized_std, _ = pooled(
            groups, "test_energy_RMSE_over_STD_mean", "test_energy_RMSE_over_STD_std"
        )
        energy_stds = {round(float(group["test_energy_std_kcal_per_mol"]), 12) for group in groups}
        if len(energy_stds) != 1:
            raise ValueError(f"Inconsistent test-set standard deviations for {(l_max, n_max, data_size)}")
        points.append(
            {
                "hiphop_l_max": l_max,
                "hiphop_n_max": n_max,
                "data_size": data_size,
                "completed_seed_count": count,
                "test_T-RMSE_mean": rmse_mean,
                "test_T-RMSE_std": rmse_std,
                "test_energy_std_kcal_per_mol": next(iter(energy_stds)),
                "test_energy_RMSE_over_STD_mean": normalized_mean,
                "test_energy_RMSE_over_STD_std": normalized_std,
                "source_csvs": " | ".join(group["source_csv"] for group in groups),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)

    os.environ.setdefault("MPLCONFIGDIR", str(args.output.parent / ".matplotlib"))
    import matplotlib.pyplot as plt

    styles = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
    configurations = sorted({(point["hiphop_l_max"], point["hiphop_n_max"]) for point in points})
    offsets = {
        # A small log-space dodge keeps equal-size configurations readable
        # without materially moving them away from their true dataset size.
        config: 10 ** ((index - (len(configurations) - 1) / 2) * 0.012)
        for index, config in enumerate(configurations)
    }

    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    for index, config in enumerate(configurations):
        config_points = sorted(
            (point for point in points if (point["hiphop_l_max"], point["hiphop_n_max"]) == config),
            key=lambda point: point["data_size"],
        )
        x = [float(point["data_size"]) * offsets[config] for point in config_points]
        y = [float(point["test_energy_RMSE_over_STD_mean"]) for point in config_points]
        yerr = [float(point["test_energy_RMSE_over_STD_std"]) for point in config_points]
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker=styles[index % len(styles)],
            markersize=8,
            capsize=5,
            linewidth=1.5,
            linestyle="-" if len(config_points) > 1 else "none",
            label=(
                f"l={config[0]}, n={config[1]}, "
                f"d={int(config_points[0]['data_size']) // 1000}k "
                f"(N={sum(int(point['completed_seed_count']) for point in config_points)})"
            ),
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(8e2, 3e6)
    ax.set_ylim(3e-3, 2.5e-1)
    ax.set_xlabel("Training set size")
    ax.set_ylabel("Energy RMSE / STD")
    ax.set_title("Held-Out Test Energy Error, Mean ± Seed STD")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right", fontsize=9, title="Configuration")
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
