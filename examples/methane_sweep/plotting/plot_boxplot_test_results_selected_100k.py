#!/usr/bin/env python3
"""Plot 100k test-result boxplots from the selected CSV."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT = Path("examples/methane_sweep/results/paper/test_results_selected_100k.csv")
DEFAULT_OUTPUT = Path("examples/methane_sweep/results/paper/boxplot_test_results_selected_100k.png")

ORDER = ["l2n4_ref", "l3n3", "l3n4", "l3n4_ref", "l3n5"]
LABELS = {
    "l2n4_ref": r"$\ell=2, n=4$ ref",
    "l3n3": r"$\ell=3, n=3$",
    "l3n4": r"$\ell=3, n=4$",
    "l3n4_ref": r"$\ell=3, n=4$ ref",
    "l3n5": r"$\ell=3, n=5$",
}
COLORS = {
    "l2n4_ref": "#d99ab5",
    "l3n3": "#56b4e9",
    "l3n4": "#862e9c",
    "l3n4_ref": "#25b99a",
    "l3n5": "#2b8a3e",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def configure_matplotlib() -> None:
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
            "lines.linewidth": 1.5,
        }
    )


def load_data(path: Path) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[row["group"]].append(float(row["test_T-RMSE"]))
    return grouped


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    grouped = load_data(args.input)

    groups = [g for g in ORDER if g in grouped]
    values = [grouped[g] for g in groups]
    labels = [LABELS.get(g, g) + f"\n(N={len(grouped[g])})" for g in groups]

    fig, ax = plt.subplots(figsize=(9.5, 6), dpi=180)
    bp = ax.boxplot(
        values,
        widths=0.6,
        patch_artist=True,
        showmeans=False,
        showfliers=False,
        boxprops=dict(edgecolor="black", linewidth=1.2),
        whiskerprops=dict(color="black", linewidth=1.2),
        capprops=dict(color="black", linewidth=1.2),
        medianprops=dict(color="darkblue", linewidth=2),
    )

    for patch, group in zip(bp["boxes"], groups):
        patch.set_facecolor(COLORS.get(group, "#cccccc"))
        patch.set_alpha(0.72)

    np.random.seed(42)
    for i, group in enumerate(groups, start=1):
        y = grouped[group]
        x = np.ones(len(y)) * i + np.random.normal(0, 0.04, size=len(y))
        ax.scatter(x, y, alpha=0.65, s=45, color="red", edgecolors="darkred", linewidth=0.5, zorder=3)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Test RMSE (kcal/mol)")
    ax.set_title("100k Test RMSE Distribution")
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=180, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    print(f"Saved {args.output}")
    for group in groups:
        arr = np.array(grouped[group], dtype=float)
        std = arr.std(ddof=1) if len(arr) > 1 else 0.0
        print(f"{group}: N={len(arr)}, mean={arr.mean():.5f}, std={std:.5f}")


if __name__ == "__main__":
    main()
