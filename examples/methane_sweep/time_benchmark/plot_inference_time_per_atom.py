"""
Plot HIP-HOP-NN random-init inference time per atom from benchmark CSV output.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def mean(values):
    return sum(values) / len(values)


def load_rows(csv_path):
    grouped = defaultdict(list)
    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = (int(row["l_max"]), int(row["n_max"]))
            timed_us_per_atom = 1.0e6 / float(row["cuda_atoms_per_second"])
            grouped[key].append(timed_us_per_atom)
    return grouped


def main(args):
    grouped = load_rows(args.input_csv)
    if not grouped:
        raise ValueError(f"No rows found in {args.input_csv}")

    configs = sorted(grouped)
    labels = [f"({l}, {n})" for l, n in configs]
    timed = [mean(grouped[key]) for key in configs]

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    x = list(range(len(configs)))
    bars = ax.bar(x, timed, width=0.68, color="#3b73b9")

    for bar, value in zip(bars, timed):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_title("HIP-HOP-NN Inference Time per Atom")
    ax.set_xlabel(r"Model config $(\ell_{max}, n_{max})$")
    ax.set_ylabel("Time/atom (us)")
    ax.set_xticks(x, labels)
    ax.grid(axis="y", alpha=0.25)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi)
    print(f"Wrote {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default="random_init_inference_results.csv")
    parser.add_argument("--output", default="inference_time_per_atom.png")
    parser.add_argument("--dpi", type=int, default=200)
    main(parser.parse_args())
