"""Plot HIP-HOP-NN random-init inference time per atom."""

import argparse
import ast
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


BASELINE_TIMES_US = {
    (0, 1): 0.7081370381913946,
    (1, 2): 1.4017859154199002,
    (2, 1): 1.3904497468259913,
    (2, 2): 2.7664624727689,
    (2, 3): 3.1545500783525275,
    (2, 4): 3.2641979364248424,
    (3, 2): 4.9360620199575935,
    (3, 3): 5.688219380801952,
    (3, 4): 7.240114127390483,
}


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


def load_benchmark_json(json_path, batch_size):
    with open(json_path) as handle:
        data = json.load(handle)

    n_atoms = int(data["n_atoms"])
    grouped = {}
    for raw_key, timings in data["metrics"].items():
        metric = dict(ast.literal_eval(raw_key))
        if int(metric["batch_size"]) != batch_size:
            continue
        key = (
            metric["tensor_model"],
            int(metric["tensor_order"]),
            int(metric["tensor_factors"]),
        )
        grouped[key] = mean(timings) * 1.0e6 / n_atoms

    if not grouped:
        raise ValueError(f"No batch-size-{batch_size} metrics found in {json_path}")
    return grouped


def config_sort_key(config):
    model, tensor_order, tensor_factors = config
    model_order = {"NONE": 0, "TS": 1, "HOP": 2}
    return model_order[model], tensor_order, tensor_factors


def config_label(config):
    model, tensor_order, tensor_factors = config
    if model == "NONE":
        return "HIP-NN"
    if model == "TS":
        return rf"TS $\ell={tensor_order}$"
    return rf"$({tensor_order}, {tensor_factors})$"


def main(args):
    if args.baseline_json or args.optimized_json:
        if not args.baseline_json or not args.optimized_json:
            raise ValueError("Both --baseline_json and --optimized_json are required")
        baseline_grouped = load_benchmark_json(args.baseline_json, args.batch_size)
        optimized_grouped = load_benchmark_json(args.optimized_json, args.batch_size)
        configs = sorted(set(baseline_grouped) | set(optimized_grouped), key=config_sort_key)
        labels = [config_label(config) for config in configs]
        baseline = [baseline_grouped.get(key, math.nan) for key in configs]
        timed = [optimized_grouped.get(key, math.nan) for key in configs]
        baseline_label = "Upstream HIP-NN kernels"
        optimized_label = "Expanded Triton kernels"
    else:
        grouped = load_rows(args.input_csv)
        if not grouped:
            raise ValueError(f"No rows found in {args.input_csv}")
        configs = sorted(set(grouped) | set(BASELINE_TIMES_US))
        labels = [f"({l}, {n})" for l, n in configs]
        timed = [mean(grouped[key]) if key in grouped else math.nan for key in configs]
        baseline = [BASELINE_TIMES_US.get(key, math.nan) for key in configs]
        baseline_label = "Baseline"
        optimized_label = "Triton optimized"

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    x = list(range(len(configs)))
    width = 0.38
    baseline_bars = ax.bar(
        [v - width / 2 for v in x],
        baseline,
        width=width,
        color="#f4b6b6",
        label=baseline_label,
    )
    optimized_bars = ax.bar(
        [v + width / 2 for v in x],
        timed,
        width=width,
        color="#3b73b9",
        label=optimized_label,
    )

    for bar, base_value, opt_value in zip(optimized_bars, baseline, timed):
        if math.isnan(base_value) or math.isnan(opt_value) or base_value <= 0 or opt_value <= 0:
            continue
        speedup = (base_value - opt_value) / base_value * 100.0
        color = "#1a7f37" if speedup >= 0 else "#c62828"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{speedup:+.0f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            color=color,
        )

    ax.set_title("HIP-HOP-NN Inference Time per Atom")
    ax.set_xlabel(r"Model architecture; HIP-HOP labels show $(\ell_{max}, n_{max})$")
    ax.set_ylabel("Time/atom (us)")
    ax.set_xticks(x, labels)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi)
    print(f"Wrote {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default="random_init_inference_results.csv")
    parser.add_argument("--baseline_json")
    parser.add_argument("--optimized_json")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--output", default="inference_time_per_atom.png")
    parser.add_argument("--dpi", type=int, default=200)
    main(parser.parse_args())
