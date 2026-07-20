#!/usr/bin/env python3
"""Summarize HIP-HOP profiler traces in terms of model architecture."""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


CATEGORIES = (
    "Neighbor list & geometry",
    "Tensor message passing",
    "Invariant polynomials",
    "Feature mixing & atom layers",
    "Invariant normalization",
    "Energy readout",
    "Other model tensor ops",
    "Memory transfer/init",
)

COLORS = {
    "Neighbor list & geometry": "#4C78A8",
    "Tensor message passing": "#E45756",
    "Invariant polynomials": "#F2CF5B",
    "Feature mixing & atom layers": "#59A14F",
    "Invariant normalization": "#B279A2",
    "Energy readout": "#76B7B2",
    "Other model tensor ops": "#9C9C9C",
    "Memory transfer/init": "#BAB0AC",
}

DESCRIPTIONS = {
    "Neighbor list & geometry": (
        "Indexes the real atoms, builds the atom-pair list inside the cutoff, and computes pair distances/directions."
    ),
    "Tensor message passing": (
        "The fused neighbor aggregation: each atom receives tensor-valued messages from nearby atoms."
    ),
    "Invariant polynomials": (
        "Turns equivariant tensor features into rotation-invariant many-body features by evaluating "
        "the HIP-HOP invariant polynomials."
    ),
    "Feature mixing & atom layers": (
        "Dense matrix multiplications that mix channels after message passing and the atom-local neural-network layers, "
        "used during the forward pass."
    ),
    "Invariant normalization": "Group-normalizes the invariant channels.",
    "Energy readout": "Maps per-atom features to atomic energies and sums them into each molecule's energy.",
    "Other model tensor ops": (
        "Activations, reductions, indexing, and tensor reshaping/copies that cannot be assigned more narrowly from these traces."
    ),
    "Memory transfer/init": "GPU memory copies, allocation initialization, and memset operations.",
}

CONFIG_RE = re.compile(r"l(?P<l_max>\d+)_n(?P<n_max>\d+)")


def _architecture_scope(event, annotations):
    """Return the innermost model/loss annotation enclosing an event."""
    ts = event.get("ts", -1)
    end = ts + event.get("dur", 0)
    matches = [
        annotation
        for annotation in annotations.get((event.get("pid"), event.get("tid")), ())
        if annotation[0] <= ts and end <= annotation[1]
    ]
    return min(matches, key=lambda item: item[1] - item[0])[2] if matches else ""


def _is_matrix_multiply(name):
    lowered = name.lower()
    return any(marker in lowered for marker in ("sgemm", "gemm", "gemv", "cutlass", "splitkreduce"))


def classify_kernel(kernel, host_event, scope):
    """Assign one non-overlapping architecture category to a GPU event."""
    name = kernel.get("name", "")
    lowered = name.lower()
    host_name = host_event.get("name", "").lower() if host_event else ""
    scope_lower = scope.lower()

    if kernel.get("cat") in ("gpu_memcpy", "gpu_memset"):
        return "Memory transfer/init"
    if "tensor_products_kernel" in lowered:
        return "Tensor message passing"
    if "evaluate_polynomials_kernel" in lowered:
        return "Invariant polynomials"
    if any(stage in scope_lower for stage in ("onehot", "paddingindexer", "pairindexer")):
        return "Neighbor list & geometry"
    if scope.startswith("model::HEnergy"):
        return "Energy readout"
    if "groupnorm" in host_name or "group_norm" in host_name or any(
        marker in lowered for marker in ("gammabeta1d", "compute1d")
    ):
        return "Invariant normalization"
    if _is_matrix_multiply(name):
        return "Feature mixing & atom layers"
    return "Other model tensor ops"


def summarize_trace(path):
    with path.open() as handle:
        trace = json.load(handle)
    events = trace["traceEvents"]

    host_annotations = defaultdict(list)
    gpu_annotations = defaultdict(list)
    host_by_external_id = {}
    for event in events:
        if event.get("ph") != "X":
            continue
        name = event.get("name", "")
        if event.get("cat") == "user_annotation" and name.startswith(("model::", "loss::")):
            key = (event.get("pid"), event.get("tid"))
            host_annotations[key].append((event["ts"], event["ts"] + event.get("dur", 0), name))
        if event.get("cat") == "gpu_user_annotation" and name.startswith(("model::", "loss::")):
            key = (event.get("pid"), event.get("tid"))
            gpu_annotations[key].append((event["ts"], event["ts"] + event.get("dur", 0), name))
        if event.get("cat") == "cpu_op":
            external_id = event.get("args", {}).get("External id")
            if external_id is not None:
                host_by_external_id[external_id] = event

    totals_us = defaultdict(float)
    kernel_totals_us = defaultdict(float)
    for event in events:
        if event.get("ph") != "X" or event.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        external_id = event.get("args", {}).get("External id")
        host_event = host_by_external_id.get(external_id)
        scope = _architecture_scope(event, gpu_annotations)
        if not scope and host_event:
            scope = _architecture_scope(host_event, host_annotations)
        # The trace was collected around training steps. Restricting to model
        # forward ranges extracts the inference workload and excludes loss,
        # parameter backpropagation, and optimizer kernels.
        if not scope.startswith("model::"):
            continue
        category = classify_kernel(event, host_event, scope)
        totals_us[category] += event.get("dur", 0)
        kernel_totals_us[event.get("name", "unknown")] += event.get("dur", 0)

    model_calls = sum(
        1
        for event in events
        if event.get("cat") == "user_annotation"
        and event.get("name", "").startswith("model::hipnn_model ")
    )
    batches = model_calls or 1
    match = CONFIG_RE.search(path.parent.name)
    if not match:
        raise ValueError(f"Could not determine l_max/n_max from directory name: {path.parent.name}")
    config = (int(match.group("l_max")), int(match.group("n_max")))
    return {
        "path": path,
        "config": config,
        "label": f"l={config[0]}, n={config[1]}",
        "batches": batches,
        "category_ms_per_batch": {category: totals_us[category] / 1000 / batches for category in CATEGORIES},
        "kernel_ms_per_batch": {name: duration / 1000 / batches for name, duration in kernel_totals_us.items()},
    }


def write_csv(results, path):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("l_max", "n_max", "profiled_batches", "category", "gpu_ms_per_batch", "percent"))
        for result in results:
            timings = result["category_ms_per_batch"]
            total = sum(timings.values())
            for category in CATEGORIES:
                value = timings[category]
                writer.writerow((*result["config"], result["batches"], category, f"{value:.6f}", f"{100 * value / total:.3f}"))


def write_report(results, path):
    lines = [
        "# HIP-HOP GPU profile by model architecture",
        "",
        "These are **forward-pass GPU kernel times extracted from batch-64 training traces**, summed without overlapping "
        "architecture ranges. Only `model::` ranges are included; loss, parameter backpropagation, optimizer updates, "
        "and force calculation are excluded. These runs used shuffled training batches and are not an absolute-speed "
        "comparison equivalent to the batch-1024 inference benchmark.",
        "",
        "| Model | Total ms | Message passing ms | Invariants ms | Feature/atom layers ms | Largest component |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for result in results:
        timings = result["category_ms_per_batch"]
        total = sum(timings.values())
        largest = max(CATEGORIES, key=timings.get)
        lines.append(
            f"| {result['label']} | {total:.2f} | {timings['Tensor message passing']:.2f} | "
            f"{timings['Invariant polynomials']:.2f} | {timings['Feature mixing & atom layers']:.2f} | "
            f"{largest} ({100 * timings[largest] / total:.1f}%) |"
        )

    lines.extend(("", "## Architectural meaning", ""))
    for category in CATEGORIES:
        lines.append(f"- **{category}:** {DESCRIPTIONS[category]}")

    lines.extend(("", "## Optimization target", ""))
    overall = {category: sum(r["category_ms_per_batch"][category] for r in results) for category in CATEGORIES}
    ranked = sorted(CATEGORIES, key=overall.get, reverse=True)
    lines.append(
        f"Across these four inference passes, the first target is **{ranked[0]}**. "
        f"The second target is **{ranked[1]}**. The absolute-time plot shows whether an optimization matters "
        "for the expensive models; the percentage plot shows how each model changes its computational balance."
    )
    l4n4 = next((result for result in results if result["config"] == (4, 4)), None)
    if l4n4:
        invariant_ms = l4n4["category_ms_per_batch"]["Invariant polynomials"]
        total_ms = sum(l4n4["category_ms_per_batch"].values())
        lines.append(
            f"Concretely, optimize the forward `evaluate_polynomials_kernel` first: "
            f"it consumes {invariant_ms:.2f} of {total_ms:.2f} GPU ms/batch in `(l_max=4, n_max=4)`."
        )
    lines.append(
        "For `(2,4)` and `(4,3)`, the largest bucket is instead the dense feature mixing and atom-local layers. "
        "That time is spread across several cuBLAS/CUTLASS matrix multiplications, rather than one identifiable kernel."
    )
    lines.extend(
        (
            "",
            "The gray `Other model tensor ops` segment is intentionally conservative. The existing traces name the fused "
            "message-passing and invariant kernels precisely, but do not put architecture labels around every internal HIP-HOP operation. "
            "It should not be treated as one kernel or one optimization target.",
            "",
            "There is no `Calculating forces` segment because force calculation is not part of this forward energy-inference view.",
        )
    )
    path.write_text("\n".join(lines) + "\n")


def plot_results(results, path):
    labels = [result["label"] for result in results]
    reference = next((result for result in results if result["config"] == (4, 4)), results[-1])
    category_order = sorted(CATEGORIES, key=reference["category_ms_per_batch"].get, reverse=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios": (1.15, 1)})

    for ax, normalized in zip(axes, (False, True)):
        bottoms = [0.0] * len(results)
        for category in category_order:
            values = [result["category_ms_per_batch"][category] for result in results]
            if normalized:
                values = [
                    100 * value / sum(result["category_ms_per_batch"].values())
                    for value, result in zip(values, results)
                ]
            ax.bar(labels, values, bottom=bottoms, label=category, color=COLORS[category], width=0.68)
            bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("HIP-HOP model configuration")
        ax.set_ylabel("GPU kernel time per batch (ms)" if not normalized else "Share of GPU kernel time (%)")
        ax.set_title("Absolute optimization impact" if not normalized else "Architectural time balance")
        ax.set_ylim(bottom=0)

    handles, legend_labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("HIP-HOP forward-pass architecture profile", fontsize=16)
    fig.text(
        0.5,
        0.925,
        "Extracted from shuffled batch-64 training traces; not the batch-1024 inference benchmark",
        ha="center",
        fontsize=10,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.14, 1, 0.91))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "traces",
        nargs="*",
        type=Path,
        help="Trace JSON files (default: ani1ccx_profile_l*_n*/profile_trace.json beside this script)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "plots")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    traces = args.traces or sorted(script_dir.glob("ani1ccx_profile_l*_n*_GPU*_SEED*/profile_trace.json"))
    if not traces:
        parser.error("No profiler traces found")

    display_order = {(2, 4): 0, (4, 3): 1, (3, 5): 2, (4, 4): 3}
    results = sorted(
        (summarize_trace(path.resolve()) for path in traces),
        key=lambda result: (display_order.get(result["config"], 100), result["config"]),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = args.output_dir / "profile_architecture_breakdown.png"
    csv_path = args.output_dir / "profile_architecture_breakdown.csv"
    report_path = args.output_dir / "profile_architecture_report.md"
    plot_results(results, plot_path)
    write_csv(results, csv_path)
    write_report(results, report_path)
    print(f"Wrote {plot_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
