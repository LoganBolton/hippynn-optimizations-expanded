#!/usr/bin/env python3
"""Plot architecture-level energy and force inference costs from profiler traces."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


CATEGORIES = (
    "Invariant construction",
    "Interaction feature mixing",
    "Atom-local neural layers",
    "Tensor message passing",
    "Tensor direction basis",
    "Radial sensitivities",
    "Invariant normalization",
    "Neighbor list & geometry",
    "Energy readout",
    "Other energy model ops",
    "Force: invariant derivatives",
    "Force: message derivatives",
    "Force: dense-layer derivatives",
    "Force: remaining coordinate gradient",
)

COLORS = {
    "Invariant construction": "#E3B341",
    "Interaction feature mixing": "#4F9D55",
    "Atom-local neural layers": "#78B159",
    "Tensor message passing": "#D95B59",
    "Tensor direction basis": "#E98B72",
    "Radial sensitivities": "#6D9DC5",
    "Invariant normalization": "#A878B3",
    "Neighbor list & geometry": "#3F6F9F",
    "Energy readout": "#70B7B1",
    "Other energy model ops": "#999999",
    "Force: invariant derivatives": "#F2D57A",
    "Force: message derivatives": "#EEAAA8",
    "Force: dense-layer derivatives": "#A7CFA9",
    "Force: remaining coordinate gradient": "#C8C8C8",
}


def enclosing_scope(event, annotations):
    ts = event.get("ts", -1)
    end = ts + event.get("dur", 0)
    matches = [item for item in annotations[(event.get("pid"), event.get("tid"))] if item[0] <= ts and end <= item[1]]
    return min(matches, key=lambda item: item[1] - item[0])[2] if matches else ""


def is_gemm(name):
    lowered = name.lower()
    return any(value in lowered for value in ("sgemm", "gemm", "gemv", "cutlass", "splitkreduce"))


def classify(event, scope):
    name = event.get("name", "")
    lowered = name.lower()
    scope_lower = scope.lower()
    is_force = "forces" in scope_lower or "gradient" in scope_lower

    if is_force:
        if "evaluate_polynomials_kernel" in lowered:
            return "Force: invariant derivatives"
        if "tensor_products_kernel" in lowered:
            return "Force: message derivatives"
        if is_gemm(name):
            return "Force: dense-layer derivatives"
        return "Force: remaining coordinate gradient"

    if "evaluate_polynomials_kernel" in lowered:
        return "Invariant construction"
    if "tensor_products_kernel" in lowered:
        return "Tensor message passing"
    if "pairindexer" in scope_lower or "paddingindexer" in scope_lower or "onehot" in scope_lower:
        return "Neighbor list & geometry"
    if "tensor_extractor" in scope_lower or "tensorextractor" in scope_lower:
        return "Tensor direction basis"
    if "sensitivity" in scope_lower or "cutoff" in scope_lower:
        return "Radial sensitivities"
    if "groupnorm" in scope_lower or "group_norm" in scope_lower:
        return "Invariant normalization"
    if "henergy" in scope_lower:
        return "Energy readout"
    if is_gemm(name):
        if ".blocks." in scope_lower:
            block_tail = scope_lower.split(".blocks.", 1)[1].split(" ", 1)[0]
            parts = block_tail.split(".")
            if len(parts) > 1 and parts[1] != "0":
                return "Atom-local neural layers"
        return "Interaction feature mixing"
    return "Other energy model ops"


def summarize(trace_path, n_atoms):
    events = json.loads(trace_path.read_text())["traceEvents"]
    annotations = defaultdict(list)
    for event in events:
        if event.get("ph") == "X" and event.get("cat") == "gpu_user_annotation":
            name = event.get("name", "")
            if name.startswith(("model::", "module::", "inference::")):
                key = (event.get("pid"), event.get("tid"))
                annotations[key].append((event["ts"], event["ts"] + event.get("dur", 0), name))

    totals_us = defaultdict(float)
    for event in events:
        if event.get("ph") != "X" or event.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        scope = enclosing_scope(event, annotations)
        if not scope:
            continue
        totals_us[classify(event, scope)] += event.get("dur", 0)
    return {category: totals_us[category] / n_atoms for category in CATEGORIES}


def main(args):
    profile_dir = args.profile_dir.resolve()
    metadata = json.loads((profile_dir / "metadata.json").read_text())
    n_atoms = metadata["n_atoms"]
    results = {
        "Energy only": summarize(profile_dir / "energy" / "profile_trace.json", n_atoms),
        "Energy + forces": summarize(profile_dir / "energy_forces" / "profile_trace.json", n_atoms),
    }

    order = sorted(CATEGORIES, key=results["Energy + forces"].get, reverse=True)
    fig, ax = plt.subplots(figsize=(10.5, 7))
    labels = list(results)
    bottoms = [0.0, 0.0]
    for category in order:
        values = [results[label][category] for label in labels]
        if max(values) == 0:
            continue
        ax.bar(labels, values, bottom=bottoms, width=0.58, color=COLORS[category], label=category)
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

    for index, total in enumerate(bottoms):
        ax.text(index, total, f"{total:.2f} us/atom", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("GPU kernel time per atom (us)")
    fig.suptitle(
        f"HIP-HOP-NN ({metadata['l_max']}, {metadata['n_max']}) inference architecture breakdown",
        y=0.98,
        fontsize=15,
    )
    ax.set_title(
        f"Batch {metadata['batch_size']}; {metadata['n_structures']:,} test structures; {n_atoms:,} atoms; {metadata['device_name']}",
        fontsize=9,
        color="#444444",
        pad=12,
    )
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    output = args.output or profile_dir / "architecture_breakdown.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)

    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("workload", "category", "gpu_us_per_atom", "percent"))
        for label in labels:
            total = sum(results[label].values())
            for category in order:
                value = results[label][category]
                writer.writerow((label, category, f"{value:.8f}", f"{100 * value / total:.4f}"))
    energy_total = sum(results["Energy only"].values())
    force_total = sum(results["Energy + forces"].values())
    force_extra = force_total - energy_total
    invariant_force = results["Energy + forces"]["Force: invariant derivatives"]
    message_force = results["Energy + forces"]["Force: message derivatives"]
    report_path = output.with_suffix(".md")
    report_path.write_text(
        "\n".join(
            (
                f"# HIP-HOP-NN ({metadata['l_max']}, {metadata['n_max']}) inference architecture profile",
                "",
                f"- Energy-only GPU kernels: **{energy_total:.2f} us/atom**",
                f"- Energy plus force GPU kernels: **{force_total:.2f} us/atom**",
                f"- Additional coordinate-gradient cost: **{force_extra:.2f} us/atom**",
                "",
                "## Primary optimization target",
                "",
                f"Differentiating the invariant polynomials costs **{invariant_force:.2f} us/atom**, "
                f"or **{100 * invariant_force / force_extra:.1f}% of the additional force cost**. "
                "This is the backward/derivative invocation of `evaluate_polynomials_kernel`.",
                "",
                f"Differentiating tensor message passing costs **{message_force:.2f} us/atom**, "
                f"or **{100 * message_force / force_extra:.1f}% of the additional force cost**. "
                "Together, invariant and message derivatives account for "
                f"**{100 * (invariant_force + message_force) / force_extra:.1f}%** of force overhead.",
                "",
                "The force pass calculates `-dE/dR`: the gradient of molecular energy with respect to atomic coordinates. "
                "It does not calculate a loss, parameter gradients, or an optimizer update.",
                "",
                f"Workload: batch {metadata['batch_size']}, {metadata['n_structures']:,} fixed test structures, "
                f"{n_atoms:,} atoms, {metadata['device_name']}.",
            )
        )
        + "\n"
    )
    print(f"Wrote {output}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile_dir", type=Path)
    parser.add_argument("--output", type=Path)
    main(parser.parse_args())
