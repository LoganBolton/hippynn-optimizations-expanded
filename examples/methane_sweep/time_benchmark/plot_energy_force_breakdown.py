"""Plot energy inference time and the additional cost of force calculation."""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from plot_inference_time_per_atom import config_label, config_sort_key, load_benchmark_pt


def comparison_label(upstream_value, triton_value):
    relative_change = (triton_value - upstream_value) / upstream_value * 100.0
    if abs(relative_change) < 0.5:
        relative_change = 0.0
    color = "#666666" if relative_change == 0 else ("#c62828" if relative_change > 0 else "#1a7f37")
    return f"{relative_change:+.0f}%", color


def main(args):
    upstream_energy = load_benchmark_pt(args.upstream_energy_pt, args.batch_size, args.n_atoms)
    upstream_total = load_benchmark_pt(args.upstream_energy_forces_pt, args.batch_size, args.n_atoms)
    triton_energy = load_benchmark_pt(args.triton_energy_pt, args.batch_size, args.n_atoms)
    for extra_path in args.extra_triton_energy_pt:
        triton_energy.update(load_benchmark_pt(extra_path, args.batch_size, args.n_atoms))
    triton_total = load_benchmark_pt(args.triton_energy_forces_pt, args.batch_size, args.n_atoms)
    for extra_path in args.extra_triton_energy_forces_pt:
        triton_total.update(load_benchmark_pt(extra_path, args.batch_size, args.n_atoms))

    configs = sorted(
        {
            config
            for config in set(upstream_energy) | set(triton_energy)
            if config[0] != "TS"
        },
        key=config_sort_key,
    )
    excluded_configs = {
        (model, int(tensor_order), int(tensor_factors))
        for model, tensor_order, tensor_factors in (
            value.split(":") for value in args.exclude_config
        )
    }
    configs = [config for config in configs if config not in excluded_configs]
    if not configs:
        raise ValueError("The four result files have no configurations in common")

    labels = [config_label(config) for config in configs]
    x = list(range(len(configs)))
    width = 0.38
    upstream_x = [value - width / 2 for value in x]
    triton_x = [value + width / 2 for value in x]

    upstream_energy_values = [upstream_energy.get(config, math.nan) for config in configs]
    upstream_total_values = [upstream_total.get(config, math.nan) for config in configs]
    triton_energy_values = [triton_energy.get(config, math.nan) for config in configs]
    triton_total_values = [triton_total.get(config, math.nan) for config in configs]

    upstream_force_values = [
        max(total - energy, 0.0) if math.isfinite(energy) and math.isfinite(total) else math.nan
        for energy, total in zip(upstream_energy_values, upstream_total_values)
    ]
    triton_force_values = [
        max(total - energy, 0.0) if math.isfinite(energy) and math.isfinite(total) else math.nan
        for energy, total in zip(triton_energy_values, triton_total_values)
    ]

    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    upstream_color = "#e99f9f"
    triton_color = "#3b73b9"

    ax.bar(upstream_x, upstream_energy_values, width=width, color=upstream_color)
    ax.bar(
        upstream_x,
        upstream_force_values,
        width=width,
        bottom=upstream_energy_values,
        color=upstream_color,
        alpha=0.42,
    )
    triton_energy_bars = ax.bar(
        triton_x, triton_energy_values, width=width, color=triton_color
    )
    triton_total_bars = ax.bar(
        triton_x,
        triton_force_values,
        width=width,
        bottom=triton_energy_values,
        color=triton_color,
        alpha=0.42,
    )

    for bar, upstream_value, triton_value in zip(
        triton_energy_bars, upstream_energy_values, triton_energy_values
    ):
        if not math.isfinite(upstream_value) or not math.isfinite(triton_value):
            continue
        if upstream_value <= 0 or triton_value <= 0:
            continue
        label, color = comparison_label(upstream_value, triton_value)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            triton_value,
            label,
            ha="center",
            va="bottom",
            fontsize=7,
            color=color,
        )

    for bar, upstream_value, triton_value in zip(
        triton_total_bars, upstream_total_values, triton_total_values
    ):
        if not math.isfinite(upstream_value) or not math.isfinite(triton_value):
            continue
        if upstream_value <= 0 or triton_value <= 0:
            continue
        label, color = comparison_label(upstream_value, triton_value)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            triton_value,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
            color=color,
        )

    for bar, energy_value, total_value in zip(
        triton_energy_bars, triton_energy_values, triton_total_values
    ):
        if math.isfinite(energy_value) and not math.isfinite(total_value):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                energy_value,
                "force N/A",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#666666",
            )

    l3n4 = ("HOP", 3, 4)
    l3n5 = ("HOP", 3, 5)
    if l3n4 in configs and l3n5 in configs:
        divider_x = (configs.index(l3n4) + configs.index(l3n5)) / 2
        ax.axvline(divider_x, color="#999999", linestyle=":", linewidth=1.2, alpha=0.75)

    legend_handles = [
        Patch(facecolor=upstream_color, label="Default energy"),
        Patch(facecolor=upstream_color, alpha=0.42, label="Default force calculation"),
        Patch(facecolor=triton_color, label="Triton energy"),
        Patch(facecolor=triton_color, alpha=0.42, label="Triton force calculation"),
    ]
    ax.legend(handles=legend_handles, ncols=2)
    ax.set_title("HIP-HOP-NN Energy and Force Inference Time per Atom")
    ax.set_xlabel(r"HIP-HOP settings: $(\ell_{max}, n_{max})$")
    ax.set_ylabel("Time/atom (us)")
    ax.set_xticks(x, labels)
    ax.grid(axis="y", alpha=0.25)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi)
    print(f"Wrote {output}")


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream_energy_pt", required=True)
    parser.add_argument("--upstream_energy_forces_pt", required=True)
    parser.add_argument("--triton_energy_pt", required=True)
    parser.add_argument(
        "--extra_triton_energy_pt",
        action="append",
        default=[],
        help="Additional Triton energy-only result file; may be supplied more than once",
    )
    parser.add_argument("--triton_energy_forces_pt", required=True)
    parser.add_argument(
        "--extra_triton_energy_forces_pt",
        action="append",
        default=[],
        help="Additional Triton energy-and-force result file; may be supplied more than once",
    )
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--n_atoms", type=int, default=676395)
    parser.add_argument(
        "--output",
        default=script_dir / "plots/energy_force_breakdown_inference_time_per_atom.png",
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--exclude_config",
        action="append",
        default=[],
        metavar="MODEL:LMAX:NMAX",
        help="Configuration to omit; may be supplied more than once",
    )
    main(parser.parse_args())
