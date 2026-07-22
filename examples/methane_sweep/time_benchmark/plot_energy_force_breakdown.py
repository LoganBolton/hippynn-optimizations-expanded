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

    if bool(args.optimized_energy_pt) != bool(args.optimized_energy_forces_pt):
        raise ValueError(
            "Both --optimized_energy_pt and --optimized_energy_forces_pt are required"
        )
    optimized_energy = (
        load_benchmark_pt(args.optimized_energy_pt, args.batch_size, args.n_atoms)
        if args.optimized_energy_pt
        else {}
    )
    optimized_total = (
        load_benchmark_pt(
            args.optimized_energy_forces_pt, args.batch_size, args.n_atoms
        )
        if args.optimized_energy_forces_pt
        else {}
    )

    show_optimized = bool(optimized_energy)
    config_set = set(upstream_energy) | set(upstream_total) | set(triton_energy) | set(triton_total)
    if show_optimized:
        config_set |= set(optimized_energy) | set(optimized_total)
    configs = sorted(
        {
            config
            for config in config_set
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
    default_width = 0.26 if show_optimized else 0.38
    width = args.bar_width if args.bar_width is not None else default_width
    upstream_x = [value - width if show_optimized else value - width / 2 for value in x]
    triton_x = x if show_optimized else [value + width / 2 for value in x]
    optimized_x = [value + width for value in x]

    upstream_energy_values = [upstream_energy.get(config, math.nan) for config in configs]
    upstream_total_values = [upstream_total.get(config, math.nan) for config in configs]
    triton_energy_values = [triton_energy.get(config, math.nan) for config in configs]
    triton_total_values = [triton_total.get(config, math.nan) for config in configs]
    optimized_energy_values = [optimized_energy.get(config, math.nan) for config in configs]
    optimized_total_values = [optimized_total.get(config, math.nan) for config in configs]

    upstream_force_values = [
        max(total - energy, 0.0) if math.isfinite(energy) and math.isfinite(total) else math.nan
        for energy, total in zip(upstream_energy_values, upstream_total_values)
    ]
    triton_force_values = [
        max(total - energy, 0.0) if math.isfinite(energy) and math.isfinite(total) else math.nan
        for energy, total in zip(triton_energy_values, triton_total_values)
    ]
    optimized_force_values = [
        max(total - energy, 0.0) if math.isfinite(energy) and math.isfinite(total) else math.nan
        for energy, total in zip(optimized_energy_values, optimized_total_values)
    ]

    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height), constrained_layout=True)
    upstream_color = "#e99f9f"
    triton_color = "#3b73b9"
    optimized_color = "#2e8b57"

    if args.total_only:
        ax.bar(upstream_x, upstream_total_values, width=width, color=upstream_color)
        triton_total_bars = ax.bar(
            triton_x, triton_total_values, width=width, color=triton_color
        )
        triton_energy_bars = []
        optimized_total_bars = ax.bar(
            optimized_x,
            optimized_total_values,
            width=width,
            color=optimized_color,
        )
        optimized_energy_bars = []
    else:
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
        optimized_energy_bars = ax.bar(
            optimized_x,
            optimized_energy_values,
            width=width,
            color=optimized_color,
        )
        optimized_total_bars = ax.bar(
            optimized_x,
            optimized_force_values,
            width=width,
            bottom=optimized_energy_values,
            color=optimized_color,
            alpha=0.42,
        )

    if not args.total_only:
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
                fontsize=6,
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
            fontsize=7,
            color=color,
        )

    if show_optimized:
        if args.optimized_compare_to == "default":
            optimized_energy_reference_values = upstream_energy_values
            optimized_total_reference_values = upstream_total_values
        elif args.optimized_compare_to == "triton":
            optimized_energy_reference_values = triton_energy_values
            optimized_total_reference_values = triton_total_values
        else:
            optimized_energy_reference_values = [
                upstream if math.isfinite(upstream) else triton
                for upstream, triton in zip(upstream_energy_values, triton_energy_values)
            ]
            optimized_total_reference_values = [
                upstream if math.isfinite(upstream) else triton
                for upstream, triton in zip(upstream_total_values, triton_total_values)
            ]

    if show_optimized and not args.total_only:
        for bar, reference_value, optimized_value in zip(
            optimized_energy_bars, optimized_energy_reference_values, optimized_energy_values
        ):
            if not math.isfinite(reference_value) or not math.isfinite(optimized_value):
                continue
            label, color = comparison_label(reference_value, optimized_value)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                optimized_value,
                label,
                ha="center",
                va="bottom",
                fontsize=6,
                color=color,
                clip_on=False,
            )

    if show_optimized:
        for bar, reference_value, optimized_value in zip(
            optimized_total_bars, optimized_total_reference_values, optimized_total_values
        ):
            if not math.isfinite(reference_value) or not math.isfinite(optimized_value):
                continue
            label, color = comparison_label(reference_value, optimized_value)
            total_height = optimized_value
            if not args.total_only:
                total_height = bar.get_y() + bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                total_height,
                label,
                ha="center",
                va="bottom",
                fontsize=7,
                color=color,
                clip_on=False,
            )

    if not args.total_only:
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
                    fontsize=6,
                    color="#666666",
                )

    l3n4 = ("HOP", 3, 4)
    l3n5 = ("HOP", 3, 5)
    if l3n4 in configs and l3n5 in configs:
        divider_x = (configs.index(l3n4) + configs.index(l3n5)) / 2
        ax.axvline(divider_x, color="#999999", linestyle=":", linewidth=1.2, alpha=0.75)

    if args.total_only:
        legend_handles = [
            Patch(facecolor=upstream_color, label="Default energy + forces"),
            Patch(facecolor=triton_color, label="Triton energy + forces"),
        ]
        if show_optimized:
            legend_handles.append(
                Patch(facecolor=optimized_color, label=args.optimized_label)
            )
        title = "HIP-HOP-NN Energy + Forces Inference Time per Atom"
    else:
        legend_handles = [
            Patch(facecolor=upstream_color, label="Default energy"),
            Patch(facecolor=upstream_color, alpha=0.42, label="Default force calculation"),
            Patch(facecolor=triton_color, label="Triton energy"),
            Patch(facecolor=triton_color, alpha=0.42, label="Triton force calculation"),
        ]
        if show_optimized:
            legend_handles.extend(
                [
                    Patch(facecolor=optimized_color, label=f"{args.optimized_label} energy"),
                    Patch(
                        facecolor=optimized_color,
                        alpha=0.42,
                        label=f"{args.optimized_label} force calculation",
                    ),
                ]
            )
        title = "HIP-HOP-NN Energy and Force Inference Time per Atom"
    ax.legend(handles=legend_handles, ncols=2)
    ax.set_title(title)
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
        "--optimized_energy_pt",
        help="Optional result file for a third, green optimized-kernel series",
    )
    parser.add_argument(
        "--optimized_energy_forces_pt",
        help="Optional force result file for the third optimized-kernel series",
    )
    parser.add_argument(
        "--optimized_label",
        default="2D polynomial Triton",
        help="Legend label for the optional green series",
    )
    parser.add_argument(
        "--optimized_compare_to",
        choices=["triton", "default", "default_then_triton"],
        default="triton",
        help="Series used for green percentage labels when the optional optimized series is shown",
    )
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
        "--bar_width",
        type=float,
        help="Optional grouped-bar width override; larger values make each column thicker",
    )
    parser.add_argument("--fig_width", type=float, default=10)
    parser.add_argument("--fig_height", type=float, default=5.5)
    parser.add_argument(
        "--total_only",
        action="store_true",
        help="Plot one energy-plus-forces bar per implementation instead of a stacked breakdown",
    )
    parser.add_argument(
        "--exclude_config",
        action="append",
        default=[],
        metavar="MODEL:LMAX:NMAX",
        help="Configuration to omit; may be supplied more than once",
    )
    main(parser.parse_args())
