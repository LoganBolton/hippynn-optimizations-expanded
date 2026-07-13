#!/usr/bin/env python3
"""Inspect and plot top validation configurations driving a methane error spike."""

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ELEMENTS = {1: "H", 6: "C"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug_dir",
        type=Path,
        default=Path("examples/TEST_METHANE_MODEL_l4_n3_d100000_seed42-b256_fresh/validation_debug"),
    )
    parser.add_argument(
        "--culprit_csv",
        type=Path,
        default=Path(
            "examples/TEST_METHANE_MODEL_l4_n3_d100000_seed42-b256_fresh/"
            "validation_debug/spike_analysis/culprits_epoch_3000_vs_2200.csv"
        ),
    )
    parser.add_argument("--epoch", type=int, default=3000)
    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument("--source_indices", nargs="*", type=int)
    parser.add_argument("--out_dir", type=Path)
    return parser.parse_args()


def read_culprits(path, top_n):
    with path.open() as csv_file:
        rows = list(csv.DictReader(csv_file))
    return rows[:top_n]


def load_epoch(debug_dir, epoch):
    path = debug_dir / f"validation_errors_epoch_{epoch}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path)


def pair_table(numbers, positions):
    pairs = []
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            distance = float(np.linalg.norm(positions[i] - positions[j]))
            pairs.append((distance, i, j, ELEMENTS.get(int(numbers[i]), str(numbers[i])), ELEMENTS.get(int(numbers[j]), str(numbers[j]))))
    return sorted(pairs)


def angle_degrees(a, b, c):
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return float("nan")
    cosine = np.dot(ba, bc) / denom
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def methane_geometry_summary(numbers, positions):
    carbon_candidates = np.where(numbers == 6)[0]
    if len(carbon_candidates) != 1:
        return {}
    carbon = int(carbon_candidates[0])
    hydrogens = [int(i) for i in np.where(numbers == 1)[0]]
    ch = [(h, float(np.linalg.norm(positions[h] - positions[carbon]))) for h in hydrogens]
    hh = []
    hch = []
    for i, h1 in enumerate(hydrogens):
        for h2 in hydrogens[i + 1 :]:
            hh.append(float(np.linalg.norm(positions[h1] - positions[h2])))
            hch.append(angle_degrees(positions[h1], positions[carbon], positions[h2]))
    return {
        "carbon_index": carbon,
        "ch_distances": ch,
        "min_ch": min(d for _, d in ch),
        "max_ch": max(d for _, d in ch),
        "mean_ch": float(np.mean([d for _, d in ch])),
        "min_hh": min(hh),
        "max_hh": max(hh),
        "mean_hh": float(np.mean(hh)),
        "min_hch_angle": min(hch),
        "max_hch_angle": max(hch),
        "mean_hch_angle": float(np.mean(hch)),
    }


def write_xyz(path, numbers, positions, comment):
    with path.open("w") as xyz:
        xyz.write(f"{len(numbers)}\n")
        xyz.write(comment + "\n")
        for number, position in zip(numbers, positions):
            symbol = ELEMENTS.get(int(number), str(int(number)))
            xyz.write(f"{symbol} {position[0]: .10f} {position[1]: .10f} {position[2]: .10f}\n")


def plot_structure(path, numbers, positions, forces, force_error, title):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    colors = ["#222222" if int(n) == 6 else "#dddddd" for n in numbers]
    sizes = [140 if int(n) == 6 else 80 for n in numbers]

    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=sizes, c=colors, edgecolors="black", depthshade=True)
    for distance, i, j, _, _ in pair_table(numbers, positions):
        if distance < 1.8:
            xs = [positions[i, 0], positions[j, 0]]
            ys = [positions[i, 1], positions[j, 1]]
            zs = [positions[i, 2], positions[j, 2]]
            ax.plot(xs, ys, zs, color="0.55", linewidth=1.5)

    scale = 0.04
    ax.quiver(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        force_error[:, 0],
        force_error[:, 1],
        force_error[:, 2],
        length=scale,
        normalize=False,
        color="#d62728",
        linewidth=1.5,
    )
    ax.quiver(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        forces[:, 0],
        forces[:, 1],
        forces[:, 2],
        length=scale,
        normalize=False,
        color="#1f77b4",
        linewidth=1.0,
        alpha=0.65,
    )

    for idx, (number, position) in enumerate(zip(numbers, positions)):
        ax.text(position[0], position[1], position[2], f" {ELEMENTS.get(int(number), number)}{idx}", fontsize=9)

    center = np.mean(positions, axis=0)
    span = max(float(np.ptp(positions[:, axis])) for axis in range(3))
    radius = max(span * 0.75, 1.2)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def set_2d_bounds(ax, positions, vectors=None, scale=0.015, pad=0.45):
    points = [positions[:, :2]]
    if vectors is not None:
        points.append(positions[:, :2] + vectors[:, :2] * scale)
    all_points = np.concatenate(points, axis=0)
    mins = np.min(all_points, axis=0)
    maxs = np.max(all_points, axis=0)
    center = (mins + maxs) / 2
    span = max(float(np.max(maxs - mins)), 1.0) + 2 * pad
    ax.set_xlim(center[0] - span / 2, center[0] + span / 2)
    ax.set_ylim(center[1] - span / 2, center[1] + span / 2)
    ax.set_aspect("equal", adjustable="box")


def draw_molecule_2d(ax, numbers, positions, dims=(0, 1), annotate_distances=True):
    projected = positions[:, dims]
    for distance, i, j, symbol_i, symbol_j in pair_table(numbers, positions):
        if distance < 1.8:
            ax.plot(
                [projected[i, 0], projected[j, 0]],
                [projected[i, 1], projected[j, 1]],
                color="0.5",
                linewidth=1.6,
                zorder=1,
            )
            if annotate_distances:
                midpoint = (projected[i] + projected[j]) / 2
                ax.text(midpoint[0], midpoint[1], f"{distance:.2f}", fontsize=8, color="0.25")
    colors = ["#222222" if int(n) == 6 else "#f4f4f4" for n in numbers]
    sizes = [150 if int(n) == 6 else 95 for n in numbers]
    ax.scatter(projected[:, 0], projected[:, 1], s=sizes, c=colors, edgecolors="black", linewidths=1, zorder=2)
    for idx, (number, point) in enumerate(zip(numbers, projected)):
        ax.text(point[0] + 0.035, point[1] + 0.035, f"{ELEMENTS.get(int(number), number)}{idx}", fontsize=9, zorder=3)


def plot_clean_views(path, numbers, positions, forces, force_error, title):
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    fig.suptitle(title)
    panels = [
        ("Geometry x-y", None, (0, 1), "none"),
        ("Geometry x-z", None, (0, 2), "none"),
        ("Target forces x-y", forces, (0, 1), "#1f77b4"),
        ("Force error x-y", force_error, (0, 1), "#d62728"),
    ]
    vector_scale = 0.015
    for ax, (panel_title, vectors, dims, color) in zip(axes.ravel(), panels):
        draw_molecule_2d(ax, numbers, positions, dims=dims, annotate_distances=vectors is None)
        if vectors is not None:
            projected = positions[:, dims]
            projected_vectors = vectors[:, dims]
            ax.quiver(
                projected[:, 0],
                projected[:, 1],
                projected_vectors[:, 0],
                projected_vectors[:, 1],
                angles="xy",
                scale_units="xy",
                scale=1 / vector_scale,
                color=color,
                width=0.006,
                zorder=4,
            )
        set_2d_bounds(ax, positions[:, dims], vectors[:, dims] if vectors is not None else None, scale=vector_scale)
        ax.set_title(panel_title)
        ax.set_xlabel("xyz"[dims[0]])
        ax.set_ylabel("xyz"[dims[1]])
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_detail_report(path, summary, numbers, positions, forces, force_error):
    geom = methane_geometry_summary(numbers, positions)
    with path.open("w") as report:
        report.write(f"rank: {summary['rank']}\n")
        report.write(f"source_index: {summary['source_index']}\n")
        report.write(f"valid_row: {summary['valid_row']}\n")
        report.write(f"target_force_rmse: {summary['target_force_rmse']}\n")
        report.write(f"baseline_force_rmse: {summary['baseline_force_rmse']}\n")
        report.write(f"delta_force_rmse: {summary['delta_force_rmse']}\n\n")
        report.write("Coordinates:\n")
        for idx, (number, position) in enumerate(zip(numbers, positions)):
            report.write(f"  {idx} {ELEMENTS.get(int(number), number):>2} {position[0]: .6f} {position[1]: .6f} {position[2]: .6f}\n")
        report.write("\nShortest pair distances:\n")
        for distance, i, j, symbol_i, symbol_j in pair_table(numbers, positions)[:10]:
            report.write(f"  {symbol_i}{i}-{symbol_j}{j}: {distance:.6f} Ang\n")
        if geom:
            report.write("\nMethane geometry:\n")
            report.write(f"  C atom: {geom['carbon_index']}\n")
            for hydrogen, distance in geom["ch_distances"]:
                report.write(f"  C-H{hydrogen}: {distance:.6f} Ang\n")
            report.write(f"  H-C-H angle range: {geom['min_hch_angle']:.3f} to {geom['max_hch_angle']:.3f} deg\n")
        report.write("\nPer-atom target force and force error magnitudes:\n")
        for idx, number in enumerate(numbers):
            fmag = float(np.linalg.norm(forces[idx]))
            emag = float(np.linalg.norm(force_error[idx]))
            report.write(f"  {ELEMENTS.get(int(number), number)}{idx}: |target_force|={fmag:.6f}, |force_error|={emag:.6f}\n")


def main():
    args = parse_args()
    out_dir = args.out_dir or args.debug_dir / "culprit_structures"
    out_dir.mkdir(parents=True, exist_ok=True)

    epoch_data = load_epoch(args.debug_dir, args.epoch)
    culprits = read_culprits(args.culprit_csv, args.top_n)
    if args.source_indices:
        wanted = set(args.source_indices)
        culprits = [row for row in culprits if int(row["source_index"]) in wanted]
        missing = wanted - {int(row["source_index"]) for row in culprits}
        if missing:
            raise ValueError(f"source indices were not present in the selected culprit rows: {sorted(missing)}")
    summary_rows = []

    for culprit in culprits:
        rank = int(culprit["rank"])
        row = int(culprit["valid_row"])
        source_index = int(culprit["source_index"])
        numbers = epoch_data["numbers"][row].astype(int)
        positions = epoch_data["positions"][row]
        forces = epoch_data["forces"][row]
        force_error = epoch_data["force_error"][row]
        geom = methane_geometry_summary(numbers, positions)
        pairs = pair_table(numbers, positions)
        prefix = f"rank_{rank:03d}_source_{source_index}_epoch_{args.epoch}"

        write_xyz(
            out_dir / f"{prefix}.xyz",
            numbers,
            positions,
            (
                f"rank={rank} source_index={source_index} valid_row={row} "
                f"target_force_rmse={culprit.get('target_force_rmse', '')}"
            ),
        )
        plot_structure(
            out_dir / f"{prefix}.png",
            numbers,
            positions,
            forces,
            force_error,
            f"rank {rank}, source {source_index}, epoch {args.epoch}",
        )
        plot_clean_views(
            out_dir / f"{prefix}_clean_views.png",
            numbers,
            positions,
            forces,
            force_error,
            f"rank {rank}, source {source_index}, epoch {args.epoch}",
        )

        summary = {
            "rank": rank,
            "source_index": source_index,
            "valid_row": row,
            "target_force_rmse": culprit.get("target_force_rmse", ""),
            "baseline_force_rmse": culprit.get("baseline_force_rmse", ""),
            "delta_force_rmse": culprit.get("delta_force_rmse", ""),
            "target_max_abs_force_component_error": culprit.get("target_max_abs_force_component_error", ""),
            "energy_error": float(epoch_data["energy_error"][row]),
            "energy": float(epoch_data["energy"][row]),
            "pred_energy": float(epoch_data["pred_energy"][row]),
            "min_pair_distance": pairs[0][0],
            "shortest_pair": f"{pairs[0][3]}{pairs[0][1]}-{pairs[0][4]}{pairs[0][2]}",
            "max_force_magnitude": float(np.max(np.linalg.norm(forces, axis=1))),
            "max_force_error_magnitude": float(np.max(np.linalg.norm(force_error, axis=1))),
            "xyz": f"{prefix}.xyz",
            "png": f"{prefix}.png",
            "clean_views_png": f"{prefix}_clean_views.png",
            "detail_report": f"{prefix}_details.txt",
        }
        if geom:
            summary.update(
                {
                    "min_ch": geom["min_ch"],
                    "max_ch": geom["max_ch"],
                    "mean_ch": geom["mean_ch"],
                    "min_hh": geom["min_hh"],
                    "max_hh": geom["max_hh"],
                    "mean_hh": geom["mean_hh"],
                    "min_hch_angle": geom["min_hch_angle"],
                    "max_hch_angle": geom["max_hch_angle"],
                    "mean_hch_angle": geom["mean_hch_angle"],
                }
            )
        summary_rows.append(summary)
        write_detail_report(out_dir / f"{prefix}_details.txt", summary, numbers, positions, forces, force_error)

    fieldnames = list(summary_rows[0])
    with (out_dir / "structure_summary.csv").open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Wrote structure summaries and plots to {out_dir}")
    print(f"Start with {out_dir / 'structure_summary.csv'}")


if __name__ == "__main__":
    main()
