#!/usr/bin/env python3
"""Create a 100k box plot for best validation RMSE across model configs."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_OUTPUT = Path("examples/methane_sweep/results/paper/boxplot_best_valid_rmse_all_configs_100k.png")
DEFAULT_CSV = Path("examples/methane_sweep/results/paper/boxplot_best_valid_rmse_all_configs_100k.csv")
DEFAULT_REFERENCE_JSON = Path("examples/methane_sweep/results/paper/methane_results.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output PNG path.")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV, help="Output CSV path.")
    parser.add_argument(
        "--reference-json",
        type=Path,
        default=DEFAULT_REFERENCE_JSON,
        help="Optional methane_results.json containing reference RMSE distributions.",
    )
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
            "legend.fontsize": 10,
            "lines.linewidth": 1.5,
        }
    )


def load_best_valid_per_run(metrics_csv: Path) -> list[dict[str, object]]:
    per_run: dict[tuple[str, str], dict[str, object]] = {}

    with metrics_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            valid_rmse = row.get("valid_T-RMSE", "")
            if not valid_rmse:
                continue

            try:
                value = float(valid_rmse)
                epoch = int(row["epoch"])
                l_max = int(row["hiphop_l_max"])
                n_max = int(row["hiphop_n_max"])
                data_size = int(row["data_size"])
            except (TypeError, ValueError, KeyError):
                continue

            run = row["run"]
            source_file = row["source_file"]
            key = (run, source_file)

            current = per_run.get(key)
            if current is None or value < current["best_valid_T-RMSE"]:
                per_run[key] = {
                    "run": run,
                    "source_file": source_file,
                    "sweep_task_id": row.get("sweep_task_id", ""),
                    "seed": row.get("seed", ""),
                    "hiphop_l_max": l_max,
                    "hiphop_n_max": n_max,
                    "data_size": data_size,
                    "best_epoch": epoch,
                    "best_valid_T-RMSE": value,
                    "metrics_csv": str(metrics_csv),
                }

    return list(per_run.values())


def metrics_sources() -> list[Path]:
    current_repo = [
        Path("examples/methane_sweep/logs/plots_l3_b256_final_test/metrics.csv"),
        Path("examples/methane_sweep/logs/plots_compare_l4n3_b256_fresh/metrics.csv"),
        Path("examples/methane_sweep/logs/plots_compare_l4n4_b256_fresh/metrics.csv"),
    ]

    l3n5_candidates = [
        Path("examples/methane_sweep/logs/plots_l3n5_100k_all8/metrics.csv"),
        Path("/vast/home/logan_bolton/Github/hippynn-lmax3-nmax5/examples/methane_sweep/logs/plots_l3n5_100k_all8/metrics.csv"),
        Path("/vast/home/logan_bolton/Github/hippynn-optimization-speedup-l3n5/examples/methane_sweep/logs/plots_l3n5_100k_all8/metrics.csv"),
    ]

    resolved: list[Path] = []
    resolved.extend(path for path in current_repo if path.exists())
    for candidate in l3n5_candidates:
        if candidate.exists():
            resolved.append(candidate)
            break

    return resolved


def load_all_best_valid_100k() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for metrics_csv in metrics_sources():
        rows.extend(load_best_valid_per_run(metrics_csv))

    rows = [row for row in rows if row["data_size"] == 100000]

    # Merge resumed/relogged fragments into one logical run per config+seed.
    merged: dict[tuple[int, int, int, str], dict[str, object]] = {}
    for row in rows:
        key = (
            int(row["hiphop_l_max"]),
            int(row["hiphop_n_max"]),
            int(row["data_size"]),
            str(row["seed"]),
        )
        current = merged.get(key)
        if current is None:
            merged[key] = {
                **row,
                "all_runs": [str(row["run"])],
                "all_source_files": [str(row["source_file"])],
                "all_metrics_csvs": [str(row["metrics_csv"])],
            }
            continue

        current["all_runs"].append(str(row["run"]))
        current["all_source_files"].append(str(row["source_file"]))
        current["all_metrics_csvs"].append(str(row["metrics_csv"]))
        if float(row["best_valid_T-RMSE"]) < float(current["best_valid_T-RMSE"]):
            current.update(
                {
                    "run": row["run"],
                    "source_file": row["source_file"],
                    "sweep_task_id": row.get("sweep_task_id", ""),
                    "best_epoch": row["best_epoch"],
                    "best_valid_T-RMSE": row["best_valid_T-RMSE"],
                    "metrics_csv": row["metrics_csv"],
                }
            )

    for row in merged.values():
        row["all_runs"] = sorted(set(row["all_runs"]))
        row["all_source_files"] = sorted(set(row["all_source_files"]))
        row["all_metrics_csvs"] = sorted(set(row["all_metrics_csvs"]))
        l_max = int(row["hiphop_l_max"])
        n_max = int(row["hiphop_n_max"])
        row["config"] = f"l{l_max}n{n_max}"
        row["config_label"] = rf"$\ell={l_max}, n={n_max}$"
        row["metric_type"] = "best_valid"
        row["sort_key"] = (l_max, n_max, 0)

    return list(merged.values())


def load_reference_rows(reference_json: Path) -> list[dict[str, object]]:
    if not reference_json.exists():
        return []

    import json

    payload = json.loads(reference_json.read_text(encoding="utf-8"))
    reference_specs = [
        ("HipHop_l2_n4", 2, 4, "l2n4_ref", rf"$\ell=2, n=4$ ref"),
        ("HipHop", 3, 4, "l3n4_ref", rf"$\ell=3, n=4$ ref"),
    ]

    rows: list[dict[str, object]] = []
    for series_key, l_max, n_max, config_key, config_label in reference_specs:
        values = payload.get(series_key, {}).get("100000", [])
        for idx, value in enumerate(values):
            rows.append(
                {
                    "hiphop_l_max": l_max,
                    "hiphop_n_max": n_max,
                    "config": config_key,
                    "config_label": config_label,
                    "seed": f"ref_{idx}",
                    "data_size": 100000,
                    "best_valid_T-RMSE": float(value),
                    "best_epoch": -1,
                    "run": f"{series_key}_100000_rep{idx}",
                    "source_file": str(reference_json),
                    "metrics_csv": str(reference_json),
                    "all_runs": [f"{series_key}_100000_rep{idx}"],
                    "all_source_files": [str(reference_json)],
                    "all_metrics_csvs": [str(reference_json)],
                    "metric_type": "reference_test",
                    "sort_key": (l_max, n_max, 1),
                }
            )
    return rows


def write_points_csv(rows: list[dict[str, object]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "hiphop_l_max",
        "hiphop_n_max",
        "config",
        "seed",
        "data_size",
        "metric_type",
        "best_valid_T-RMSE",
        "best_epoch",
        "run",
        "source_file",
        "metrics_csv",
        "all_runs",
        "all_source_files",
        "all_metrics_csvs",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["hiphop_l_max"], r["hiphop_n_max"], str(r["seed"]))):
            writer.writerow(
                {
                    "hiphop_l_max": row["hiphop_l_max"],
                    "hiphop_n_max": row["hiphop_n_max"],
                    "config": f"l{row['hiphop_l_max']}n{row['hiphop_n_max']}",
                    "seed": row["seed"],
                    "data_size": row["data_size"],
                    "metric_type": row.get("metric_type", "best_valid"),
                    "best_valid_T-RMSE": row["best_valid_T-RMSE"],
                    "best_epoch": row["best_epoch"],
                    "run": row["run"],
                    "source_file": row["source_file"],
                    "metrics_csv": row["metrics_csv"],
                    "all_runs": ";".join(row.get("all_runs", [])),
                    "all_source_files": ";".join(row.get("all_source_files", [])),
                    "all_metrics_csvs": ";".join(row.get("all_metrics_csvs", [])),
                }
            )


def create_boxplot(rows: list[dict[str, object]], output: Path) -> None:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    config_meta: dict[str, tuple[str, tuple[int, int, int]]] = {}
    for row in rows:
        config = str(row["config"])
        grouped[config].append(row)
        config_meta[config] = (str(row["config_label"]), tuple(row["sort_key"]))

    configs = sorted(grouped, key=lambda key: config_meta[key][1])
    if not configs:
        raise RuntimeError("No 100k RMSE data found.")

    plot_data: list[list[float]] = []
    labels: list[str] = []
    summary_rows: list[str] = []
    for config in configs:
        entries = sorted(grouped[config], key=lambda r: str(r["seed"]))
        values = [float(entry["best_valid_T-RMSE"]) for entry in entries]
        plot_data.append(values)
        label_base = config_meta[config][0]
        labels.append(label_base + f"\n(N={len(values)})")
        metric_type = str(entries[0].get("metric_type", "best_valid"))
        summary_rows.append(
            f"{config} [{metric_type}]: "
            f"N={len(values)}, mean={np.mean(values):.5f}, std={np.std(values, ddof=1) if len(values) > 1 else 0.0:.5f}, "
            f"values={[round(v, 5) for v in values]}"
        )

    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
    bp = ax.boxplot(
        plot_data,
        widths=0.6,
        patch_artist=True,
        showmeans=False,
        showfliers=False,
        boxprops=dict(edgecolor="black", linewidth=1.2),
        whiskerprops=dict(color="black", linewidth=1.2),
        capprops=dict(color="black", linewidth=1.2),
        medianprops=dict(color="darkblue", linewidth=2),
    )

    colors = ["#f2c14e", "#56b4e9", "#e07a7a", "#25b99a", "#d99ab5"]
    for patch, color in zip(bp["boxes"], colors * len(bp["boxes"])):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    np.random.seed(42)
    for i, values in enumerate(plot_data, start=1):
        x = np.ones(len(values)) * i + np.random.normal(0, 0.04, size=len(values))
        ax.scatter(x, values, alpha=0.65, s=50, color="red", zorder=3, edgecolors="darkred", linewidth=0.5)

    ax.set_ylabel("RMSE (kcal/mol)")
    ax.set_title("100k RMSE Distribution\n(best validation + reference test groups)", fontsize=16, fontweight="bold")
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    print(f"Saved {output}")
    print("100k best-validation summary:")
    for line in summary_rows:
        print(f"  {line}")


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    rows = load_all_best_valid_100k()
    rows.extend(load_reference_rows(args.reference_json))
    rows = [
        row
        for row in rows
        if str(row["config"]) not in {"l4n3", "l4n4"}
    ]
    write_points_csv(rows, args.output_csv)
    print(f"Wrote {args.output_csv}")
    create_boxplot(rows, args.output)


if __name__ == "__main__":
    main()
