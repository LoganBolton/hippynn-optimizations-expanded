#!/usr/bin/env python3
"""Combine L3N5/L3N4 sequential +80k and random-common test results."""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path


METHANE_DIR = Path(__file__).resolve().parents[1]
SEQUENTIAL_CSV = (
    METHANE_DIR
    / "results/sequential_heldout_test/l3n5_l3n4_1m_training_log_final_test_metrics.csv"
)
COMMON_L3N4_CSV = METHANE_DIR / "results/external_test/best_l3n4_l4n4_common_test.csv"
COMMON_L3N5_CSV = METHANE_DIR / "results/external_test/best_l3n5_l4n3_common_test.csv"
OUTPUT_DIR = METHANE_DIR / "results/test_set_comparison"
OUTPUT_CSV = OUTPUT_DIR / "l3n5_l3n4_1m_both_test_sets.csv"
OUTPUT_MD = OUTPUT_DIR / "l3n5_l3n4_1m_both_test_sets.md"
METRICS = ("T-RMSE", "T-MAE", "F-RMSE", "F-MAE")


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required input is not ready: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    output_rows: list[dict[str, str | int | float]] = []

    for row in read_rows(SEQUENTIAL_CSV):
        output_rows.append(
            {
                "architecture": row["architecture"],
                "hiphop_l_max": row["hiphop_l_max"],
                "hiphop_n_max": row["hiphop_n_max"],
                "data_size": row["data_size"],
                "seed": row["seed"],
                "test_set": "sequential_plus_80k",
                "test_definition": "frames [1000000,1080000)",
                "checkpoint_selection": "training-log final best model",
                **{f"test_{metric}": row[f"test_{metric}"] for metric in METRICS},
                "source": row["source_log"],
            }
        )

    for path, architecture in ((COMMON_L3N4_CSV, "L3N4"), (COMMON_L3N5_CSV, "L3N5")):
        expected_l, expected_n = (3, 4) if architecture == "L3N4" else (3, 5)
        for row in read_rows(path):
            if int(row["hiphop_l_max"]) != expected_l or int(row["hiphop_n_max"]) != expected_n:
                continue
            if row.get("status") != "ok":
                continue
            output_rows.append(
                {
                    "architecture": architecture,
                    "hiphop_l_max": expected_l,
                    "hiphop_n_max": expected_n,
                    "data_size": row["data_size"],
                    "seed": row["seed"],
                    "test_set": "reserved_random_common_80k",
                    "test_definition": row["test_indices_path"],
                    "checkpoint_selection": "best validation T-MAE",
                    **{f"test_{metric}": row[f"test_{metric}"] for metric in METRICS},
                    "source": row["checkpoint_path"],
                }
            )

    output_rows.sort(key=lambda row: (str(row["architecture"]), str(row["test_set"]), int(row["seed"])))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = list(output_rows[0])
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    grouped: dict[tuple[str, str], list[dict[str, str | int | float]]] = defaultdict(list)
    for row in output_rows:
        grouped[(str(row["architecture"]), str(row["test_set"]))].append(row)

    lines = [
        "# L3N5/L3N4 1M test-set comparison",
        "",
        "The sequential `+80k` and reserved random-common results are reported separately.",
        "Seed coverage is not identical. The sole sequential L3N4 row is an older batch-16384 run;",
        "the random-common L3N4 rows are newer batch-256 lineages, so architecture-level means are not paired estimates.",
        "",
        "| Architecture | Test set | Seeds | T-RMSE mean ± SD | T-MAE mean ± SD |",
        "|---|---|---:|---:|---:|",
    ]
    for (architecture, test_set), rows in sorted(grouped.items()):
        seeds = ", ".join(str(row["seed"]) for row in rows)
        values = {}
        for metric in ("T-RMSE", "T-MAE"):
            metric_values = [float(row[f"test_{metric}"]) for row in rows]
            std = statistics.stdev(metric_values) if len(metric_values) > 1 else 0.0
            values[metric] = f"{statistics.mean(metric_values):.6g} ± {std:.6g}"
        lines.append(
            f"| {architecture} | {test_set} | {seeds} | {values['T-RMSE']} | {values['T-MAE']} |"
        )
    lines.extend(("", f"Detailed rows: `{OUTPUT_CSV.name}`", ""))
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
