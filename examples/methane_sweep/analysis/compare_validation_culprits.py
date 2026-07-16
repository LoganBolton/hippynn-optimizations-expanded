#!/usr/bin/env python3
"""Compare ranked per-configuration errors from two validation runs."""

import argparse
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    args = parse_args()
    reference = read_rows(args.reference)
    candidate = read_rows(args.candidate)
    reference_by_source = {int(row["source_index"]): row for row in reference}
    candidate_by_source = {int(row["source_index"]): row for row in candidate}
    reference_top = {int(row["source_index"]) for row in reference[: args.top_n]}
    candidate_top = {int(row["source_index"]) for row in candidate[: args.top_n]}
    overlap = sorted(reference_top & candidate_top)

    lines = [
        f"reference: {args.reference}",
        f"candidate: {args.candidate}",
        f"top_n: {args.top_n}",
        f"top-{args.top_n} overlap count: {len(overlap)}",
        f"top-{args.top_n} overlap sources: {overlap}",
        "",
        "Earlier three culprits in candidate validation ranking:",
    ]
    for source in (32801, 1136, 60963):
        old = reference_by_source.get(source)
        new = candidate_by_source.get(source)
        if new is None:
            lines.append(f"source {source}: not in candidate validation split")
            continue
        lines.append(
            f"source {source}: old rank={old['rank'] if old else 'n/a'}, "
            f"old RMSE={old['force_rmse'] if old else 'n/a'}; "
            f"new rank={new['rank']}, new RMSE={new['force_rmse']}"
        )

    lines.extend(["", f"Candidate top {min(20, len(candidate))}:"])
    for row in candidate[:20]:
        source = int(row["source_index"])
        old = reference_by_source.get(source)
        lines.append(
            f"rank {row['rank']}: source {source}, RMSE={row['force_rmse']}, "
            f"reference rank={old['rank'] if old else 'not in reference validation'}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
