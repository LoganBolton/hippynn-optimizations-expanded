#!/usr/bin/env python3
"""Calculate and persist energy statistics for deterministic methane test slices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import ase.io
import numpy as np


HARTREE_TO_KCAL_PER_MOL = 627.5096080305927
ENERGY_SHIFT_KCAL_PER_MOL = -25042.327220945674


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("datasets/methane.extxyz"))
    parser.add_argument("--training-size", type=int, action="append", required=True)
    parser.add_argument("--test-size", type=int, default=80_000)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("examples/methane_sweep/data_statistics")
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for training_size in sorted(set(args.training_size)):
        output = args.output_dir / f"test_energy_d{training_size}_n{args.test_size}_sequential.json"
        if output.exists():
            print(f"Already stored: {output}")
            continue
        energies = np.fromiter(
            (
                frame.get_total_energy() * HARTREE_TO_KCAL_PER_MOL - ENERGY_SHIFT_KCAL_PER_MOL
                for frame in ase.io.iread(
                    args.data, index=f"{training_size}:{training_size + args.test_size}"
                )
            ),
            dtype=np.float64,
            count=args.test_size,
        )
        if energies.size != args.test_size:
            raise RuntimeError(f"Expected {args.test_size} test energies; read {energies.size}")
        statistics = {
            "count": int(energies.size),
            "data_source": str(args.data.resolve()),
            "energy_mean_kcal_per_mol": float(energies.mean()),
            "energy_shift_kcal_per_mol": ENERGY_SHIFT_KCAL_PER_MOL,
            "energy_std_kcal_per_mol": float(energies.std(ddof=0)),
            "energy_units": "kcal/mol",
            "seed": None,
            "selection": "sequential",
            "std_ddof": 0,
            "test_size": args.test_size,
            "training_size": training_size,
        }
        temporary = output.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(statistics, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(output)
        print(f"Stored {output}: STD={statistics['energy_std_kcal_per_mol']:.12g} kcal/mol")


if __name__ == "__main__":
    main()
