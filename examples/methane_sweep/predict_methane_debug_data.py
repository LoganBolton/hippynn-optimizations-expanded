#!/usr/bin/env python3
"""Run a trained methane HIP-HOP model on saved debug geometries."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

import hippynn
from hippynn.graphs import Predictor
from hippynn.graphs import inputs, targets, physics
from hippynn.graphs.nodes.networks import HipHopnn


DEFAULT_MODEL_DIR = Path(
    "examples/TEST_METHANE_MODEL_l4_n3_d100000_seed42-b256_fresh"
)
DEFAULT_DATA_DIR = DEFAULT_MODEL_DIR / "validation_debug/test_data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint path. Defaults to model-dir/best_checkpoint.pt.")
    parser.add_argument("--output-dir", type=Path, help="Output directory. Defaults to model-dir/validation_debug/predictions.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--l-max", type=int, default=4)
    parser.add_argument("--n-max", type=int, default=3)
    return parser.parse_args()


def build_predictor(l_max: int, n_max: int, device: torch.device) -> Predictor:
    species = inputs.SpeciesNode(name="species", db_name="numbers")
    positions = inputs.PositionsNode(name="positions", db_name="positions")
    network = HipHopnn(
        "network",
        (species, positions),
        module_kwargs={
            "possible_species": [0, 1, 6],
            "n_features": 32,
            "n_sensitivities": 20,
            "dist_soft_min": 0.4,
            "dist_soft_max": 9.0,
            "dist_hard_max": 10.3,
            "n_interaction_layers": 1,
            "n_atom_layers": 3,
            "l_max": l_max,
            "n_max": n_max,
        },
    )
    henergy = targets.HEnergyNode("HEnergy", network, db_name="energy", first_is_interacting=True)
    force = physics.GradientNode("forces", (henergy, positions), sign=-1, db_name="forces")
    return Predictor(
        [species, positions],
        [henergy, force],
        model_device=device,
        return_device=torch.device("cpu"),
        requires_grad=False,
        name="Methane debug predictor",
    )


def load_state(predictor: Predictor, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    missing, unexpected = predictor.graph.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(
            "Checkpoint did not match predictor graph:\n"
            f"missing={list(missing)}\n"
            f"unexpected={list(unexpected)}"
        )
    if unexpected:
        print(f"Ignoring {len(unexpected)} checkpoint tensors unused by the predictor graph.")
    predictor.graph.eval()


def predict_file(
    predictor: Predictor,
    path: Path,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
) -> None:
    positions = np.load(path).astype(np.float32)
    if positions.ndim != 3 or positions.shape[1:] != (5, 3):
        raise ValueError(f"{path} must have shape (n_frames, 5, 3), got {positions.shape}")

    n_frames = positions.shape[0]
    numbers = np.array([6, 1, 1, 1, 1], dtype=np.int64)
    all_numbers = np.repeat(numbers[None, :], n_frames, axis=0)

    energies = []
    forces = []
    for start in range(0, n_frames, batch_size):
        stop = min(start + batch_size, n_frames)
        batch_numbers = torch.as_tensor(all_numbers[start:stop], dtype=torch.long, device=device)
        batch_positions = torch.as_tensor(positions[start:stop], dtype=torch.float32, device=device)
        with torch.enable_grad():
            output = predictor(numbers=batch_numbers, positions=batch_positions)
        energies.append(output["energy"].detach().cpu().numpy())
        forces.append(output["forces"].detach().cpu().numpy())

    pred_energy = np.concatenate(energies, axis=0).reshape(n_frames)
    pred_forces = np.concatenate(forces, axis=0).reshape(n_frames, 5, 3)

    np.savez_compressed(
        output_dir / f"{path.stem}_predictions.npz",
        positions=positions,
        numbers=all_numbers,
        pred_energy=pred_energy,
        pred_forces=pred_forces,
    )

    with (output_dir / f"{path.stem}_energies.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "pred_energy"])
        for frame, energy in enumerate(pred_energy):
            writer.writerow([frame, f"{energy:.12g}"])

    with (output_dir / f"{path.stem}_forces.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "atom", "Z", "x", "y", "z", "pred_fx", "pred_fy", "pred_fz"])
        for frame in range(n_frames):
            for atom in range(5):
                writer.writerow(
                    [
                        frame,
                        atom,
                        int(numbers[atom]),
                        f"{positions[frame, atom, 0]:.12g}",
                        f"{positions[frame, atom, 1]:.12g}",
                        f"{positions[frame, atom, 2]:.12g}",
                        f"{pred_forces[frame, atom, 0]:.12g}",
                        f"{pred_forces[frame, atom, 1]:.12g}",
                        f"{pred_forces[frame, atom, 2]:.12g}",
                    ]
                )

    print(f"{path.name}: wrote {n_frames} predictions")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint_path = args.checkpoint or (args.model_dir / "best_checkpoint.pt")
    output_dir = args.output_dir or (args.model_dir / "validation_debug/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)

    predictor = build_predictor(args.l_max, args.n_max, device)
    load_state(predictor, checkpoint_path, device)

    files = sorted(args.data_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files found in {args.data_dir}")
    for path in files:
        predict_file(predictor, path, output_dir, device, args.batch_size)
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
