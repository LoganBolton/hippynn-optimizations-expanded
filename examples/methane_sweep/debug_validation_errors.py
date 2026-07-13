#!/usr/bin/env python3
"""Dump per-validation methane prediction errors for saved checkpoints."""

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import ase.io
import numpy as np
import torch

import hippynn
from hippynn.custom_kernels import CustomKernelError
from hippynn.experiment import assemble_for_training, setup_training
from hippynn.experiment.controllers import PatienceController, RaiseBatchSizeOnPlateau
from hippynn.graphs import inputs, loss, physics, targets
from hippynn.graphs.nodes.networks import HipHopnn
from hippynn.pretraining import set_e0_values


ENERGY_MEAN = -25042.327220945674
FORCE_CONVERSION = 51.42208619083232 * 23.060541945329334
ENERGY_CONVERSION = 627.5096080305927


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hiphop_l_max", type=int, default=4)
    parser.add_argument("--hiphop_n_max", type=int, default=3)
    parser.add_argument("--data_size", type=int, default=100_000)
    parser.add_argument("--test_set_size", type=int, default=80_000)
    parser.add_argument("--eval_batch_size", type=int, default=1024)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--epochs", nargs="+", type=int, default=[2180, 2190, 2200, 2210, 2220])
    parser.add_argument("--run_dir", type=Path)
    parser.add_argument("--data_src", type=Path)
    parser.add_argument("--out_dir", type=Path)
    return parser.parse_args()


def require_triton_if_cuda(device):
    if device.type != "cuda":
        return
    try:
        import triton  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError("CUDA evaluation for this model needs triton installed.") from exc
    try:
        active_kernel = hippynn.custom_kernels.set_custom_kernels("triton")
    except CustomKernelError as exc:
        raise RuntimeError("Could not activate hippynn Triton custom kernels.") from exc
    print(f"Using hippynn custom kernels: {active_kernel}", flush=True)


def prepare_data(data_src, train_size, test_size):
    train_dict = {"numbers": [], "positions": [], "forces": [], "energy": []}
    test_dict = {"numbers": [], "positions": [], "forces": [], "energy": []}
    for idx, frame in enumerate(ase.io.iread(data_src)):
        target = train_dict if idx < train_size else test_dict
        if idx >= train_size + test_size:
            break
        target["numbers"].append(frame.get_atomic_numbers())
        target["positions"].append(frame.get_positions())
        target["forces"].append(frame.get_forces() * FORCE_CONVERSION)
        target["energy"].append(frame.get_total_energy() * ENERGY_CONVERSION - ENERGY_MEAN)

    for target in (train_dict, test_dict):
        target["numbers"] = np.asarray(target["numbers"], dtype=np.int64)
        target["positions"] = np.asarray(target["positions"], dtype=np.float32)
        target["forces"] = np.asarray(target["forces"], dtype=np.float32)
        target["energy"] = np.asarray(target["energy"], dtype=np.float32)
    return train_dict, test_dict


def build_model(args, device):
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
            "l_max": args.hiphop_l_max,
            "n_max": args.hiphop_n_max,
        },
    )
    henergy = targets.HEnergyNode("HEnergy", network, db_name="energy", first_is_interacting=True)
    force = physics.GradientNode("forces", (henergy, positions), sign=-1, db_name="forces")

    rmse_force = loss.MSELoss.of_node(force) ** (1 / 2)
    mae_force = loss.MAELoss.of_node(force)
    rmse_energy = loss.MSELoss.of_node(henergy) ** (1 / 2)
    mae_energy = loss.MAELoss.of_node(henergy)
    total_loss = rmse_energy + mae_energy + rmse_force + mae_force + 1e-6 * loss.l2reg(network)
    validation_losses = {
        "T-RMSE": rmse_energy,
        "T-MAE": mae_energy,
        "F-RMSE": rmse_force,
        "F-MAE": mae_force,
        "Loss": total_loss,
    }
    training_modules, db_info = assemble_for_training(total_loss, validation_losses)
    optimizer = torch.optim.Adam(training_modules.model.parameters(), lr=2.5e-3)
    controller = PatienceController(
        optimizer=optimizer,
        scheduler=RaiseBatchSizeOnPlateau(optimizer=optimizer, max_batch_size=4096, patience=150, factor=0.5),
        batch_size=256,
        eval_batch_size=args.eval_batch_size,
        max_epochs=10_000,
        stopping_key="T-MAE",
        termination_patience=300,
        fraction_train_eval=1,
    )
    training_modules, controller, _ = setup_training(
        training_modules=training_modules,
        setup_params=hippynn.experiment.SetupParams(controller=controller, device=device),
    )
    return training_modules.model, henergy, db_info


def make_train_database(train_dict, db_info, seed):
    train_database = hippynn.databases.Database(
        arr_dict=train_dict,
        seed=seed,
        pin_memory=True,
        test_size=0.1,
        valid_size=0.1,
        quiet=True,
        **db_info,
    )
    return train_database


def evaluate_checkpoint(model, checkpoint_path, valid, device, batch_size):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    pred_energy_chunks = []
    pred_force_chunks = []

    numbers = torch.as_tensor(valid["numbers"], dtype=torch.long, device=device)
    positions_all = torch.as_tensor(valid["positions"], dtype=torch.float32, device=device)
    for start in range(0, len(numbers), batch_size):
        stop = min(start + batch_size, len(numbers))
        positions = positions_all[start:stop].detach().clone().requires_grad_(True)
        with torch.enable_grad():
            outputs = model(numbers[start:stop], positions)
        pred_force_chunks.append(outputs[1].detach().cpu().numpy())
        pred_energy_chunks.append(outputs[2].detach().cpu().numpy().reshape(-1))

    pred_energy = np.concatenate(pred_energy_chunks)
    pred_forces = np.concatenate(pred_force_chunks)
    force_error = pred_forces - valid["forces"]
    energy_error = pred_energy - valid["energy"]
    per_config_force_rmse = np.sqrt(np.mean(force_error**2, axis=(1, 2)))
    per_config_force_mae = np.mean(np.abs(force_error), axis=(1, 2))
    return {
        "pred_energy": pred_energy,
        "pred_forces": pred_forces,
        "energy_error": energy_error,
        "force_error": force_error,
        "per_config_force_rmse": per_config_force_rmse,
        "per_config_force_mae": per_config_force_mae,
        "force_rmse": float(np.sqrt(np.mean(force_error**2))),
        "force_mae": float(np.mean(np.abs(force_error))),
        "energy_rmse": float(np.sqrt(np.mean(energy_error**2))),
        "energy_mae": float(np.mean(np.abs(energy_error))),
    }


def write_ranked_csv(path, valid, result, epoch):
    order = np.argsort(result["per_config_force_rmse"])[::-1]
    with path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "rank",
                "epoch",
                "source_index",
                "valid_row",
                "force_rmse",
                "force_mae",
                "energy_target",
                "energy_pred",
                "energy_error",
                "max_abs_force_component_error",
            ]
        )
        for rank, row in enumerate(order, start=1):
            writer.writerow(
                [
                    rank,
                    epoch,
                    int(valid["source_indices"][row]),
                    int(row),
                    float(result["per_config_force_rmse"][row]),
                    float(result["per_config_force_mae"][row]),
                    float(valid["energy"][row]),
                    float(result["pred_energy"][row]),
                    float(result["energy_error"][row]),
                    float(np.max(np.abs(result["force_error"][row]))),
                ]
            )


def main():
    args = parse_args()
    repo_root = Path(__file__).parents[2]
    data_src = args.data_src or repo_root / "datasets" / "methane.extxyz"
    run_dir = args.run_dir or repo_root / "examples" / (
        f"TEST_METHANE_MODEL_l{args.hiphop_l_max}_n{args.hiphop_n_max}_d{args.data_size}_seed{args.seed}-b256_fresh"
    )
    out_dir = args.out_dir or run_dir / "validation_debug"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    require_triton_if_cuda(device)
    torch.random.manual_seed(args.seed)

    print(f"Loading data from {data_src}", flush=True)
    train_dict, _ = prepare_data(data_src, args.data_size, args.test_set_size)
    print("Building model and deterministic validation split", flush=True)
    model, henergy, db_info = build_model(args, device)
    train_database = make_train_database(train_dict, db_info, args.seed)
    set_e0_values(henergy, train_database, trainable_after=False)
    valid_indices = train_database.splits["valid"]["indices"].cpu().numpy()
    valid = {
        "source_indices": valid_indices,
        "numbers": train_dict["numbers"][valid_indices],
        "positions": train_dict["positions"][valid_indices],
        "forces": train_dict["forces"][valid_indices],
        "energy": train_dict["energy"][valid_indices],
    }
    np.savez_compressed(out_dir / "validation_targets.npz", **valid)

    summary_rows = []
    for epoch in args.epochs:
        checkpoint_path = run_dir / f"checkpoint_epoch_{epoch}.pt"
        if not checkpoint_path.exists():
            print(f"Skipping missing checkpoint {checkpoint_path}", flush=True)
            continue
        print(f"Evaluating {checkpoint_path.name}", flush=True)
        result = evaluate_checkpoint(model, checkpoint_path, valid, device, args.eval_batch_size)
        npz_path = out_dir / f"validation_errors_epoch_{epoch}.npz"
        csv_path = out_dir / f"validation_errors_epoch_{epoch}_ranked.csv"
        np.savez_compressed(npz_path, **valid, **result)
        write_ranked_csv(csv_path, valid, result, epoch)
        summary_rows.append(
            {
                "epoch": epoch,
                "force_rmse": result["force_rmse"],
                "force_mae": result["force_mae"],
                "energy_rmse": result["energy_rmse"],
                "energy_mae": result["energy_mae"],
                "npz": npz_path.name,
                "ranked_csv": csv_path.name,
            }
        )
        print(
            f"epoch {epoch}: F-RMSE={result['force_rmse']:.6g} "
            f"F-MAE={result['force_mae']:.6g} T-RMSE={result['energy_rmse']:.6g}",
            flush=True,
        )

    with (out_dir / "summary.csv").open("w", newline="") as csv_file:
        fieldnames = ["epoch", "force_rmse", "force_mae", "energy_rmse", "energy_mae", "npz", "ranked_csv"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote validation debug outputs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
