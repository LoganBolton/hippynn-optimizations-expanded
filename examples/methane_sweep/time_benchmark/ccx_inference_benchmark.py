"""
Random-initialized HIP-HOP-NN inference benchmark for ANI-1ccx-style inputs.

This script does not train. It builds the requested architecture, loads a
deterministic test-like subset from ANI-1x, warms up the GPU, and times forward
passes for system energy prediction.
"""

import argparse
import csv
import fcntl
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

import hippynn
from hippynn.custom_kernels import CustomKernelError
from hippynn.graphs import GraphModule
from hippynn.graphs import inputs, networks, targets
from hippynn.layers.hiplayers.invariants import triton_available_with_gather

sys.path.append(str(Path(__file__).resolve().parent))
import pyanitools


def build_energy_model(args):
    species = inputs.SpeciesNode(db_name="atomic_numbers")
    positions = inputs.PositionsNode(db_name="coordinates")
    network_params = {
        "possible_species": [0, 1, 6, 7, 8],
        "n_features": args.n_features,
        "n_sensitivities": args.n_sensitivities,
        "dist_soft_min": args.lower_cutoff,
        "dist_soft_max": args.cutoff_distance - 1,
        "dist_hard_max": args.cutoff_distance,
        "n_interaction_layers": args.n_interactions,
        "n_atom_layers": args.n_atom_layers,
        "l_max": args.tensor_order,
        "n_max": args.tensor_factors,
    }
    network = networks.HipHopnn("hipnn_model", (species, positions), module_kwargs=network_params)
    henergy = targets.HEnergyNode("HEnergy", network)
    return GraphModule([species, positions], [henergy.main_output])


def load_ccx_subset(path, seed, test_fraction, max_configs):
    loader = pyanitools.anidataloader(path)
    species_blocks = []
    coord_blocks = []
    max_atoms = 0
    ccx_count = 0

    try:
        for group in loader:
            required = ("atomic_numbers", "coordinates", "ccsd(t)_cbs.energy", "wb97x_dz.energy")
            if not all(key in group for key in required):
                continue

            ccsd_energy = np.asarray(group["ccsd(t)_cbs.energy"])
            wb97x_energy = np.asarray(group["wb97x_dz.energy"])
            found = np.isfinite(ccsd_energy) & np.isfinite(wb97x_energy)
            if not np.any(found):
                continue

            atomic_numbers = np.asarray(group["atomic_numbers"], dtype=np.int64)
            coordinates = np.asarray(group["coordinates"], dtype=np.float32)[found]
            n_configs = coordinates.shape[0]
            n_atoms = atomic_numbers.shape[0]

            species_blocks.append(np.broadcast_to(atomic_numbers, (n_configs, n_atoms)).copy())
            coord_blocks.append(coordinates)
            max_atoms = max(max_atoms, n_atoms)
            ccx_count += n_configs
    finally:
        loader.cleanup()

    if ccx_count == 0:
        raise ValueError(f"No ccx-compatible configurations found in {path}")

    rng = np.random.default_rng(seed)
    n_test = max(1, int(round(ccx_count * test_fraction)))
    selected_flat = rng.permutation(ccx_count)[:n_test]
    if max_configs is not None:
        selected_flat = selected_flat[:max_configs]
    selected_flat = np.sort(selected_flat)

    selected_species = np.zeros((len(selected_flat), max_atoms), dtype=np.int64)
    selected_coordinates = np.zeros((len(selected_flat), max_atoms, 3), dtype=np.float32)

    block_start = 0
    out_start = 0
    for species_block, coord_block in zip(species_blocks, coord_blocks):
        block_stop = block_start + species_block.shape[0]
        local_mask = (selected_flat >= block_start) & (selected_flat < block_stop)
        if np.any(local_mask):
            local_indices = selected_flat[local_mask] - block_start
            out_stop = out_start + len(local_indices)
            n_atoms = species_block.shape[1]
            selected_species[out_start:out_stop, :n_atoms] = species_block[local_indices]
            selected_coordinates[out_start:out_stop, :n_atoms] = coord_block[local_indices]
            out_start = out_stop
        block_start = block_stop

    species = torch.as_tensor(selected_species)
    coordinates = torch.as_tensor(selected_coordinates)
    return species, coordinates, ccx_count, len(selected_flat)


def batch_slices(n_items, batch_size):
    for start in range(0, n_items, batch_size):
        yield slice(start, min(start + batch_size, n_items))


def run_pass(model, species, coordinates, batch_size):
    n_batches = 0
    n_configs = species.shape[0]
    with torch.inference_mode():
        for batch in batch_slices(n_configs, batch_size):
            model(species[batch], coordinates[batch])
            n_batches += 1
    return n_batches, n_configs


def timed_cuda_passes(model, species, coordinates, batch_size, passes, atoms_per_pass):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    start_event.record()
    total_batches = 0
    total_configs = 0
    for _ in range(passes):
        n_batches, n_configs = run_pass(model, species, coordinates, batch_size)
        total_batches += n_batches
        total_configs += n_configs
    end_event.record()
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - wall_start
    cuda_seconds = start_event.elapsed_time(end_event) / 1000.0
    return {
        "wall_seconds": wall_seconds,
        "cuda_seconds": cuda_seconds,
        "batches": total_batches,
        "configs": total_configs,
        "atoms": atoms_per_pass * passes,
    }


def append_csv(path, row):
    fieldnames = list(row)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    exists = output_path.exists()
    with open(output_path, "a", newline="") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        fcntl.flock(handle, fcntl.LOCK_UN)


def require_cuda_triton_polynomial_invariants():
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")

    try:
        import triton
    except ModuleNotFoundError as exc:
        raise RuntimeError("l_max=4 requires Triton polynomial invariants, but triton is not installed.") from exc

    try:
        active_kernel = hippynn.custom_kernels.set_custom_kernels("triton")
    except CustomKernelError as exc:
        raise RuntimeError("Triton custom kernels were requested, but hippynn could not activate them.") from exc

    if active_kernel != "triton":
        raise RuntimeError(f"Expected Triton custom kernels, but hippynn activated {active_kernel!r}.")

    if not hippynn.settings.USE_POLYNOMIAL_INVARIANTS:
        raise RuntimeError(
            "l_max=4 requires HIPPYNN_USE_POLYNOMIAL_INVARIANTS=True. "
            "The pure PyTorch invariant fallback has incomplete l_max=4 support."
        )

    if not triton_available_with_gather:
        raise RuntimeError(
            f"l_max=4 requires Triton polynomial invariants with gather support. "
            f"Current triton version is {triton.__version__}; this repo checks for triton >= 3.3.0."
        )

    return active_kernel


def main(args):
    if args.tensor_factors < 1:
        raise ValueError("--tensor_factors/--n_max must be >= 1")

    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float32)
    hippynn.settings.WARN_LOW_DISTANCES = False
    hippynn.settings.PROGRESS = None
    active_kernel = require_cuda_triton_polynomial_invariants()
    import triton
    hippynn.settings.USE_TENSOR_MESSAGE_PASSING = args.use_triton_message_passing

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This benchmark expects a CUDA GPU.")
    torch.cuda.set_device(device)

    print(f"Using Python model device: {device}")
    print(f"custom kernels: {active_kernel}")
    print(f"triton version: {triton.__version__}")
    print(f"triton file: {triton.__file__}")
    print(f"polynomial invariants: {hippynn.settings.USE_POLYNOMIAL_INVARIANTS}")
    print(f"triton polynomial gather: {triton_available_with_gather}")
    print(f"triton tensor message passing: {hippynn.settings.USE_TENSOR_MESSAGE_PASSING}")
    print(f"architecture: l_max={args.tensor_order}, n_max={args.tensor_factors}")

    model = build_energy_model(args).to(device).eval()
    parameter_count = sum(p.numel() for p in model.parameters())

    species, coordinates, ccx_count, selected_count = load_ccx_subset(
        args.anidata_location,
        seed=args.split_seed,
        test_fraction=args.test_fraction,
        max_configs=args.max_configs,
    )
    atoms_per_pass = int((species > 0).sum().item())
    species = species.to(device, non_blocking=True)
    coordinates = coordinates.to(device, non_blocking=True)
    torch.cuda.synchronize()

    print(f"ccx-compatible configs: {ccx_count}")
    print(f"timed configs: {selected_count}")
    print(f"batch size: {args.batch_size}")
    print(f"warmup passes: {args.warmup_passes}")
    print(f"timed passes: {args.timed_passes}")
    print(f"parameters: {parameter_count}")

    total_start = time.perf_counter()
    warmup = timed_cuda_passes(model, species, coordinates, args.batch_size, args.warmup_passes, atoms_per_pass)
    timed = timed_cuda_passes(model, species, coordinates, args.batch_size, args.timed_passes, atoms_per_pass)
    total_with_warmup_wall = time.perf_counter() - total_start
    total_with_warmup_cuda = warmup["cuda_seconds"] + timed["cuda_seconds"]

    seconds_per_pass = timed["cuda_seconds"] / args.timed_passes
    atoms_per_second = timed["atoms"] / timed["cuda_seconds"]
    configs_per_second = timed["configs"] / timed["cuda_seconds"]
    batches_per_pass = math.ceil(selected_count / args.batch_size)

    row = {
        "tag": args.tag,
        "l_max": args.tensor_order,
        "n_max": args.tensor_factors,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "batch_size": args.batch_size,
        "selected_configs": selected_count,
        "batches_per_pass": batches_per_pass,
        "parameter_count": parameter_count,
        "warmup_passes": args.warmup_passes,
        "timed_passes": args.timed_passes,
        "warmup_cuda_seconds": warmup["cuda_seconds"],
        "warmup_wall_seconds": warmup["wall_seconds"],
        "timed_cuda_seconds": timed["cuda_seconds"],
        "timed_wall_seconds": timed["wall_seconds"],
        "total_with_warmup_cuda_seconds": total_with_warmup_cuda,
        "total_with_warmup_wall_seconds": total_with_warmup_wall,
        "cuda_seconds_per_timed_pass": seconds_per_pass,
        "cuda_configs_per_second": configs_per_second,
        "cuda_atoms_per_second": atoms_per_second,
        "device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "custom_kernels": str(hippynn.custom_kernels.CUSTOM_KERNELS_ACTIVE),
    }

    print(json.dumps(row, indent=2))
    append_csv(args.output_csv, row)
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="random_init_inference")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42, help="random init seed")
    parser.add_argument("--split_seed", type=int, default=42, help="deterministic test subset seed")

    parser.add_argument("--n_interactions", type=int, default=2)
    parser.add_argument("--n_atom_layers", type=int, default=5)
    parser.add_argument("--n_features", type=int, default=256)
    parser.add_argument("--n_sensitivities", type=int, default=20)
    parser.add_argument("--cutoff_distance", type=float, default=6.5)
    parser.add_argument("--lower_cutoff", type=float, default=0.75)
    parser.add_argument("--tensor_order", type=int, default=2, help="l_max")
    parser.add_argument("--tensor_factors", type=int, default=4, help="n_max")

    parser.add_argument("--anidata_location", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--test_fraction", type=float, default=0.01)
    parser.add_argument("--max_configs", type=int, default=None)
    parser.add_argument("--warmup_passes", type=int, default=2)
    parser.add_argument("--timed_passes", type=int, default=5)
    parser.add_argument("--output_csv", type=str, default="random_init_inference_results.csv")
    parser.add_argument(
        "--use_triton_message_passing",
        action="store_true",
        default=False,
        help="Use Triton tensor message passing for HIP-HOP.",
    )
    main(parser.parse_args())
