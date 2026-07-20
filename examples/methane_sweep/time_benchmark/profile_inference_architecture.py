#!/usr/bin/env python3
"""Profile the same inference workload used by the energy/force benchmark."""

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
NICK_DIR = SCRIPT_DIR / "nick_speed_comparison"
sys.path.insert(0, str(NICK_DIR))

import hippynn  # noqa: E402
from hippynn.graphs.gops import search_by_name  # noqa: E402
from training_script import load_db, make_model  # noqa: E402


def register_ranges(graph):
    """Add graph-node and nested neural-module labels to a profiler trace."""
    handles = []
    active = collections.defaultdict(list)

    def add_hooks(module, label):
        def enter(current_module, inputs, range_label=label):
            context = torch.profiler.record_function(range_label)
            context.__enter__()
            active[current_module].append(context)

        def exit_range(current_module, inputs, output):
            active[current_module].pop().__exit__(None, None, None)

        handles.append(module.register_forward_pre_hook(enter))
        handles.append(module.register_forward_hook(exit_range, always_call=True))

    if all(hasattr(graph, attr) for attr in ("forward_output_list", "names_dict", "moddict")):
        for node in graph.forward_output_list:
            module_key = graph.names_dict[node]
            if module_key in graph.moddict:
                module = graph.moddict[module_key]
                add_hooks(module, f"model::{node.name} [{type(module).__name__}]")

    # Graph hooks identify pair construction, energy readout, and forces. Nested
    # module hooks separate interaction blocks, atom-local layers, normalization,
    # sensitivity functions, and dense transforms inside HIP-HOP.
    for name, module in graph.named_modules():
        if not name or module is graph or module in graph.moddict.values():
            continue
        if len(list(module.children())) == 0 or type(module).__name__ in {
            "HOPInteractionLayer",
            "ResNetWrapper",
        }:
            add_hooks(module, f"module::{name} [{type(module).__name__}]")
    return handles


def profile_predictor(predictor, db, mode, output_dir, batch_size, warmups):
    output_dir.mkdir(parents=True, exist_ok=True)
    predictor.graph.eval()

    def apply_once():
        result = predictor.apply_to_database(db, batch_size=batch_size)
        del result

    print(f"Warming up {mode} for {warmups} full test-set passes")
    for _ in range(warmups):
        apply_once()
    torch.cuda.synchronize()

    handles = register_ranges(predictor.graph)
    trace_path = output_dir / "profile_trace.json"
    print(f"Profiling one full {mode} test-set pass")
    start = time.perf_counter()
    try:
        with torch.profiler.profile(
            activities=(torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_flops=False,
            with_modules=True,
        ) as profiler:
            with torch.profiler.record_function(f"inference::{mode}"):
                apply_once()
        torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()
    elapsed = time.perf_counter() - start
    profiler.export_chrome_trace(str(trace_path))
    print(f"Saved {trace_path} ({elapsed:.3f} s profiled wall time)")
    return {"trace": str(trace_path), "profiled_wall_seconds": elapsed}


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.set_device(args.gpu)
    torch.set_default_dtype(torch.float32)
    hippynn.settings.WARN_LOW_DISTANCES = False

    network_parameters = {
        "possible_species": [0, 1, 6, 7, 8],
        "n_features": 128,
        "n_sensitivities": 20,
        "dist_soft_min": 0.75,
        "dist_soft_max": 5.5,
        "dist_hard_max": 6.5,
        "n_interaction_layers": 2,
        "n_atom_layers": 3,
    }
    en_name = "wb97x_dz.energy"
    force_name = "wb97x_dz.forces"
    db_info = {"inputs": ["atomic_numbers", "coordinates"], "targets": [en_name, force_name]}
    db = load_db(db_info, en_name, force_name, args.seed, args.anidata_location, 0, True)
    db.send_to_device(f"cuda:{args.gpu}")
    del db.splits["valid"]
    del db.splits["train"]

    species_array = db.splits["test"]["atomic_numbers"]
    n_structures = len(species_array)
    n_atoms = int((species_array > 0).sum().item())
    print(f"Fixed test split: {n_structures} structures, {n_atoms} atoms")

    henergy, force = make_model(
        network_parameters,
        tensor_model="HOP",
        tensor_order=args.l_max,
        tensor_factors=args.n_max,
        atomization_consistent=False,
        group_norm=True,
    )
    species = search_by_name([henergy], "atomic_numbers")
    coordinates = search_by_name([henergy], "coordinates")

    output_root = Path(args.output_dir).resolve()
    metadata = {
        "l_max": args.l_max,
        "n_max": args.n_max,
        "batch_size": args.batch_size,
        "warmups": args.warmups,
        "n_structures": n_structures,
        "n_atoms": n_atoms,
        "device_name": torch.cuda.get_device_name(args.gpu),
        "network_parameters": network_parameters,
        "profiles": {},
    }

    modes = ("energy", "energy_forces") if args.mode == "both" else (args.mode,)
    for mode in modes:
        outputs = [henergy.main_output] if mode == "energy" else [henergy.main_output, force]
        predictor = hippynn.Predictor(
            [species, coordinates],
            outputs,
            model_device=f"cuda:{args.gpu}",
            return_device=f"cuda:{args.gpu}",
        )
        mode_dir = output_root / mode
        metadata["profiles"][mode] = profile_predictor(
            predictor, db, mode, mode_dir, args.batch_size, args.warmups
        )

    metadata_path = output_root / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anidata_location", required=True)
    parser.add_argument("--output_dir", default=SCRIPT_DIR / "inference_profile_l4_n4")
    parser.add_argument("--l_max", type=int, default=4)
    parser.add_argument("--n_max", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--mode", choices=("energy", "energy_forces", "both"), default="both")
    main(parser.parse_args())
