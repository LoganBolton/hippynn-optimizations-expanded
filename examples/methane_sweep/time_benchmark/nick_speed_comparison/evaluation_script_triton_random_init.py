"""
Random-initialized timing using Nick's database/predictor benchmark path.

This intentionally follows evaluation_script.py: build a model, keep only the
ANI-1ccx test split, and time Predictor.apply_to_database. It does not train or
load model weights.
"""

import argparse
import json
import time
from pathlib import Path

import hippynn
import torch

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable

from hippynn.graphs.gops import search_by_name
from training_script import load_db, make_model


def parse_config(spec):
    parts = spec.upper().split(":")
    model = parts[0]
    if model == "NONE" and len(parts) == 1:
        return dict(tensor_model="NONE", tensor_order=0, tensor_factors=0)
    if model == "TS" and len(parts) == 2:
        return dict(tensor_model="TS", tensor_order=int(parts[1]), tensor_factors=0)
    if model == "HOP" and len(parts) == 3:
        return dict(tensor_model="HOP", tensor_order=int(parts[1]), tensor_factors=int(parts[2]))
    raise ValueError(
        f"Invalid config {spec!r}; expected NONE, TS:l_max, or HOP:l_max:n_max."
    )


def metric_key(config, batch_size):
    key = list(sorted(config.items(), key=lambda kv: kv[0]))
    return (*key, ("batch_size", batch_size))


def jsonify_metrics(metrics):
    return {repr(key): value for key, value in metrics.items()}


def configure_kernels(kernel_mode, use_triton_message_passing):
    hippynn.settings.WARN_LOW_DISTANCES = False
    hippynn.settings.PROGRESS = None
    if kernel_mode == "upstream":
        hippynn.settings.USE_POLYNOMIAL_INVARIANTS = False
        hippynn.settings.USE_TENSOR_MESSAGE_PASSING = False
        active_kernel = hippynn.custom_kernels.set_custom_kernels("auto")
        triton_version = None
        triton_file = None
        if active_kernel == "triton":
            import triton

            triton_version = triton.__version__
            triton_file = triton.__file__
        return {
            "kernel_mode": kernel_mode,
            "custom_kernels": active_kernel,
            "triton_version": triton_version,
            "triton_file": triton_file,
            "use_polynomial_invariants": hippynn.settings.USE_POLYNOMIAL_INVARIANTS,
            "use_tensor_message_passing": hippynn.settings.USE_TENSOR_MESSAGE_PASSING,
        }

    if kernel_mode == "pytorch":
        hippynn.settings.USE_POLYNOMIAL_INVARIANTS = False
        hippynn.settings.USE_TENSOR_MESSAGE_PASSING = False
        active_kernel = hippynn.custom_kernels.set_custom_kernels(False)
        if active_kernel != "pytorch" or hippynn.custom_kernels.kernel_active != "pytorch":
            raise RuntimeError("Failed to select the pure PyTorch HIP-NN kernel paths.")
        return {
            "kernel_mode": kernel_mode,
            "custom_kernels": active_kernel,
            "triton_version": None,
            "triton_file": None,
            "use_polynomial_invariants": hippynn.settings.USE_POLYNOMIAL_INVARIANTS,
            "use_tensor_message_passing": hippynn.settings.USE_TENSOR_MESSAGE_PASSING,
        }

    if kernel_mode in ("triton", "triton_legacy"):
        hippynn.settings.USE_POLYNOMIAL_INVARIANTS = True
        hippynn.settings.USE_TENSOR_MESSAGE_PASSING = use_triton_message_passing
        active_kernel = hippynn.custom_kernels.set_custom_kernels("triton")
        if active_kernel != "triton" or hippynn.custom_kernels.kernel_active != "triton":
            raise RuntimeError(f"Expected Triton custom kernels, got {active_kernel!r}.")

        import triton

        if kernel_mode == "triton_legacy":
            import hippynn.layers.hiplayers.invariants as invariants_module
            from hippynn.custom_kernels.poly_triton import _EvaluatePolynomialsLegacy

            # HopInvariantLayer resolves this module global at call time. This
            # keeps the model, data, and all non-polynomial kernels identical.
            invariants_module.EvaluatePolynomials = _EvaluatePolynomialsLegacy

        return {
            "kernel_mode": kernel_mode,
            "custom_kernels": active_kernel,
            "triton_version": triton.__version__,
            "triton_file": triton.__file__,
            "use_polynomial_invariants": hippynn.settings.USE_POLYNOMIAL_INVARIANTS,
            "use_tensor_message_passing": hippynn.settings.USE_TENSOR_MESSAGE_PASSING,
        }

    raise ValueError(f"Unknown kernel mode: {kernel_mode}")


def main(args):
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float32)
    torch.cuda.set_device(args.gpu)

    kernel_info = configure_kernels(args.kernel_mode, args.use_triton_message_passing)

    config_list = [parse_config(spec) for spec in args.configs]
    print("all configs", config_list)
    print("kernel info", kernel_info)

    network_parameters = {
        "possible_species": [0, 1, 6, 7, 8],
        "n_features": args.n_features,
        "n_sensitivities": args.n_sensitivities,
        "dist_soft_min": args.lower_cutoff,
        "dist_soft_max": args.cutoff_distance - 1,
        "dist_hard_max": args.cutoff_distance,
        "n_interaction_layers": args.n_interactions,
        "n_atom_layers": args.n_atom_layers,
    }

    en_name = "wb97x_dz.energy"
    force_name = "wb97x_dz.forces"
    db_info = dict(inputs=["atomic_numbers", "coordinates"], targets=[en_name, force_name])
    db = load_db(
        db_info,
        en_name,
        force_name,
        seed=args.split_seed,
        anidata_location=args.anidata_location,
        n_workers=args.n_workers,
        use_ccx_subset=True,
    )
    db.send_to_device(f"cuda:{args.gpu}")
    del db.splits["valid"]
    del db.splits["train"]

    n_configs = len(db.splits["test"]["coordinates"])
    n_atoms = int((db.splits["test"]["atomic_numbers"] > 0).sum().item())
    print(f"test configs: {n_configs}")
    print(f"test atoms: {n_atoms}")

    all_times = {}
    parameter_counts = {}

    for config in tqdm(config_list, desc="configs"):
        henergy, force = make_model(
            network_parameters.copy(),
            **config,
            atomization_consistent=False,
            group_norm=args.group_norm,
        )
        species = search_by_name([henergy], "atomic_numbers")
        coords = search_by_name([henergy], "coordinates")

        outputs = [henergy.main_output, force] if args.include_forces else [henergy.main_output]
        predictor = hippynn.Predictor(
            [species, coords],
            outputs,
            model_device=f"cuda:{args.gpu}",
            return_device=f"cuda:{args.gpu}",
        )
        parameter_count = sum(p.numel() for p in predictor.graph.parameters())
        parameter_counts[repr(tuple(sorted(config.items())))] = parameter_count

        print("Configuration:", config, "parameters:", parameter_count)
        for batch_size in tqdm(args.batch_sizes, desc="batch_sizes"):
            timing_list = []
            for rep in tqdm(range(args.reps + args.warmups), desc="reps"):
                torch.cuda.synchronize()
                time_start = time.time()
                predictor.apply_to_database(db, batch_size=batch_size)
                torch.cuda.synchronize()
                this_time = time.time() - time_start
                if rep < args.warmups:
                    continue
                timing_list.append(this_time)
            all_times[metric_key(config, batch_size)] = timing_list

    info = dict(
        metrics=all_times,
        metrics_jsonable=jsonify_metrics(all_times),
        parameter_counts=parameter_counts,
        device_name=torch.cuda.get_device_name(args.gpu),
        n_configs=n_configs,
        n_atoms=n_atoms,
        batch_sizes=args.batch_sizes,
        reps=args.reps,
        warmups=args.warmups,
        seed=args.seed,
        split_seed=args.split_seed,
        include_forces=args.include_forces,
        network_parameters=network_parameters,
        kernel_info=kernel_info,
    )

    print(info)
    torch.save(info, args.output)
    with open(args.output_json, "w") as handle:
        json.dump({**info, "metrics": jsonify_metrics(all_times)}, handle, indent=2)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    results_dir = Path(__file__).resolve().parent.parent / "results/nick"
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--anidata_location", type=str, required=True)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "HOP:3:4",
            "HOP:3:3",
            "HOP:3:2",
            "HOP:2:4",
            "HOP:2:3",
            "HOP:2:2",
            "HOP:1:2",
            "TS:1",
            "TS:2",
            "NONE",
        ],
        help="Config specs as NONE, TS:l_max, or HOP:l_max:n_max.",
    )
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[2048])
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--n_interactions", type=int, default=2)
    parser.add_argument("--n_atom_layers", type=int, default=3)
    parser.add_argument("--n_features", type=int, default=128)
    parser.add_argument("--n_sensitivities", type=int, default=20)
    parser.add_argument("--cutoff_distance", type=float, default=6.5)
    parser.add_argument("--lower_cutoff", type=float, default=0.75)
    parser.add_argument("--n_workers", type=int, default=0)
    parser.add_argument("--group_norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include_forces", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--kernel_mode",
        choices=["upstream", "pytorch", "triton", "triton_legacy"],
        default="upstream",
    )
    parser.add_argument("--use_triton_message_passing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=str, default=results_dir / "triton_random_init_speed_eval.pt")
    parser.add_argument("--output_json", type=str, default=results_dir / "triton_random_init_speed_eval.json")
    main(parser.parse_args())
