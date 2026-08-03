#!/usr/bin/env python3
"""Evaluate the current best checkpoint for unfinished methane sweep runs.

This script is meant for cases where some sweep tasks have already finished and
written final test metrics, while others are still training. It:

1. Reads ``sweep_task_map.csv`` and ``final_test_metrics.csv`` from a methane
   plotting directory.
2. Infers which sweep tasks do not yet have final test results.
3. Locates the corresponding run directories under ``examples/``.
4. Loads each run's current best checkpoint (``best_checkpoint.pt`` by default).
5. Rebuilds the held-out methane test set used by the training script.
6. Evaluates the checkpoint on the test split and writes a CSV summary.

Example
-------
python examples/methane_sweep/analysis/evaluate_current_best_checkpoints.py \
    --plot-dir examples/methane_sweep/logs/plots_l3n5_1m_all_current

Optional task restriction:
python examples/methane_sweep/analysis/evaluate_current_best_checkpoints.py \
    --plot-dir examples/methane_sweep/logs/plots_l3n5_1m_all_current \
    --task-ids 0 1 2 6 7
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import ase.io
import numpy as np
import torch

import hippynn
from hippynn.experiment import test_model
from hippynn.experiment.serialization import load_checkpoint


SCRIPT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PLOT_DIR = SCRIPT_ROOT / "examples" / "methane_sweep" / "logs" / "plots_l3n5_1m_all_current"
DEFAULT_DATASET_PATH = SCRIPT_ROOT / "datasets" / "methane.extxyz"
ENERGY_MEAN = -25042.327220945674
DEFAULT_TEST_SET_SIZE = 80_000
DEFAULT_OUTPUT_NAME = "current_best_checkpoint_test_metrics.csv"
RUN_DIR_RE = re.compile(r"TEST_METHANE_MODEL_l(\d+)_n(\d+)_d(\d+)_seed(\d+)")
LIGHTNING_BEST_RE = re.compile(r"best-epoch=(\d+)-valid_T-MAE=([0-9.eE+-]+)\.ckpt$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=DEFAULT_PLOT_DIR,
        help="Directory containing sweep_task_map.csv and final_test_metrics.csv",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=SCRIPT_ROOT,
        help="Repository/worktree root used to locate examples/ and datasets/",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="Override directory searched for TEST_METHANE_MODEL_* run folders. Defaults to <repo-root>/examples",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Override methane.extxyz path. Defaults to <repo-root>/datasets/methane.extxyz",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to <plot-dir>/current_best_checkpoint_test_metrics.csv",
    )
    parser.add_argument(
        "--task-ids",
        type=int,
        nargs="+",
        default=None,
        help="Specific sweep task ids to evaluate. If omitted, infer unfinished tasks.",
    )
    parser.add_argument(
        "--run-dirs",
        type=Path,
        nargs="+",
        default=None,
        help="Explicit TEST_METHANE_MODEL_* directories to evaluate instead of inferring sweep tasks.",
    )
    parser.add_argument(
        "--checkpoint-paths",
        type=Path,
        nargs="+",
        default=None,
        help="Explicit checkpoint for each --run-dirs entry, in the same order.",
    )
    parser.add_argument(
        "--checkpoint-target-epochs",
        type=int,
        nargs="+",
        default=None,
        help="Optional validation-selection epoch for each explicit checkpoint (recorded in the output).",
    )
    parser.add_argument(
        "--checkpoint-kind",
        choices=("best", "latest"),
        default="best",
        help="Which checkpoint to evaluate. 'best' uses best_checkpoint.pt, 'latest' prefers checkpoint_epoch_*.pt.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Override evaluation batch size. Defaults to the checkpoint controller value.",
    )
    parser.add_argument(
        "--test-set-size",
        type=int,
        default=DEFAULT_TEST_SET_SIZE,
        help="Held-out test set size used by methane_configurations.py",
    )
    parser.add_argument(
        "--test-indices",
        type=Path,
        default=None,
        help="Optional .npy file of absolute methane frame indices (for example the reserved common external test set).",
    )
    parser.add_argument(
        "--per-point-output-dir",
        type=Path,
        default=None,
        help="Write one CSV summary and compressed raw prediction/error NPZ per evaluated checkpoint.",
    )
    parser.add_argument(
        "--per-point-only",
        action="store_true",
        help="Only write per-point records and their directly recomputed aggregate RMSE/MAE values.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Evaluation device: auto, cpu, cuda, cuda:0, ...",
    )
    parser.add_argument(
        "--retry-attempts",
        type=int,
        default=5,
        help="Retries when a checkpoint is being rewritten while training.",
    )
    parser.add_argument(
        "--retry-sleep",
        type=float,
        default=5.0,
        help="Seconds to sleep between checkpoint load retries.",
    )
    parser.add_argument(
        "--strict-run-dir",
        action="store_true",
        help="Fail if multiple candidate run directories are found for a task instead of choosing the newest.",
    )
    return parser.parse_args()


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_arg)


def maybe_enable_triton() -> None:
    if not torch.cuda.is_available():
        return
    try:
        import triton  # noqa: F401
        active = hippynn.custom_kernels.set_custom_kernels("triton")
        print(f"Using hippynn custom kernels: {active}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: could not activate triton custom kernels: {exc}", flush=True)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def infer_target_tasks(plot_dir: Path, selected_task_ids: list[int] | None) -> list[dict[str, Any]]:
    sweep_map_path = plot_dir / "sweep_task_map.csv"
    final_test_path = plot_dir / "final_test_metrics.csv"

    if not sweep_map_path.exists():
        raise FileNotFoundError(f"Missing sweep task map: {sweep_map_path}")
    if not final_test_path.exists():
        raise FileNotFoundError(f"Missing final test metrics: {final_test_path}")

    sweep_rows = read_csv_rows(sweep_map_path)
    finished_rows = read_csv_rows(final_test_path)
    finished_task_ids = {int(row["sweep_task_id"]) for row in finished_rows}

    tasks = []
    for row in sweep_rows:
        task_id = int(row["sweep_task_id"])
        if selected_task_ids is None:
            if task_id in finished_task_ids:
                continue
        elif task_id not in selected_task_ids:
            continue

        tasks.append(
            {
                "sweep_task_id": task_id,
                "seed": int(row["seed"]),
                "hiphop_l_max": int(row["hiphop_l_max"]),
                "hiphop_n_max": int(row["hiphop_n_max"]),
                "data_size": int(row["data_size"]),
                "total_params": int(row["total_params"]) if row.get("total_params") else "",
                "source_file": row.get("source_file", ""),
            }
        )

    tasks.sort(key=lambda row: row["sweep_task_id"])
    return tasks


def tasks_from_run_dirs(
    run_dirs: list[Path],
    checkpoint_paths: list[Path] | None = None,
    checkpoint_target_epochs: list[int] | None = None,
) -> list[dict[str, Any]]:
    if checkpoint_paths is not None and len(checkpoint_paths) != len(run_dirs):
        raise ValueError("--checkpoint-paths must contain one path per --run-dirs entry")
    if checkpoint_target_epochs is not None and len(checkpoint_target_epochs) != len(run_dirs):
        raise ValueError("--checkpoint-target-epochs must contain one epoch per --run-dirs entry")
    tasks: list[dict[str, Any]] = []
    for index, raw_path in enumerate(run_dirs):
        path = raw_path.resolve()
        match = RUN_DIR_RE.search(path.name)
        if not match:
            raise ValueError(f"Cannot infer l/n/data-size/seed from run directory: {path}")
        l_max, n_max, data_size, seed = (int(value) for value in match.groups())
        task = {
                "sweep_task_id": index,
                "seed": seed,
                "hiphop_l_max": l_max,
                "hiphop_n_max": n_max,
                "data_size": data_size,
                "total_params": "",
                "source_file": "",
                "run_dir": path,
            }
        if checkpoint_paths is not None:
            task["checkpoint_path"] = checkpoint_paths[index].resolve()
        if checkpoint_target_epochs is not None:
            task["checkpoint_target_epoch"] = checkpoint_target_epochs[index]
        tasks.append(task)
    return tasks


def _checkpoint_score(path: Path) -> tuple[int, float]:
    stem = path.stem
    if stem.startswith("checkpoint_epoch_"):
        try:
            return int(stem.rsplit("_", 1)[-1]), path.stat().st_mtime
        except ValueError:
            pass
    return -1, path.stat().st_mtime


def latest_checkpoint(run_dir: Path) -> Path | None:
    epoch_checkpoints = sorted(run_dir.glob("checkpoint_epoch_*.pt"), key=_checkpoint_score)
    if epoch_checkpoints:
        return epoch_checkpoints[-1]
    best_checkpoint = run_dir / "best_checkpoint.pt"
    if best_checkpoint.exists():
        return best_checkpoint
    return None


def choose_checkpoint(run_dir: Path, checkpoint_kind: str) -> Path | None:
    best_checkpoint = run_dir / "best_checkpoint.pt"
    latest = latest_checkpoint(run_dir)

    lightning_checkpoints = list((run_dir / "lightning_checkpoints").glob("best-*.ckpt"))
    lightning_best: Path | None = None
    if lightning_checkpoints:
        def lightning_score(path: Path) -> tuple[float, int, float]:
            match = LIGHTNING_BEST_RE.match(path.name)
            if match:
                return float(match.group(2)), -int(match.group(1)), -path.stat().st_mtime
            return float("inf"), 0, -path.stat().st_mtime

        lightning_best = min(lightning_checkpoints, key=lightning_score)

    if checkpoint_kind == "best":
        if lightning_best is not None:
            return lightning_best
        if best_checkpoint.exists():
            return best_checkpoint
        return latest

    last_lightning = run_dir / "lightning_checkpoints" / "last.ckpt"
    if last_lightning.exists():
        return last_lightning
    return latest or lightning_best


def find_run_dir(task: dict[str, Any], run_root: Path, strict: bool = False) -> Path:
    if "run_dir" in task:
        return Path(task["run_dir"])
    prefix = (
        f"TEST_METHANE_MODEL_l{task['hiphop_l_max']}_n{task['hiphop_n_max']}_"
        f"d{task['data_size']}_seed{task['seed']}"
    )
    candidates = [path for path in run_root.glob(prefix + "*") if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directory found for prefix: {prefix}")

    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    if len(candidates) > 1:
        checkpoint_candidates = [path for path in candidates if choose_checkpoint(path, "best") is not None]
        if checkpoint_candidates:
            candidates = checkpoint_candidates
        if strict and len(candidates) > 1:
            formatted = "\n  ".join(str(path) for path in candidates)
            raise RuntimeError(f"Multiple candidate run directories found for {prefix}:\n  {formatted}")
        print(
            "Warning: multiple candidate run directories found; choosing newest:\n"
            + "\n".join(f"  {path}" for path in candidates),
            flush=True,
        )
    return candidates[0]


def prepare_test_dict(
    data_size: int,
    test_set_size: int,
    dataset_path: Path,
    test_indices: Path | None = None,
) -> dict[str, Any]:
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Methane dataset not found at {dataset_path}. "
            "Download methane.extxyz into datasets/ first or pass --dataset-path."
        )

    test_dict: dict[str, list[Any] | np.ndarray] = {
        "source_indices": [],
        "numbers": [],
        "positions": [],
        "forces": [],
        "energy": [],
    }

    selected_indices: set[int] | None = None
    if test_indices is not None:
        indices = np.load(test_indices)
        selected_indices = {int(index) for index in indices.tolist()}
        if len(selected_indices) != len(indices):
            raise ValueError(f"Test index file contains duplicate indices: {test_indices}")
        test_set_size = len(selected_indices)

    start_idx = data_size if selected_indices is None else min(selected_indices)
    stop_idx = data_size + test_set_size if selected_indices is None else max(selected_indices) + 1

    for idx, frame in enumerate(ase.io.iread(dataset_path)):
        if idx < start_idx:
            continue
        if idx >= stop_idx:
            break
        if selected_indices is not None and idx not in selected_indices:
            continue

        species = frame.get_atomic_numbers()
        positions = frame.get_positions()
        forces = frame.get_forces()
        energy = frame.get_total_energy()

        forces = forces * 51.42208619083232 * 23.060541945329334
        energy = energy * 627.5096080305927
        energy -= ENERGY_MEAN

        test_dict["source_indices"].append(idx)
        test_dict["numbers"].append(species)
        test_dict["positions"].append(positions)
        test_dict["forces"].append(forces)
        test_dict["energy"].append(energy)

    test_dict["source_indices"] = np.asarray(test_dict["source_indices"], dtype=np.int64)
    test_dict["numbers"] = np.asarray(test_dict["numbers"], dtype=np.int64)
    for key in ("positions", "forces", "energy"):
        test_dict[key] = np.array(test_dict[key], dtype=np.float32)

    if len(test_dict["numbers"]) != test_set_size:
        raise RuntimeError(
            f"Expected {test_set_size} test structures for data_size={data_size}, "
            f"but read {len(test_dict['numbers'])}."
        )

    return test_dict


def build_test_database(test_dict: dict[str, Any], db_info: dict[str, Any], seed: int, device: torch.device):
    database = hippynn.databases.Database(
        arr_dict=test_dict,
        seed=seed + 1,
        pin_memory=(device.type == "cuda"),
        **db_info,
    )
    database.split_the_rest("test")
    database.send_to_device(device)
    return database


def load_bundle_with_retry(
    structure_path: Path,
    checkpoint_path: Path,
    device: torch.device,
    attempts: int,
    sleep_seconds: float,
):
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            return load_checkpoint(
                str(structure_path),
                str(checkpoint_path),
                restart_db=False,
                model_device=str(device),
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == attempts:
                break
            print(
                f"Load attempt {attempt}/{attempts} failed for {checkpoint_path}: {exc}\n"
                f"Sleeping {sleep_seconds}s before retry...",
                flush=True,
            )
            time.sleep(sleep_seconds)
    raise RuntimeError(f"Failed to load checkpoint after {attempts} attempts: {checkpoint_path}") from last_error


def metric_value(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        return value.item()
    return value


def save_per_point_predictions(
    *,
    evaluator,
    test_dict: dict[str, Any],
    device: torch.device,
    batch_size: int,
    output_dir: Path,
    task: dict[str, Any],
) -> tuple[dict[str, float], Path, Path]:
    """Persist lossless per-structure targets, predictions, and errors."""
    numbers = np.asarray(test_dict["numbers"], dtype=np.int64)
    positions = np.asarray(test_dict["positions"], dtype=np.float32)
    true_energy = np.asarray(test_dict["energy"], dtype=np.float32).reshape(-1)
    true_forces = np.asarray(test_dict["forces"], dtype=np.float32)
    source_indices = np.asarray(test_dict["source_indices"], dtype=np.int64)
    n_structures, n_atoms = numbers.shape

    pred_energy_chunks: list[np.ndarray] = []
    pred_force_chunks: list[np.ndarray] = []
    evaluator.model.eval()
    for start in range(0, n_structures, batch_size):
        stop = min(start + batch_size, n_structures)
        # Force outputs are gradients of the predicted energy with respect to
        # positions, so the position leaf must retain autograd information even
        # though this is evaluation rather than training.
        batch_positions = (
            torch.as_tensor(positions[start:stop], dtype=torch.float32, device=device)
            .detach()
            .clone()
            .requires_grad_(True)
        )
        batch_values = {
            "numbers": torch.as_tensor(numbers[start:stop], dtype=torch.long, device=device),
            "positions": batch_positions,
        }
        model_inputs = tuple(batch_values[node.db_name] for node in evaluator.model.input_nodes)
        with torch.enable_grad():
            outputs = evaluator.model(*model_inputs)
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        batch_count = stop - start
        energy_outputs = [output for output in outputs if output.numel() == batch_count]
        force_outputs = [output for output in outputs if output.numel() == batch_count * n_atoms * 3]
        if len(energy_outputs) != 1 or len(force_outputs) != 1:
            shapes = [tuple(output.shape) for output in outputs]
            raise RuntimeError(f"Could not identify unique energy/force predictions; output shapes={shapes}")
        pred_energy_chunks.append(energy_outputs[0].detach().cpu().numpy().reshape(batch_count))
        pred_force_chunks.append(force_outputs[0].detach().cpu().numpy().reshape(batch_count, n_atoms, 3))

    pred_energy = np.concatenate(pred_energy_chunks).astype(np.float32, copy=False)
    pred_forces = np.concatenate(pred_force_chunks).astype(np.float32, copy=False)
    energy_error = pred_energy - true_energy
    force_error = pred_forces - true_forces
    per_structure_force_rmse = np.sqrt(np.mean(force_error.astype(np.float64) ** 2, axis=(1, 2)))
    per_structure_force_mae = np.mean(np.abs(force_error.astype(np.float64)), axis=(1, 2))
    per_structure_max_abs_force_error = np.max(np.abs(force_error), axis=(1, 2))

    stem = (
        f"l{task['hiphop_l_max']}_n{task['hiphop_n_max']}_d{task['data_size']}_"
        f"seed{task['seed']}_task{task['sweep_task_id']}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / f"{stem}_per_point.npz"
    csv_path = output_dir / f"{stem}_per_point.csv"
    np.savez_compressed(
        npz_path,
        source_indices=source_indices,
        numbers=numbers,
        positions=positions,
        true_energy=true_energy,
        pred_energy=pred_energy,
        energy_error=energy_error,
        true_forces=true_forces,
        pred_forces=pred_forces,
        force_error=force_error,
        per_structure_force_rmse=per_structure_force_rmse,
        per_structure_force_mae=per_structure_force_mae,
        per_structure_max_abs_force_error=per_structure_max_abs_force_error,
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "test_row",
                "source_index",
                "energy_target",
                "energy_prediction",
                "energy_error",
                "abs_energy_error",
                "force_rmse",
                "force_mae",
                "max_abs_force_component_error",
            ]
        )
        for row in range(n_structures):
            writer.writerow(
                [
                    row,
                    int(source_indices[row]),
                    float(true_energy[row]),
                    float(pred_energy[row]),
                    float(energy_error[row]),
                    float(abs(energy_error[row])),
                    float(per_structure_force_rmse[row]),
                    float(per_structure_force_mae[row]),
                    float(per_structure_max_abs_force_error[row]),
                ]
            )

    aggregates = {
        "test_T-RMSE": float(np.sqrt(np.mean(energy_error.astype(np.float64) ** 2))),
        "test_T-MAE": float(np.mean(np.abs(energy_error.astype(np.float64)))),
        "test_F-RMSE": float(np.sqrt(np.mean(force_error.astype(np.float64) ** 2))),
        "test_F-MAE": float(np.mean(np.abs(force_error.astype(np.float64)))),
    }
    print(f"Wrote per-point records: {csv_path}", flush=True)
    print(f"Wrote raw predictions/errors: {npz_path}", flush=True)
    return aggregates, csv_path, npz_path


def evaluate_task(
    task: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    test_dict_cache: dict[tuple[int, int], dict[str, Any]],
) -> dict[str, Any]:
    run_dir = find_run_dir(task, run_root=args.run_root, strict=args.strict_run_dir)
    explicit_checkpoint = "checkpoint_path" in task
    checkpoint_path = Path(task["checkpoint_path"]) if explicit_checkpoint else choose_checkpoint(
        run_dir, args.checkpoint_kind
    )
    structure_path = run_dir / "experiment_structure.pt"

    if checkpoint_path is not None and checkpoint_path.suffix == ".ckpt" and not structure_path.exists():
        metadata = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        embedded_structure = metadata.get("hippynn_structure_file")
        if embedded_structure:
            structure_path = Path(embedded_structure)
        del metadata

    if not structure_path.exists():
        raise FileNotFoundError(f"Missing experiment structure: {structure_path}")
    if checkpoint_path is None or not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found in run directory: {run_dir}")

    print("=" * 80, flush=True)
    print(f"Evaluating task {task['sweep_task_id']} (seed={task['seed']})", flush=True)
    print(f"Run dir:      {run_dir}", flush=True)
    print(f"Checkpoint:   {checkpoint_path}", flush=True)
    print(f"Structure:    {structure_path}", flush=True)

    if checkpoint_path.suffix == ".ckpt":
        from hippynn.experiment import HippynnLightningModule
        from hippynn.experiment.evaluator import Evaluator

        lightning_module = HippynnLightningModule.load_from_checkpoint(
            str(checkpoint_path),
            structure_file=str(structure_path),
            map_location="cpu",
            weights_only=False,
        )
        lightning_module.model.to(device)
        evaluator = Evaluator(
            lightning_module.model,
            lightning_module.eval_loss,
            lightning_module.eval_names,
            db_info={"inputs": list(lightning_module.inputs), "targets": list(lightning_module.targets)},
        )
        controller = lightning_module.controller
        metric_tracker = lightning_module.metric_tracker
        del lightning_module
    else:
        bundle = load_bundle_with_retry(
            structure_path=structure_path,
            checkpoint_path=checkpoint_path,
            device=device,
            attempts=args.retry_attempts,
            sleep_seconds=args.retry_sleep,
        )
        training_modules = bundle["training_modules"]
        controller = bundle["controller"]
        metric_tracker = bundle["metric_tracker"]
        if explicit_checkpoint:
            print("Evaluating exact model weights from explicitly requested checkpoint", flush=True)
        elif getattr(metric_tracker, "best_model", None):
            print("Loading best-so-far weights from metric_tracker.best_model", flush=True)
            training_modules.model.load_state_dict(metric_tracker.best_model)
            training_modules.evaluator.model.load_state_dict(metric_tracker.best_model)
        evaluator = training_modules.evaluator

    db_info = evaluator.db_info
    cache_key = (task["data_size"], args.test_set_size)
    if cache_key not in test_dict_cache:
        print(
            f"Preparing methane test split once for data_size={task['data_size']} test_size={args.test_set_size}",
            flush=True,
        )
        test_dict_cache[cache_key] = prepare_test_dict(
            task["data_size"], args.test_set_size, args.dataset_path, args.test_indices
        )

    eval_batch_size = args.eval_batch_size or getattr(controller, "eval_batch_size", None) or 512
    when = "CurrentBestCheckpoint"
    per_point_metrics: dict[str, float] = {}
    per_point_csv: Path | str = ""
    per_point_npz: Path | str = ""
    if args.per_point_output_dir is not None:
        per_point_metrics, per_point_csv, per_point_npz = save_per_point_predictions(
            evaluator=evaluator,
            test_dict=test_dict_cache[cache_key],
            device=device,
            batch_size=eval_batch_size,
            output_dir=args.per_point_output_dir,
            task=task,
        )

    if args.per_point_only:
        metrics = {key.removeprefix("test_"): value for key, value in per_point_metrics.items()}
        results_tracker = metric_tracker
    else:
        test_database = build_test_database(
            test_dict=test_dict_cache[cache_key],
            db_info=db_info,
            seed=task["seed"],
            device=device,
        )

        torch.cuda.empty_cache()
        results_tracker = test_model(
            test_database,
            evaluator,
            batch_size=eval_batch_size,
            when=when,
            metric_tracker=metric_tracker,
        )
        metrics = results_tracker.other_metric_values[when]["test"]
    best_valid = results_tracker.best_metric_values.get("valid", {})

    row = {
        "run": f"task {task['sweep_task_id']}: l={task['hiphop_l_max']} n={task['hiphop_n_max']} d={task['data_size']}",
        "sweep_task_id": task["sweep_task_id"],
        "seed": task["seed"],
        "hiphop_l_max": task["hiphop_l_max"],
        "hiphop_n_max": task["hiphop_n_max"],
        "data_size": task["data_size"],
        "total_params": task["total_params"],
        "run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_kind_requested": args.checkpoint_kind,
        "checkpoint_file_used": checkpoint_path.name,
        "checkpoint_target_epoch": task.get("checkpoint_target_epoch", ""),
        "eval_batch_size": eval_batch_size,
        "test_indices_path": str(args.test_indices or ""),
        "test_set_size": len(test_dict_cache[cache_key]["numbers"]),
        "per_point_csv": str(per_point_csv),
        "per_point_npz": str(per_point_npz),
        "checkpoint_epoch_count": getattr(results_tracker, "current_epoch", ""),
        "best_valid_T-RMSE": metric_value(best_valid.get("T-RMSE", "")),
        "best_valid_T-MAE": metric_value(best_valid.get("T-MAE", "")),
        "best_valid_F-RMSE": metric_value(best_valid.get("F-RMSE", "")),
        "best_valid_F-MAE": metric_value(best_valid.get("F-MAE", "")),
        "test_T-RMSE": metric_value(metrics.get("T-RMSE", "")),
        "test_T-MAE": metric_value(metrics.get("T-MAE", "")),
        "test_T-RSQ": metric_value(metrics.get("T-RSQ", "")),
        "test_F-RMSE": metric_value(metrics.get("F-RMSE", "")),
        "test_F-MAE": metric_value(metrics.get("F-MAE", "")),
        "test_F-RSQ": metric_value(metrics.get("F-RSQ", "")),
        "test_Error Loss": metric_value(metrics.get("Error Loss", "")),
        "test_L2": metric_value(metrics.get("L2", "")),
        "test_Loss": metric_value(metrics.get("Loss", "")),
        "status": "ok",
    }
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    plot_dir = args.plot_dir.resolve()
    args.repo_root = args.repo_root.resolve()
    args.run_root = (args.run_root or (args.repo_root / "examples")).resolve()
    args.dataset_path = (args.dataset_path or (args.repo_root / "datasets" / "methane.extxyz")).resolve()
    if args.test_indices is not None:
        args.test_indices = args.test_indices.resolve()
    if args.per_point_output_dir is not None:
        args.per_point_output_dir = args.per_point_output_dir.resolve()
    if args.per_point_only and args.per_point_output_dir is None:
        raise ValueError("--per-point-only requires --per-point-output-dir")
    output_csv = (args.output_csv or (plot_dir / DEFAULT_OUTPUT_NAME)).resolve()
    device = choose_device(args.device)

    print(f"Plot dir:      {plot_dir}", flush=True)
    print(f"Repo root:     {args.repo_root}", flush=True)
    print(f"Run root:      {args.run_root}", flush=True)
    print(f"Dataset path:  {args.dataset_path}", flush=True)
    print(f"Test indices:  {args.test_indices or 'sequential legacy slice'}", flush=True)
    print(f"Output CSV:    {output_csv}", flush=True)
    print(f"Device:        {device}", flush=True)

    maybe_enable_triton()

    tasks = (
        tasks_from_run_dirs(args.run_dirs, args.checkpoint_paths, args.checkpoint_target_epochs)
        if args.run_dirs
        else infer_target_tasks(plot_dir, args.task_ids)
    )
    if not tasks:
        print("No tasks selected for evaluation.", flush=True)
        return 0

    print("Tasks to evaluate:", flush=True)
    for task in tasks:
        print(
            f"  task {task['sweep_task_id']}: seed={task['seed']} "
            f"l={task['hiphop_l_max']} n={task['hiphop_n_max']} d={task['data_size']}",
            flush=True,
        )

    rows: list[dict[str, Any]] = []
    test_dict_cache: dict[tuple[int, int], dict[str, Any]] = {}

    for task in tasks:
        try:
            row = evaluate_task(task, args, device, test_dict_cache)
        except Exception as exc:  # noqa: BLE001
            row = {
                "run": f"task {task['sweep_task_id']}: l={task['hiphop_l_max']} n={task['hiphop_n_max']} d={task['data_size']}",
                "sweep_task_id": task["sweep_task_id"],
                "seed": task["seed"],
                "hiphop_l_max": task["hiphop_l_max"],
                "hiphop_n_max": task["hiphop_n_max"],
                "data_size": task["data_size"],
                "total_params": task["total_params"],
                "status": f"error: {exc}",
            }
            print(f"ERROR for task {task['sweep_task_id']}: {exc}", flush=True)
        rows.append(row)

    write_csv(output_csv, rows)
    print(f"Wrote {output_csv}", flush=True)

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if ok_rows:
        print("\nSuccessful evaluations:", flush=True)
        for row in ok_rows:
            print(
                f"  task {row['sweep_task_id']}: "
                f"test_T-MAE={row.get('test_T-MAE', '')} "
                f"test_F-MAE={row.get('test_F-MAE', '')}",
                flush=True,
            )

    failed_rows = [row for row in rows if row.get("status") != "ok"]
    if failed_rows:
        print("\nFailed evaluations:", flush=True)
        for row in failed_rows:
            print(f"  task {row['sweep_task_id']}: {row['status']}", flush=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
