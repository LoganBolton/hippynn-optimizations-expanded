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
                "total_params": int(row["total_params"]),
                "source_file": row.get("source_file", ""),
            }
        )

    tasks.sort(key=lambda row: row["sweep_task_id"])
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

    if checkpoint_kind == "best":
        if best_checkpoint.exists():
            return best_checkpoint
        return latest

    return latest


def find_run_dir(task: dict[str, Any], run_root: Path, strict: bool = False) -> Path:
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


def prepare_test_dict(data_size: int, test_set_size: int, dataset_path: Path) -> dict[str, Any]:
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Methane dataset not found at {dataset_path}. "
            "Download methane.extxyz into datasets/ first or pass --dataset-path."
        )

    test_dict: dict[str, list[Any] | np.ndarray] = {
        "numbers": [],
        "positions": [],
        "forces": [],
        "energy": [],
    }

    start_idx = data_size
    stop_idx = data_size + test_set_size

    for idx, frame in enumerate(ase.io.iread(dataset_path)):
        if idx < start_idx:
            continue
        if idx >= stop_idx:
            break

        species = frame.get_atomic_numbers()
        positions = frame.get_positions()
        forces = frame.get_forces()
        energy = frame.get_total_energy()

        forces = forces * 51.42208619083232 * 23.060541945329334
        energy = energy * 627.5096080305927
        energy -= ENERGY_MEAN

        test_dict["numbers"].append(species)
        test_dict["positions"].append(positions)
        test_dict["forces"].append(forces)
        test_dict["energy"].append(energy)

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


def evaluate_task(
    task: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    test_dict_cache: dict[tuple[int, int], dict[str, Any]],
) -> dict[str, Any]:
    run_dir = find_run_dir(task, run_root=args.run_root, strict=args.strict_run_dir)
    structure_path = run_dir / "experiment_structure.pt"
    checkpoint_path = choose_checkpoint(run_dir, args.checkpoint_kind)

    if not structure_path.exists():
        raise FileNotFoundError(f"Missing experiment structure: {structure_path}")
    if checkpoint_path is None or not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found in run directory: {run_dir}")

    print("=" * 80, flush=True)
    print(f"Evaluating task {task['sweep_task_id']} (seed={task['seed']})", flush=True)
    print(f"Run dir:      {run_dir}", flush=True)
    print(f"Checkpoint:   {checkpoint_path}", flush=True)
    print(f"Structure:    {structure_path}", flush=True)

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

    if getattr(metric_tracker, "best_model", None):
        print("Loading best-so-far weights from metric_tracker.best_model", flush=True)
        training_modules.model.load_state_dict(metric_tracker.best_model)
        training_modules.evaluator.model.load_state_dict(metric_tracker.best_model)
    else:
        print("Warning: checkpoint does not contain metric_tracker.best_model; evaluating checkpoint model state directly", flush=True)

    db_info = training_modules.evaluator.db_info
    cache_key = (task["data_size"], args.test_set_size)
    if cache_key not in test_dict_cache:
        print(
            f"Preparing methane test split once for data_size={task['data_size']} test_size={args.test_set_size}",
            flush=True,
        )
        test_dict_cache[cache_key] = prepare_test_dict(task["data_size"], args.test_set_size, args.dataset_path)

    test_database = build_test_database(
        test_dict=test_dict_cache[cache_key],
        db_info=db_info,
        seed=task["seed"],
        device=device,
    )

    eval_batch_size = args.eval_batch_size or getattr(controller, "eval_batch_size", None) or 512
    when = "CurrentBestCheckpoint"

    torch.cuda.empty_cache()
    results_tracker = test_model(
        test_database,
        training_modules.evaluator,
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
        "eval_batch_size": eval_batch_size,
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
    output_csv = (args.output_csv or (plot_dir / DEFAULT_OUTPUT_NAME)).resolve()
    device = choose_device(args.device)

    print(f"Plot dir:      {plot_dir}", flush=True)
    print(f"Repo root:     {args.repo_root}", flush=True)
    print(f"Run root:      {args.run_root}", flush=True)
    print(f"Dataset path:  {args.dataset_path}", flush=True)
    print(f"Output CSV:    {output_csv}", flush=True)
    print(f"Device:        {device}", flush=True)

    maybe_enable_triton()

    tasks = infer_target_tasks(plot_dir, args.task_ids)
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
