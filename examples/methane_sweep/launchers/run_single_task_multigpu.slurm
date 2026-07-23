#!/bin/bash
#SBATCH --job-name=methane_single_multigpu
#SBATCH --output=./logs/submissions/%j_methane_single_multigpu.out
#SBATCH --error=./logs/submissions/%j_methane_single_multigpu.err
#SBATCH --time=2-00:00:00
#SBATCH --qos=long
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --open-mode=append
#SBATCH --partition=shared-gpu-ampere
#SBATCH --constraint=gpu_cc:8.0
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G

# Train exactly one sweep task using one synchronized model replica per GPU.
#
# Run from examples/methane_sweep. TASK_ID is required. Examples:
#
# Resume seed 7 (task 2 in the 8-seed config) across all GPUs on the node:
#   TASK_ID=2 NUM_GPUS=4 \
#     SWEEP_CONFIG=configs/methane-l4-n3-1m-8seeds.yml \
#     MODEL_SUFFIX=b256_fresh \
#     sbatch --constraint='gpu_cc:8.0&gpu_count:4' launchers/run_single_task_multigpu.slurm
#
# Resume seed 42 (task 0 in either current config) on a 2-GPU node:
#   TASK_ID=0 NUM_GPUS=2 MODEL_SUFFIX=b256_fresh \
#     sbatch --constraint='gpu_cc:8.0&gpu_count:2' launchers/run_single_task_multigpu.slurm
#
# RESUME defaults to 1. When no checkpoint exists, --resume retains the
# trainer's existing behavior and starts the task from scratch.

METHANE_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
PROJECT_ROOT="$(cd "$METHANE_DIR/../.." && pwd)"
TRAINING_SCRIPT="$METHANE_DIR/training/methane_configurations.py"
SWEEP_CONFIG="${SWEEP_CONFIG:-$METHANE_DIR/configs/methane-l4-n4.yml}"
if [[ "$SWEEP_CONFIG" != /* ]]; then
    SWEEP_CONFIG="$METHANE_DIR/$SWEEP_CONFIG"
fi
CONDA_ENV="${CONDA_ENV:-hippynn-expanded-lmax}"
TASK_ID="${TASK_ID:-}"
NUM_GPUS="${NUM_GPUS:-}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10}"
WANDB_MODE="${WANDB_MODE:-offline}"
MODEL_SUFFIX="${MODEL_SUFFIX:-}"
RESUME="${RESUME:-1}"
ATTEMPT_ID="${SLURM_RESTART_COUNT:-0}"
children=()

if ! [[ "$TASK_ID" =~ ^[0-9]+$ ]]; then
    echo "Error: TASK_ID is required and must be a non-negative integer, got '$TASK_ID'"
    exit 1
fi

case "$WANDB_MODE" in
    online|offline|disabled)
        ;;
    *)
        echo "Error: WANDB_MODE must be one of: online, offline, disabled"
        exit 1
        ;;
esac

if ! [[ "$CHECKPOINT_EVERY" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: CHECKPOINT_EVERY must be a positive integer, got '$CHECKPOINT_EVERY'"
    exit 1
fi

case "$RESUME" in
    0|1)
        ;;
    *)
        echo "Error: RESUME must be 0 or 1, got '$RESUME'"
        exit 1
        ;;
esac

if [ "$#" -ne 0 ]; then
    echo "Error: configure this launcher with environment variables, not positional arguments"
    exit 1
fi

set -e

requeue_before_timeout() {
    echo ""
    echo "Received Slurm pre-timeout signal at $(date). Requeueing job $SLURM_JOB_ID..."
    for child in "${children[@]:-}"; do
        if kill -0 "$child" 2>/dev/null; then
            kill -TERM "$child" 2>/dev/null || true
        fi
    done

    if scontrol requeue "$SLURM_JOB_ID"; then
        echo "Requeued job $SLURM_JOB_ID. The next allocation will resume from checkpoints."
        exit 0
    fi

    echo "WARNING: scontrol requeue failed; exiting so this failure is visible."
    exit 1
}

trap requeue_before_timeout USR1

echo "Job ID:        $SLURM_JOB_ID"
echo "Restart count: ${SLURM_RESTART_COUNT:-0}"
echo "Node:          $(hostname)"
echo "Partition:     $SLURM_JOB_PARTITION"
echo "Working dir:   $(pwd)"
echo "Methane dir:   $METHANE_DIR"
echo "Sweep config:  $SWEEP_CONFIG"
echo "Task id:       $TASK_ID"
echo "Resume:        $RESUME"
echo "Checkpoint:    every $CHECKPOINT_EVERY epoch(s)"
echo "W&B mode:      $WANDB_MODE"
echo "Model suffix:  ${MODEL_SUFFIX:-none}"
echo "Start time:    $(date)"
echo "=========================================="
echo ""

echo "Checking GPU availability..."
nvidia-smi
echo ""

AVAILABLE_GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if [ -z "$NUM_GPUS" ]; then
    NUM_GPUS="$AVAILABLE_GPUS"
fi
if ! [[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]] || [ "$NUM_GPUS" -gt "$AVAILABLE_GPUS" ]; then
    echo "Error: NUM_GPUS must be between 1 and the $AVAILABLE_GPUS available GPUs, got '$NUM_GPUS'"
    exit 1
fi
if [ "$NUM_GPUS" != "1" ] && [ "$NUM_GPUS" != "2" ] && [ "$NUM_GPUS" != "4" ]; then
    echo "Error: NUM_GPUS must be 1, 2, or 4, got '$NUM_GPUS'"
    exit 1
fi
echo "GPUs for this run: $NUM_GPUS / $AVAILABLE_GPUS visible"
echo ""

echo "Loading conda environment..."
module load miniconda3

if conda info --envs | grep -q "$CONDA_ENV"; then
    source ~/.bashrc
    echo "Activating conda environment '$CONDA_ENV'..."
    conda activate "$CONDA_ENV"
else
    echo "Conda environment '$CONDA_ENV' not found. Please create it first"
    exit 1
fi

echo "Python location: $(which python)"
echo "Python version:  $(python --version)"
echo ""

export PYTHONPATH="$PROJECT_ROOT"
# Keep newer packages installed in ~/.local from shadowing the CUDA-compatible
# PyTorch build that belongs to this Conda environment. This matters on the
# driver-575 V100/Titan V nodes.
export PYTHONNOUSERSITE=1
export WANDB_DIR="${WANDB_DIR:-$(pwd)/wandb}"
mkdir -p "$WANDB_DIR"
mkdir -p logs "logs/runs/${SLURM_JOB_ID}"

SWEEP_COUNT="$(python - "$SWEEP_CONFIG" <<'PY'
import itertools
import sys

import yaml


def parameter_values(spec):
    if "values" in spec:
        return spec["values"]
    if "value" in spec:
        return [spec["value"]]
    if "min" in spec and "max" in spec:
        step = spec.get("step", 1)
        return list(range(int(spec["min"]), int(spec["max"]) + 1, int(step)))
    raise ValueError("sweep parameter needs value, values, or min/max")


with open(sys.argv[1], "r") as config_file:
    parameters = yaml.safe_load(config_file)["parameters"]

grid_values = [
    parameter_values(parameters["seed"]),
    parameter_values(parameters["hiphop_l_max"]),
    parameter_values(parameters["hiphop_n_max"]),
    parameter_values(parameters["data_size"]),
]
print(len(list(itertools.product(*grid_values))))
PY
)"

if [ "$TASK_ID" -ge "$SWEEP_COUNT" ]; then
    echo "Error: task id $TASK_ID is outside the configured sweep size ($SWEEP_COUNT)"
    exit 1
fi

resume_args=()
if [ "$RESUME" = "1" ]; then
    resume_args=(--resume)
fi

suffix_args=()
if [ -n "$MODEL_SUFFIX" ]; then
    suffix_args=(--model_suffix "$MODEL_SUFFIX")
fi

output_base="logs/runs/${SLURM_JOB_ID}/${SLURM_JOB_ID}_r${ATTEMPT_ID}_${TASK_ID}_methane_single_${NUM_GPUS}gpu"
echo "Starting sweep task $TASK_ID / $SWEEP_COUNT on $NUM_GPUS synchronized GPUs"
python -u "$TRAINING_SCRIPT" \
    --sweep_config "$SWEEP_CONFIG" \
    --sweep_task_id "$TASK_ID" \
    --num_gpus "$NUM_GPUS" \
    --wandb_mode "$WANDB_MODE" \
    --checkpoint_every "$CHECKPOINT_EVERY" \
    "${resume_args[@]}" \
    "${suffix_args[@]}" \
    > "${output_base}.out" 2> "${output_base}.err" &
children+=("$!")

EXIT_CODE=0
if ! wait "${children[0]}"; then
    EXIT_CODE=1
fi

echo "Single-task multi-GPU run finished at: $(date)"
echo "Exit code: $EXIT_CODE"
exit "$EXIT_CODE"
