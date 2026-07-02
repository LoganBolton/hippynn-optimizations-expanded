#!/bin/bash
#SBATCH --job-name=methane_sweep
#SBATCH --output=./logs/%j_methane_sweep.out
#SBATCH --error=./logs/%j_methane_sweep.err
#SBATCH --time=2-00:00:00          # Max for Darwin long QoS
#SBATCH --qos=long
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --open-mode=append
#SBATCH --partition=shared-gpu-ampere  # GPU partition name (check with: sinfo)
#SBATCH --constraint=gpu1_model:NVIDIA_A100-SXM4-40GB&gpu_count:4
#SBATCH --cpus-per-task=32         # CPUs for four local workers
#SBATCH --mem=96G                  # RAM for four local workers

# Usage: sbatch run_sweep.slurm
#
# This is a local/offline replacement for `wandb agent`. W&B offline mode logs
# each run locally, but it cannot pull sweep jobs from the online W&B backend.

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TRAINING_SCRIPT="$SCRIPT_DIR/methane_configurations.py"
SWEEP_CONFIG="$SCRIPT_DIR/methane-l4-n4.yml"
CONDA_ENV="hippynn-expanded-lmax"       # Your conda environment name
MAX_PARALLEL=4
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10}"
WANDB_MODE="${WANDB_MODE:-online}"
ATTEMPT_ID="${SLURM_RESTART_COUNT:-0}"
children=()

case "$WANDB_MODE" in
    online|offline|disabled)
        ;;
    off)
        echo "Error: WANDB_MODE=off is not valid. Use WANDB_MODE=disabled, offline, or online."
        exit 1
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

if [ "$#" -ne 0 ]; then
    echo "Error: this local sweep does not take a W&B sweep ID"
    echo "Usage: sbatch run_sweep.slurm"
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
echo "Script dir:    $SCRIPT_DIR"
echo "Start time:    $(date)"
echo "=========================================="
echo ""

echo "Checking GPU availability..."
nvidia-smi
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
export WANDB_DIR="${WANDB_DIR:-$(pwd)/wandb}"
mkdir -p "$WANDB_DIR"
mkdir -p logs

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
]
print(len(list(itertools.product(*grid_values))))
PY
)"

echo "Starting local sweep"
echo "Sweep config: $SWEEP_CONFIG"
echo "Sweep tasks:  $SWEEP_COUNT"
echo "Parallelism:  $MAX_PARALLEL"
echo "Checkpoint:   every $CHECKPOINT_EVERY epoch(s)"
echo "W&B mode:     $WANDB_MODE"
echo "=========================================="
echo ""

EXIT_CODE=0

for ((TASK_START = 0; TASK_START < SWEEP_COUNT; TASK_START += MAX_PARALLEL)); do
    pids=()

    for ((GPU_ID = 0; GPU_ID < MAX_PARALLEL; GPU_ID++)); do
        TASK_ID=$((TASK_START + GPU_ID))
        if [ "$TASK_ID" -ge "$SWEEP_COUNT" ]; then
            break
        fi

        echo "Starting sweep task $TASK_ID / $SWEEP_COUNT on GPU $GPU_ID"
        (
            export CUDA_VISIBLE_DEVICES="$GPU_ID"
            python -u "$TRAINING_SCRIPT" \
                --sweep_config "$SWEEP_CONFIG" \
                --sweep_task_id "$TASK_ID" \
                --wandb_mode "$WANDB_MODE" \
                --resume \
                --checkpoint_every "$CHECKPOINT_EVERY"
        ) > "logs/${SLURM_JOB_ID}_r${ATTEMPT_ID}_${TASK_ID}_methane_sweep.out" 2> "logs/${SLURM_JOB_ID}_r${ATTEMPT_ID}_${TASK_ID}_methane_sweep.err" &
        pids+=("$!")
        children+=("$!")
    done

    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            EXIT_CODE=1
        fi
    done
done

echo "Local sweep finished at: $(date)"
echo "Exit code: $EXIT_CODE"

exit "$EXIT_CODE"
