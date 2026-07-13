#!/bin/bash
#SBATCH --job-name=methane_sweep_2gpu
#SBATCH --output=./logs/%j_methane_sweep_2gpu.out
#SBATCH --error=./logs/%j_methane_sweep_2gpu.err
#SBATCH --time=2-00:00:00
#SBATCH --qos=long
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --open-mode=append
#SBATCH --partition=shared-gpu-ampere
#SBATCH --constraint=gpu1_model:NVIDIA_A100-PCIE-40GB&gpu_count:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# Usage:
#   RESUME=0 MODEL_SUFFIX=b256_fresh TASK_IDS="8 9 10 11 12 13 14 15" sbatch run_sweep_2gpu.slurm
#
# This is the 2-GPU A100 variant of run_sweep.slurm. It is useful when the
# 4-GPU SXM node is busy but 2-GPU A100-PCIE nodes are available.

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TRAINING_SCRIPT="$SCRIPT_DIR/methane_configurations.py"
SWEEP_CONFIG="$SCRIPT_DIR/methane-l4-n4.yml"
CONDA_ENV="hippynn-expanded-lmax"
MAX_PARALLEL=2
TASK_IDS="${TASK_IDS:-}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10}"
WANDB_MODE="${WANDB_MODE:-offline}"
MODEL_SUFFIX="${MODEL_SUFFIX:-}"
RESUME="${RESUME:-1}"
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

case "$RESUME" in
    0|1)
        ;;
    *)
        echo "Error: RESUME must be 0 or 1, got '$RESUME'"
        exit 1
        ;;
esac

if [ "$#" -ne 0 ]; then
    echo "Error: this local sweep does not take a W&B sweep ID"
    echo "Usage: sbatch run_sweep_2gpu.slurm"
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
    parameter_values(parameters["data_size"]),
]
print(len(list(itertools.product(*grid_values))))
PY
)"

echo "Starting local sweep"
echo "Sweep config: $SWEEP_CONFIG"
echo "Sweep tasks:  $SWEEP_COUNT"
echo "Parallelism:  $MAX_PARALLEL"
echo "Task ids:     ${TASK_IDS:-0..$((SWEEP_COUNT - 1))}"
echo "Checkpoint:   every $CHECKPOINT_EVERY epoch(s)"
echo "W&B mode:     $WANDB_MODE"
echo "Model suffix: ${MODEL_SUFFIX:-none}"
echo "Resume:       $RESUME"
echo "=========================================="
echo ""

if [ -n "$TASK_IDS" ]; then
    read -r -a task_ids <<< "$TASK_IDS"
else
    task_ids=()
    for ((TASK_ID = 0; TASK_ID < SWEEP_COUNT; TASK_ID++)); do
        task_ids+=("$TASK_ID")
    done
fi

run_task_on_gpu() {
    local gpu_id="$1"
    local task_id="$2"
    local worker_id="$3"

    if ! [[ "$task_id" =~ ^[0-9]+$ ]]; then
        echo "Error: task id must be an integer, got '$task_id'"
        return 1
    fi
    if [ "$task_id" -ge "$SWEEP_COUNT" ]; then
        echo "Error: task id $task_id is outside the configured sweep size ($SWEEP_COUNT)"
        return 1
    fi

    resume_args=()
    if [ "$RESUME" = "1" ]; then
        resume_args=(--resume)
    fi

    suffix_args=()
    if [ -n "$MODEL_SUFFIX" ]; then
        suffix_args=(--model_suffix "$MODEL_SUFFIX")
    fi

    echo "Starting sweep task $task_id / $SWEEP_COUNT on GPU $gpu_id"
    (
        export CUDA_VISIBLE_DEVICES="$gpu_id"
        python -u "$TRAINING_SCRIPT" \
            --sweep_config "$SWEEP_CONFIG" \
            --sweep_task_id "$task_id" \
            --wandb_mode "$WANDB_MODE" \
            --checkpoint_every "$CHECKPOINT_EVERY" \
            "${resume_args[@]}" \
            "${suffix_args[@]}"
    ) > "logs/${SLURM_JOB_ID}_r${ATTEMPT_ID}_${task_id}_w${worker_id}_methane_sweep_2gpu.out" 2> "logs/${SLURM_JOB_ID}_r${ATTEMPT_ID}_${task_id}_w${worker_id}_methane_sweep_2gpu.err"
}

worker_loop() {
    local gpu_id="$1"
    local task_index="$gpu_id"
    local exit_code=0

    while [ "$task_index" -lt "${#task_ids[@]}" ]; do
        TASK_ID="${task_ids[$task_index]}"
        if ! [[ "$TASK_ID" =~ ^[0-9]+$ ]]; then
            echo "Error: task id must be an integer, got '$TASK_ID'"
            return 1
        fi
        if [ "$TASK_ID" -ge "$SWEEP_COUNT" ]; then
            echo "Error: task id $TASK_ID is outside the configured sweep size ($SWEEP_COUNT)"
            return 1
        fi

        if ! run_task_on_gpu "$gpu_id" "$TASK_ID" "$gpu_id"; then
            exit_code=1
        fi

        task_index=$((task_index + MAX_PARALLEL))
    done

    return "$exit_code"
}

EXIT_CODE=0
for ((GPU_ID = 0; GPU_ID < MAX_PARALLEL; GPU_ID++)); do
    worker_loop "$GPU_ID" &
    children+=("$!")
done

for pid in "${children[@]}"; do
    if ! wait "$pid"; then
        EXIT_CODE=1
    fi
done

echo "Local 2-GPU sweep finished at: $(date)"
echo "Exit code: $EXIT_CODE"

exit "$EXIT_CODE"
