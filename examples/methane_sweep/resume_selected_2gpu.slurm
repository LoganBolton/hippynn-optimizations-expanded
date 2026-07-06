#!/bin/bash
#SBATCH --job-name=methane_resume_selected
#SBATCH --output=./logs/%j_methane_resume_selected.out
#SBATCH --error=./logs/%j_methane_resume_selected.err
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
#   TASK_IDS="6 7" sbatch resume_selected_2gpu.slurm
#
# Keep this focused on task ids that are not already running elsewhere.

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TRAINING_SCRIPT="$SCRIPT_DIR/methane_configurations.py"
SWEEP_CONFIG="$SCRIPT_DIR/methane-l4-n4.yml"
CONDA_ENV="hippynn-expanded-lmax"
MAX_PARALLEL=2
TASK_IDS="${TASK_IDS:-6 7}"
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
echo "Task ids:      $TASK_IDS"
echo "Parallelism:   $MAX_PARALLEL"
echo "Checkpoint:    every $CHECKPOINT_EVERY epoch(s)"
echo "W&B mode:      $WANDB_MODE"
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

read -r -a task_ids <<< "$TASK_IDS"
EXIT_CODE=0

for ((TASK_START = 0; TASK_START < ${#task_ids[@]}; TASK_START += MAX_PARALLEL)); do
    pids=()

    for ((GPU_ID = 0; GPU_ID < MAX_PARALLEL; GPU_ID++)); do
        TASK_INDEX=$((TASK_START + GPU_ID))
        if [ "$TASK_INDEX" -ge "${#task_ids[@]}" ]; then
            break
        fi

        TASK_ID="${task_ids[$TASK_INDEX]}"
        if ! [[ "$TASK_ID" =~ ^[0-9]+$ ]]; then
            echo "Error: task id must be an integer, got '$TASK_ID'"
            exit 1
        fi

        echo "Starting sweep task $TASK_ID on GPU $GPU_ID"
        (
            export CUDA_VISIBLE_DEVICES="$GPU_ID"
            python -u "$TRAINING_SCRIPT" \
                --sweep_config "$SWEEP_CONFIG" \
                --sweep_task_id "$TASK_ID" \
                --wandb_mode "$WANDB_MODE" \
                --resume \
                --checkpoint_every "$CHECKPOINT_EVERY"
        ) > "logs/${SLURM_JOB_ID}_r${ATTEMPT_ID}_${TASK_ID}_methane_resume_selected.out" 2> "logs/${SLURM_JOB_ID}_r${ATTEMPT_ID}_${TASK_ID}_methane_resume_selected.err" &
        pids+=("$!")
        children+=("$!")
    done

    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            EXIT_CODE=1
        fi
    done
done

echo "Selected resume finished at: $(date)"
echo "Exit code: $EXIT_CODE"

exit "$EXIT_CODE"
