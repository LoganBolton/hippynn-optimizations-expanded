#!/bin/bash
#SBATCH --job-name=methane_l4_titanv
#SBATCH --output=./logs/titanv_l4/%A_%a_methane_l4_titanv.slurm.out
#SBATCH --error=./logs/titanv_l4/%A_%a_methane_l4_titanv.slurm.err
#SBATCH --time=2-00:00:00
#SBATCH --qos=long
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --open-mode=append
#SBATCH --array=0-7
#SBATCH --partition=volta-x86
#SBATCH --constraint="[gpu1_model:NVIDIA_TITAN_V|gpu1_model:Tesla_V100S-PCIE-32GB|gpu1_model:Tesla_V100-PCIE-32GB|gpu1_model:Tesla_V100-SXM2-16GB]&gpu_count:1"
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# Usage:
#   mkdir -p logs/titanv_l4
#   RESUME=0 MODEL_SUFFIX=b256_fresh_volta TASK_IDS="8 9 10 11 12 13 14 15" sbatch --array=0-7 run_l4_titanv_1gpu.slurm
#
# This is the Volta one-GPU companion for methane-l4-n4.yml. Darwin exposes
# these GPUs as Slurm features rather than GRES, so the constraint accepts
# Titan V and one-GPU V100 variants.
# Use disjoint TASK_IDS from any simultaneous A100 submission to avoid
# overlapping checkpoints and duplicate W&B runs.

METHANE_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
PROJECT_ROOT="$(cd "$METHANE_DIR/../.." && pwd)"
TRAINING_SCRIPT="$METHANE_DIR/training/methane_configurations.py"
SWEEP_CONFIG="$METHANE_DIR/configs/methane-l4-n4.yml"
CONDA_ENV="hippynn-expanded-lmax"
TASK_IDS="${TASK_IDS:-8 9 10 11 12 13 14 15}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10}"
WANDB_MODE="${WANDB_MODE:-offline}"
MODEL_SUFFIX="${MODEL_SUFFIX:-b256_fresh_volta}"
RESUME="${RESUME:-1}"
ATTEMPT_ID="${SLURM_RESTART_COUNT:-0}"
LOG_DIR="${LOG_DIR:-$METHANE_DIR/logs/titanv_l4}"
WANDB_DIR="${WANDB_DIR:-$METHANE_DIR/wandb_titanv_l4}"
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
echo "Task ids:      $TASK_IDS"
echo "Array task:    ${SLURM_ARRAY_TASK_ID:-0}"
echo "Checkpoint:    every $CHECKPOINT_EVERY epoch(s)"
echo "W&B mode:      $WANDB_MODE"
echo "Model suffix:  $MODEL_SUFFIX"
echo "Resume:        $RESUME"
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
export PYTHONNOUSERSITE=1
mkdir -p "$WANDB_DIR"
mkdir -p "$LOG_DIR"

read -r -a task_ids <<< "$TASK_IDS"
ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if ! [[ "$ARRAY_INDEX" =~ ^[0-9]+$ ]]; then
    echo "Error: SLURM_ARRAY_TASK_ID must be an integer, got '$ARRAY_INDEX'"
    exit 1
fi

if [ "$ARRAY_INDEX" -ge "${#task_ids[@]}" ]; then
    echo "Error: array index $ARRAY_INDEX has no matching task id in TASK_IDS='$TASK_IDS'"
    exit 1
fi

TASK_ID="${task_ids[$ARRAY_INDEX]}"
if ! [[ "$TASK_ID" =~ ^[0-9]+$ ]]; then
    echo "Error: task id must be an integer, got '$TASK_ID'"
    exit 1
fi

resume_args=()
if [ "$RESUME" = "1" ]; then
    resume_args=(--resume)
fi

echo "Starting L4 Titan V sweep task $TASK_ID on GPU 0"
export CUDA_VISIBLE_DEVICES=0
python -u "$TRAINING_SCRIPT" \
    --sweep_config "$SWEEP_CONFIG" \
    --sweep_task_id "$TASK_ID" \
    --wandb_mode "$WANDB_MODE" \
    --model_suffix "$MODEL_SUFFIX" \
    --checkpoint_every "$CHECKPOINT_EVERY" \
    "${resume_args[@]}" \
    > "$LOG_DIR/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}_r${ATTEMPT_ID}_${TASK_ID}_methane_l4_titanv.out" \
    2> "$LOG_DIR/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}_r${ATTEMPT_ID}_${TASK_ID}_methane_l4_titanv.err"
EXIT_CODE=$?

echo "L4 Titan V sweep task finished at: $(date)"
echo "Exit code: $EXIT_CODE"

exit "$EXIT_CODE"
