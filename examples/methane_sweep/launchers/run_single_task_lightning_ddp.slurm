#!/bin/bash
#SBATCH --job-name=methane_lightning_ddp
#SBATCH --output=./logs/submissions/%j_methane_lightning_ddp.out
#SBATCH --error=./logs/submissions/%j_methane_lightning_ddp.err
#SBATCH --time=2-00:00:00
#SBATCH --qos=long
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --open-mode=append
#SBATCH --partition=shared-gpu-ampere
#SBATCH --constraint=gpu_cc:8.0
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G

# One methane model trained with one persistent Lightning DDP process per GPU.
#
# Submit from examples/methane_sweep and request a node with the desired GPU
# count through the site constraint:
#
#   TASK_ID=0 NUM_GPUS=2 MODEL_SUFFIX=lightning_ddp \
#     sbatch --constraint='gpu_cc:8.0&gpu_count:2' \
#     launchers/run_single_task_lightning_ddp.slurm

set -euo pipefail

METHANE_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
PROJECT_ROOT="$(cd "$METHANE_DIR/../.." && pwd)"
TRAINING_SCRIPT="$METHANE_DIR/training/methane_configurations.py"
SWEEP_CONFIG="${SWEEP_CONFIG:-$METHANE_DIR/configs/methane-l4-n4.yml}"
CONDA_ENV="${CONDA_ENV:-hippynn-expanded-lmax}"
TASK_ID="${TASK_ID:-}"
NUM_GPUS="${NUM_GPUS:-}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10}"
WANDB_MODE="${WANDB_MODE:-offline}"
MODEL_SUFFIX="${MODEL_SUFFIX:-lightning_ddp}"
RESUME="${RESUME:-1}"
POLYNOMIAL_CACHE_DEBUG="${POLYNOMIAL_CACHE_DEBUG:-1}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-}"
LIMIT_TEST_BATCHES="${LIMIT_TEST_BATCHES:-}"
N_EPOCHS="${N_EPOCHS:-}"
ATTEMPT_ID="${SLURM_RESTART_COUNT:-0}"
training_pid=""

if [[ "$SWEEP_CONFIG" != /* ]]; then
    SWEEP_CONFIG="$METHANE_DIR/$SWEEP_CONFIG"
fi
if ! [[ "$TASK_ID" =~ ^[0-9]+$ ]]; then
    echo "Error: TASK_ID is required and must be a non-negative integer, got '$TASK_ID'"
    exit 1
fi
if ! [[ "$CHECKPOINT_EVERY" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: CHECKPOINT_EVERY must be a positive integer"
    exit 1
fi
case "$WANDB_MODE" in
    online|offline|disabled) ;;
    *) echo "Error: WANDB_MODE must be online, offline, or disabled"; exit 1 ;;
esac
case "$RESUME" in
    0|1) ;;
    *) echo "Error: RESUME must be 0 or 1"; exit 1 ;;
esac
case "$POLYNOMIAL_CACHE_DEBUG" in
    0|1) ;;
    *) echo "Error: POLYNOMIAL_CACHE_DEBUG must be 0 or 1"; exit 1 ;;
esac

requeue_before_timeout() {
    echo "Received Slurm pre-timeout signal at $(date)."
    if [ -n "$training_pid" ] && kill -0 "$training_pid" 2>/dev/null; then
        kill -TERM "$training_pid" 2>/dev/null || true
        wait "$training_pid" 2>/dev/null || true
    fi
    if scontrol requeue "$SLURM_JOB_ID"; then
        echo "Requeued job $SLURM_JOB_ID; the next allocation will load last.ckpt."
        exit 0
    fi
    echo "ERROR: scontrol requeue failed."
    exit 1
}
trap requeue_before_timeout USR1

echo "Job ID:        $SLURM_JOB_ID"
echo "Restart count: ${SLURM_RESTART_COUNT:-0}"
echo "Node:          $(hostname)"
echo "Sweep config:  $SWEEP_CONFIG"
echo "Task id:       $TASK_ID"
echo "Resume:        $RESUME"
echo "W&B mode:      $WANDB_MODE"
echo "Start time:    $(date)"

nvidia-smi
AVAILABLE_GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if [ -z "$NUM_GPUS" ]; then
    NUM_GPUS="$AVAILABLE_GPUS"
fi
if ! [[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]] || [ "$NUM_GPUS" -gt "$AVAILABLE_GPUS" ]; then
    echo "Error: NUM_GPUS must be between 1 and $AVAILABLE_GPUS, got '$NUM_GPUS'"
    exit 1
fi
echo "DDP processes/GPUs: $NUM_GPUS"

module load miniconda3
# Darwin's system bashrc references optional variables, so source it before
# enabling nounset again.
set +u
source ~/.bashrc
set -u
conda activate "$CONDA_ENV"
export PYTHONPATH="$PROJECT_ROOT"
export PYTHONNOUSERSITE=1
python -c 'import pytorch_lightning as pl; print("PyTorch Lightning:", pl.__version__)' || {
    echo "Error: '$CONDA_ENV' does not provide pytorch_lightning."
    echo "Install HIPPYNN's compatible full optional dependencies before resubmitting."
    exit 1
}

export WANDB_DIR="${WANDB_DIR:-$METHANE_DIR/wandb}"
mkdir -p "$WANDB_DIR" "$METHANE_DIR/logs" "$METHANE_DIR/logs/runs/${SLURM_JOB_ID}"

resume_args=()
if [ "$RESUME" = "1" ]; then
    resume_args=(--resume)
fi
cache_args=()
if [ "$POLYNOMIAL_CACHE_DEBUG" = "1" ]; then
    cache_args=(--polynomial_cache_debug)
fi
suffix_args=()
if [ -n "$MODEL_SUFFIX" ]; then
    suffix_args=(--model_suffix "$MODEL_SUFFIX")
fi
limit_args=()
if [ -n "$LIMIT_TRAIN_BATCHES" ]; then
    limit_args+=(--limit_train_batches "$LIMIT_TRAIN_BATCHES")
fi
if [ -n "$LIMIT_VAL_BATCHES" ]; then
    limit_args+=(--limit_val_batches "$LIMIT_VAL_BATCHES")
fi
if [ -n "$LIMIT_TEST_BATCHES" ]; then
    limit_args+=(--limit_test_batches "$LIMIT_TEST_BATCHES")
fi
epoch_args=()
if [ -n "$N_EPOCHS" ]; then
    epoch_args=(--n_epochs "$N_EPOCHS")
fi

output_base="$METHANE_DIR/logs/runs/${SLURM_JOB_ID}/${SLURM_JOB_ID}_r${ATTEMPT_ID}_${TASK_ID}_methane_lightning_${NUM_GPUS}gpu"
echo "Launching torchrun; output base: $output_base"
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="$NUM_GPUS" \
    "$TRAINING_SCRIPT" \
    --parallel_backend lightning-ddp \
    --sweep_config "$SWEEP_CONFIG" \
    --sweep_task_id "$TASK_ID" \
    --num_gpus "$NUM_GPUS" \
    --wandb_mode "$WANDB_MODE" \
    --checkpoint_every "$CHECKPOINT_EVERY" \
    "${resume_args[@]}" \
    "${cache_args[@]}" \
    "${suffix_args[@]}" \
    "${limit_args[@]}" \
    "${epoch_args[@]}" \
    > "${output_base}.out" 2> "${output_base}.err" &
training_pid="$!"

set +e
wait "$training_pid"
exit_code=$?
set -e
echo "Lightning DDP run finished at $(date), exit code $exit_code"
exit "$exit_code"
