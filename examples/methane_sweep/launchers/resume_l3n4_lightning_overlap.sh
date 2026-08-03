#!/bin/bash
#
# Resume a stopped L3N4 Lightning lineage inside an existing 2xGPU Slurm
# allocation. The caller is responsible for starting this through
# `srun --jobid=... --overlap`.
#
# Usage:
#   resume_l3n4_lightning_overlap.sh seed1
#   resume_l3n4_lightning_overlap.sh seed202

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 {seed1|seed202}" >&2
    exit 2
fi

TARGET="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHANE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$METHANE_DIR/../.." && pwd)"
CONDA_ENV="${CONDA_ENV:-hippynn-expanded-lmax}"
PYTHON="$HOME/.conda/envs/$CONDA_ENV/bin/python"
TORCHRUN="$HOME/.conda/envs/$CONDA_ENV/bin/torchrun"

case "$TARGET" in
    seed1)
        SEED=1
        MODEL_SUFFIX=lightning_ddp_a100x2
        ;;
    seed202)
        SEED=202
        MODEL_SUFFIX=cn2_2gpu_lightning_fresh
        ;;
    *)
        echo "Unknown target '$TARGET'; expected seed1 or seed202." >&2
        exit 2
        ;;
esac

if [ ! -x "$PYTHON" ] || [ ! -x "$TORCHRUN" ]; then
    echo "Missing Python or torchrun in conda environment '$CONDA_ENV'." >&2
    exit 1
fi

cd "$METHANE_DIR"

export PYTHONPATH="$PROJECT_ROOT"
export PYTHONNOUSERSITE=1
export WANDB_DIR="${WANDB_DIR:-$METHANE_DIR/wandb}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-l3n4-${SEED}-${SLURM_JOB_ID:-manual}}"
mkdir -p "$WANDB_DIR" "$MPLCONFIGDIR"

AVAILABLE_GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if [ "$AVAILABLE_GPUS" -lt 2 ]; then
    echo "Expected at least 2 visible GPUs, found $AVAILABLE_GPUS." >&2
    exit 1
fi

CHECKPOINT="$PROJECT_ROOT/examples/TEST_METHANE_MODEL_l3_n4_d1000000_seed${SEED}-${MODEL_SUFFIX}/lightning_checkpoints/last.ckpt"
if [ ! -s "$CHECKPOINT" ]; then
    echo "Resume checkpoint does not exist or is empty: $CHECKPOINT" >&2
    exit 1
fi

echo "L3N4 Lightning overlap resume"
echo "Target:            $TARGET"
echo "Seed:              $SEED"
echo "Model suffix:      $MODEL_SUFFIX"
echo "Checkpoint:        $CHECKPOINT"
echo "Parent Slurm job:  ${SLURM_JOB_ID:-none}"
echo "Slurm step:        ${SLURM_STEP_ID:-none}"
echo "Host:              $(hostname)"
echo "Visible GPUs:      ${CUDA_VISIBLE_DEVICES:-all}"
echo "Start time:        $(date)"

# These explicit values reproduce the original 1M-sample L3N4 Lightning
# configurations. Resume restores model, optimizer, scheduler, controller,
# callback, and RNG state from last.ckpt. Deliberately do not reset early
# stopping or the batch schedule.
exec "$TORCHRUN" \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=2 \
    "$METHANE_DIR/training/methane_configurations.py" \
    --parallel_backend lightning-ddp \
    --num_gpus 2 \
    --seed "$SEED" \
    --hiphop_l_max 3 \
    --hiphop_n_max 4 \
    --run_name "methane-l3-n4-d1000000-seed${SEED}" \
    --model_suffix "$MODEL_SUFFIX" \
    --wandb_mode offline \
    --n_epochs 10000 \
    --termination_patience 450 \
    --data_size 1000000 \
    --test_set_size 80000 \
    --batch_size 256 \
    --eval_batch_size 512 \
    --max_batch_size 512 \
    --activation_max_batches 10 \
    --activation_max_values 200000 \
    --invariant_max_batches 10 \
    --invariant_max_values 1000000 \
    --checkpoint_every 10 \
    --polynomial_cache_debug \
    --resume
