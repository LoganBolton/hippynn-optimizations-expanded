#!/bin/bash
set -euo pipefail

METHANE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$METHANE_DIR/../.." && pwd)"
PYTHON="/vast/home/logan_bolton/.conda/envs/hippynn-expanded-lmax/bin/python"

cd "$METHANE_DIR"

export CUDA_VISIBLE_DEVICES=2
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_ROOT"
export WANDB_DIR="${WANDB_DIR:-$METHANE_DIR/wandb}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-seed202-${SLURM_JOB_ID:-manual}}"

mkdir -p "$WANDB_DIR" "$MPLCONFIGDIR"

echo "Seed 202 continuation started at: $(date)"
echo "Host: $(hostname)"
echo "Parent Slurm job: ${SLURM_JOB_ID:-none}"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

exec "$PYTHON" -u training/methane_configurations.py \
    --sweep_config configs/methane-l4-n3-1m-8seeds.yml \
    --sweep_task_id 5 \
    --wandb_mode offline \
    --resume \
    --reset_early_stopping_patience \
    --checkpoint_every 10 \
    --model_suffix b256_fresh
