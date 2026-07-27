#!/bin/bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 GPU_ID TASK_ID RUN_NAME"
    exit 2
fi

GPU_ID="$1"
TASK_ID="$2"
RUN_NAME="$3"

METHANE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$METHANE_DIR/../.." && pwd)"
PYTHON="/vast/home/logan_bolton/.conda/envs/hippynn-expanded-lmax/bin/python"

cd "$METHANE_DIR"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_ROOT"
export WANDB_DIR="${WANDB_DIR:-$METHANE_DIR/wandb}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${RUN_NAME}-${SLURM_JOB_ID:-manual}}"

echo "Watcher started: $(date)"
echo "Host: $(hostname)"
echo "Physical GPU: $GPU_ID"
echo "Sweep task: $TASK_ID"
echo "Run name: $RUN_NAME"

# A free GPU on these nodes normally uses only a few MiB.
# Require three consecutive checks below 100 MiB to avoid launching
# during a brief idle period between training and validation.
FREE_CHECKS=0

while true; do
    MEMORY_USED="$(
        nvidia-smi \
            --id="$GPU_ID" \
            --query-gpu=memory.used \
            --format=csv,noheader,nounits |
        tr -d ' '
    )"

    UTILIZATION="$(
        nvidia-smi \
            --id="$GPU_ID" \
            --query-gpu=utilization.gpu \
            --format=csv,noheader,nounits |
        tr -d ' '
    )"

    echo "$(date): GPU $GPU_ID: ${MEMORY_USED} MiB, ${UTILIZATION}% utilization"

    if (( MEMORY_USED < 100 )); then
        FREE_CHECKS=$((FREE_CHECKS + 1))
        echo "Free check ${FREE_CHECKS}/3"
    else
        FREE_CHECKS=0
    fi

    if (( FREE_CHECKS >= 3 )); then
        break
    fi

    sleep 60
done

echo "GPU $GPU_ID is free. Starting fresh training at $(date)."

export CUDA_VISIBLE_DEVICES="$GPU_ID"

exec "$PYTHON" -u training/methane_configurations.py \
    --sweep_config configs/methane-l3-n4-1m-seeds0-7.yml \
    --sweep_task_id "$TASK_ID" \
    --wandb_mode offline \
    --model_suffix b256_fresh
