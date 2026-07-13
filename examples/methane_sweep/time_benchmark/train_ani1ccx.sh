#!/bin/bash -l
#SBATCH --output=job_%x_%j.log
#SBATCH --partition=shared-gpu-ampere
#SBATCH --qos=long
#SBATCH --time=2-00:00:00

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [ "$(basename "$SCRIPT_DIR")" != "time_benchmark" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

CONDA_ENV="${CONDA_ENV:-hippynn-expanded-lmax}"
ANI_DATA="${ANI_DATA:-$PROJECT_ROOT/release/datasets/ani1x_release/ani1x-release.h5}"
GPU_LIST="${GPU_LIST:-0}"
SEEDS="${SEEDS:-42 1776 7 0 1234 2024}"
TENSOR_ORDERS="${TENSOR_ORDERS:-0 1 2 3}"
TENSOR_FACTORS="${TENSOR_FACTORS:-1 2 3 4}"
CONFIGS="${CONFIGS:-}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
TAG="${TAG:-${SLURM_JOB_NAME:-ani1ccx}}"

if [ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]; then
    module load miniconda3
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

export PYTHONPATH="$PROJECT_ROOT"
export PYTHONNOUSERSITE=1
cd "$SCRIPT_DIR"

echo "Using Python: $(which python)"
python -c "import sys, numpy, h5py, torch, hippynn; print(sys.executable); print('numpy', numpy.__version__); print('h5py', h5py.__version__); print('torch', torch.__version__)"

read -r -a GPUS <<< "$GPU_LIST"
if [ "${#GPUS[@]}" -eq 0 ]; then
    echo "GPU_LIST must contain at least one GPU id."
    exit 1
fi

run_one() {
    local gpu="$1"
    local tensor_order="$2"
    local tensor_factor="$3"
    local seed="$4"
    local run_tag="${TAG}_l${tensor_order}_n${tensor_factor}"

    CUDA_VISIBLE_DEVICES="$gpu" python -u ccx_training.py \
        --tag "$run_tag" \
        --gpu 0 \
        --seed "$seed" \
        --tensor_order "$tensor_order" \
        --tensor_factors "$tensor_factor" \
        --batch_size "$BATCH_SIZE" \
        --anidata_location "$ANI_DATA"
}

job_index=0
if [ -n "$CONFIGS" ]; then
    for CONFIG in $CONFIGS; do
        TENSOR_ORDER="${CONFIG%%:*}"
        TENSOR_FACTOR="${CONFIG##*:}"
        for SEED in $SEEDS; do
            GPU="${GPUS[$((job_index % ${#GPUS[@]}))]}"
            run_one "$GPU" "$TENSOR_ORDER" "$TENSOR_FACTOR" "$SEED" &
            job_index=$((job_index + 1))
            if [ "$((job_index % ${#GPUS[@]}))" -eq 0 ]; then
                wait
            fi
        done
    done
else
    for TENSOR_ORDER in $TENSOR_ORDERS; do
        for TENSOR_FACTOR in $TENSOR_FACTORS; do
            for SEED in $SEEDS; do
                GPU="${GPUS[$((job_index % ${#GPUS[@]}))]}"
                run_one "$GPU" "$TENSOR_ORDER" "$TENSOR_FACTOR" "$SEED" &
                job_index=$((job_index + 1))
                if [ "$((job_index % ${#GPUS[@]}))" -eq 0 ]; then
                    wait
                fi
            done
        done
    done
fi

wait
