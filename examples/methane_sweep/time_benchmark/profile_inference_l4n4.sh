#!/bin/bash -l

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CONDA_ENV="${CONDA_ENV:-hippynn-expanded-lmax}"
ANI_DATA="${ANI_DATA:-$PROJECT_ROOT/datasets/ani1x-release.h5}"
GPU_ID="${GPU_ID:-0}"
EXPECTED_GPU_NAME="${EXPECTED_GPU_NAME:-NVIDIA A100-PCIE-40GB}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/inference_profile_l4_n4}"

if [ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]; then
    module load miniconda3
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

export PYTHONPATH="$PROJECT_ROOT"
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="$GPU_ID"
cd "$SCRIPT_DIR"

ACTIVE_GPU_PIDS="$(nvidia-smi --id="$GPU_ID" --query-compute-apps=pid --format=csv,noheader,nounits)"
if [ -n "$ACTIVE_GPU_PIDS" ]; then
    echo "Refusing to profile on busy GPU $GPU_ID (active PIDs: $ACTIVE_GPU_PIDS)." >&2
    exit 2
fi

ACTUAL_GPU_NAME="$(nvidia-smi --id="$GPU_ID" --query-gpu=name --format=csv,noheader | xargs)"
if [ "$ACTUAL_GPU_NAME" != "$EXPECTED_GPU_NAME" ]; then
    echo "Refusing to profile on '$ACTUAL_GPU_NAME'; expected '$EXPECTED_GPU_NAME'." >&2
    exit 3
fi

echo "Profiling exact benchmark architecture on physical GPU $GPU_ID ($ACTUAL_GPU_NAME)"
echo "Configuration: l_max=4 n_max=4, batch=1024, energy and energy+forces"
python -u profile_inference_architecture.py \
    --anidata_location "$ANI_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --l_max 4 \
    --n_max 4 \
    --batch_size 1024 \
    --warmups 2 \
    --seed 0 \
    --gpu 0 \
    --mode both

MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-profile}" \
    python plot_inference_architecture.py "$OUTPUT_DIR"
