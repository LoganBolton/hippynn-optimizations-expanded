#!/bin/bash -l
#SBATCH --output=profile_%x_%j.log
#SBATCH --partition=shared-gpu-ampere
#SBATCH --qos=long
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --constraint=gpu1_model:NVIDIA_A100-PCIE-40GB

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [ "$(basename "$SCRIPT_DIR")" != "time_benchmark" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

CONDA_ENV="${CONDA_ENV:-hippynn-expanded-lmax}"
ANI_DATA="${ANI_DATA:-$PROJECT_ROOT/datasets/ani1x-release.h5}"
CONFIGS="${CONFIGS:-3:5 4:3 4:4}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-64}"
PROFILE_EPOCHS="${PROFILE_EPOCHS:-1}"
PROFILE_BATCHES="${PROFILE_BATCHES:-5}"
TAG_PREFIX="${TAG_PREFIX:-ani1ccx_profile}"
GPU_ID="${GPU_ID:-0}"
EXPECTED_GPU_NAME="${EXPECTED_GPU_NAME:-NVIDIA A100-PCIE-40GB}"

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
    echo "Resubmit with GPU_ID set to an idle GPU." >&2
    exit 2
fi

ACTUAL_GPU_NAME="$(nvidia-smi --id="$GPU_ID" --query-gpu=name --format=csv,noheader | xargs)"
if [ "$ACTUAL_GPU_NAME" != "$EXPECTED_GPU_NAME" ]; then
    echo "Refusing to profile on '$ACTUAL_GPU_NAME'; expected '$EXPECTED_GPU_NAME'." >&2
    exit 3
fi

echo "Using Python: $(which python)"
echo "Profiling on physical GPU $GPU_ID ($ACTUAL_GPU_NAME)"
echo "Profiling configurations: $CONFIGS"

for CONFIG in $CONFIGS; do
    if ! [[ "$CONFIG" =~ ^[0-9]+:[0-9]+$ ]]; then
        echo "Invalid configuration '$CONFIG'; expected l_max:n_max." >&2
        exit 4
    fi

    TENSOR_ORDER="${CONFIG%%:*}"
    TENSOR_FACTOR="${CONFIG##*:}"
    TAG="${TAG_PREFIX}_l${TENSOR_ORDER}_n${TENSOR_FACTOR}"

    echo "Profiling l_max=$TENSOR_ORDER n_max=$TENSOR_FACTOR for $PROFILE_EPOCHS x $PROFILE_BATCHES batches"
    python -u ccx_training.py \
        --profile \
        --profile_epochs "$PROFILE_EPOCHS" \
        --profile_batches "$PROFILE_BATCHES" \
        --profile_record_shapes \
        --profile_memory \
        --profile_trace profile_trace.json \
        --tag "$TAG" \
        --gpu 0 \
        --seed "$SEED" \
        --tensor_order "$TENSOR_ORDER" \
        --tensor_factors "$TENSOR_FACTOR" \
        --batch_size "$BATCH_SIZE" \
        --anidata_location "$ANI_DATA"
done
