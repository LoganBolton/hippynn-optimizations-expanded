#!/bin/bash -l
#SBATCH --job-name=poly_triton_l4n4
#SBATCH --output=slurm_logs/poly_triton_%j.out
#SBATCH --error=slurm_logs/poly_triton_%j.err
#SBATCH --partition=shared-gpu-ampere
#SBATCH --qos=long
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --constraint=gpu1_model:NVIDIA_A100-PCIE-40GB

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONDA_ENV="${CONDA_ENV:-hippynn-expanded-lmax}"

module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export PYTHONPATH="$PROJECT_ROOT"
export PYTHONNOUSERSITE=1
cd "$PROJECT_ROOT"

mkdir -p slurm_logs benchmarks/results
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
python -u benchmarks/benchmark_poly_triton.py \
    --output "benchmarks/results/poly_triton_a100_${SLURM_JOB_ID}.json"
