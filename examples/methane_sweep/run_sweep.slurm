#!/bin/bash
#SBATCH --job-name=methane_sweep
#SBATCH --output=./logs/%A_%a_methane_sweep.out
#SBATCH --error=./logs/%A_%a_methane_sweep.err
#SBATCH --time=10:00:00            # Job duration (adjust as needed)
#SBATCH --partition=volta-x86      # GPU partition name (check with: sinfo)
#SBATCH --constraint=gpu1_model:Tesla_V100-PCIE-32GB  # Optional: specific GPU
#SBATCH --cpus-per-task=16         # CPUs per task
#SBATCH --mem=64G                  # RAM
#SBATCH --array=0-3                # Run 4 local sweep jobs in parallel

# Usage: sbatch run_sweep.slurm
#
# This is a local/offline replacement for `wandb agent`. W&B offline mode logs
# each run locally, but it cannot pull sweep jobs from the online W&B backend.

TRAINING_SCRIPT="examples/methane_sweep/methane_configurations.py"
CONDA_ENV="hippynn-expanded-lmax"       # Your conda environment name
WANDB_MODE="offline"
HIPHOP_L_MAX=4
HIPHOP_N_MAX=4
SEEDS=(2025 2026 2027 2028)

if [ "$#" -ne 0 ]; then
    echo "Error: this local sweep does not take a W&B sweep ID"
    echo "Usage: sbatch run_sweep.slurm"
    exit 1
fi

set -e 

echo "Job ID:        $SLURM_JOB_ID"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

echo "Array Task:    $TASK_ID / $SLURM_ARRAY_TASK_COUNT"
echo "Node:          $(hostname)"
echo "Partition:     $SLURM_JOB_PARTITION"
echo "Working dir:   $(pwd)"
echo "Start time:    $(date)"
echo "=========================================="
echo ""

echo "Checking GPU availability..."
nvidia-smi
echo ""

# Load conda
echo "Loading conda environment..."
module load miniconda3

# Check if conda environment exists
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

SEED="${SEEDS[$TASK_ID]}"
RUN_NAME="methane-l${HIPHOP_L_MAX}-n${HIPHOP_N_MAX}-seed${SEED}"

if [ -z "$SEED" ]; then
    echo "No local sweep seed configured for array task $TASK_ID"
    exit 1
fi

echo "Starting local WandB run"
echo "Mode:    $WANDB_MODE"
echo "Run:     $RUN_NAME"
echo "Seed:    $SEED"
echo "l_max:   $HIPHOP_L_MAX"
echo "n_max:   $HIPHOP_N_MAX"
echo "=========================================="
echo ""

# Set environment variables
export PYTHONPATH="$(pwd)"
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE="$WANDB_MODE"
export WANDB_DIR="${WANDB_DIR:-$(pwd)/wandb}"
mkdir -p "$WANDB_DIR"

set +e
python "$TRAINING_SCRIPT" \
    --seed "$SEED" \
    --hiphop_l_max "$HIPHOP_L_MAX" \
    --hiphop_n_max "$HIPHOP_N_MAX" \
    --run_name "$RUN_NAME" \
    --wandb_mode "$WANDB_MODE"
EXIT_CODE=$?
set -e

echo "Local sweep task finished at: $(date)"
echo "Exit code: $EXIT_CODE"

exit $EXIT_CODE
