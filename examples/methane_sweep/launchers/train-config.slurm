#!/bin/bash
#SBATCH --job-name=methane_l4_n4
#SBATCH --output=./logs/submissions/%j_methane_l4_n4.out
#SBATCH --time=10:00:00            # Job duration
#SBATCH --partition=volta-x86      # GPU partition name
#SBATCH --gres=gpu:1               # Number of GPUs
#SBATCH --cpus-per-task=16         # CPUs per task
#SBATCH --mem=64G                  # RAM

METHANE_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
TRAINING_SCRIPT="$METHANE_DIR/training/methane_configurations.py"
CONDA_ENV="hippynn-expanded-lmax"
SEED=2025
HIPHOP_L_MAX=4
HIPHOP_N_MAX=4
WANDB_MODE="offline"
RUN_NAME="methane-l${HIPHOP_L_MAX}-n${HIPHOP_N_MAX}-seed${SEED}"

# Set single NVIDIA GPU
echo "Starting training job on $(hostname) with job ID: $SLURM_JOB_ID, partition: $SLURM_JOB_PARTITION"
echo "Running in directory: $(pwd)"
# nvidia-smi
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

# Run the training script
echo "Running training script. Python=`which python`"
# Set PYTHONPATH to the project root
export PYTHONPATH=$(pwd)
export WANDB_MODE="$WANDB_MODE"
export WANDB_DIR="${WANDB_DIR:-$(pwd)/wandb}"
mkdir -p "$WANDB_DIR"
echo "Methane training settings: seed=$SEED, l_max=$HIPHOP_L_MAX, n_max=$HIPHOP_N_MAX"
CUDA_VISIBLE_DEVICES=0 python "$TRAINING_SCRIPT" --seed "$SEED" --hiphop_l_max "$HIPHOP_L_MAX" --hiphop_n_max "$HIPHOP_N_MAX" --run_name "$RUN_NAME" --wandb_mode "$WANDB_MODE"
