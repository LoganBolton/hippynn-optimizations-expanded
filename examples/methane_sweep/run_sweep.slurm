#!/bin/bash
#SBATCH --job-name=methane_l4_n4
#SBATCH --output=./logs/%A_%a_methane_l4_n4.out
#SBATCH --time=10:00:00            # Job duration
#SBATCH --partition=volta-x86      # GPU partition name
#SBATCH --constraint=gpu1_model:Tesla_V100-PCIE-32GB
#SBATCH --cpus-per-task=16         # CPUs per task
#SBATCH --mem=64G                  # RAM
#SBATCH --array=0-3                # Run repeated methane jobs

# Environment
SWEEPID="$1"
CONDA_ENV="hippynn"

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters"
    exit 1
fi

# Getting the environment ready
echo "Starting training job on $(hostname) with job ID: $SLURM_JOB_ID, partition: $SLURM_JOB_PARTITION"
echo "Running in directory: $(pwd)"
nvidia-smi
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
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$(pwd)" wandb agent --entity "karella" --project "hippynn" "$SWEEPID"
