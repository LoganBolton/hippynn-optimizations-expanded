#!/bin/bash
#SBATCH --job-name=methane_l4n4_seed7_resume
#SBATCH --output=./logs/submissions/%j_methane_l4n4_seed7_resume.out
#SBATCH --error=./logs/submissions/%j_methane_l4n4_seed7_resume.err
#SBATCH --time=2-00:00:00
#SBATCH --qos=long
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --open-mode=append
#SBATCH --partition=volta-x86
#SBATCH --constraint='gpu_cc:7.0&gpu_count:2'
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G

# Resume the exact checkpoint lineage currently trained by job 17152172:
#   TEST_METHANE_MODEL_l4_n4_d1000000_seed7-lightning_ddp_v100
#
# Submit from examples/methane_sweep:
#   sbatch launchers/resume_task10_lightning_ddp_v100.slurm
#
# RESET_EARLY_STOPPING_PATIENCE remains 0 so the checkpoint's current patience
# state is preserved. Override it at submission only if a fresh patience window
# is intentional:
#   RESET_EARLY_STOPPING_PATIENCE=1 \
#     sbatch launchers/resume_task10_lightning_ddp_v100.slurm

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHANE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# The generic launcher uses SLURM_SUBMIT_DIR as the methane working directory.
# Pin it here so this dedicated script is safe to submit from another directory.
export SLURM_SUBMIT_DIR="$METHANE_DIR"
export SWEEP_CONFIG="$METHANE_DIR/configs/methane-l4-n4.yml"
export TASK_ID=10
export NUM_GPUS=2
export MODEL_SUFFIX=lightning_ddp_v100
export RESUME=1
export RESET_EARLY_STOPPING_PATIENCE="${RESET_EARLY_STOPPING_PATIENCE:-0}"
export RESET_BATCH_SCHEDULE_ON_RESUME=0
export CHECKPOINT_EVERY=10
export WANDB_MODE=offline
export POLYNOMIAL_CACHE_DEBUG=1

cd "$METHANE_DIR"
exec bash "$SCRIPT_DIR/run_single_task_lightning_ddp_v100.slurm"
