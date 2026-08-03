#!/bin/bash
#SBATCH --job-name=methane_rmse_external
#SBATCH --output=./logs/submissions/%j_methane_rmse_external.out
#SBATCH --error=./logs/submissions/%j_methane_rmse_external.err
#SBATCH --time=1-00:00:00
#SBATCH --qos=long
#SBATCH --partition=volta-x86
#SBATCH --constraint='gpu_cc:7.0&gpu_count:2'
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G

# Evaluate checkpoints selected by minimum validation T-RMSE. Lightning only
# persisted periodic checkpoints every 10 epochs, so these are the closest
# available periodic checkpoints to the target epochs listed below.

set -euo pipefail

METHANE_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
PROJECT_ROOT="$(cd "$METHANE_DIR/../.." && pwd)"
PYTHON="$HOME/.conda/envs/hippynn-expanded-lmax/bin/python"
EVALUATOR="$METHANE_DIR/analysis/evaluate_current_best_checkpoints.py"
TEST_INDICES="$METHANE_DIR/test_sets/methane_common_test_n80000_seed20260720.npy"
OUTPUT_CSV="$METHANE_DIR/results/external_test/best_rmse_l3n4_l4n4_common_test.csv"

cd "$METHANE_DIR"
module load miniconda3
set +u
source "$HOME/.bashrc"
set -u
conda activate hippynn-expanded-lmax

export PYTHONPATH="$PROJECT_ROOT"
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" "$(dirname "$OUTPUT_CSV")" logs/submissions

echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $(hostname)"
echo "Selection:    minimum validation T-RMSE (nearest saved periodic checkpoint)"
echo "Test indices: $TEST_INDICES"
echo "Output CSV:   $OUTPUT_CSV"
nvidia-smi

"$PYTHON" "$EVALUATOR" \
  --device cuda:0 \
  --test-indices "$TEST_INDICES" \
  --output-csv "$OUTPUT_CSV" \
  --checkpoint-target-epochs 4761 2280 3393 2190 3461 1951 6547 6795 2170 5070 \
  --run-dirs \
    "$PROJECT_ROOT/examples/TEST_METHANE_MODEL_l3_n4_d1000000_seed0-lightning_ddp_a100_4gpu_fresh" \
    "$PROJECT_ROOT/examples/TEST_METHANE_MODEL_l3_n4_d1000000_seed1-lightning_ddp_a100x2" \
    "$PROJECT_ROOT/examples/TEST_METHANE_MODEL_l3_n4_d1000000_seed7-b256_fresh" \
    "$PROJECT_ROOT/examples/TEST_METHANE_MODEL_l3_n4_d1000000_seed42-lightning_ddp_v100" \
    "$PROJECT_ROOT/examples/TEST_METHANE_MODEL_l3_n4_d1000000_seed101-lightning_ddp_a100_2gpu_fresh" \
    "$PROJECT_ROOT/examples/TEST_METHANE_MODEL_l3_n4_d1000000_seed202-cn2_2gpu_lightning_fresh" \
    "$PROJECT_ROOT/examples/TEST_METHANE_MODEL_l4_n4_d1000000_seed0-lightning_ddp_v100_batchfix" \
    "$PROJECT_ROOT/examples/TEST_METHANE_MODEL_l4_n4_d1000000_seed7-lightning_ddp_v100" \
    "$PROJECT_ROOT/examples/TEST_METHANE_MODEL_l4_n4_d1000000_seed42-lightning_ddp_v100_batchfix" \
    "$PROJECT_ROOT/examples/TEST_METHANE_MODEL_l4_n4_d1000000_seed1776-lightning_ddp_v100_batchfix" \
  --checkpoint-paths \
    "$PROJECT_ROOT/examples/TEST_METHANE_MODEL_l3_n4_d1000000_seed0-lightning_ddp_a100_4gpu_fresh/lightning_checkpoints/periodic-epoch=4759.ckpt" \
    "$PROJECT_ROOT/examples/TEST_METHANE_MODEL_l3_n4_d1000000_seed1-lightning_ddp_a100x2/lightning_checkpoints/periodic-epoch=2279.ckpt" \
    "$PROJECT_ROOT/examples/TEST_METHANE_MODEL_l3_n4_d1000000_seed7-b256_fresh/checkpoint_epoch_3390.pt" \
    "$PROJECT_ROOT/examples/TEST_METHANE_MODEL_l3_n4_d1000000_seed42-lightning_ddp_v100/lightning_checkpoints/periodic-epoch=2189.ckpt" \
    "$PROJECT_ROOT/examples/TEST_METHANE_MODEL_l3_n4_d1000000_seed101-lightning_ddp_a100_2gpu_fresh/lightning_checkpoints/periodic-epoch=3459.ckpt" \
    "$PROJECT_ROOT/examples/TEST_METHANE_MODEL_l3_n4_d1000000_seed202-cn2_2gpu_lightning_fresh/lightning_checkpoints/periodic-epoch=1949.ckpt" \
    "$PROJECT_ROOT/examples/TEST_METHANE_MODEL_l4_n4_d1000000_seed0-lightning_ddp_v100_batchfix/lightning_checkpoints/periodic-epoch=6549.ckpt" \
    "$PROJECT_ROOT/examples/TEST_METHANE_MODEL_l4_n4_d1000000_seed7-lightning_ddp_v100/lightning_checkpoints/periodic-epoch=6799.ckpt" \
    "$PROJECT_ROOT/examples/TEST_METHANE_MODEL_l4_n4_d1000000_seed42-lightning_ddp_v100_batchfix/lightning_checkpoints/periodic-epoch=2169-v1.ckpt" \
    "$PROJECT_ROOT/examples/TEST_METHANE_MODEL_l4_n4_d1000000_seed1776-lightning_ddp_v100_batchfix/lightning_checkpoints/periodic-epoch=5069.ckpt"
