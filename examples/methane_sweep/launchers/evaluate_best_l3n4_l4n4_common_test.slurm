#!/bin/bash
#SBATCH --job-name=methane_external_test
#SBATCH --output=./logs/submissions/%j_methane_external_test.out
#SBATCH --error=./logs/submissions/%j_methane_external_test.err
#SBATCH --time=1-00:00:00
#SBATCH --qos=long
#SBATCH --partition=volta-x86
#SBATCH --constraint='gpu_cc:7.0&gpu_count:2'
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G

# Evaluate the best validation checkpoint for one run per seed:
#   L3N4: seeds 0, 1, 7, 42, 101, 202
#   L4N4: seeds 0, 7, 42, 1776
# The test frames come from the reserved common external set and are disjoint
# from the training/validation and legacy sequential held-out slices.

set -euo pipefail

METHANE_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
PROJECT_ROOT="$(cd "$METHANE_DIR/../.." && pwd)"
PYTHON="$HOME/.conda/envs/hippynn-expanded-lmax/bin/python"
EVALUATOR="$METHANE_DIR/analysis/evaluate_current_best_checkpoints.py"
TEST_INDICES="$METHANE_DIR/test_sets/methane_common_test_n80000_seed20260720.npy"
OUTPUT_CSV="$METHANE_DIR/results/external_test/best_l3n4_l4n4_common_test.csv"

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
echo "Test indices: $TEST_INDICES"
echo "Output CSV:   $OUTPUT_CSV"
nvidia-smi

"$PYTHON" "$EVALUATOR" \
  --device cuda:0 \
  --checkpoint-kind best \
  --test-indices "$TEST_INDICES" \
  --output-csv "$OUTPUT_CSV" \
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
    "$PROJECT_ROOT/examples/TEST_METHANE_MODEL_l4_n4_d1000000_seed1776-lightning_ddp_v100_batchfix"
