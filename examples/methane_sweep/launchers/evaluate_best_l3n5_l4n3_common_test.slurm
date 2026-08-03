#!/bin/bash
#SBATCH --job-name=methane_l3n5_l4n3_ext
#SBATCH --output=./logs/submissions/%j_methane_l3n5_l4n3_ext.out
#SBATCH --error=./logs/submissions/%j_methane_l3n5_l4n3_ext.err
#SBATCH --time=1-00:00:00
#SBATCH --qos=long
#SBATCH --partition=volta-x86
#SBATCH --constraint='gpu_cc:7.0&gpu_count:2'
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G

# Evaluate best-validation-MAE checkpoints on the reserved common external set:
#   L3N5: seeds 42, 1776, 7, 0, 1, 2, 3, 4
#   L4N3: seeds 42, 1776, 7, 101, 202, 303

set -euo pipefail

METHANE_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
PROJECT_ROOT="$(cd "$METHANE_DIR/../.." && pwd)"
PYTHON="$HOME/.conda/envs/hippynn-expanded-lmax/bin/python"
EVALUATOR="$METHANE_DIR/analysis/evaluate_current_best_checkpoints.py"
TEST_INDICES="$METHANE_DIR/test_sets/methane_common_test_n80000_seed20260720.npy"
OUTPUT_CSV="$METHANE_DIR/results/external_test/best_l3n5_l4n3_common_test.csv"
L3N5_ROOT="$HOME/Github/hippynn-lmax3-nmax5/examples"
L3N5_SPEEDUP_ROOT="$HOME/Github/hippynn-optimization-speedup-l3n5/examples"

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
echo "Selection:    best validation T-MAE checkpoint"
echo "Test indices: $TEST_INDICES"
echo "Output CSV:   $OUTPUT_CSV"
nvidia-smi

"$PYTHON" "$EVALUATOR" \
  --device cuda:0 \
  --checkpoint-kind best \
  --test-indices "$TEST_INDICES" \
  --output-csv "$OUTPUT_CSV" \
  --run-dirs \
    "$L3N5_SPEEDUP_ROOT/TEST_METHANE_MODEL_l3_n5_d1000000_seed42-b256_fresh_1m" \
    "$L3N5_ROOT/TEST_METHANE_MODEL_l3_n5_d1000000_seed1776-b256_fresh_1m" \
    "$L3N5_SPEEDUP_ROOT/TEST_METHANE_MODEL_l3_n5_d1000000_seed7-b256_fresh_1m" \
    "$L3N5_ROOT/TEST_METHANE_MODEL_l3_n5_d1000000_seed0-b256_fresh_1m" \
    "$L3N5_ROOT/TEST_METHANE_MODEL_l3_n5_d1000000_seed1-b256_fresh_1m" \
    "$L3N5_ROOT/TEST_METHANE_MODEL_l3_n5_d1000000_seed2-b256_fresh_1m" \
    "$L3N5_ROOT/TEST_METHANE_MODEL_l3_n5_d1000000_seed3-b256_fresh_1m" \
    "$L3N5_ROOT/TEST_METHANE_MODEL_l3_n5_d1000000_seed4-b256_fresh_1m" \
    "$PROJECT_ROOT/examples/TEST_METHANE_MODEL_l4_n3_d1000000_seed42-b256_fresh" \
    "$PROJECT_ROOT/examples/TEST_METHANE_MODEL_l4_n3_d1000000_seed1776-b256_fresh" \
    "$PROJECT_ROOT/examples/TEST_METHANE_MODEL_l4_n3_d1000000_seed7-b256_fresh" \
    "$PROJECT_ROOT/examples/TEST_METHANE_MODEL_l4_n3_d1000000_seed101-b256_fresh" \
    "$PROJECT_ROOT/examples/TEST_METHANE_MODEL_l4_n3_d1000000_seed202-b256_fresh" \
    "$PROJECT_ROOT/examples/TEST_METHANE_MODEL_l4_n3_d1000000_seed303-b256_fresh"
