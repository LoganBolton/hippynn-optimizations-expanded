#!/bin/bash
#SBATCH --job-name=methane_l4_per_point
#SBATCH --output=./logs/submissions/%j_methane_l4_per_point.out
#SBATCH --error=./logs/submissions/%j_methane_l4_per_point.err
#SBATCH --time=1-00:00:00
#SBATCH --qos=long
#SBATCH --partition=volta-x86
#SBATCH --constraint='gpu_cc:7.0&gpu_count:2'
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G

# Save per-structure predictions/errors for every selected L4N3 and L4N4 run
# on the same reserved random-common 80k external test set.

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -d "$SUBMIT_DIR/analysis" && -d "$SUBMIT_DIR/test_sets" ]]; then
  METHANE_DIR="$SUBMIT_DIR"
elif [[ -d "$SUBMIT_DIR/../analysis" && -d "$SUBMIT_DIR/../test_sets" ]]; then
  METHANE_DIR="$(cd "$SUBMIT_DIR/.." && pwd)"
else
  echo "Could not locate examples/methane_sweep from submission directory: $SUBMIT_DIR" >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$METHANE_DIR/../.." && pwd)"
PYTHON="$HOME/.conda/envs/hippynn-expanded-lmax/bin/python"
EVALUATOR="$METHANE_DIR/analysis/evaluate_current_best_checkpoints.py"
TEST_INDICES="$METHANE_DIR/test_sets/methane_common_test_n80000_seed20260720.npy"
OUTPUT_DIR="$METHANE_DIR/results/per_point/random_common_80k"

RUN_DIRS=(
  "$PROJECT_ROOT/examples/TEST_METHANE_MODEL_l4_n3_d1000000_seed42-b256_fresh"
  "$PROJECT_ROOT/examples/TEST_METHANE_MODEL_l4_n3_d1000000_seed1776-b256_fresh"
  "$PROJECT_ROOT/examples/TEST_METHANE_MODEL_l4_n3_d1000000_seed7-b256_fresh"
  "$PROJECT_ROOT/examples/TEST_METHANE_MODEL_l4_n3_d1000000_seed101-b256_fresh"
  "$PROJECT_ROOT/examples/TEST_METHANE_MODEL_l4_n3_d1000000_seed202-b256_fresh"
  "$PROJECT_ROOT/examples/TEST_METHANE_MODEL_l4_n3_d1000000_seed303-b256_fresh"
  "$PROJECT_ROOT/examples/TEST_METHANE_MODEL_l4_n4_d1000000_seed0-lightning_ddp_v100_batchfix"
  "$PROJECT_ROOT/examples/TEST_METHANE_MODEL_l4_n4_d1000000_seed7-lightning_ddp_v100"
  "$PROJECT_ROOT/examples/TEST_METHANE_MODEL_l4_n4_d1000000_seed42-lightning_ddp_v100_batchfix"
  "$PROJECT_ROOT/examples/TEST_METHANE_MODEL_l4_n4_d1000000_seed1776-lightning_ddp_v100_batchfix"
)

cd "$METHANE_DIR"
module load miniconda3
set +u
source "$HOME/.bashrc"
set -u
conda activate hippynn-expanded-lmax

export PYTHONPATH="$PROJECT_ROOT"
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" "$OUTPUT_DIR" logs/submissions

echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $(hostname)"
echo "Selection:    best validation T-MAE checkpoint"
echo "Test indices: $TEST_INDICES"
echo "Output dir:   $OUTPUT_DIR"
nvidia-smi

"$PYTHON" "$EVALUATOR" \
  --device cuda:0 \
  --checkpoint-kind best \
  --test-indices "$TEST_INDICES" \
  --per-point-only \
  --per-point-output-dir "$OUTPUT_DIR" \
  --output-csv "$OUTPUT_DIR/l4n3_l4n4_aggregate_metrics.csv" \
  --run-dirs "${RUN_DIRS[@]}"
