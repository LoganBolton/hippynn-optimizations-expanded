#!/bin/bash
#SBATCH --job-name=methane_per_point
#SBATCH --output=./logs/submissions/%j_methane_per_point.out
#SBATCH --error=./logs/submissions/%j_methane_per_point.err
#SBATCH --time=1-00:00:00
#SBATCH --qos=long
#SBATCH --partition=volta-x86
#SBATCH --constraint='gpu_cc:7.0&gpu_count:2'
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G

# Preserve per-structure predictions and errors for the same current L3N5 and
# L3N4 checkpoints on both the sequential +80k and reserved random test sets.

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
RESULT_ROOT="$METHANE_DIR/results/per_point"
L3N5_ROOT="$HOME/Github/hippynn-lmax3-nmax5/examples"
L3N5_SPEEDUP_ROOT="$HOME/Github/hippynn-optimization-speedup-l3n5/examples"
L3N5_CODE_ROOT="$HOME/Github/hippynn-optimization-speedup-l3n5"

L3N5_RUN_DIRS=(
  "$L3N5_SPEEDUP_ROOT/TEST_METHANE_MODEL_l3_n5_d1000000_seed42-b256_fresh_1m"
  "$L3N5_ROOT/TEST_METHANE_MODEL_l3_n5_d1000000_seed1776-b256_fresh_1m"
  "$L3N5_SPEEDUP_ROOT/TEST_METHANE_MODEL_l3_n5_d1000000_seed7-b256_fresh_1m"
  "$L3N5_ROOT/TEST_METHANE_MODEL_l3_n5_d1000000_seed0-b256_fresh_1m"
  "$L3N5_ROOT/TEST_METHANE_MODEL_l3_n5_d1000000_seed1-b256_fresh_1m"
  "$L3N5_ROOT/TEST_METHANE_MODEL_l3_n5_d1000000_seed2-b256_fresh_1m"
  "$L3N5_ROOT/TEST_METHANE_MODEL_l3_n5_d1000000_seed3-b256_fresh_1m"
  "$L3N5_ROOT/TEST_METHANE_MODEL_l3_n5_d1000000_seed4-b256_fresh_1m"
)

L3N4_RUN_DIRS=(
  "$PROJECT_ROOT/examples/TEST_METHANE_MODEL_l3_n4_d1000000_seed0-lightning_ddp_a100_4gpu_fresh"
  "$PROJECT_ROOT/examples/TEST_METHANE_MODEL_l3_n4_d1000000_seed1-lightning_ddp_a100x2"
  "$PROJECT_ROOT/examples/TEST_METHANE_MODEL_l3_n4_d1000000_seed7-b256_fresh"
  "$PROJECT_ROOT/examples/TEST_METHANE_MODEL_l3_n4_d1000000_seed42-lightning_ddp_v100"
  "$PROJECT_ROOT/examples/TEST_METHANE_MODEL_l3_n4_d1000000_seed101-lightning_ddp_a100_2gpu_fresh"
  "$PROJECT_ROOT/examples/TEST_METHANE_MODEL_l3_n4_d1000000_seed202-cn2_2gpu_lightning_fresh"
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
mkdir -p "$MPLCONFIGDIR" "$RESULT_ROOT/random_common_80k" "$RESULT_ROOT/sequential_plus_80k" logs/submissions

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
nvidia-smi

echo "Evaluating reserved random-common 80k set"
PYTHONPATH="$L3N5_CODE_ROOT" "$PYTHON" "$EVALUATOR" \
  --device cuda:0 \
  --checkpoint-kind best \
  --test-indices "$TEST_INDICES" \
  --per-point-only \
  --per-point-output-dir "$RESULT_ROOT/random_common_80k" \
  --output-csv "$RESULT_ROOT/random_common_80k/l3n5_aggregate_metrics.csv" \
  --run-dirs "${L3N5_RUN_DIRS[@]}"

PYTHONPATH="$PROJECT_ROOT" "$PYTHON" "$EVALUATOR" \
  --device cuda:0 \
  --checkpoint-kind best \
  --test-indices "$TEST_INDICES" \
  --per-point-only \
  --per-point-output-dir "$RESULT_ROOT/random_common_80k" \
  --output-csv "$RESULT_ROOT/random_common_80k/l3n4_aggregate_metrics.csv" \
  --run-dirs "${L3N4_RUN_DIRS[@]}"

echo "Evaluating sequential +80k set"
PYTHONPATH="$L3N5_CODE_ROOT" "$PYTHON" "$EVALUATOR" \
  --device cuda:0 \
  --checkpoint-kind best \
  --test-set-size 80000 \
  --per-point-only \
  --per-point-output-dir "$RESULT_ROOT/sequential_plus_80k" \
  --output-csv "$RESULT_ROOT/sequential_plus_80k/l3n5_aggregate_metrics.csv" \
  --run-dirs "${L3N5_RUN_DIRS[@]}"

PYTHONPATH="$PROJECT_ROOT" "$PYTHON" "$EVALUATOR" \
  --device cuda:0 \
  --checkpoint-kind best \
  --test-set-size 80000 \
  --per-point-only \
  --per-point-output-dir "$RESULT_ROOT/sequential_plus_80k" \
  --output-csv "$RESULT_ROOT/sequential_plus_80k/l3n4_aggregate_metrics.csv" \
  --run-dirs "${L3N4_RUN_DIRS[@]}"
