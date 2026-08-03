#!/bin/bash
#SBATCH --job-name=methane_rmse_points
#SBATCH --output=./logs/submissions/%j_methane_rmse_points.out
#SBATCH --error=./logs/submissions/%j_methane_rmse_points.err
#SBATCH --time=1-00:00:00
#SBATCH --qos=long
#SBATCH --partition=volta-x86
#SBATCH --constraint='gpu_cc:7.0&gpu_count:2'
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G

# Evaluate all selected L3N5, L3N4, L4N3, and L4N4 runs on the reserved
# random-common 80k set using the checkpoint nearest each run's minimum
# validation energy RMSE (T-RMSE). Save complete per-structure records.

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
OUTPUT_DIR="$METHANE_DIR/results/per_point/random_common_80k_best_valid_t_rmse"
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
L3N5_TARGET_EPOCHS=(3893 3604 3875 1370 5244 1681 5679 5271)
L3N5_CHECKPOINTS=(
  "${L3N5_RUN_DIRS[0]}/checkpoint_epoch_3890.pt"
  "${L3N5_RUN_DIRS[1]}/checkpoint_epoch_3600.pt"
  "${L3N5_RUN_DIRS[2]}/checkpoint_epoch_3870.pt"
  "${L3N5_RUN_DIRS[3]}/checkpoint_epoch_1370.pt"
  "${L3N5_RUN_DIRS[4]}/checkpoint_epoch_5240.pt"
  "${L3N5_RUN_DIRS[5]}/checkpoint_epoch_1680.pt"
  "${L3N5_RUN_DIRS[6]}/checkpoint_epoch_5680.pt"
  "${L3N5_RUN_DIRS[7]}/checkpoint_epoch_5270.pt"
)

CURRENT_RUN_DIRS=(
  "$PROJECT_ROOT/examples/TEST_METHANE_MODEL_l3_n4_d1000000_seed0-lightning_ddp_a100_4gpu_fresh"
  "$PROJECT_ROOT/examples/TEST_METHANE_MODEL_l3_n4_d1000000_seed1-lightning_ddp_a100x2"
  "$PROJECT_ROOT/examples/TEST_METHANE_MODEL_l3_n4_d1000000_seed7-b256_fresh"
  "$PROJECT_ROOT/examples/TEST_METHANE_MODEL_l3_n4_d1000000_seed42-lightning_ddp_v100"
  "$PROJECT_ROOT/examples/TEST_METHANE_MODEL_l3_n4_d1000000_seed101-lightning_ddp_a100_2gpu_fresh"
  "$PROJECT_ROOT/examples/TEST_METHANE_MODEL_l3_n4_d1000000_seed202-cn2_2gpu_lightning_fresh"
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
CURRENT_TARGET_EPOCHS=(4761 2280 3393 2190 3461 1951 6045 6432 5563 5018 5101 4203 6547 6795 2170 5070)
CURRENT_CHECKPOINTS=(
  "${CURRENT_RUN_DIRS[0]}/lightning_checkpoints/periodic-epoch=4759.ckpt"
  "${CURRENT_RUN_DIRS[1]}/lightning_checkpoints/periodic-epoch=2279.ckpt"
  "${CURRENT_RUN_DIRS[2]}/checkpoint_epoch_3390.pt"
  "${CURRENT_RUN_DIRS[3]}/lightning_checkpoints/periodic-epoch=2189.ckpt"
  "${CURRENT_RUN_DIRS[4]}/lightning_checkpoints/periodic-epoch=3459.ckpt"
  "${CURRENT_RUN_DIRS[5]}/lightning_checkpoints/periodic-epoch=1949.ckpt"
  "${CURRENT_RUN_DIRS[6]}/checkpoint_epoch_6040.pt"
  "${CURRENT_RUN_DIRS[7]}/checkpoint_epoch_6430.pt"
  "${CURRENT_RUN_DIRS[8]}/checkpoint_epoch_5560.pt"
  "${CURRENT_RUN_DIRS[9]}/checkpoint_epoch_5020.pt"
  "${CURRENT_RUN_DIRS[10]}/checkpoint_epoch_5100.pt"
  "${CURRENT_RUN_DIRS[11]}/checkpoint_epoch_4200.pt"
  "${CURRENT_RUN_DIRS[12]}/lightning_checkpoints/periodic-epoch=6549.ckpt"
  "${CURRENT_RUN_DIRS[13]}/lightning_checkpoints/periodic-epoch=6799.ckpt"
  "${CURRENT_RUN_DIRS[14]}/lightning_checkpoints/periodic-epoch=2169-v1.ckpt"
  "${CURRENT_RUN_DIRS[15]}/lightning_checkpoints/periodic-epoch=5069.ckpt"
)

cd "$METHANE_DIR"
module load miniconda3
set +u
source "$HOME/.bashrc"
set -u
conda activate hippynn-expanded-lmax

export PYTHONNOUSERSITE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" "$OUTPUT_DIR" logs/submissions

echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $(hostname)"
echo "Selection:    minimum validation energy T-RMSE"
echo "Test indices: $TEST_INDICES"
echo "Output dir:   $OUTPUT_DIR"
nvidia-smi

# L3N5 structures require the compatible HipHop implementation from their
# training worktree. All other structures use this worktree.
PYTHONPATH="$L3N5_CODE_ROOT" "$PYTHON" "$EVALUATOR" \
  --device cuda:0 \
  --test-indices "$TEST_INDICES" \
  --per-point-only \
  --per-point-output-dir "$OUTPUT_DIR" \
  --output-csv "$OUTPUT_DIR/l3n5_aggregate_metrics.csv" \
  --checkpoint-target-epochs "${L3N5_TARGET_EPOCHS[@]}" \
  --run-dirs "${L3N5_RUN_DIRS[@]}" \
  --checkpoint-paths "${L3N5_CHECKPOINTS[@]}"

PYTHONPATH="$PROJECT_ROOT" "$PYTHON" "$EVALUATOR" \
  --device cuda:0 \
  --test-indices "$TEST_INDICES" \
  --per-point-only \
  --per-point-output-dir "$OUTPUT_DIR" \
  --output-csv "$OUTPUT_DIR/l3n4_l4n3_l4n4_aggregate_metrics.csv" \
  --checkpoint-target-epochs "${CURRENT_TARGET_EPOCHS[@]}" \
  --run-dirs "${CURRENT_RUN_DIRS[@]}" \
  --checkpoint-paths "${CURRENT_CHECKPOINTS[@]}"
