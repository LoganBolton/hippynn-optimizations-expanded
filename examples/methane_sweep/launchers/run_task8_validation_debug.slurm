#!/bin/bash
#SBATCH --job-name=task8_valid_debug
#SBATCH --partition=shared-gpu-ampere
#SBATCH --qos=long
#SBATCH --account=y2020-bf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --constraint=gpu_cc:8.0
#SBATCH --output=logs/%j_task8_valid_debug.out
#SBATCH --error=logs/%j_task8_valid_debug.err

set -euo pipefail

METHANE_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO_ROOT="$(cd "$METHANE_DIR/../.." && pwd)"
RUN_DIR="$REPO_ROOT/examples/TEST_METHANE_MODEL_l4_n3_d1000000_seed7-b256_fresh"
REFERENCE="$REPO_ROOT/examples/TEST_METHANE_MODEL_l4_n3_d100000_seed42-b256_fresh/validation_debug/validation_errors_epoch_3000_ranked.csv"

PYTHON="/vast/home/logan_bolton/.conda/envs/hippynn-expanded-lmax/bin/python"
export PYTHONNOUSERSITE=1

LATEST_CHECKPOINT="$(find "$RUN_DIR" -maxdepth 1 -type f -name 'checkpoint_epoch_*.pt' -printf '%f\n' | sort -V | tail -1)"
LATEST_EPOCH="${LATEST_CHECKPOINT#checkpoint_epoch_}"
LATEST_EPOCH="${LATEST_EPOCH%.pt}"
EPOCH="${EPOCH_OVERRIDE:-$LATEST_EPOCH}"
OUT_DIR="$RUN_DIR/validation_debug"

echo "Evaluating task 8 checkpoint epoch $EPOCH"
"$PYTHON" -u "$METHANE_DIR/analysis/debug_validation_errors.py" \
    --seed 7 \
    --hiphop_l_max 4 \
    --hiphop_n_max 3 \
    --data_size 1000000 \
    --test_set_size 80000 \
    --eval_batch_size 4096 \
    --device cuda:0 \
    --reuse_targets \
    --epochs "$EPOCH" \
    --run_dir "$RUN_DIR" \
    --out_dir "$OUT_DIR"

"$PYTHON" -u "$METHANE_DIR/analysis/compare_validation_culprits.py" \
    --reference "$REFERENCE" \
    --candidate "$OUT_DIR/validation_errors_epoch_${EPOCH}_ranked.csv" \
    --top-n 100 \
    --out "$OUT_DIR/culprit_overlap_epoch_${EPOCH}_vs_100k_epoch_3000.txt"

echo "Diagnostic complete for epoch $EPOCH"
