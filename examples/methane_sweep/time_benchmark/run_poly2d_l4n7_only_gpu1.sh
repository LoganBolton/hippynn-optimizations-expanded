#!/bin/bash
set -euo pipefail

# Runs the new Triton polynomial-evaluation kernel on exactly the missing
# (l_max, n_max) = (4, 7) benchmark point, using the same settings as the
# existing poly_2d_real_data batch-1024 runs.
#
# Safe for worktrees/branches: every path is resolved from this script's
# location rather than the caller's current working directory.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$SCRIPT_DIR/run_poly2d_new_configs_gpu1.sh"
RESULT_DIR="$SCRIPT_DIR/results/poly_2d_real_data"
LOG_DIR="$SCRIPT_DIR/logs/poly_2d_real_data"

GPU_ID="${GPU_ID:-1}"
EXPECTED_GPU_NAME="${EXPECTED_GPU_NAME:-NVIDIA A100-PCIE-40GB}"
SKIP_GPU_CHECK="${SKIP_GPU_CHECK:-0}"

BASE_ENERGY_PT="${MERGE_BASE_ENERGY_PT:-$RESULT_DIR/triton_energy_b1024_autotune_full_l3n5_refresh.pt}"
BASE_FORCES_PT="${MERGE_BASE_FORCES_PT:-$RESULT_DIR/triton_energy_forces_b1024_autotune_full_l3n5_refresh.pt}"

if [ ! -f "$RUNNER" ]; then
    echo "Missing runner: $RUNNER"
    exit 1
fi

mkdir -p "$RESULT_DIR" "$LOG_DIR"

if [ "$SKIP_GPU_CHECK" != "1" ]; then
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia-smi not found; cannot validate GPU selection."
        exit 1
    fi

    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$GPU_ID" | head -n 1 | xargs)"
    if [ -z "$GPU_NAME" ]; then
        echo "Could not query GPU $GPU_ID via nvidia-smi."
        exit 1
    fi

    if [ "$GPU_NAME" != "$EXPECTED_GPU_NAME" ]; then
        echo "Refusing to run on GPU $GPU_ID."
        echo "Expected: $EXPECTED_GPU_NAME"
        echo "Found:    $GPU_NAME"
        echo "Override with EXPECTED_GPU_NAME=... or SKIP_GPU_CHECK=1 if intentional."
        exit 1
    fi

    GPU_PIDS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU_ID" 2>/dev/null | grep -v 'No running processes found' | sed '/^$/d' || true)"
    if [ -n "$GPU_PIDS" ]; then
        echo "Refusing to run because GPU $GPU_ID already has compute processes:"
        echo "$GPU_PIDS"
        echo "Pick another GPU or rerun with SKIP_GPU_CHECK=1 if you really want to share."
        exit 1
    fi
fi

if [ ! -f "$BASE_ENERGY_PT" ]; then
    echo "Missing merge-base energy file: $BASE_ENERGY_PT"
    exit 1
fi
if [ ! -f "$BASE_FORCES_PT" ]; then
    echo "Missing merge-base forces file: $BASE_FORCES_PT"
    exit 1
fi

echo "Running exact-match poly Triton benchmark for HOP:4:7"
echo "GPU_ID=$GPU_ID"
echo "EXPECTED_GPU_NAME=$EXPECTED_GPU_NAME"
echo "Merge base energy: $BASE_ENERGY_PT"
echo "Merge base forces: $BASE_FORCES_PT"
echo

CONFIGS="HOP:4:7" \
RUN_TAG="autotune_l4n7_only" \
ENERGY_STEM="triton_energy_b1024_autotune_l4n7_only" \
FORCES_STEM="triton_energy_forces_b1024_autotune_l4n7_only" \
MERGE_BASE_ENERGY_PT="$BASE_ENERGY_PT" \
MERGE_BASE_FORCES_PT="$BASE_FORCES_PT" \
MERGED_ENERGY_STEM="triton_energy_b1024_autotune_full_l4n7_refresh" \
MERGED_FORCES_STEM="triton_energy_forces_b1024_autotune_full_l4n7_refresh" \
CONDA_ENV="${CONDA_ENV:-hippynn-expanded-lmax}" \
GPU_ID="$GPU_ID" \
BATCH_SIZE="1024" \
WARMUPS="2" \
REPS="5" \
SEED="0" \
SPLIT_SEED="0" \
ENABLE_AUTOTUNING_PRINT="1" \
AUTOTUNING_LOG_DIR="$LOG_DIR" \
bash "$RUNNER"

python - <<'PY' "$RESULT_DIR"
import json
import statistics
import sys
from pathlib import Path

result_dir = Path(sys.argv[1])
checks = [
    result_dir / "triton_energy_b1024_autotune_l4n7_only.json",
    result_dir / "triton_energy_forces_b1024_autotune_l4n7_only.json",
    result_dir / "triton_energy_b1024_autotune_full_l4n7_refresh.json",
    result_dir / "triton_energy_forces_b1024_autotune_full_l4n7_refresh.json",
]
expected = "(('tensor_factors', 7), ('tensor_model', 'HOP'), ('tensor_order', 4), ('batch_size', 1024))"

for path in checks:
    data = json.loads(path.read_text())
    if expected not in data["metrics"]:
        raise SystemExit(f"Missing expected l4n7 key in {path}")
    timings = data["metrics"][expected]
    print(f"{path.name}: median={statistics.median(timings):.6f}s reps={len(timings)}")

print("Validated HOP:4:7 in both standalone and merged result files.")
PY

echo
echo "Standalone outputs:"
echo "  $RESULT_DIR/triton_energy_b1024_autotune_l4n7_only.pt"
echo "  $RESULT_DIR/triton_energy_forces_b1024_autotune_l4n7_only.pt"
echo "Merged outputs:"
echo "  $RESULT_DIR/triton_energy_b1024_autotune_full_l4n7_refresh.pt"
echo "  $RESULT_DIR/triton_energy_forces_b1024_autotune_full_l4n7_refresh.pt"
echo
echo "Plot with:"
echo "python $SCRIPT_DIR/plot_energy_force_breakdown.py \\
  --upstream_energy_pt $SCRIPT_DIR/results/nick/nick_exact_upstream_energy_only_b1024_2warmups_speed_eval.pt \\
  --upstream_energy_forces_pt $SCRIPT_DIR/results/nick/nick_exact_upstream_energy_forces_b1024_2warmups_speed_eval.pt \\
  --triton_energy_pt $SCRIPT_DIR/results/nick/nick_exact_triton_all_energy_only_b1024_2warmups_speed_eval.pt \\
  --extra_triton_energy_pt $SCRIPT_DIR/results/nick/nick_exact_triton_l3_n5_energy_only_b1024_2warmups_speed_eval.pt \\
  --extra_triton_energy_pt $SCRIPT_DIR/results/nick/nick_exact_triton_l4_n7_energy_only_b1024_2warmups_speed_eval.pt \\
  --triton_energy_forces_pt $SCRIPT_DIR/results/nick/nick_exact_triton_all_energy_forces_b1024_2warmups_speed_eval.pt \\
  --extra_triton_energy_forces_pt $SCRIPT_DIR/results/nick/nick_exact_triton_l3_n5_energy_forces_b1024_2warmups_speed_eval.pt \\
  --extra_triton_energy_forces_pt $SCRIPT_DIR/results/nick/nick_exact_triton_l4_n7_energy_forces_b1024_2warmups_speed_eval.pt \\
  --optimized_energy_pt $RESULT_DIR/triton_energy_b1024_autotune_full_l4n7_refresh.pt \\
  --optimized_energy_forces_pt $RESULT_DIR/triton_energy_forces_b1024_autotune_full_l4n7_refresh.pt \\
  --optimized_label 'New 2D polynomial Triton' \\
  --batch_size 1024 \\
  --output $SCRIPT_DIR/plots/energy_force_breakdown_b1024_with_poly_2d_green_autotune_full_l4n7_refresh.png"
