#!/bin/bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="/vast/home/logan_bolton/.conda/envs/hippynn-expanded-lmax/bin/python"

mkdir -p "$BENCHMARK_DIR/results/nick" "$BENCHMARK_DIR/logs/nick"

cd "$SCRIPT_DIR"

(
    unset HOP_CONFIGS
    export INCLUDE_L4_CONFIGS=True
    export INCLUDE_FORCES=False
    export BATCH_SIZE=1024
    export HIPPYNN_USE_CUSTOM_KERNELS=triton
    export HIPPYNN_USE_POLYNOMIAL_INVARIANTS=True
    export HIPPYNN_USE_TENSOR_MESSAGE_PASSING=True
    export CUDA_VISIBLE_DEVICES=0
    export SPEED_EVAL_OUTPUT="$BENCHMARK_DIR/results/nick/nick_exact_triton_all_energy_only_b1024_2warmups_speed_eval.pt"
    "$PYTHON" -u evaluation_script.py > "$BENCHMARK_DIR/logs/nick/triton_all_energy_only_b1024.log" 2>&1
) &
triton_pid=$!

(
    unset HOP_CONFIGS
    unset INCLUDE_L4_CONFIGS
    export INCLUDE_FORCES=False
    export BATCH_SIZE=1024
    export HIPPYNN_USE_CUSTOM_KERNELS=auto
    export HIPPYNN_USE_POLYNOMIAL_INVARIANTS=False
    export HIPPYNN_USE_TENSOR_MESSAGE_PASSING=False
    export CUDA_VISIBLE_DEVICES=1
    export SPEED_EVAL_OUTPUT="$BENCHMARK_DIR/results/nick/nick_exact_upstream_energy_only_b1024_2warmups_speed_eval.pt"
    "$PYTHON" -u evaluation_script.py > "$BENCHMARK_DIR/logs/nick/upstream_energy_only_b1024.log" 2>&1
) &
upstream_pid=$!

echo "Triton all-config PID: $triton_pid (GPU 0)"
echo "Upstream PID:          $upstream_pid (GPU 1)"
echo "Logs: $BENCHMARK_DIR/logs/nick/triton_all_energy_only_b1024.log"
echo "      $BENCHMARK_DIR/logs/nick/upstream_energy_only_b1024.log"

wait "$triton_pid"
triton_status=$?
wait "$upstream_pid"
upstream_status=$?

echo "Triton all-config exit status: $triton_status"
echo "Upstream exit status:          $upstream_status"

if (( triton_status != 0 || upstream_status != 0 )); then
    exit 1
fi
