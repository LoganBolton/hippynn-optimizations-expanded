#!/bin/bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="/vast/home/logan_bolton/.conda/envs/hippynn-expanded-lmax/bin/python"

mkdir -p "$BENCHMARK_DIR/results/nick" "$BENCHMARK_DIR/logs/nick"

cd "$SCRIPT_DIR"

(
    export HOP_CONFIGS="3:5"
    unset INCLUDE_L4_CONFIGS
    export INCLUDE_FORCES=False
    export BATCH_SIZE=1024
    export HIPPYNN_USE_CUSTOM_KERNELS=triton
    export HIPPYNN_USE_POLYNOMIAL_INVARIANTS=True
    export HIPPYNN_USE_TENSOR_MESSAGE_PASSING=True
    export CUDA_VISIBLE_DEVICES=0
    export SPEED_EVAL_OUTPUT="$BENCHMARK_DIR/results/nick/nick_exact_triton_l3_n5_energy_only_b1024_2warmups_speed_eval.pt"
    "$PYTHON" -u evaluation_script.py > "$BENCHMARK_DIR/logs/nick/triton_l3_n5_energy_only_b1024.log" 2>&1
) &
triton_pid=$!

echo "Triton l3n5 PID: $triton_pid (GPU 0)"
echo "Log: $BENCHMARK_DIR/logs/nick/triton_l3_n5_energy_only_b1024.log"

wait "$triton_pid"
triton_status=$?

echo "Triton l3n5 exit status: $triton_status"

if (( triton_status != 0 )); then
    exit 1
fi
