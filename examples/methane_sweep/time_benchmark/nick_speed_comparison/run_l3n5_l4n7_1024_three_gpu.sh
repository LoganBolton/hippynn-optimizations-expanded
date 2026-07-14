#!/bin/bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="/vast/home/logan_bolton/.conda/envs/hippynn-expanded-lmax/bin/python"

mkdir -p "$BENCHMARK_DIR/results/nick" "$BENCHMARK_DIR/logs/nick"
cd "$SCRIPT_DIR"

# l3n5 energy + forces on GPU 1
(
    export PYTHONNOUSERSITE=1
    export HOP_CONFIGS="3:5"
    unset INCLUDE_L4_CONFIGS
    export INCLUDE_FORCES=True
    export BATCH_SIZE=1024
    export HIPPYNN_USE_CUSTOM_KERNELS=triton
    export HIPPYNN_USE_POLYNOMIAL_INVARIANTS=True
    export HIPPYNN_USE_TENSOR_MESSAGE_PASSING=True
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export CUDA_VISIBLE_DEVICES=1
    export SPEED_EVAL_OUTPUT="$BENCHMARK_DIR/results/nick/nick_exact_triton_l3_n5_energy_forces_b1024_2warmups_speed_eval.pt"
    "$PYTHON" -u evaluation_script.py > "$BENCHMARK_DIR/logs/nick/triton_l3_n5_energy_forces_b1024.log" 2>&1
) &
l3n5_forces_pid=$!

# l4n7 energy only on GPU 2
(
    export PYTHONNOUSERSITE=1
    export HOP_CONFIGS="4:7"
    unset INCLUDE_L4_CONFIGS
    export INCLUDE_FORCES=False
    export BATCH_SIZE=1024
    export HIPPYNN_USE_CUSTOM_KERNELS=triton
    export HIPPYNN_USE_POLYNOMIAL_INVARIANTS=True
    export HIPPYNN_USE_TENSOR_MESSAGE_PASSING=True
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export CUDA_VISIBLE_DEVICES=2
    export SPEED_EVAL_OUTPUT="$BENCHMARK_DIR/results/nick/nick_exact_triton_l4_n7_energy_only_b1024_2warmups_speed_eval.pt"
    "$PYTHON" -u evaluation_script.py > "$BENCHMARK_DIR/logs/nick/triton_l4_n7_energy_only_b1024.log" 2>&1
) &
l4n7_energy_pid=$!

# l4n7 energy + forces on GPU 3
(
    export PYTHONNOUSERSITE=1
    export HOP_CONFIGS="4:7"
    unset INCLUDE_L4_CONFIGS
    export INCLUDE_FORCES=True
    export BATCH_SIZE=1024
    export HIPPYNN_USE_CUSTOM_KERNELS=triton
    export HIPPYNN_USE_POLYNOMIAL_INVARIANTS=True
    export HIPPYNN_USE_TENSOR_MESSAGE_PASSING=True
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export CUDA_VISIBLE_DEVICES=3
    export SPEED_EVAL_OUTPUT="$BENCHMARK_DIR/results/nick/nick_exact_triton_l4_n7_energy_forces_b1024_2warmups_speed_eval.pt"
    "$PYTHON" -u evaluation_script.py > "$BENCHMARK_DIR/logs/nick/triton_l4_n7_energy_forces_b1024.log" 2>&1
) &
l4n7_forces_pid=$!

echo "l3n5 energy+forces PID: $l3n5_forces_pid (GPU 1)"
echo "l4n7 energy-only PID:   $l4n7_energy_pid (GPU 2)"
echo "l4n7 energy+forces PID: $l4n7_forces_pid (GPU 3)"
echo "Logs: $BENCHMARK_DIR/logs/nick/triton_l3_n5_energy_forces_b1024.log"
echo "      $BENCHMARK_DIR/logs/nick/triton_l4_n7_energy_only_b1024.log"
echo "      $BENCHMARK_DIR/logs/nick/triton_l4_n7_energy_forces_b1024.log"

wait "$l3n5_forces_pid"
l3n5_forces_status=$?
wait "$l4n7_energy_pid"
l4n7_energy_status=$?
wait "$l4n7_forces_pid"
l4n7_forces_status=$?

echo "l3n5 energy+forces exit status: $l3n5_forces_status"
echo "l4n7 energy-only exit status:   $l4n7_energy_status"
echo "l4n7 energy+forces exit status: $l4n7_forces_status"

if (( l3n5_forces_status != 0 || l4n7_energy_status != 0 || l4n7_forces_status != 0 )); then
    exit 1
fi
