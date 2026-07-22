#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONFIGS="${CONFIGS:-HOP:3:5}" \
RUN_TAG="${RUN_TAG:-autotune_l3n5_only}" \
ENERGY_STEM="${ENERGY_STEM:-triton_energy_b1024_autotune_l3n5_only}" \
FORCES_STEM="${FORCES_STEM:-triton_energy_forces_b1024_autotune_l3n5_only}" \
GPU_ID="${GPU_ID:-1}" \
BATCH_SIZE="${BATCH_SIZE:-1024}" \
WARMUPS="${WARMUPS:-2}" \
REPS="${REPS:-5}" \
ENABLE_AUTOTUNING_PRINT="${ENABLE_AUTOTUNING_PRINT:-1}" \
bash "$SCRIPT_DIR/run_poly2d_new_configs_gpu1.sh"
