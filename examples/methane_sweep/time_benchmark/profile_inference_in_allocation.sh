#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_ID="${JOB_ID:-${1:-}}"
GPU_ID="${GPU_ID:-0}"
LOG_FILE="${LOG_FILE:-$SCRIPT_DIR/profile_inference_step_${JOB_ID:-unknown}_gpu${GPU_ID}.log}"

if [ -z "$JOB_ID" ]; then
    echo "Usage: JOB_ID=<running-job-id> GPU_ID=<idle-physical-gpu> bash profile_inference_in_allocation.sh" >&2
    exit 1
fi

echo "Starting exact inference profiler in allocation $JOB_ID on physical GPU $GPU_ID"
echo "Step output: $LOG_FILE"

exec srun \
    --jobid="$JOB_ID" \
    --overlap \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=4 \
    --open-mode=append \
    --output="$LOG_FILE" \
    --error="$LOG_FILE" \
    env GPU_ID="$GPU_ID" bash "$SCRIPT_DIR/profile_inference_l4n4.sh"
