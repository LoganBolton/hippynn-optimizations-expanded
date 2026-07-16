#!/bin/bash
set -euo pipefail

METHANE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$METHANE_DIR"

if squeue -h -u "$USER" -n methane_sweep | grep -q .; then
    echo "Refusing to submit: a methane_sweep job is already queued or running for $USER."
    echo "Wait for it to finish, or inspect it with: squeue -u $USER"
    exit 1
fi

echo "Submitting the two original l4/n3+n4 b256_fresh task groups with checkpoint resume enabled."

TASK_IDS="0 2 3 4 5 6 7" \
MODEL_SUFFIX="b256_fresh" \
RESUME=1 \
sbatch launchers/run_sweep.slurm

TASK_IDS="8 10 11 12 13 14 15" \
MODEL_SUFFIX="b256_fresh" \
RESUME=1 \
sbatch launchers/run_sweep.slurm
