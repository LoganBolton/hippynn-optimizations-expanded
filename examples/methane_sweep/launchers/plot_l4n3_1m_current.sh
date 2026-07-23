#!/bin/bash
set -euo pipefail

METHANE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$METHANE_DIR"

TMP_LOG_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_LOG_DIR"' EXIT
mkdir -p "$TMP_LOG_DIR/runs"

# Current b256_fresh logs, including completed and active jobs.
find logs/runs -type f -name '*_methane_*.out' -exec ln -s "$(pwd)/{}" "$TMP_LOG_DIR/runs/" \;

python plotting/plot_methane_sweep_logs.py \
  --log-dir "$TMP_LOG_DIR" \
  --log-pattern 'runs/*_methane_*.out' \
  --output-dir logs/plots_l4n3_1m_all_current \
  --sweep-config configs/methane-l4-n3-1m-8seeds.yml \
  --data-size 1000000 \
  --hiphop-l-max 4 \
  --hiphop-n-max 3 \
  --requested-train-batch-size 256 \
  --label-mode seed-config
