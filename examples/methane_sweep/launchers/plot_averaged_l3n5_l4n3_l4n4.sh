#!/bin/bash
set -euo pipefail

METHANE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$METHANE_DIR"

# Rebuild every input plot directory so active logs are reflected in the
# averaged comparison rather than using a stale metrics.csv.
python plotting/plot_methane_sweep_logs.py \
  --log-dir logs \
  --log-pattern 'combined_l3n5_1m_current/*.out' \
  --output-dir logs/plots_l3n5_1m_updated \
  --sweep-config configs/methane-l3-n5-1m-seeds.yml \
  --data-size 1000000 \
  --hiphop-l-max 3 \
  --hiphop-n-max 5 \
  --label-mode task

./launchers/plot_l4n3_1m_current.sh

# Bugged batch size: intermediate pre-rollback L4N4 runs.
# --extra-log-pattern 'runs/17148141/*.out'
# --extra-log-pattern 'runs/17148142/*.out'
# --extra-log-pattern 'runs/17148143/*.out'
# --extra-log-pattern 'runs/17148144/*.out'
# --extra-log-pattern 'runs/17148163/*.out'
# --extra-log-pattern 'runs/17148165/*.out'
# --extra-log-pattern 'runs/17148175/*.out'
# --extra-log-pattern 'runs/17148164/*.out'
# --extra-log-pattern 'runs/17148192/*.out'
# --extra-log-pattern 'runs/17148166/*.out'

# Post-rollback L4N4 resumes on Volta.
python plotting/plot_methane_sweep_logs.py \
  --log-dir logs \
  --log-pattern 'runs/17145567/*.out' \
  --extra-log-pattern 'runs/17145568/*.out' \
  --extra-log-pattern 'runs/17145569/*.out' \
  --extra-log-pattern 'runs/17145570/*.out' \
  --extra-log-pattern 'runs/17148565/*.out' \
  --extra-log-pattern 'runs/17148566/*.out' \
  --extra-log-pattern 'runs/17148567/*.out' \
  --extra-log-pattern 'runs/17148568/*.out' \
  --output-dir logs/plots_lightning_ddp_v100 \
  --sweep-config configs/methane-l4-n4.yml \
  --data-size 1000000 \
  --hiphop-l-max 4 \
  --hiphop-n-max 4 \
  --label-mode task

# The l3n4 logs have descriptive filenames rather than the array-task naming
# convention expected by the parser. Give them temporary conventional names
# so they are included; embedded run metadata still takes precedence when
# available (as it does for the Lightning run).
L3N4_LOG_DIR="$(mktemp -d)"
trap 'rm -rf "$L3N4_LOG_DIR"' EXIT
mkdir -p "$L3N4_LOG_DIR/runs"
ln -s "$METHANE_DIR/logs/runs/17146276/17146276_l3n4_seed7_fresh_restart.out" \
  "$L3N4_LOG_DIR/runs/17146276_r0_1_methane_l3n5.out"
ln -s "$METHANE_DIR/logs/runs/17148178/17148178_r0_2_methane_l3n4_redstone4_lightning.out" \
  "$L3N4_LOG_DIR/runs/17148178_r0_2_methane_lightning_l3n4-redstone4.out"

python plotting/plot_methane_sweep_logs.py \
  --log-dir "$L3N4_LOG_DIR" \
  --log-pattern 'runs/*.out' \
  --output-dir logs/plots_l3n4_1m_current \
  --sweep-config configs/methane-l3-n4-1m-seeds0-7.yml \
  --data-size 1000000 \
  --hiphop-l-max 3 \
  --hiphop-n-max 4 \
  --label-mode seed-config

# Average the available seeds within each architecture, then compare the
# architecture-level mean ± standard deviation curves.
python plotting/plot_averaged_methane_sweep_metrics.py \
  --input-dir logs/plots_l4n3_1m_all_current \
  --input-dir logs/plots_l3n5_1m_updated \
  --input-dir logs/plots_lightning_ddp_v100 \
  --input-dir logs/plots_l3n4_1m_current \
  --output-dir logs/plots_averaged_l3n5_l4n3_l4n4
