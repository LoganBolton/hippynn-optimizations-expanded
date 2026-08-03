#!/bin/bash
set -euo pipefail

METHANE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$METHANE_DIR"


# finish l4n3
# ./launchers/plot_l4n3_1m_current.sh

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
  --extra-log-pattern 'runs/17150716/*.out' \
  --extra-log-pattern 'runs/17152172/*.out' \
  --output-dir logs/plots_lightning_ddp_v100 \
  --sweep-config configs/methane-l4-n4.yml \
  --data-size 1000000 \
  --hiphop-l-max 4 \
  --hiphop-n-max 4 \
  --label-mode task

# The l3n4 logs have descriptive or generic filenames rather than the strict
# array-task naming convention expected by the parser. Give each persisted
# training log a conventional name.
#
# Use label-mode=task here so we can:
#   - merge true continuations/resumes by reusing the same synthetic task id
#   - keep distinct fresh runs separate even if they share the same seed
L3N4_LOG_DIR="$(mktemp -d)"
trap 'rm -rf "$L3N4_LOG_DIR"' EXIT
mkdir -p "$L3N4_LOG_DIR/runs"
ln -s "$METHANE_DIR/logs/runs/17146276/17146276_l3n4_seed7_fresh_restart.out" \
  "$L3N4_LOG_DIR/runs/17146276_r0_7_methane_sweep.out"
if [[ -f "$METHANE_DIR/logs/runs/17146291/17146291_l3n4_seed7_live_continue.out" ]]; then
  ln -s "$METHANE_DIR/logs/runs/17146291/17146291_l3n4_seed7_live_continue.out" \
    "$L3N4_LOG_DIR/runs/17146291_r0_7_methane_sweep_seed7-live.out"
fi
if [[ -f "$METHANE_DIR/logs/runs/17146291/17146291_l3n4_seed7_gpu3_resume.out" ]]; then
  ln -s "$METHANE_DIR/logs/runs/17146291/17146291_l3n4_seed7_gpu3_resume.out" \
    "$L3N4_LOG_DIR/runs/17146291_r0_7_methane_sweep_seed7-gpu3-resume.out"
fi
if [[ -f "$METHANE_DIR/logs/runs/17149246/17149246_r0_1_methane_single_4gpu.out" ]]; then
  # This direct four-GPU continuation belongs to the same seed-7 lineage.
  ln -s "$METHANE_DIR/logs/runs/17149246/17149246_r0_1_methane_single_4gpu.out" \
    "$L3N4_LOG_DIR/runs/17149246_r0_7_methane_sweep_l3n4-seed7-a100x4.out"
fi
ln -s "$METHANE_DIR/logs/runs/17148178/17148178_r0_2_methane_l3n4_redstone4_lightning.out" \
  "$L3N4_LOG_DIR/runs/17148178_r0_42_methane_lightning_l3n4-redstone4.out"
if [[ -f "$METHANE_DIR/logs/runs/17146291/17146291_l3n4_seed42_gpu3_resume.out" ]]; then
  ln -s "$METHANE_DIR/logs/runs/17146291/17146291_l3n4_seed42_gpu3_resume.out" \
    "$L3N4_LOG_DIR/runs/17146291_r0_42_methane_lightning_l3n4-seed42-resume.out"
fi
if [[ -f "$METHANE_DIR/logs/runs/17153996/17153996_r42_2_methane_lightning_2gpu.out" ]]; then
  ln -s "$METHANE_DIR/logs/runs/17153996/17153996_r42_2_methane_lightning_2gpu.out" \
    "$L3N4_LOG_DIR/runs/17153996_r0_42_methane_lightning_l3n4-seed42-v100-overlap.out"
fi
ln -s "$METHANE_DIR/logs/runs/17146291/17146291_r0_2_methane_lightning_2gpu.out" \
  "$L3N4_LOG_DIR/runs/17146291_r0_101_methane_lightning_l3n4-seed101.out"
if [[ -f "$METHANE_DIR/logs/runs/17153996/17153996_r0_2_methane_lightning_2gpu.out" ]]; then
  ln -s "$METHANE_DIR/logs/runs/17153996/17153996_r0_2_methane_lightning_2gpu.out" \
    "$L3N4_LOG_DIR/runs/17153996_r0_101_methane_lightning_l3n4-seed101-v100-resume.out"
fi
if [[ -f "$METHANE_DIR/logs/runs/17149896/17149896_r0_0_methane_lightning_4gpu.out" ]]; then
  ln -s "$METHANE_DIR/logs/runs/17149896/17149896_r0_0_methane_lightning_4gpu.out" \
    "$L3N4_LOG_DIR/runs/17149896_r0_0_methane_lightning_l3n4-seed0-a100x4.out"
fi
if [[ -f "$METHANE_DIR/logs/runs/17154612/17154612_r0_0_methane_lightning_2gpu.out" ]]; then
  ln -s "$METHANE_DIR/logs/runs/17154612/17154612_r0_0_methane_lightning_2gpu.out" \
    "$L3N4_LOG_DIR/runs/17154612_r0_0_methane_lightning_l3n4-seed0-a100x2-resume.out"
fi
if [[ -f "$METHANE_DIR/logs/runs/17150844/17150844_r0_1_w0_methane_sweep.out" ]]; then
  ln -s "$METHANE_DIR/logs/runs/17150844/17150844_r0_1_w0_methane_sweep.out" \
    "$L3N4_LOG_DIR/runs/17150844_r0_1_methane_sweep_l3n4-seed1.out"
fi
if [[ -f "$METHANE_DIR/logs/runs/17150844/17150844_l3n4_seed1_lightning_2gpu_step.out" ]]; then
  ln -s "$METHANE_DIR/logs/runs/17150844/17150844_l3n4_seed1_lightning_2gpu_step.out" \
    "$L3N4_LOG_DIR/runs/17150844_r0_1002_methane_sweep_l3n4-seed1-a100x2.out"
fi
if [[ -f "$METHANE_DIR/logs/runs/17148565/17148565_overlap_l3n4_seed1.out" ]]; then
  # Continue the direct-launcher 2xGPU seed-1 lineage under the same synthetic
  # task id so the resume extends, rather than duplicates, that curve.
  ln -s "$METHANE_DIR/logs/runs/17148565/17148565_overlap_l3n4_seed1.out" \
    "$L3N4_LOG_DIR/runs/17148565_r0_1002_methane_lightning_l3n4-seed1-v100-overlap.out"
fi
if [[ -f "$METHANE_DIR/logs/runs/17150844/17150844_l3n4_seed1_single_gpu3_step.out" ]]; then
  ln -s "$METHANE_DIR/logs/runs/17150844/17150844_l3n4_seed1_single_gpu3_step.out" \
    "$L3N4_LOG_DIR/runs/17150844_r0_1001_methane_sweep_l3n4-seed1-single-a100.out"
fi
if [[ -f "$METHANE_DIR/logs/runs/17150844/17150844_seed202_2gpu_step.out" ]]; then
  ln -s "$METHANE_DIR/logs/runs/17150844/17150844_seed202_2gpu_step.out" \
    "$L3N4_LOG_DIR/runs/17150844_r0_3_methane_lightning_l3n4-seed202-cn2-a100x2.out"
fi
if [[ -f "$METHANE_DIR/logs/runs/17148568/17148568_overlap_l3n4_seed202.out" ]]; then
  # Seed 202 is task 3 in methane-l3-n4-1m-seeds0-7.yml; keep that task id so
  # the Volta overlap resume is merged into the same plotted run.
  ln -s "$METHANE_DIR/logs/runs/17148568/17148568_overlap_l3n4_seed202.out" \
    "$L3N4_LOG_DIR/runs/17148568_r0_3_methane_lightning_l3n4-seed202-v100-overlap.out"
fi
if [[ -f "$METHANE_DIR/slurm-17154725.out" ]]; then
  ln -s "$METHANE_DIR/slurm-17154725.out" \
    "$L3N4_LOG_DIR/runs/17154725_r0_42_methane_lightning_l3n4-seed42-redstone4.out"
fi

python plotting/plot_methane_sweep_logs.py \
  --log-dir "$L3N4_LOG_DIR" \
  --log-pattern 'runs/*.out' \
  --output-dir logs/plots_l3n4_1m_current \
  --sweep-config configs/methane-l3-n4-1m-seeds0-7.yml \
  --data-size 1000000 \
  --hiphop-l-max 3 \
  --hiphop-n-max 4 \
  --label-mode task

# Rebuild every input plot directory so active logs are reflected in the
# averaged comparison rather than using a stale metrics.csv.

# not actively running so juts use the cached
# python plotting/plot_methane_sweep_logs.py \
#   --log-dir logs \
#   --log-pattern 'combined_l3n5_1m_current/*.out' \
#   --output-dir logs/plots_l3n5_1m_updated \
#   --sweep-config configs/methane-l3-n5-1m-seeds.yml \
#   --data-size 1000000 \
#   --hiphop-l-max 3 \
#   --hiphop-n-max 5 \
#   --label-mode task

# Average the available seeds within each architecture, then compare the
# architecture-level mean ± standard deviation curves.
AVERAGED_OUTPUT_DIR="logs/plots_averaged_l3n5_l4n3_l4n4"

python plotting/plot_averaged_methane_sweep_metrics.py \
  --input-dir logs/plots_l4n3_1m_all_current \
  --input-dir logs/plots_l3n5_1m_updated \
  --input-dir logs/plots_lightning_ddp_v100 \
  --input-dir logs/plots_l3n4_1m_current \
  --output-dir "$AVERAGED_OUTPUT_DIR"

# Also write a companion averaged comparison with the l4n3 1M seed-7 outlier
# excluded.
python plotting/plot_averaged_methane_sweep_metrics.py \
  --input-dir logs/plots_l4n3_1m_all_current \
  --input-dir logs/plots_l3n5_1m_updated \
  --input-dir logs/plots_lightning_ddp_v100 \
  --input-dir logs/plots_l3n4_1m_current \
  --exclude-run 'l=4 n=3 d=1M seed=7' \
  --output-dir "$AVERAGED_OUTPUT_DIR/without_seed7"
