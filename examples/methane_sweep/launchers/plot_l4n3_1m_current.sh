#!/bin/bash
set -euo pipefail

METHANE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$METHANE_DIR"

TMP_LOG_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_LOG_DIR"' EXIT

mkdir -p "$TMP_LOG_DIR/runs"

# Include the original, completed, and currently active methane logs.
find logs/runs -type f -name '*_methane_*.out' \
    -exec ln -s "$(pwd)/{}" "$TMP_LOG_DIR/runs/" \;

# Include both separately launched seed-202 continuation segments.  They fill
# the gap between the original run (ending near epoch 1,793) and the current
# worker log (resuming near epoch 4,021).  Distinct link names retain task id 5
# so the parser merges and de-duplicates the epoch records.
for continuation_spec in \
    "17146291_seed202_continue.out early" \
    "17146291_seed202_continue_detached.out detached"; do
    read -r continuation tag <<< "$continuation_spec"
    RESUME_LOG="logs/runs/17146291/$continuation"
    RESUME_LINK="$TMP_LOG_DIR/runs/17146291_r1_5_w2_methane_sweep_${tag}.out"
    if [[ -f "$RESUME_LOG" ]]; then
        ln -s "$(pwd)/$RESUME_LOG" "$RESUME_LINK"
        echo "Including seed-202 continuation: $RESUME_LOG"
    else
        echo "warning: seed-202 continuation log not found: $RESUME_LOG" >&2
    fi
done

# Keep this diagnostic concise; it makes missing continuation links obvious
# when the launcher is run interactively.
echo "Seed-202 plot links:"
find "$TMP_LOG_DIR/runs" -maxdepth 1 -type l -name '*_r1_5_w2_methane_sweep_*.out' -printf '  %f -> %l\n'

python plotting/plot_methane_sweep_logs.py \
    --log-dir "$TMP_LOG_DIR" \
    --log-pattern 'runs/*.out' \
    --output-dir logs/plots_l4n3_1m_all_current \
    --sweep-config configs/methane-l4-n3-1m-8seeds.yml \
    --data-size 1000000 \
    --hiphop-l-max 4 \
    --hiphop-n-max 3 \
    --label-mode seed-config
