#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/nick_speed_comparison/evaluation_script_triton_random_init.py"
RESULT_DIR="$SCRIPT_DIR/results/poly_2d_real_data"

CONDA_ENV="${CONDA_ENV:-hippynn-expanded-lmax}"
GPU_ID="${GPU_ID:-1}"
ANI_DATA="${ANI_DATA:-$PROJECT_ROOT/datasets/ani1x-release.h5}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
WARMUPS="${WARMUPS:-2}"
REPS="${REPS:-5}"
SEED="${SEED:-0}"
SPLIT_SEED="${SPLIT_SEED:-0}"
CONFIGS_STRING="${CONFIGS:-HOP:3:4 HOP:3:5 HOP:4:3 HOP:4:4}"
ENABLE_AUTOTUNING_PRINT="${ENABLE_AUTOTUNING_PRINT:-1}"
AUTOTUNING_LOG_DIR="${AUTOTUNING_LOG_DIR:-$SCRIPT_DIR/logs/poly_2d_real_data}"
RUN_TAG="${RUN_TAG:-autotune_rerun}"
ENERGY_STEM="${ENERGY_STEM:-triton_energy_b${BATCH_SIZE}_${RUN_TAG}}"
FORCES_STEM="${FORCES_STEM:-triton_energy_forces_b${BATCH_SIZE}_${RUN_TAG}}"
MERGE_BASE_ENERGY_PT="${MERGE_BASE_ENERGY_PT:-}"
MERGE_BASE_ENERGY_JSON="${MERGE_BASE_ENERGY_JSON:-}"
MERGE_BASE_FORCES_PT="${MERGE_BASE_FORCES_PT:-}"
MERGE_BASE_FORCES_JSON="${MERGE_BASE_FORCES_JSON:-}"
MERGED_ENERGY_STEM="${MERGED_ENERGY_STEM:-triton_energy_b${BATCH_SIZE}_${RUN_TAG}_merged}"
MERGED_FORCES_STEM="${MERGED_FORCES_STEM:-triton_energy_forces_b${BATCH_SIZE}_${RUN_TAG}_merged}"

if [ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]; then
    set +u
    source ~/.bashrc
    conda activate "$CONDA_ENV"
    set -u
fi

mkdir -p "$RESULT_DIR" "$AUTOTUNING_LOG_DIR"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT"
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [ "$ENABLE_AUTOTUNING_PRINT" = "1" ]; then
    export TRITON_PRINT_AUTOTUNING=1
else
    unset TRITON_PRINT_AUTOTUNING
fi

ENERGY_LOG="$AUTOTUNING_LOG_DIR/${ENERGY_STEM}.log"
FORCES_LOG="$AUTOTUNING_LOG_DIR/${FORCES_STEM}.log"

if [ -n "$MERGE_BASE_ENERGY_PT" ] && [ -z "$MERGE_BASE_ENERGY_JSON" ]; then
    MERGE_BASE_ENERGY_JSON="${MERGE_BASE_ENERGY_PT%.pt}.json"
fi
if [ -n "$MERGE_BASE_FORCES_PT" ] && [ -z "$MERGE_BASE_FORCES_JSON" ]; then
    MERGE_BASE_FORCES_JSON="${MERGE_BASE_FORCES_PT%.pt}.json"
fi

read -r -a CONFIG_ARRAY <<< "$CONFIGS_STRING"
if [ "${#CONFIG_ARRAY[@]}" -eq 0 ]; then
    echo "No configs provided. Set CONFIGS like: CONFIGS='HOP:3:4 HOP:3:5 HOP:4:3 HOP:4:4'"
    exit 1
fi

echo "============================================"
echo "Benchmarking optimized Triton configs on GPU $GPU_ID"
echo "Environment: $CONDA_ENV"
echo "ANI data:    $ANI_DATA"
echo "Batch size:  $BATCH_SIZE"
echo "Warmups:     $WARMUPS"
echo "Reps:        $REPS"
echo "Seed:        $SEED"
echo "Split seed:  $SPLIT_SEED"
echo "Configs:     ${CONFIG_ARRAY[*]}"
echo "Architecture: n_features=128, n_atom_layers=3, n_interactions=2"
echo "Outputs:"
echo "  $RESULT_DIR/${ENERGY_STEM}.pt"
echo "  $RESULT_DIR/${FORCES_STEM}.pt"
if [ -n "$MERGE_BASE_ENERGY_PT" ] || [ -n "$MERGE_BASE_FORCES_PT" ]; then
    echo "Merge base:"
    echo "  $MERGE_BASE_ENERGY_PT"
    echo "  $MERGE_BASE_FORCES_PT"
    echo "Merged outputs:"
    echo "  $RESULT_DIR/${MERGED_ENERGY_STEM}.pt"
    echo "  $RESULT_DIR/${MERGED_FORCES_STEM}.pt"
fi
echo "Logs:"
echo "  $ENERGY_LOG"
echo "  $FORCES_LOG"
if [ "$ENABLE_AUTOTUNING_PRINT" = "1" ]; then
    echo "TRITON_PRINT_AUTOTUNING=1"
fi
echo "============================================"
echo

CUDA_VISIBLE_DEVICES="$GPU_ID" python -u "$PYTHON_SCRIPT" \
    --gpu 0 \
    --seed "$SEED" \
    --split_seed "$SPLIT_SEED" \
    --anidata_location "$ANI_DATA" \
    --configs "${CONFIG_ARRAY[@]}" \
    --batch_sizes "$BATCH_SIZE" \
    --warmups "$WARMUPS" \
    --reps "$REPS" \
    --n_interactions 2 \
    --n_atom_layers 3 \
    --n_features 128 \
    --n_sensitivities 20 \
    --cutoff_distance 6.5 \
    --lower_cutoff 0.75 \
    --kernel_mode triton \
    --use_triton_message_passing \
    --no-include_forces \
    --output "$RESULT_DIR/${ENERGY_STEM}.pt" \
    --output_json "$RESULT_DIR/${ENERGY_STEM}.json" \
    2>&1 | tee "$ENERGY_LOG"

echo
echo "✓ Energy-only benchmark complete"
echo

CUDA_VISIBLE_DEVICES="$GPU_ID" python -u "$PYTHON_SCRIPT" \
    --gpu 0 \
    --seed "$SEED" \
    --split_seed "$SPLIT_SEED" \
    --anidata_location "$ANI_DATA" \
    --configs "${CONFIG_ARRAY[@]}" \
    --batch_sizes "$BATCH_SIZE" \
    --warmups "$WARMUPS" \
    --reps "$REPS" \
    --n_interactions 2 \
    --n_atom_layers 3 \
    --n_features 128 \
    --n_sensitivities 20 \
    --cutoff_distance 6.5 \
    --lower_cutoff 0.75 \
    --kernel_mode triton \
    --use_triton_message_passing \
    --include_forces \
    --output "$RESULT_DIR/${FORCES_STEM}.pt" \
    --output_json "$RESULT_DIR/${FORCES_STEM}.json" \
    2>&1 | tee "$FORCES_LOG"

echo
echo "✓ Energy+forces benchmark complete"
echo

PLOT_ENERGY_PT="$RESULT_DIR/${ENERGY_STEM}.pt"
PLOT_FORCES_PT="$RESULT_DIR/${FORCES_STEM}.pt"

if [ -n "$MERGE_BASE_ENERGY_PT" ] || [ -n "$MERGE_BASE_FORCES_PT" ]; then
    python - "$MERGE_BASE_ENERGY_PT" "$RESULT_DIR/${ENERGY_STEM}.pt" "$RESULT_DIR/${MERGED_ENERGY_STEM}.pt" \
              "$MERGE_BASE_ENERGY_JSON" "$RESULT_DIR/${ENERGY_STEM}.json" "$RESULT_DIR/${MERGED_ENERGY_STEM}.json" \
              "$MERGE_BASE_FORCES_PT" "$RESULT_DIR/${FORCES_STEM}.pt" "$RESULT_DIR/${MERGED_FORCES_STEM}.pt" \
              "$MERGE_BASE_FORCES_JSON" "$RESULT_DIR/${FORCES_STEM}.json" "$RESULT_DIR/${MERGED_FORCES_STEM}.json" <<'PY'
import json
import sys
from pathlib import Path

import torch


def merge_pt(base_path, new_path, out_path):
    base = torch.load(base_path, map_location="cpu", weights_only=False)
    new = torch.load(new_path, map_location="cpu", weights_only=False)
    merged = dict(base)
    merged["metrics"] = dict(base.get("metrics", {}))
    merged["metrics"].update(new.get("metrics", {}))
    if "metrics_jsonable" in base or "metrics_jsonable" in new:
        merged["metrics_jsonable"] = dict(base.get("metrics_jsonable", {}))
        merged["metrics_jsonable"].update(new.get("metrics_jsonable", {}))
    if "parameter_counts" in base or "parameter_counts" in new:
        merged["parameter_counts"] = dict(base.get("parameter_counts", {}))
        merged["parameter_counts"].update(new.get("parameter_counts", {}))
    for field in ("batch_sizes", "reps", "warmups", "seed", "split_seed", "include_forces", "network_parameters", "kernel_info"):
        if field in new:
            merged[field] = new[field]
    torch.save(merged, out_path)
    print(f"Wrote {out_path}")


def merge_json(base_path, new_path, out_path):
    base = json.loads(Path(base_path).read_text())
    new = json.loads(Path(new_path).read_text())
    merged = dict(base)
    merged["metrics"] = dict(base.get("metrics", {}))
    merged["metrics"].update(new.get("metrics", {}))
    if "metrics_jsonable" in base or "metrics_jsonable" in new:
        merged["metrics_jsonable"] = dict(base.get("metrics_jsonable", {}))
        merged["metrics_jsonable"].update(new.get("metrics_jsonable", {}))
    if "parameter_counts" in base or "parameter_counts" in new:
        merged["parameter_counts"] = dict(base.get("parameter_counts", {}))
        merged["parameter_counts"].update(new.get("parameter_counts", {}))
    for field in ("batch_sizes", "reps", "warmups", "seed", "split_seed", "include_forces", "network_parameters", "kernel_info"):
        if field in new:
            merged[field] = new[field]
    Path(out_path).write_text(json.dumps(merged, indent=2) + "\n")
    print(f"Wrote {out_path}")

(
    base_energy_pt,
    new_energy_pt,
    out_energy_pt,
    base_energy_json,
    new_energy_json,
    out_energy_json,
    base_forces_pt,
    new_forces_pt,
    out_forces_pt,
    base_forces_json,
    new_forces_json,
    out_forces_json,
) = sys.argv[1:]

merge_pt(base_energy_pt, new_energy_pt, out_energy_pt)
merge_json(base_energy_json, new_energy_json, out_energy_json)
merge_pt(base_forces_pt, new_forces_pt, out_forces_pt)
merge_json(base_forces_json, new_forces_json, out_forces_json)
PY

    PLOT_ENERGY_PT="$RESULT_DIR/${MERGED_ENERGY_STEM}.pt"
    PLOT_FORCES_PT="$RESULT_DIR/${MERGED_FORCES_STEM}.pt"
    echo
    echo "✓ Merged benchmark files"
    echo
fi

echo "Plot with:"
echo "python $SCRIPT_DIR/plot_energy_force_breakdown.py \\
  --upstream_energy_pt $SCRIPT_DIR/results/nick/nick_exact_upstream_energy_only_b1024_2warmups_speed_eval.pt \\
  --upstream_energy_forces_pt $SCRIPT_DIR/results/nick/nick_exact_upstream_energy_forces_b1024_2warmups_speed_eval.pt \\
  --triton_energy_pt $SCRIPT_DIR/results/nick/nick_exact_triton_all_energy_only_b1024_2warmups_speed_eval.pt \\
  --extra_triton_energy_pt $SCRIPT_DIR/results/nick/nick_exact_triton_l3_n5_energy_only_b1024_2warmups_speed_eval.pt \\
  --triton_energy_forces_pt $SCRIPT_DIR/results/nick/nick_exact_triton_all_energy_forces_b1024_2warmups_speed_eval.pt \\
  --extra_triton_energy_forces_pt $SCRIPT_DIR/results/nick/nick_exact_triton_l3_n5_energy_forces_b1024_2warmups_speed_eval.pt \\
  --optimized_energy_pt $PLOT_ENERGY_PT \\
  --optimized_energy_forces_pt $PLOT_FORCES_PT \\
  --optimized_label 'New 2D polynomial Triton' \\
  --batch_size 1024 \\
  --exclude_config HOP:4:7 \\
  --output $SCRIPT_DIR/plots/energy_force_breakdown_b1024_with_poly_2d_green_${RUN_TAG}.png"
