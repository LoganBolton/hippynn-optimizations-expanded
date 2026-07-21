#!/bin/bash -l
#SBATCH --job-name=poly_2d_real_l4n4
#SBATCH --output=slurm_logs/poly_2d_real_%j.out
#SBATCH --error=slurm_logs/poly_2d_real_%j.err
#SBATCH --partition=shared-gpu-ampere
#SBATCH --qos=long
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --constraint=gpu1_model:NVIDIA_A100-PCIE-40GB

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
BENCHMARK_DIR="$PROJECT_ROOT/examples/methane_sweep/time_benchmark"
SCRIPT_DIR="$BENCHMARK_DIR/nick_speed_comparison"
ANI_DATA="${ANI_DATA:-$PROJECT_ROOT/datasets/ani1x-release.h5}"
CONDA_ENV="${CONDA_ENV:-hippynn-expanded-lmax}"
RESULT_DIR="$BENCHMARK_DIR/results/poly_2d_real_data"

module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export PYTHONPATH="$PROJECT_ROOT"
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$PROJECT_ROOT"
mkdir -p slurm_logs "$RESULT_DIR"

nvidia-smi --id=0 --query-gpu=name,driver_version --format=csv,noheader

run_case() {
    local kernel_mode="$1"
    local force_flag="$2"
    local workload="$3"
    local output_stem="$RESULT_DIR/${kernel_mode}_${workload}_b512"

    python -u "$SCRIPT_DIR/evaluation_script_triton_random_init.py" \
        --gpu 0 \
        --seed 0 \
        --split_seed 0 \
        --anidata_location "$ANI_DATA" \
        --configs HOP:4:4 \
        --batch_sizes 512 \
        --warmups 2 \
        --reps 3 \
        --kernel_mode "$kernel_mode" \
        --use_triton_message_passing \
        "$force_flag" \
        --output "${output_stem}.pt" \
        --output_json "${output_stem}.json"
}

run_case triton_legacy --no-include_forces energy
run_case triton --no-include_forces energy
run_case triton_legacy --include_forces energy_forces
run_case triton --include_forces energy_forces

python - "$RESULT_DIR" <<'PY'
import json
import statistics
import sys
from pathlib import Path

result_dir = Path(sys.argv[1])
summary = {}
for path in sorted(result_dir.glob("*_b512.json")):
    data = json.loads(path.read_text())
    timings = next(iter(data["metrics"].values()))
    summary[path.stem] = {
        "seconds_median": statistics.median(timings),
        "seconds_mean": statistics.mean(timings),
        "seconds_repetitions": timings,
        "n_atoms": data["n_atoms"],
        "n_configs": data["n_configs"],
        "microseconds_per_atom_median": 1e6 * statistics.median(timings) / data["n_atoms"],
    }

for workload in ("energy", "energy_forces"):
    legacy = summary[f"triton_legacy_{workload}_b512"]
    optimized = summary[f"triton_{workload}_b512"]
    optimized["speedup_vs_legacy"] = legacy["seconds_median"] / optimized["seconds_median"]

output = result_dir / "summary_b512.json"
output.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
print(f"Wrote {output}")
PY
