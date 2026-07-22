#!/bin/bash
#SBATCH -p shared-gpu-ampere
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --constraint=gpu1_model:NVIDIA_A100-PCIE-40GB
#SBATCH -t 03:00:00
#SBATCH -J benchmark_new_configs
#SBATCH -o logs/benchmark_new_configs_%j.log
#SBATCH -e logs/benchmark_new_configs_%j.err

set -e
set -u

module purge
source ~/.bashrc
conda activate hippynn

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$SCRIPT_DIR"
ANI_DATA="/vast/home/logan_bolton/ani1ccx/ani1x-release.h5"

mkdir -p "$BENCHMARK_DIR/results/nick"
mkdir -p "$BENCHMARK_DIR/logs"

cd "$SCRIPT_DIR/nick_speed_comparison"

echo "============================================"
echo "Benchmarking new configurations: (4,3), (3,5), (3,4)"
echo "Using existing evaluation_script_triton_random_init.py"
echo "============================================"
echo ""

# Energy-only benchmark
echo "Running ENERGY-ONLY benchmark..."
python evaluation_script_triton_random_init.py \
    --gpu 0 \
    --seed 42 \
    --split_seed 42 \
    --anidata_location "$ANI_DATA" \
    --configs "HOP:4:3" "HOP:3:5" "HOP:3:4" \
    --batch_sizes 2048 \
    --warmups 2 \
    --reps 5 \
    --n_interactions 2 \
    --n_atom_layers 5 \
    --n_features 256 \
    --n_sensitivities 20 \
    --kernel_mode triton \
    --use_triton_message_passing \
    --no-include_forces \
    --output "$BENCHMARK_DIR/results/nick/new_configs_energy_only.pt" \
    --output_json "$BENCHMARK_DIR/results/nick/new_configs_energy_only.json"

if [ $? -ne 0 ]; then
    echo "ERROR: Energy-only benchmark failed"
    exit 1
fi

echo ""
echo "✓ Energy-only benchmark completed"
echo ""

# Energy + Forces benchmark
echo "Running ENERGY+FORCES benchmark..."
python evaluation_script_triton_random_init.py \
    --gpu 0 \
    --seed 42 \
    --split_seed 42 \
    --anidata_location "$ANI_DATA" \
    --configs "HOP:4:3" "HOP:3:5" "HOP:3:4" \
    --batch_sizes 2048 \
    --warmups 2 \
    --reps 5 \
    --n_interactions 2 \
    --n_atom_layers 5 \
    --n_features 256 \
    --n_sensitivities 20 \
    --kernel_mode triton \
    --use_triton_message_passing \
    --include_forces \
    --output "$BENCHMARK_DIR/results/nick/new_configs_energy_forces.pt" \
    --output_json "$BENCHMARK_DIR/results/nick/new_configs_energy_forces.json"

if [ $? -ne 0 ]; then
    echo "ERROR: Energy+forces benchmark failed"
    exit 1
fi

echo ""
echo "✓ Energy+forces benchmark completed"
echo ""

echo "============================================"
echo "ALL BENCHMARKS COMPLETED SUCCESSFULLY!"
echo "============================================"
echo ""
echo "Results saved to:"
echo "  Energy-only:   $BENCHMARK_DIR/results/nick/new_configs_energy_only.pt"
echo "  Energy+forces: $BENCHMARK_DIR/results/nick/new_configs_energy_forces.pt"
echo ""
echo "To plot results with existing (4,4) data, use plot_energy_force_breakdown.py"
