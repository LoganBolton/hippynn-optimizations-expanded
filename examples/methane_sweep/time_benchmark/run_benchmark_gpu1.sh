#!/bin/bash
set -e

cd /vast/home/logan_bolton/Github/hippynn-optimizations-expanded/examples/methane_sweep/time_benchmark/nick_speed_comparison

source ~/.bashrc
conda activate hippynn-expanded-lmax

ANI_DATA="/vast/home/logan_bolton/Github/hippynn-optimizations-expanded/datasets/ani1x-release.h5"

echo "============================================"
echo "Benchmarking on GPU 1 (free A100)"
echo "Configs: (4,3), (3,5), (3,4)"
echo "============================================"
echo ""

# Energy-only benchmark
echo "Running ENERGY-ONLY benchmark..."
CUDA_VISIBLE_DEVICES=1 python evaluation_script_triton_random_init.py \
    --gpu 0 \
    --seed 42 \
    --split_seed 42 \
    --anidata_location "$ANI_DATA" \
    --configs "HOP:4:3" "HOP:3:5" "HOP:3:4" \
    --batch_sizes 1024 \
    --warmups 2 \
    --reps 5 \
    --n_interactions 2 \
    --n_atom_layers 5 \
    --n_features 256 \
    --n_sensitivities 20 \
    --kernel_mode triton \
    --use_triton_message_passing \
    --no-include_forces \
    --output "../results/nick/new_configs_energy_only.pt" \
    --output_json "../results/nick/new_configs_energy_only.json"

echo ""
echo "✓ Energy-only completed"
echo ""

# Energy + Forces benchmark
echo "Running ENERGY+FORCES benchmark..."
CUDA_VISIBLE_DEVICES=1 python evaluation_script_triton_random_init.py \
    --gpu 0 \
    --seed 42 \
    --split_seed 42 \
    --anidata_location "$ANI_DATA" \
    --configs "HOP:4:3" "HOP:3:5" "HOP:3:4" \
    --batch_sizes 1024 \
    --warmups 2 \
    --reps 5 \
    --n_interactions 2 \
    --n_atom_layers 5 \
    --n_features 256 \
    --n_sensitivities 20 \
    --kernel_mode triton \
    --use_triton_message_passing \
    --include_forces \
    --output "../results/nick/new_configs_energy_forces.pt" \
    --output_json "../results/nick/new_configs_energy_forces.json"

echo ""
echo "============================================"
echo "ALL BENCHMARKS COMPLETED!"
echo "============================================"
