# INCONSISTENT ARCHITECTURE - DO NOT USE

These benchmark results were run with **inconsistent network architectures** and cannot be fairly compared with the old results.

## Problem

The old benchmark results in `results/nick/` have **no metadata** about network architecture, making fair comparison impossible.

The new benchmarks were run with:
- **256 features**
- **5 atom layers**
- **2 interaction layers**

But we don't know what the old benchmarks used (likely the same, but unverified).

Additionally, the `poly_2d_real_data` (4,4) results use:
- **128 features** ⚠️ DIFFERENT!
- **3 atom layers** ⚠️ DIFFERENT!
- **2 interaction layers**

## Files in this directory

- `new_configs_energy_only.pt/json` - (3,4), (3,5), (4,3) with 256 feat, 5 layers
- `new_configs_energy_forces.pt/json` - (3,4), (3,5), (4,3) with 256 feat, 5 layers

## What to do

To get consistent, comparable results:
1. Re-run ALL configs with the **same architecture** (document it!)
2. Include architecture metadata in all result files
3. Use consistent: features, layers, interactions, dataset, batch size

## Benchmark run details

- Date: 2026-07-21
- GPU: A100-PCIE-40GB
- Batch size: 1024
- Dataset: ANI-1x (48,957 test configs, 676,583 atoms)
- Warmups: 2
- Reps: 5
- Architecture: 256 features, 5 atom layers, 2 interactions ⚠️
