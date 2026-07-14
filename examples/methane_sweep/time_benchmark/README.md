# Inference timing artifacts

Generated benchmark artifacts are separated from source code:

- `results/csv/`: CSV outputs from the direct inference benchmark.
- `results/nick/`: Nick-style `.pt` and `.json` result files.
- `plots/`: generated figures.
- `logs/nick/`: Nick-style benchmark logs.
- `nick_speed_comparison/`: benchmark source and reference notebooks.

Files containing `energy_only` omit force calculation. Other Nick-style
results include energy and force prediction. Do not directly compare those two
workloads.

The L4 energy-plus-force run exhausted GPU memory and produced no result. The
L4 energy-only Triton run completed. Its L4-only result is preserved as
`results/nick/nick_exact_triton_l4_energy_only_2warmups_speed_eval.pt`.

The all-config Triton energy-only run writes
`results/nick/nick_exact_triton_all_energy_only_2warmups_speed_eval.pt` and
contains the original ten models plus HOP `(4,3)` and `(4,4)`.

To run energy and force inference at batch size 1024 on two GPUs, use:

```bash
bash nick_speed_comparison/run_energy_forces_1024_two_gpu.sh
```

This runs HIP-NN and all HIP-HOP configurations, including the two L4 models,
with Triton on GPU 0. The upstream path on GPU 1 omits the L4 models. TS models
are excluded from future benchmark runs. It uses two warmups and five
measured repetitions. The outputs include `energy_forces_b1024` in their names
and do not overwrite the batch-2048 results.

`plot_energy_force_breakdown.py` compares matching energy-only and
energy-plus-force result files. Solid bar segments show energy inference time;
the translucent segments above them show the additional force-calculation
time. The generated batch-2048 comparison is
`plots/energy_force_breakdown_upstream_vs_triton_inference_time_per_atom.png`.
