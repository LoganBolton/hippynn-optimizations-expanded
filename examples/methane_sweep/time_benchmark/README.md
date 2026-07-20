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

## Profile training

`ccx_training.py` has an opt-in short profiling mode. Normal training is
unchanged unless `--profile` is passed. The profiling run performs one warmup,
then records a small number of training batches and exits instead of starting a
full training run.

The recorded inference results in this directory used
`NVIDIA A100-PCIE-40GB`. The profiler validates that exact GPU model by
default so its results remain comparable.

To avoid consuming another job slot, run the profiler as a step inside one of
your existing allocations. Specify a physical GPU that is idle in that
allocation:

```bash
JOB_ID=12345678 GPU_ID=1 bash profile_in_allocation.sh
```

This uses `srun --jobid ... --overlap`, so it is a job step rather than another
submitted job and does not count against the partition's maximum job count.
The command stays attached until profiling finishes; its output is also saved
to `profile_step_${JOB_ID}_gpu${GPU_ID}.log`.

By default, the launcher profiles only `(l_max, n_max)` configurations `(3,5)`,
`(4,3)`, and `(4,4)`, sequentially on the selected GPU. It does not include
`(4,7)`. Adjust the workload with environment variables:

```bash
JOB_ID=12345678 GPU_ID=1 BATCH_SIZE=64 PROFILE_BATCHES=10 \
    bash profile_in_allocation.sh
```

`CONFIGS` accepts a space-separated list in `l_max:n_max` form when a smaller
subset is needed, for example `CONFIGS="3:5"`.

The shared GPU partition does not expose GPUs as Slurm GRES. Both launchers
bind the process to `GPU_ID` (default `0`) and exit before importing PyTorch if
`nvidia-smi` reports an existing compute process on that GPU. They also refuse
to run on a different GPU model unless `EXPECTED_GPU_NAME` is explicitly
changed.

When no suitable existing allocation is available, submit one profiling
configuration as a separate job with `sbatch profile_ani1ccx.sh`.

The run directory is named like `ani1ccx_profile_l3_n4_GPU0_SEED42/`. It
contains `training_log.txt` with tables ranked by GPU and CPU time, plus
`profile_trace.json`. Open the trace in Chrome's `chrome://tracing` or the
Perfetto trace viewer. Ranges prefixed with `model::` separate graph stages
such as indexing, pair construction, HIP-HOP message passing, and energy
prediction; `loss::` ranges identify loss calculation. CUDA kernels nested
under each range show which model stage launched the expensive work.

`--profile_with_stack` and `--profile_with_flops` are available for deeper
investigation, but add profiler overhead. Do not compare profiled wall-clock
time to the unprofiled benchmark.

These source edits do not alter existing Python processes because those jobs
have already loaded their source. The allocation-step launcher attaches only a
new process to the allocation; it does not attach a profiler to the training
process itself. The busy-GPU check prevents that process from intentionally
sharing a GPU with a running job.
