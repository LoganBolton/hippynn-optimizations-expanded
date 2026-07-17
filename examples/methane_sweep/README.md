# Methane sweep workspace

Run commands from this directory so logs, W&B files, and Slurm output stay in
their existing locations.

## Layout

- `training/`: the methane model training program
- `configs/`: sweep and environment YAML files
- `launchers/`: Slurm launchers and resume helpers
- `analysis/`: validation, invariant, and outlier diagnostics
- `plotting/`: plot-generation and model-comparison scripts
- `results/paper/`: checked-in paper data and figures
- `logs/runs/<job-id>/`: worker and historical launcher logs grouped by originating Slurm job
- `logs/submissions/`: top-level Slurm launcher output for new submissions
- `logs/archive/`: older archived experiment sets
- `logs/plots_*` and `logs/metric_plots*`: generated metric plots
- `wandb*/`: offline W&B run data
- `time_benchmark/`: the self-contained timing benchmark workflow

The three root-level links to `methane_configurations.py` and the two sweep
YAML files are temporary compatibility entry points for jobs submitted before
this reorganization. Do not remove them until jobs `17122751` and `17120983`
are permanently finished and will not be requeued.

## Common commands

From `examples/methane_sweep`:

```bash
# Start or resume the main A100 sweep.
MODEL_SUFFIX=b256_fresh RESUME=1 sbatch launchers/run_sweep.slurm

# Resume the exact task groups used by the current l4/n3 and l4/n4 runs.
./launchers/resume_l4_b256_fresh.sh

# Refresh the 1M comparison plots.
python plotting/plot_methane_sweep_logs.py \
  --log-dir logs \
  --log-pattern 'runs/**/*_methane_*.out' \
  --output-dir logs/plots_compare_data_1M_b256_fresh \
  --sweep-config configs/methane-l4-n4.yml \
  --data-size 1000000

# Refresh the 100k comparison plots.
python plotting/plot_methane_sweep_logs.py \
  --log-dir logs \
  --log-pattern 'runs/**/*_methane_*.out' \
  --output-dir logs/plots_compare_data_100k_b256_fresh \
  --sweep-config configs/methane-l4-n4.yml \
  --data-size 100000
```

Checkpoint directories intentionally remain in `examples/TEST_METHANE_MODEL_*`
so existing runs resume from exactly the same locations.
