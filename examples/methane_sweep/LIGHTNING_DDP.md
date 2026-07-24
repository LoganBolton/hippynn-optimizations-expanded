# Methane Lightning DDP

The multi-GPU methane path uses one persistent Lightning DDP process per GPU.
The original one-GPU command remains the default. The former threaded
`DataParallel` implementation is available only through the explicit
`--parallel_backend data-parallel` debugging option.

## Environment prerequisite

The training environment must provide the `pytorch_lightning` import used by
HIPPYNN. The repository declares it through the `full` optional dependencies.
Check the intended Conda environment before submitting:

```bash
conda run -n hippynn-expanded-lmax python -c \
  'import pytorch_lightning as pl; print(pl.__version__)'
```

Do not upgrade Lightning independently of the HIPPYNN environment.

## Two-GPU smoke run

Run from `examples/methane_sweep`. This uses only 20 train batches and two
validation/test batches; it is intended to verify DDP and the HIP-HOP caches,
not model quality.

```bash
TASK_ID=0 \
NUM_GPUS=2 \
MODEL_SUFFIX=lightning_ddp_smoke \
RESUME=0 \
WANDB_MODE=disabled \
LIMIT_TRAIN_BATCHES=20 \
LIMIT_VAL_BATCHES=2 \
LIMIT_TEST_BATCHES=2 \
POLYNOMIAL_CACHE_DEBUG=1 \
sbatch --constraint='gpu_cc:8.0&gpu_count:2' \
  launchers/run_single_task_lightning_ddp.slurm
```

The log must contain two `DDP_RANK_INFO` lines with world size 2 and distinct
local CUDA devices, plus `DistributedSampler` for the train and validation
loaders. For each invariant layer and rank, the final
`POLYNOMIAL_CACHE_SUMMARY` must report one collection build and no derivative
level built more than once.

The global command-line batch sizes are divided by world size before creating
the controller and loaders. For example, a global train batch of 256 becomes
128 per rank on two GPUs. Future `RaiseBatchSizeOnPlateau` changes remain in
per-rank units, so the effective global size grows consistently.

## Checkpoints

Lightning checkpoints are stored under:

```text
examples/TEST_METHANE_MODEL_.../lightning_checkpoints/
```

They include `last.ckpt`, the best `valid_T-MAE` checkpoint, and periodic
checkpoints. `experiment_structure.pt` is stored in the parent model
directory. `--resume` loads `last.ckpt`, including Lightning optimizer and
scheduler state plus HIPPYNN controller and metric-tracker state.

Use `--legacy_checkpoint PATH` to import weights from an older HIPPYNN `.pt`
file. This is deliberately weights-only and does not claim to restore the old
epoch, optimizer, scheduler, controller, metrics, or RNG state.

Final activation plots and invariant-value exports remain on the one-GPU
evaluation path because HIPPYNN's Lightning adapter does not support
`PlotMaker`.
