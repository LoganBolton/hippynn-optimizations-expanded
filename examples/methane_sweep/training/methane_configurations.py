"""
This script is designed to accompany
Allen, A. E. A., Shinkle, E., Bujack, R., & Lubbers, N. (2025).
Optimal invariant bases for atomistic machine learning.
arXiv preprint arXiv:2503.23515. https://arxiv.org/abs/2503.23515

In the above paper, a methane dataset of ~7M configurations is used to test the expressive
capacity of different HIP-NN variants on different sizes of data. We find that for small
dataset sizes, different HIP-NN architecture variants produce similar performance. As more
data becomes available, HIP-HOP-NN is able to learn far more detail about geometries
in the environment, significantly surpassing HIP-NN-TS and HIP-NN. (See Figure 4.)

BEFORE RUNNING:
1. Download the file methane.extxyz.gz from https://archive.materialscloud.org/records/kz78r-6nx43
2. Unzip the file: $ gunzip methane.extxyz.gz
3. Place the resulting file in a folder called datasets/ at the same level as hippynn/
   or change ``data_src`` below

NOTE: The methane.extxyz file will be very slow to read, so this script only uses first 1000
configurations for training and subsequent 80,000 for testing. You can adjust this with
the ``data_size`` variable and by setting ``random_subset = True`` below. If you want to read
the methane.extxyz file repeatedly, I strongly suggest to first convert it into another format
(eg., .traj, .npz) that will be faster to read. You can do this by setting ``random_subset = True``
below, which will create a methane.traj file automatically for future use. But this conversion
may take a while (~1hr)."
"""

import argparse
import itertools
import json
import os
import subprocess
from pathlib import Path

def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--hiphop_l_max", type=int, choices=range(0, 5))
    parser.add_argument("--hiphop_n_max", type=int, choices=range(1, 5))
    parser.add_argument("--run_name", type=str)
    parser.add_argument("--model_suffix", type=str, default="")
    parser.add_argument("--wandb_mode", choices=("online", "offline", "disabled"), default="offline")
    parser.add_argument("--sweep_config", type=str)
    parser.add_argument("--sweep_task_id", type=int)
    parser.add_argument("--n_epochs", type=positive_int, default=200)
    parser.add_argument("--data_size", type=positive_int, default=1000)
    parser.add_argument("--test_set_size", type=positive_int, default=80_000)
    parser.add_argument("--batch_size", type=positive_int, default=256)
    parser.add_argument("--eval_batch_size", type=positive_int, default=2048)
    parser.add_argument("--max_batch_size", type=positive_int, default=2048)
    parser.add_argument(
        "--parallel_backend",
        choices=("single", "lightning-ddp", "data-parallel"),
        default="single",
        help=(
            "Training implementation. 'single' preserves the original one-GPU "
            "HIPPYNN loop; 'lightning-ddp' uses one persistent process per GPU. "
            "'data-parallel' retains the old threaded path for debugging only."
        ),
    )
    parser.add_argument(
        "--num_gpus",
        type=positive_int,
        default=1,
        help="Number of GPUs used by the selected parallel backend.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the selected backend's latest native checkpoint.",
    )
    parser.add_argument(
        "--legacy_checkpoint",
        type=Path,
        help="Import model weights from a legacy HIPPYNN .pt checkpoint.",
    )
    parser.add_argument("--checkpoint_every", type=positive_int, default=10)
    parser.add_argument(
        "--limit_train_batches",
        type=positive_int,
        help="Lightning smoke-test limit; omit for the full training split.",
    )
    parser.add_argument(
        "--limit_val_batches",
        type=positive_int,
        help="Lightning smoke-test limit; omit for the full validation split.",
    )
    parser.add_argument(
        "--limit_test_batches",
        type=positive_int,
        help="Lightning smoke-test limit; omit for the full test split.",
    )
    parser.add_argument("--activation_max_batches", type=positive_int, default=10)
    parser.add_argument("--activation_max_values", type=positive_int, default=200_000)
    parser.add_argument("--invariant_max_batches", type=positive_int, default=10)
    parser.add_argument("--invariant_max_values", type=positive_int, default=1_000_000)
    parser.add_argument(
        "--polynomial_cache_debug",
        action="store_true",
        help="Print one-time HIP-HOP polynomial collection/derivative cache builds on every rank.",
    )
    return parser


parser = build_parser()
args = None


def _sweep_parameter_values(name, spec):
    if not isinstance(spec, dict):
        raise ValueError(f"parameter {name!r} must be a mapping")
    if "values" in spec:
        values = spec["values"]
    elif "value" in spec:
        values = [spec["value"]]
    elif "min" in spec and "max" in spec:
        step = spec.get("step", 1)
        values = list(range(int(spec["min"]), int(spec["max"]) + 1, int(step)))
    else:
        raise ValueError(f"parameter {name!r} needs value, values, or min/max")
    if not values:
        raise ValueError(f"parameter {name!r} has no values")
    return values


def _apply_sweep_task(args):
    if args.sweep_config is None:
        missing_args = [
            name
            for name in ("seed", "hiphop_l_max", "hiphop_n_max", "run_name")
            if getattr(args, name) is None
        ]
        if missing_args:
            parser.error(f"missing required arguments: {', '.join('--' + name for name in missing_args)}")
        return

    if args.sweep_task_id is None:
        parser.error("--sweep_task_id is required when --sweep_config is used")

    import yaml

    with open(args.sweep_config, "r") as config_file:
        sweep = yaml.safe_load(config_file)

    parameters = sweep.get("parameters", {})
    parser_defaults = {action.dest: action.default for action in parser._actions}
    for name, spec in parameters.items():
        if name in ("seed", "hiphop_l_max", "hiphop_n_max", "run_name"):
            continue
        if hasattr(args, name) and isinstance(spec, dict) and "value" in spec:
            if getattr(args, name) == parser_defaults.get(name):
                setattr(args, name, spec["value"])

    grid_names = ["seed", "hiphop_l_max", "hiphop_n_max", "data_size"]
    grid_values = [_sweep_parameter_values(name, parameters[name]) for name in grid_names]
    jobs = list(itertools.product(*grid_values))

    if args.sweep_task_id >= len(jobs):
        print(f"Sweep task {args.sweep_task_id} is outside the configured sweep size ({len(jobs)}); exiting.")
        raise SystemExit(0)

    seed, hiphop_l_max, hiphop_n_max, data_size = jobs[args.sweep_task_id]
    args.seed = seed
    args.hiphop_l_max = hiphop_l_max
    args.hiphop_n_max = hiphop_n_max
    args.data_size = data_size
    if args.run_name is None:
        args.run_name = f"methane-l{hiphop_l_max}-n{hiphop_n_max}-d{data_size}-seed{seed}"
    args.sweep_count = len(jobs)

import ase
import matplotlib.pyplot as plt
import torch
import numpy as np


class _DisabledWandbRun:
    def __init__(self):
        self.summary = {}

    def finish(self):
        pass


class _DisabledWandb:
    Image = staticmethod(lambda figure: figure)

    def init(self, *args, **kwargs):
        return _DisabledWandbRun()

    def define_metric(self, *args, **kwargs):
        pass

    def log(self, *args, **kwargs):
        pass

    def watch(self, *args, **kwargs):
        pass

    def save(self, *args, **kwargs):
        pass


try:
    import wandb
    _WANDB_AVAILABLE = True
except ModuleNotFoundError:
    _WANDB_AVAILABLE = False
    wandb = _DisabledWandb()

import hippynn
from hippynn.custom_kernels import CustomKernelError
from hippynn.graphs import inputs, targets, physics
from hippynn.graphs.nodes.networks import HipHopnn, Hipnn, HipnnVec, HipnnQuad
from hippynn.experiment import setup_training, train_model, test_model
from hippynn.graphs import loss
from hippynn.experiment.controllers import RaiseBatchSizeOnPlateau, PatienceController
from hippynn.layers.hiplayers.invariants import HopInvariantLayer
from hippynn.plotting import PlotMaker, Hist2D, SensitivityPlot
from hippynn.pretraining import set_e0_values
from hippynn.tools import active_directory

WANDB_PATH = os.path.join(Path.home(), ".wandb_key.json")
torch.set_float32_matmul_precision("high")


def configure_wandb_auth(wandb_mode):
    if wandb_mode != "online":
        os.environ.setdefault("WANDB_MODE", wandb_mode)
        return
    assert os.path.exists(WANDB_PATH), f"Wandb json not found at. {WANDB_PATH}"
    with open(WANDB_PATH, "r") as f:
        config = json.load(f)
    os.environ["WANDB_API_KEY"] = config["WANDB_API_KEY"]
    if "WANDB_HOST" in config:
        os.environ["WANDB_HOST"] = config["WANDB_HOST"]


def get_git_sha():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _metric_value(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        return value.item()
    return value


def _flatten_metrics(prefix, metrics):
    flat_metrics = {}
    for split_name, split_metrics in metrics.items():
        for metric_name, value in split_metrics.items():
            flat_metrics[f"{prefix}/{split_name}/{metric_name}"] = _metric_value(value)
    return flat_metrics


def _model_size_metrics(model):
    params = list(model.parameters())
    buffers = list(model.buffers())
    total_params = sum(p.numel() for p in params)
    trainable_params = sum(p.numel() for p in params if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    param_bytes = sum(p.numel() * p.element_size() for p in params)
    buffer_bytes = sum(b.numel() * b.element_size() for b in buffers)
    total_bytes = param_bytes + buffer_bytes
    return {
        "model/params_total": total_params,
        "model/params_trainable": trainable_params,
        "model/params_non_trainable": non_trainable_params,
        "model/params_mb": param_bytes / (1024 ** 2),
        "model/buffers_mb": buffer_bytes / (1024 ** 2),
        "model/total_mb": total_bytes / (1024 ** 2),
    }


def collect_invariant_metadata(model):
    metadata = {}
    for name, module in model.named_modules():
        if isinstance(module, HopInvariantLayer):
            layer_metadata = module.invariant_metadata()
            layer_metadata["polynomial_sizes"] = layer_metadata["polynomial_sizes"].cpu().tolist()
            metadata[name] = layer_metadata
    return metadata


def collect_invariant_outputs(model, dataloader, device, n_inputs, max_batches=10, max_values=1_000_000):
    model.eval()
    model = model.to(device)
    invariant_values = {}
    invariant_shapes = {}
    hooks = []
    invariant_modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, HopInvariantLayer)
    ]

    def get_invariant_hook(name):
        def hook(module, input, output):
            if not hasattr(output, "detach"):
                return
            output = output.detach().float().cpu()
            invariant_shapes.setdefault(name, list(output.shape))
            values = output.reshape(-1, output.shape[-1]).numpy()
            current_size = sum(item.size for item in invariant_values.get(name, []))
            if current_size < max_values:
                max_rows = (max_values - current_size) // values.shape[1]
                if max_rows > 0:
                    invariant_values.setdefault(name, []).append(values[:max_rows])

        return hook

    try:
        for name, module in invariant_modules:
            hooks.append(module.register_forward_hook(get_invariant_hook(name)))

        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            batch = [item.to(device=device, non_blocking=True) for item in batch]
            with torch.enable_grad():
                model(*batch[:n_inputs])
    finally:
        for hook in hooks:
            hook.remove()

    return {
        name: np.concatenate(values, axis=0)
        for name, values in invariant_values.items()
        if values and np.concatenate(values).size
    }, invariant_shapes


def save_invariant_tracking(model, dataloader, device, n_inputs, output_prefix="FinalTraining", max_batches=10, max_values=1_000_000):
    metadata = collect_invariant_metadata(model)
    values, output_shapes = collect_invariant_outputs(
        model=model,
        dataloader=dataloader,
        device=device,
        n_inputs=n_inputs,
        max_batches=max_batches,
        max_values=max_values,
    )

    metadata_path = f"{output_prefix}_invariant_metadata.pt"
    values_path = f"{output_prefix}_invariant_values.npz"
    value_key_map = {
        f"layer_{idx}": layer_name
        for idx, layer_name in enumerate(values)
    }

    torch.save(
        {
            "metadata": metadata,
            "output_shapes": output_shapes,
            "value_key_map": value_key_map,
            "max_batches": max_batches,
            "max_values": max_values,
        },
        metadata_path,
    )
    np.savez_compressed(values_path, **{key: values[layer_name] for key, layer_name in value_key_map.items()})

    summary = {}
    for layer_name, layer_metadata in metadata.items():
        metric_prefix = f"invariants/{layer_name}"
        summary[f"{metric_prefix}/count"] = layer_metadata["n_invariants"]
        summary[f"{metric_prefix}/input_dimension"] = layer_metadata["input_dimension"]
        if layer_name in values:
            summary[f"{metric_prefix}/saved_values"] = int(values[layer_name].size)

    if summary:
        wandb.log(summary)
    wandb.save(metadata_path)
    wandb.save(values_path)
    return metadata_path, values_path


def collect_activations(model, dataloader, device, n_inputs, max_batches=10, max_values=200_000):
    model.eval()
    model = model.to(device)
    activations = {}
    hooks = []
    leaf_modules = [
        (name, module)
        for name, module in model.named_modules()
        if name and len(list(module.children())) == 0
    ]

    def get_activation_hook(name):
        def hook(module, input, output):
            if isinstance(output, (tuple, list)):
                output = output[0]
            if not hasattr(output, "detach"):
                return
            values = output.detach().float().cpu().reshape(-1).numpy()
            if values.size > max_values:
                values = values[:max_values]
            current_size = sum(item.size for item in activations.get(name, []))
            if current_size < max_values:
                activations.setdefault(name, []).append(values[: max_values - current_size])

        return hook

    try:
        for name, module in leaf_modules:
            hooks.append(module.register_forward_hook(get_activation_hook(name)))

        # Force outputs are computed as energy gradients with respect to positions,
        # so this forward pass must build an autograd graph even though hooks detach
        # the activation tensors they record.
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            batch = [item.to(device=device, non_blocking=True) for item in batch]
            with torch.enable_grad():
                model(*batch[:n_inputs])
    finally:
        for hook in hooks:
            hook.remove()

    return {
        name: np.concatenate(values)
        for name, values in activations.items()
        if values and np.concatenate(values).size
    }


def plot_activation_distributions(activations, prefix="FinalTraining"):
    if not activations:
        return
    n_layers = len(activations)
    fig, axes = plt.subplots(n_layers, 1, figsize=(10, max(3, 3 * n_layers)))
    axes = np.atleast_1d(axes)
    for ax, (layer_name, acts) in zip(axes, activations.items()):
        ax.hist(acts, bins=50, density=True, alpha=0.7, color="C0", edgecolor="black")
        ax.set_title(f"{layer_name}\n(mean={np.mean(acts):.3f}, var={np.var(acts):.3f})")
        ax.set_xlabel("Activation values")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    wandb.log({f"{prefix}_activations": wandb.Image(fig)})
    plt.close(fig)


class WandbEpochLogger:
    def __init__(self, metric_tracker, controller):
        self.metric_tracker = metric_tracker
        self.controller = controller

    def __call__(self, epoch, new_best):
        if not self.metric_tracker.epoch_metric_values:
            return
        epoch_metrics = _flatten_metrics("epoch", self.metric_tracker.epoch_metric_values[-1])
        valid_metrics = self.metric_tracker.epoch_metric_values[-1].get("valid", {})
        if "T-MAE" in valid_metrics:
            epoch_metrics["T-MAE"] = _metric_value(valid_metrics["T-MAE"])
        epoch_metrics.update(
            {
                "epoch": epoch,
                "epoch/new_best": bool(new_best),
                "epoch/batch_size": self.controller.batch_size,
                "epoch/eval_batch_size": self.controller.eval_batch_size,
            }
        )
        if self.metric_tracker.epoch_times:
            epoch_metrics["epoch/time_sec"] = self.metric_tracker.epoch_times[-1]
        for idx, group in enumerate(self.controller.optimizer.param_groups):
            epoch_metrics[f"optimizer/lr_group_{idx}"] = group["lr"]
        wandb.log(epoch_metrics, step=epoch)


def latest_checkpoint(run_dir):
    epoch_checkpoints = []
    for path in run_dir.glob("checkpoint_epoch_*.pt"):
        try:
            epoch = int(path.stem.rsplit("_", 1)[-1])
        except ValueError:
            continue
        epoch_checkpoints.append((epoch, path.stat().st_mtime, path))
    if epoch_checkpoints:
        return max(epoch_checkpoints)[2]

    best_checkpoint = run_dir / "best_checkpoint.pt"
    if best_checkpoint.exists():
        return best_checkpoint
    return None


def model_state_for_target(model, state_dict):
    """Match state-dict key prefixes to a wrapped or unwrapped target model."""
    prefix = "module."
    keys = tuple(state_dict)
    source_is_wrapped = bool(keys) and all(key.startswith(prefix) for key in keys)
    target_is_wrapped = isinstance(
        model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)
    )

    if source_is_wrapped and not target_is_wrapped:
        return {key[len(prefix):]: value for key, value in state_dict.items()}
    if target_is_wrapped and not source_is_wrapped:
        return {prefix + key: value for key, value in state_dict.items()}
    return state_dict


def load_model_state_portably(model, state_dict):
    """Load checkpoints written by either wrapped or unwrapped models.

    ``torch.nn.DataParallel.state_dict`` prefixes every key with ``module.``.
    Existing methane checkpoints predate multi-GPU-per-run training and do not
    have that prefix. Matching the prefix to the target makes checkpoints
    portable in both directions.
    """
    model.load_state_dict(model_state_for_target(model, state_dict))


def require_cuda_triton_kernels():
    if not torch.cuda.is_available():
        raise RuntimeError("This training script requires CUDA because it is configured to use Triton custom kernels.")

    try:
        import triton  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError("This training script requires the triton package for custom CUDA kernels.") from exc

    try:
        active_kernel = hippynn.custom_kernels.set_custom_kernels("triton")
    except CustomKernelError as exc:
        raise RuntimeError("Triton custom kernels were requested, but hippynn could not activate them.") from exc

    if active_kernel != "triton":
        raise RuntimeError(f"Expected Triton custom kernels, but hippynn activated {active_kernel!r}.")

    print(f"Using hippynn custom kernels: {active_kernel}")


def run_lightning_training(
    *,
    args,
    lightning_module,
    train_database,
    test_database,
    model_save_folder,
    run_name,
    wandb_config,
    world_size,
):
    """Run the already-assembled experiment through Lightning DDP."""
    try:
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import Callback, ModelCheckpoint
        from pytorch_lightning.loggers import WandbLogger
        from pytorch_lightning.strategies import DDPStrategy
    except ImportError as exc:
        raise RuntimeError(
            "The lightning-ddp backend requires pytorch_lightning. Install "
            "HIPPYNN's 'full' optional dependencies in the training environment."
        ) from exc

    class MethaneHippynnDataModule(pl.LightningDataModule):
        """HIPPYNN loaders with distinct per-rank train/evaluation batch sizes."""

        def __init__(self, train_database, test_database, train_batch_size, eval_batch_size):
            super().__init__()
            self.train_database = train_database
            self.test_database = test_database
            self._train_batch_size = train_batch_size
            self.eval_batch_size = eval_batch_size

        @property
        def batch_size(self):
            # HippynnLightningModule updates this attribute when
            # RaiseBatchSizeOnPlateau changes controller.batch_size.
            return self._train_batch_size

        @batch_size.setter
        def batch_size(self, value):
            self._train_batch_size = int(value)

        def train_dataloader(self):
            return self.train_database.make_generator(
                "train", "train", self._train_batch_size
            )

        def val_dataloader(self):
            return self.train_database.make_generator(
                "valid", "eval", self.eval_batch_size
            )

        def test_dataloader(self):
            return self.test_database.make_generator(
                "test", "eval", self.eval_batch_size
            )

    class DistributedRunDiagnostics(Callback):
        def on_fit_start(self, trainer, pl_module):
            # Checkpoint restoration happens before this hook. Ensure a resumed
            # controller's dynamic batch size is reflected by the first loader
            # of the continued run.
            trainer.datamodule.batch_size = pl_module.controller.batch_size
            trainer.datamodule.eval_batch_size = pl_module.controller.eval_batch_size

        def on_train_start(self, trainer, pl_module):
            loader = trainer.train_dataloader
            sampler = getattr(loader, "sampler", None)
            samples = len(sampler) if sampler is not None and hasattr(sampler, "__len__") else "unknown"
            print(
                "DDP_DATALOADER "
                f"global_rank={trainer.global_rank} batches={len(loader)} "
                f"samples_per_rank={samples} batch_size_per_rank={loader.batch_size} "
                f"sampler={type(sampler).__name__}",
                flush=True,
            )
            if world_size > 1 and "DistributedSampler" not in type(sampler).__name__:
                raise RuntimeError(
                    "Lightning did not install a distributed sampler; ranks would "
                    "train on overlapping data."
                )

        def on_validation_start(self, trainer, pl_module):
            loader = trainer.val_dataloaders
            if isinstance(loader, (tuple, list)):
                loader = loader[0]
            sampler = getattr(loader, "sampler", None)
            print(
                "DDP_VALIDATION_LOADER "
                f"global_rank={trainer.global_rank} batches={len(loader)} "
                f"batch_size_per_rank={loader.batch_size} "
                f"sampler={type(sampler).__name__}",
                flush=True,
            )

        def on_train_end(self, trainer, pl_module):
            if not args.polynomial_cache_debug:
                return
            for layer_name, layer in pl_module.model.named_modules():
                if not isinstance(layer, HopInvariantLayer):
                    continue
                derivative_counts = (
                    {} if layer.polynomials is None else layer.polynomials.derivative_build_counts
                )
                print(
                    "POLYNOMIAL_CACHE_SUMMARY "
                    f"rank={trainer.global_rank} layer={layer_name!r} "
                    f"collection_builds={layer._polynomial_collection_build_count} "
                    f"derivative_builds={derivative_counts}",
                    flush=True,
                )
                if layer._polynomial_collection_build_count > 1 or any(
                    count > 1 for count in derivative_counts.values()
                ):
                    raise RuntimeError(
                        f"Polynomial cache was rebuilt on rank {trainer.global_rank}, "
                        f"layer {layer_name!r}"
                    )

    checkpoint_dir = model_save_folder / "lightning_checkpoints"
    if int(os.environ.get("RANK", "0")) == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    datamodule = MethaneHippynnDataModule(
        train_database=train_database,
        test_database=test_database,
        train_batch_size=lightning_module.controller.batch_size,
        eval_batch_size=lightning_module.controller.eval_batch_size,
    )

    if args.legacy_checkpoint is not None:
        checkpoint = torch.load(args.legacy_checkpoint, map_location="cpu", weights_only=False)
        state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        load_model_state_portably(lightning_module.model, state)
        print(
            f"Imported model weights from legacy checkpoint {args.legacy_checkpoint}; "
            "optimizer, scheduler, epoch, controller, metrics, and RNG state were not resumed.",
            flush=True,
        )

    best_checkpointer = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best-{epoch:04d}-{valid_T-MAE:.8f}",
        monitor="valid_T-MAE",
        mode="min",
        save_top_k=1,
        save_last=True,
        every_n_epochs=1,
    )
    periodic_checkpointer = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="periodic-{epoch:04d}",
        save_top_k=-1,
        every_n_epochs=args.checkpoint_every,
        save_on_train_epoch_end=False,
    )

    if args.wandb_mode == "disabled":
        lightning_logger = False
    else:
        if not _WANDB_AVAILABLE:
            raise RuntimeError("wandb logging was requested, but wandb is not installed")
        lightning_logger = WandbLogger(
            name=run_name,
            id=run_name,
            project=os.environ.get("WANDB_PROJECT", "methane_sweep"),
            save_dir=os.environ.get("WANDB_DIR", str(model_save_folder / "wandb")),
            offline=args.wandb_mode == "offline",
            resume="allow" if args.resume else None,
            log_model=False,
        )
        if int(os.environ.get("RANK", "0")) == 0:
            lightning_logger.log_hyperparams(wandb_config)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.num_gpus,
        num_nodes=1,
        strategy=DDPStrategy(find_unused_parameters=False),
        max_epochs=args.n_epochs,
        logger=lightning_logger,
        callbacks=[best_checkpointer, periodic_checkpointer, DistributedRunDiagnostics()],
        use_distributed_sampler=True,
        default_root_dir=model_save_folder,
        log_every_n_steps=10,
        precision="32-true",
        # Force predictions differentiate energy with respect to positions.
        # set_grad_enabled(True) cannot override Lightning's inference_mode.
        inference_mode=False,
        limit_train_batches=args.limit_train_batches or 1.0,
        limit_val_batches=args.limit_val_batches or 1.0,
        limit_test_batches=args.limit_test_batches or 1.0,
    )

    resume_path = checkpoint_dir / "last.ckpt"
    if args.resume and not resume_path.exists():
        print(f"No Lightning checkpoint found at {resume_path}; starting from scratch.", flush=True)
        resume_path = None
    elif args.resume:
        print(f"Resuming Lightning state from {resume_path}", flush=True)
        resume_metadata = torch.load(resume_path, map_location="cpu", weights_only=False)
        controller_state = resume_metadata.get("controller_state", {})
        datamodule.batch_size = controller_state.get("batch_size", datamodule.batch_size)
        datamodule.eval_batch_size = controller_state.get(
            "eval_batch_size", datamodule.eval_batch_size
        )
        del resume_metadata
    else:
        resume_path = None

    trainer.fit(
        model=lightning_module,
        datamodule=datamodule,
        ckpt_path=str(resume_path) if resume_path is not None else None,
    )
    trainer.test(
        model=lightning_module,
        datamodule=datamodule,
        ckpt_path=best_checkpointer.best_model_path or None,
    )


def run_training(args):
    """Build and run one methane experiment with the selected backend."""

    # ----- Constants -----
    TOTAL_NUM_SAMPLES = 7_732_488
    TEST_SET_SIZE = args.test_set_size
    ENERGY_MEAN = -25042.327220945674

    # ----- User parameters -----
    seed = args.seed
    run_name = args.run_name
    wandb_mode = args.wandb_mode
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    data_root = PROJECT_ROOT / "datasets"
    data_src = data_root / "methane.extxyz"
    processed_src = data_root / "methane.traj"
    n_epochs = args.n_epochs  # reduce to decrease the run time of the script
    data_size = args.data_size
    random_subset = False  # whether to use a random subset of data or the first data_size sample
    # network_class = Hipnn # Original HIP-NN
    # network_class = HipnnVec # HIP-NN-TS, l=1
    # network_class = HipnnQuad # HIP-NN-TS, l=2
    network_class = HipHopnn  # HIP-HOP
    hiphop_l_max = args.hiphop_l_max  # these will not be used if network_class != HipHopnn
    hiphop_n_max = args.hiphop_n_max  # these will not be used if network_class != HipHopnn
    model_save_folder = (
        PROJECT_ROOT / "examples"
        / Path(f"TEST_METHANE_MODEL_l{hiphop_l_max}_n{hiphop_n_max}_d{data_size}_seed{seed}{('-' + args.model_suffix.strip('-_')) if args.model_suffix.strip('-_') else ''}")
    )
    sha = get_git_sha()

    network_params = {
        "possible_species": [0, 1, 6],
        "n_features": 32,
        "n_sensitivities": 20,
        "dist_soft_min": 0.4,
        "dist_soft_max": 9.0,
        "dist_hard_max": 10.3,  # diagonal of 6x6x6 cube
        "n_interaction_layers": 1,
        "n_atom_layers": 3,
    }

    if network_class == HipHopnn:
        network_params.update(
            {
                "l_max": hiphop_l_max,
                "n_max": hiphop_n_max,
            }
        )

    is_lightning = args.parallel_backend == "lightning-ddp"
    if args.parallel_backend == "single" and args.num_gpus != 1:
        parser.error("--parallel_backend=single requires --num_gpus=1")
    if args.parallel_backend == "data-parallel" and args.num_gpus < 2:
        parser.error("--parallel_backend=data-parallel requires --num_gpus>=2")
    if args.legacy_checkpoint is not None and args.resume:
        parser.error("--legacy_checkpoint imports weights only and cannot be combined with --resume")
    if args.polynomial_cache_debug:
        os.environ["HIPPYNN_POLYNOMIAL_CACHE_DEBUG"] = "1"

    configure_wandb_auth(wandb_mode)
    require_cuda_triton_kernels()

    visible_gpu_count = torch.cuda.device_count()
    if args.num_gpus > visible_gpu_count:
        parser.error(
            f"--num_gpus={args.num_gpus} requested, but only {visible_gpu_count} CUDA "
            "device(s) are visible"
        )
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    global_rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1")) if is_lightning else 1
    if is_lightning:
        if world_size not in (1, args.num_gpus):
            parser.error(
                f"torchrun WORLD_SIZE={world_size} does not match --num_gpus={args.num_gpus}"
            )
        if local_rank >= visible_gpu_count:
            parser.error(f"LOCAL_RANK={local_rank} has no matching visible CUDA device")
        torch.cuda.set_device(local_rank)
        setup_device = torch.device("cpu")
        device = torch.device(f"cuda:{local_rank}")
        print(
            "DDP_RANK_INFO "
            f"global_rank={global_rank} local_rank={local_rank} world_size={world_size} "
            f"hostname={os.uname().nodename} cuda_device={device} "
            f"cuda_name={torch.cuda.get_device_name(local_rank)!r} pid={os.getpid()}",
            flush=True,
        )
    elif args.parallel_backend == "data-parallel":
        setup_device = list(range(args.num_gpus))
        device = torch.device(f"cuda:{setup_device[0]}")
    else:
        setup_device = torch.device("cuda:0")
        device = setup_device
    print(
        f"Training one model on {args.num_gpus} GPU(s); "
        f"visible CUDA devices={visible_gpu_count}, setup device={setup_device}",
        flush=True,
    )

    wandb_config = {
            "parallel_backend": args.parallel_backend,
            "batch_size/global": args.batch_size,
            "batch_size/per_rank": args.batch_size // world_size,
            "eval_batch_size/global": args.eval_batch_size,
            "eval_batch_size/per_rank": args.eval_batch_size // world_size,
            "max_batch_size/global": args.max_batch_size,
            "max_batch_size/per_rank": args.max_batch_size // world_size,
            "world_size": world_size,
            "run_name": run_name,
            "wandb_mode": wandb_mode,
            "seed": seed,
            "sha": sha,
            "activation_max_batches": args.activation_max_batches,
            "activation_max_values": args.activation_max_values,
            "invariant_max_batches": args.invariant_max_batches,
            "invariant_max_values": args.invariant_max_values,
            "float_precision": "high",
            "data_src": str(data_src),
            "processed_src": str(processed_src),
            "random_subset": random_subset,
            "data_size": data_size,
            "test_set_size": TEST_SET_SIZE,
            "energy_mean": ENERGY_MEAN,
            "network_class": network_class.__name__,
            "network_params": network_params,
            "hiphop_l_max": hiphop_l_max,
            "hiphop_n_max": hiphop_n_max,
            "n_epochs": n_epochs,
            "num_gpus": args.num_gpus,
            "resume": args.resume,
            "legacy_checkpoint": str(args.legacy_checkpoint) if args.legacy_checkpoint else None,
            "checkpoint_every": args.checkpoint_every,
            "limit_train_batches": args.limit_train_batches,
            "limit_val_batches": args.limit_val_batches,
            "limit_test_batches": args.limit_test_batches,
            "optimizer_name": "Adam",
            "optimizer_hparams": {"lr": 2.5e-3},
            "scheduler_name": "RaiseBatchSizeOnPlateau",
            "scheduler_hparams": {"max_batch_size": args.max_batch_size, "patience": 150, "factor": 0.5},
            "controller_name": "PatienceController",
            "controller_hparams": {
                "batch_size": args.batch_size,
                "eval_batch_size": args.eval_batch_size,
                "max_epochs": n_epochs,
                "stopping_key": "T-MAE",
                "termination_patience": 300,
                "fraction_train_eval": 1,
            },
            "model_save_folder": str(model_save_folder),
        }

    if is_lightning:
        run_wandb = _DisabledWandb()
        wandb_run = run_wandb.init()
    else:
        if not _WANDB_AVAILABLE and wandb_mode != "disabled":
            raise RuntimeError("wandb is not installed; use --wandb_mode=disabled or install wandb")
        run_wandb = wandb
        wandb_run = run_wandb.init(
        name=run_name,
        id=run_name,
        resume="allow" if args.resume else None,
        mode=wandb_mode,
        config=wandb_config,
        )
        run_wandb.define_metric("epoch")
        run_wandb.define_metric("epoch/*", step_metric="epoch")
        run_wandb.define_metric("optimizer/*", step_metric="epoch")
        run_wandb.define_metric("T-MAE", summary="min")

    # ----- Prepare data -----
    def prepare_data(data_src, train_size, test_size, random_subset=random_subset):
        assert os.path.exists(data_src), f"Data source {data_src} does not exist! Please download methane.extxyz from https://archive.materialscloud.org/records/kz78r-6nx43 !"
        train_dict = {
            "numbers": [],
            "positions": [],
            "forces": [],
            "energy": [],
        }
        test_dict = {
            "numbers": [],
            "positions": [],
            "forces": [],
            "energy": [],
        }
        if not random_subset:
            generator = ase.io.iread(data_src)
        else:
            if os.path.exists(processed_src):
                data_src = processed_src  # use the .traj file if it exists
            else:
                print(f"Converting {data_src} to {processed_src} for faster reading next time, this may take a while (~1hr)...")
                frames = ase.io.read(data_src, index=':')
                ase.io.write(processed_src, frames)
                data_src = processed_src
                del frames
            indices = np.arange(TOTAL_NUM_SAMPLES)
            np.random.seed(seed)
            np.random.shuffle(indices)
            with ase.io.trajectory.Trajectory(processed_src) as raw_data:
                generator = [raw_data[i] for i in indices[:data_size + TEST_SET_SIZE]]

        for idx, frame in enumerate(generator):
            species = frame.get_atomic_numbers()
            positions = frame.get_positions()
            forces = frame.get_forces()
            energy = frame.get_total_energy()
            # Change units
            forces = forces * 51.42208619083232 * 23.060541945329334  # Hartrees/Bohr --> eV/Ang --> kcal/mol/Ang
            energy = energy * 627.5096080305927  # Hartrees --> kcal/mol
            # Shift energy mean
            energy -= ENERGY_MEAN
            if idx < train_size:
                train_dict["numbers"].append(species)
                train_dict["positions"].append(positions)
                train_dict["forces"].append(forces)
                train_dict["energy"].append(energy)
            elif idx < train_size + test_size:
                test_dict["numbers"].append(species)
                test_dict["positions"].append(positions)
                test_dict["forces"].append(forces)
                test_dict["energy"].append(energy)
            else:
                break
        # Convert to arrays
        for key in train_dict:
            if key != "numbers":
                train_dict[key] = np.array(train_dict[key], dtype=np.float32)
                test_dict[key] = np.array(test_dict[key], dtype=np.float32)
        return train_dict, test_dict


    def load_or_store_test_energy_statistics(test_energies, statistics_path, metadata):
        """Calculate the held-out target statistics once, then reuse the stored values."""
        if statistics_path.exists():
            with statistics_path.open("r", encoding="utf-8") as handle:
                statistics = json.load(handle)
        else:
            energies = np.asarray(test_energies, dtype=np.float64).reshape(-1)
            statistics = {
                **metadata,
                "count": int(energies.size),
                "energy_mean_kcal_per_mol": float(np.mean(energies)),
                # Population standard deviation: the test split is the complete
                # evaluation population, rather than a sample used to estimate it.
                "energy_std_kcal_per_mol": float(np.std(energies, ddof=0)),
                "std_ddof": 0,
            }
            if global_rank == 0:
                statistics_path.parent.mkdir(parents=True, exist_ok=True)
                temporary_path = statistics_path.with_suffix(
                    statistics_path.suffix + f".{os.getpid()}.tmp"
                )
                with temporary_path.open("w", encoding="utf-8") as handle:
                    json.dump(statistics, handle, indent=2, sort_keys=True)
                    handle.write("\n")
                temporary_path.replace(statistics_path)
        return statistics

    # ----- Construct model -----
    torch.random.manual_seed(seed)

    species = inputs.SpeciesNode(name="species", db_name="numbers")
    positions = inputs.PositionsNode(name="positions", db_name="positions")

    network = network_class("network", (species, positions), module_kwargs=network_params)
    henergy = targets.HEnergyNode(
        "HEnergy", network, db_name="energy", first_is_interacting=True
    )

    force = physics.GradientNode("forces", (henergy, positions), sign=-1, db_name="forces")

    # define loss quantities
    mse_force = loss.MSELoss.of_node(force)
    rmse_force = mse_force ** (1 / 2)
    mae_force = loss.MAELoss.of_node(force)
    rsq_force = loss.Rsq.of_node(force)

    rmse_energy = loss.MSELoss.of_node(henergy) ** (1 / 2)
    mae_energy = loss.MAELoss.of_node(henergy)
    rsq_energy = loss.Rsq.of_node(henergy)

    loss_energy = rmse_energy + mae_energy
    loss_force = rmse_force + mae_force
    loss_error = loss_energy + loss_force
    l2_reg = 1e-6 * loss.l2reg(network)

    total_loss = loss_error + l2_reg

    validation_losses = {
        "T-RMSE": rmse_energy,
        "T-MAE": mae_energy,
        "T-RSQ": rsq_energy,
        "F-RMSE": rmse_force,
        "F-MAE": mae_force,
        "F-RSQ": rsq_force,
        "Error Loss": loss_error,
        "L2": l2_reg,
        "Loss": total_loss,
    }

    if is_lightning:
        # HIPPYNN's experimental Lightning adapter does not support PlotMaker.
        # Final activation/invariant analysis remains on the established
        # single-GPU path and can be run separately for DDP checkpoints.
        plot_maker = None
    else:
        plotters = [
            Hist2D.compare(henergy, saved="energy", shown=False),
            Hist2D.compare(force, saved="force", shown=False),
            SensitivityPlot(
                network.torch_module.sensitivity_layers[0],
                saved="sensitivity",
                shown=False,
            ),
        ]
        plot_maker = PlotMaker(*plotters, plot_every=10)

    training_modules, db_info = hippynn.experiment.assemble_for_training(
        total_loss, validation_losses, plot_maker=plot_maker
    )

    if is_lightning:
        for name, global_value in (
            ("batch_size", args.batch_size),
            ("eval_batch_size", args.eval_batch_size),
            ("max_batch_size", args.max_batch_size),
        ):
            if global_value % world_size:
                raise ValueError(
                    f"Global {name}={global_value} must be divisible by world size {world_size}"
                )
    per_rank_batch_size = args.batch_size // world_size
    per_rank_eval_batch_size = args.eval_batch_size // world_size
    per_rank_max_batch_size = args.max_batch_size // world_size

    optimizer = torch.optim.Adam(training_modules.model.parameters(), lr=2.5e-3)
    scheduler = RaiseBatchSizeOnPlateau(
        optimizer=optimizer,
        max_batch_size=per_rank_max_batch_size,
        patience=150,
        factor=0.5,
    )

    controller = PatienceController(
        optimizer=optimizer,
        scheduler=scheduler,
        batch_size=per_rank_batch_size,
        eval_batch_size=per_rank_eval_batch_size,
        max_epochs=n_epochs,
        stopping_key="T-MAE",
        termination_patience=300,
        fraction_train_eval=1,
    )

    experiment_params = hippynn.experiment.SetupParams(
        controller=controller,
        device=setup_device,
    )

    metric_tracker = None
    lightning_module = None

    accelerator = "cpu"
    gpu_info = {}
    if torch.cuda.is_available():
        accelerator = "gpu"
        gpu_info["gpu/count"] = torch.cuda.device_count()
        for gpu_idx in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(gpu_idx)
            gpu_info[f"gpu/{gpu_idx}/name"] = torch.cuda.get_device_name(gpu_idx)
            gpu_info[f"gpu/{gpu_idx}/memory_mb"] = properties.total_memory / 1e6
    elif torch.backends.mps.is_available():
        accelerator = "mps"
    gpu_info["accelerator"] = accelerator
    wandb_run.summary.update(gpu_info)

    # ----- Load data -----
    train_dict, test_dict = prepare_data(data_src,
                                         data_size,
                                         TEST_SET_SIZE)

    test_statistics_path = (
        PROJECT_ROOT
        / "examples/methane_sweep/data_statistics"
        / f"test_energy_d{data_size}_n{TEST_SET_SIZE}_{'random_seed' + str(seed) if random_subset else 'sequential'}.json"
    )
    test_energy_statistics = load_or_store_test_energy_statistics(
        test_dict["energy"],
        test_statistics_path,
        {
            "data_source": str(data_src),
            "training_size": data_size,
            "test_size": TEST_SET_SIZE,
            "selection": "random" if random_subset else "sequential",
            "seed": seed if random_subset else None,
            "energy_units": "kcal/mol",
            "energy_shift_kcal_per_mol": ENERGY_MEAN,
        },
    )
    print(
        "TEST_SET_STATISTICS_JSON: "
        + json.dumps({"path": str(test_statistics_path), **test_energy_statistics}, sort_keys=True),
        flush=True,
    )
    wandb_run.summary.update(
        {
            "test_set/energy_std_kcal_per_mol": test_energy_statistics["energy_std_kcal_per_mol"],
            "test_set/energy_mean_kcal_per_mol": test_energy_statistics["energy_mean_kcal_per_mol"],
            "test_set/count": test_energy_statistics["count"],
            "test_set/statistics_path": str(test_statistics_path),
        }
    )

    train_database = hippynn.databases.Database(
        arr_dict=train_dict,
        seed=seed,
        pin_memory=True,
        test_size=0.1,
        valid_size=0.1,
        **db_info,
    )

    test_database = hippynn.databases.Database(
        arr_dict=test_dict,
        seed=seed + 1,
        pin_memory=True,
        **db_info,
    )
    test_database.split_the_rest("test")

    set_e0_values(henergy, train_database, trainable_after=False)

    if is_lightning:
        try:
            from hippynn.experiment import HippynnLightningModule
        except ImportError as exc:
            raise RuntimeError(
                "The lightning-ddp backend requires the compatible 'lightning' "
                "package from HIPPYNN's full dependencies."
            ) from exc

        lightning_module, _ = HippynnLightningModule.from_experiment_setup(
            training_modules=training_modules,
            database=train_database,
            setup_params=experiment_params,
        )
        training_modules = type(training_modules)(
            lightning_module.model,
            lightning_module.loss,
            training_modules.evaluator,
        )
        controller = lightning_module.controller
        metric_tracker = lightning_module.metric_tracker
        if isinstance(
            lightning_module.model,
            (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel),
        ):
            raise RuntimeError(
                "Lightning received a pre-wrapped model; setup_device must remain CPU "
                "so Lightning can apply DDP itself."
            )
    else:
        train_database.send_to_device(device)
        test_database.send_to_device(device)
        training_modules, controller, metric_tracker = setup_training(
            training_modules=training_modules,
            setup_params=experiment_params,
        )

    wandb_size_metrics = _model_size_metrics(training_modules.model)
    run_wandb.log(wandb_size_metrics, step=0)
    wandb_run.summary.update(wandb_size_metrics)

    if args.legacy_checkpoint is not None and not is_lightning:
        checkpoint = torch.load(args.legacy_checkpoint, map_location=device, weights_only=False)
        state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        load_model_state_portably(training_modules.model, state)
        print(
            f"Imported model weights from legacy checkpoint {args.legacy_checkpoint}; "
            "optimizer, scheduler, epoch, controller, metrics, and RNG state were not resumed.",
            flush=True,
        )

    if args.resume and not is_lightning:
        checkpoint_path = latest_checkpoint(model_save_folder)
        if checkpoint_path is not None:
            print(f"Resuming {run_name} from {checkpoint_path}", flush=True)
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            load_model_state_portably(training_modules.model, checkpoint["model"])
            controller.load_state_dict(checkpoint["controller"])
            metric_tracker = checkpoint["metric_tracker"]
            if metric_tracker.best_model:
                # ``train_model`` restores this state itself before final testing,
                # so its keys must match the current wrapped/unwrapped model too.
                metric_tracker.best_model = model_state_for_target(
                    training_modules.model, metric_tracker.best_model
                )
            if "torch_rng_state" in checkpoint:
                try:
                    rng_state = checkpoint["torch_rng_state"]
                    if hasattr(rng_state, "cpu"):
                        rng_state = rng_state.cpu()
                    if not isinstance(rng_state, torch.ByteTensor):
                        if hasattr(rng_state, "byte"):
                            rng_state = rng_state.byte()
                        else:
                            rng_state = torch.tensor(rng_state, dtype=torch.uint8)
                    torch.random.set_rng_state(rng_state)
                except (TypeError, RuntimeError, ValueError) as exc:
                    print(f"Warning: could not restore torch RNG state: {exc}. Continuing resume.", flush=True)
            if metric_tracker.current_epoch >= n_epochs:
                print(f"{run_name} already reached {metric_tracker.current_epoch} epochs; nothing to do.", flush=True)
                wandb_run.summary["status"] = "already_complete"
                wandb_run.finish()
                raise SystemExit(0)
            run_wandb.log(
                {
                    "resume/from_checkpoint": str(checkpoint_path),
                    "resume/start_epoch": metric_tracker.current_epoch,
                },
                step=metric_tracker.current_epoch,
            )
            wandb_run.summary["resumed_from_checkpoint"] = str(checkpoint_path)
            wandb_run.summary["resume_start_epoch"] = metric_tracker.current_epoch
        else:
            print(f"No checkpoint found for {run_name}; starting from scratch.", flush=True)

    train_split_size = len(train_database.splits["train"]["indices"])
    valid_split_size = len(train_database.splits["valid"]["indices"])
    test_split_size = len(test_database.splits["test"]["indices"])
    print(
        "Batch config: "
        f"global_train_batch={args.batch_size} "
        f"per_rank_train_batch={controller.batch_size} "
        f"global_eval_batch={args.eval_batch_size} "
        f"per_rank_eval_batch={controller.eval_batch_size} "
        f"world_size={world_size} "
        f"train_split={train_split_size} "
        f"valid_split={valid_split_size} "
        f"test_split={test_split_size} "
        f"train_batches_per_epoch={(train_split_size + controller.batch_size - 1) // controller.batch_size} "
        f"valid_batches={(valid_split_size + controller.eval_batch_size - 1) // controller.eval_batch_size} "
        f"test_batches={(test_split_size + controller.eval_batch_size - 1) // controller.eval_batch_size}",
        flush=True,
    )

    if is_lightning:
        run_lightning_training(
            args=args,
            lightning_module=lightning_module,
            train_database=train_database,
            test_database=test_database,
            model_save_folder=model_save_folder,
            run_name=run_name,
            wandb_config=wandb_config,
            world_size=world_size,
        )
        return

    # ----- Train model -----
    wandb_out_status = "aborted"
    try:
        with active_directory(model_save_folder):
            metric_tracker = train_model(
                training_modules=training_modules,
                database=train_database,
                controller=controller,
                metric_tracker=metric_tracker,
                callbacks=[WandbEpochLogger(metric_tracker, controller)],
                batch_callbacks=None,
                store_all_better=False,
                store_best=True,
                store_every=args.checkpoint_every,
                quiet=False,
            )

            # ----- Evaluate model -----
            evaluator = training_modules.evaluator
            best_model = metric_tracker.best_model
            if best_model:
                load_model_state_portably(evaluator.model, best_model)

            print("Testing model...")
            torch.cuda.empty_cache()
            test_model(
                test_database,
                evaluator,
                when="FinalTraining",
                batch_size=controller.eval_batch_size,
                metric_tracker=metric_tracker,
            )

            save_invariant_tracking(
                model=evaluator.model,
                dataloader=test_database.make_generator("test", "eval", controller.eval_batch_size),
                device=device,
                n_inputs=len(test_database.inputs),
                output_prefix="FinalTraining",
                max_batches=args.invariant_max_batches,
                max_values=args.invariant_max_values,
            )

            activations = collect_activations(
                model=evaluator.model,
                dataloader=test_database.make_generator("test", "eval", controller.eval_batch_size),
                device=device,
                n_inputs=len(test_database.inputs),
                max_batches=args.activation_max_batches,
                max_values=args.activation_max_values,
            )
            plot_activation_distributions(activations, prefix="FinalTraining")

            wandb_metrics = _flatten_metrics("best", metric_tracker.best_metric_values)
            wandb_metrics.update(_flatten_metrics("FinalTraining", metric_tracker.other_metric_values.get("FinalTraining", {})))
            best_valid_metrics = metric_tracker.best_metric_values.get("valid", {})
            if "T-MAE" in best_valid_metrics:
                wandb_metrics["T-MAE"] = _metric_value(best_valid_metrics["T-MAE"])
            run_wandb.log(wandb_metrics)
            wandb_run.summary.update(wandb_metrics)

            for artifact_path in (
                "best_model.pt",
                "best_checkpoint.pt",
                "training_metrics.pt",
                "training_metrics.pkl",
                "experiment_structure.pt",
            ):
                if os.path.exists(artifact_path):
                    run_wandb.save(artifact_path)

        wandb_out_status = "success"
    finally:
        wandb_run.summary["status"] = wandb_out_status
        wandb_run.finish()


def main():
    parsed_args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Ignoring unknown launcher arguments: {unknown}", flush=True)
    _apply_sweep_task(parsed_args)
    if parsed_args.model_suffix:
        suffix = parsed_args.model_suffix.strip("-_")
        if suffix:
            parsed_args.run_name = f"{parsed_args.run_name}-{suffix}"
    run_training(parsed_args)


if __name__ == "__main__":
    main()
