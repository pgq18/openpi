"""
Standalone training script for fine-tuning pi05 on the warmup TFDS dataset.

This script does NOT modify any existing openpi files. It constructs the training
pipeline directly, using WarmupRldsDataset for data loading and the standard
pi05 model architecture.

Usage:
    cd openpi

    # Step 1: Compute normalization stats (first time only)
    uv run scripts/compute_warmup_norm_stats.py \
        --rlds_data_dir ../datasets \
        --output_dir ./assets/pi05_warmup/warmup

    # Step 2: Train
    uv run scripts/train_warmup.py \
        --rlds_data_dir ../datasets \
        --norm_stats_dir ./assets/pi05_warmup/warmup \
        --checkpoint_path /path/to/pi05_base_params \
        --exp_name warmup_test \
        --batch_size 32 \
        --num_train_steps 20000
"""

import argparse
import dataclasses
import functools
import logging
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.tokenizer as _tokenizer
import openpi.policies.warmup_policy as warmup_policy
import openpi.shared.array_typing as at
import openpi.shared.download as _download
import openpi.shared.nnx_utils as nnx_utils
import openpi.shared.normalize as _normalize
import openpi.training.checkpoints as _checkpoints
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.warmup_rlds_dataset as warmup_rlds_dataset
import openpi.training.weight_loaders as _weight_loaders
import openpi.transforms as _transforms


def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


ACTION_TYPE_TO_DATASET_KEY = {
    "raw_actions": "actions",
    "skeleton_actions": "skeleton_actions",
    "residual_actions": "residual_actions",
}


def create_warmup_data_loader(
    rlds_data_dir: str,
    norm_stats_dir: str,
    batch_size: int,
    action_horizon: int,
    model_config: pi0_config.Pi0Config,
    *,
    action_type: str = "raw_actions",
    use_relative_state: bool = False,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = True,
):
    """Create a data loader for the warmup TFDS dataset."""
    action_key = ACTION_TYPE_TO_DATASET_KEY[action_type]
    logging.info(f"Action type: {action_type} -> dataset key: {action_key}")

    # Load norm stats
    norm_stats_dir = str(_download.maybe_download(norm_stats_dir))
    norm_stats = _normalize.load(norm_stats_dir)
    logging.info(f"Loaded norm stats from {norm_stats_dir}")

    # When using relative_state, remap norm_stats key so Normalize sees "state"
    if use_relative_state:
        if "relative_state" in norm_stats:
            norm_stats = dict(norm_stats)
            norm_stats["state"] = norm_stats.pop("relative_state")
            logging.info("Using relative_state normalization stats (remapped to 'state')")

    # Remap action norm stats to "actions" key for downstream transforms
    # and remove unused action keys so they don't cause strict-mode errors at eval
    all_action_keys = {"actions", "skeleton_actions", "residual_actions"}
    unused_action_keys = all_action_keys - {action_key}
    if action_key != "actions" or unused_action_keys:
        norm_stats = dict(norm_stats)
        if action_key != "actions" and action_key in norm_stats:
            norm_stats["actions"] = norm_stats.pop(action_key)
            logging.info(f"Using {action_key} normalization stats (remapped to 'actions')")
        for k in unused_action_keys:
            norm_stats.pop(k, None)
        logging.info(f"Norm stats keys after action type filter: {list(norm_stats.keys())}")

    # Create the RLDS dataset
    dataset = warmup_rlds_dataset.WarmupRldsDataset(
        data_dir=rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        datasets=[
            warmup_rlds_dataset.WarmupRLDSDataset(name="warmup", version="5.0.0", split="train"),
        ],
    )

    # Build transform pipeline
    state_source = "observation/relative_state" if use_relative_state else "observation/state"
    transform_pipeline = [
        _transforms.RepackTransform(
            {
                "observation/image": "observation/image",
                "observation/wrist_image": "observation/wrist_image",
                "observation/state": state_source,
                "actions": action_key,
                "prompt": "prompt",
            }
        ),
        warmup_policy.WarmupInputs(model_type=_model.ModelType.PI05),
        _transforms.Normalize(norm_stats, use_quantiles=True),
        _transforms.InjectDefaultPrompt(None),
        _transforms.ResizeImages(224, 224),
        _transforms.TokenizePrompt(
            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
            discrete_state_input=model_config.discrete_state_input,
        ),
        _transforms.PadStatesAndActions(model_config.action_dim),
    ]

    transformed_dataset = _data_loader.IterableTransformedDataset(
        dataset,
        transform_pipeline,
        is_batched=True,
    )

    data_loader = _data_loader.RLDSDataLoader(
        transformed_dataset,
        sharding=sharding,
    )

    # Create a DataConfig for checkpoint saving
    data_config = _data_loader._config.DataConfig(
        repo_id="warmup",
        asset_id="warmup",
        norm_stats=norm_stats,
        use_quantile_norm=True,
    )

    return _data_loader.DataLoaderImpl(data_config, data_loader)


@at.typecheck
def init_train_state(
    model_config: pi0_config.Pi0Config,
    optimizer_config: _optimizer.OptimizerConfig,
    lr_schedule: _optimizer.LRScheduleConfig,
    weight_loader: _weight_loaders.WeightLoader,
    ema_decay: float | None,
    freeze_filter: nnx.filterlib.Filter,
    init_rng: at.KeyArrayLike,
    mesh: jax.sharding.Mesh,
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(optimizer_config, lr_schedule, weight_decay_mask=None)

    trainable_filter = nnx.All(nnx.Param, nnx.Not(freeze_filter))

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        model = model_config.create(model_rng)

        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        params = nnx_utils.state_map(params, freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(trainable_filter)),
            ema_decay=ema_decay,
            ema_params=None if ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    partial_params = _load_weights_and_validate(weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    train_state = jax.jit(
        init,
        donate_argnums=(1,),
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


def _load_weights_and_validate(loader, params_shape):
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def train_step(
    model_config: pi0_config.Pi0Config,
    ema_decay: float | None,
    trainable_filter: nnx.filterlib.Filter,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(model, rng, observation, actions):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    diff_state = nnx.DiffState(0, trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: ema_decay * old + (1 - ema_decay) * new, state.ema_params, new_params
            ),
        )

    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


def main():
    parser = argparse.ArgumentParser(description="Train pi05 on warmup TFDS dataset")
    parser.add_argument("--rlds_data_dir", type=str, required=True, help="Path to dir containing warmup/1.0.0/")
    parser.add_argument("--norm_stats_dir", type=str, required=True, help="Path to dir with norm_stats.json")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to pi05 base checkpoint")
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_train_steps", type=int, default=20_000)
    parser.add_argument("--lr", type=float, default=5e-5, help="Peak learning rate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--keep_period", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--fsdp_devices", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--checkpoint_base_dir", type=str, default="./checkpoints")
    parser.add_argument("--project_name", type=str, default="openpi_warmup")
    parser.add_argument(
        "--lora",
        choices=["full", "action_expert_only"],
        default=None,
        help="Enable LoRA fine-tuning. 'full'=both paligemma and action expert, 'action_expert_only'=action expert only",
    )
    parser.add_argument(
        "--action_type",
        choices=["raw_actions", "skeleton_actions", "residual_actions"],
        default="raw_actions",
        help="Which action field to train on: raw_actions (standard delta actions), skeleton_actions (low-freq DCT), residual_actions (high-freq DCT)",
    )
    parser.add_argument("--use_relative_state", action="store_true", help="Use relative_state instead of state as model input")
    args = parser.parse_args()

    if args.resume and args.overwrite:
        raise ValueError("Cannot resume and overwrite at the same time.")

    init_logging()
    logging.info(f"Running on: {platform.node()}")

    # Model config
    paligemma_variant = "gemma_2b"
    action_expert_variant = "gemma_300m"
    if args.lora == "full":
        paligemma_variant = "gemma_2b_lora"
        action_expert_variant = "gemma_300m_lora"
    elif args.lora == "action_expert_only":
        action_expert_variant = "gemma_300m_lora"

    model_config = pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=10,
        paligemma_variant=paligemma_variant,
        action_expert_variant=action_expert_variant,
    )

    # Construct a minimal TrainConfig-like object for checkpoint compat
    from openpi.training.config import TrainConfig

    if args.lora is not None:
        freeze_filter = model_config.get_freeze_filter()
        ema_decay = None
    else:
        freeze_filter = nnx.Nothing
        ema_decay = args.ema_decay

    if args.lora:
        logging.info(f"LoRA mode: {args.lora} (paligemma={paligemma_variant}, action_expert={action_expert_variant})")

    config = TrainConfig(
        name="pi05_warmup",
        project_name=args.project_name,
        exp_name=args.exp_name,
        model=model_config,
        weight_loader=_weight_loaders.CheckpointWeightLoader(args.checkpoint_path),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=args.lr,
            decay_steps=100_000,
            decay_lr=args.lr,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=ema_decay,
        freeze_filter=freeze_filter,
        batch_size=args.batch_size,
        num_train_steps=args.num_train_steps,
        seed=args.seed,
        save_interval=args.save_interval,
        keep_period=args.keep_period,
        log_interval=args.log_interval,
        overwrite=args.overwrite,
        resume=args.resume,
        wandb_enabled=not args.no_wandb,
        checkpoint_base_dir=args.checkpoint_base_dir,
        fsdp_devices=args.fsdp_devices,
    )

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )

    # Wandb
    if not config.wandb_enabled:
        wandb.init(mode="disabled")
    else:
        if resuming:
            run_id = (config.checkpoint_dir / "wandb_id.txt").read_text().strip()
            wandb.init(id=run_id, resume="must", project=config.project_name)
        else:
            wandb.init(
                name=config.exp_name,
                config=dataclasses.asdict(config),
                project=config.project_name,
            )
            config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            (config.checkpoint_dir / "wandb_id.txt").write_text(wandb.run.id)

    # Data loader (warmup-specific)
    data_loader = create_warmup_data_loader(
        rlds_data_dir=args.rlds_data_dir,
        norm_stats_dir=args.norm_stats_dir,
        batch_size=config.batch_size,
        action_horizon=model_config.action_horizon,
        model_config=model_config,
        action_type=args.action_type,
        use_relative_state=args.use_relative_state,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    # Init train state
    train_state, train_state_sharding = init_train_state(
        model_config=model_config,
        optimizer_config=config.optimizer,
        lr_schedule=config.lr_schedule,
        weight_loader=config.weight_loader,
        ema_decay=config.ema_decay,
        freeze_filter=config.freeze_filter,
        init_rng=init_rng,
        mesh=mesh,
    )
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(
            train_step,
            model_config,
            config.ema_decay,
            config.trainable_filter,
        ),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main()
