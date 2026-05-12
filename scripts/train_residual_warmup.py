"""Train a residual action correction network on top of a frozen pi05 backbone.

Usage:
    cd openpi
    uv run scripts/train_residual_warmup.py \
        --rlds_data_dir ../datasets \
        --norm_stats_dir ./assets/pi05_warmup/warmup \
        --pi05_checkpoint_path ./checkpoints/pi05_warmup/.../20000 \
        --exp_name residual_test \
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
import jax
import jax.numpy as jnp
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.residual_actor as residual_actor
import openpi.shared.array_typing as at
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.checkpoints as _checkpoints
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


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


# Reuse data loading from train_warmup
from train_warmup import create_warmup_data_loader


def init_pi05_model(pi05_config: pi0_config.Pi0Config, weight_loader: _weight_loaders.WeightLoader, rng: at.KeyArrayLike):
    """Initialize pi05 model and load frozen weights."""
    model = pi05_config.create(rng)

    # Load weights
    import flax.traverse_util as traverse_util
    params_shape = nnx.state(model)
    loaded_params = weight_loader.load(params_shape.to_pure_dict())
    at.check_pytree_equality(expected=params_shape.to_pure_dict(), got=loaded_params, check_shapes=True, check_dtypes=True)
    partial_params = traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )

    graphdef, state = nnx.split(model)
    state.replace_by_pure_dict(partial_params)
    model = nnx.merge(graphdef, state)

    model.eval()
    model.requires_grad_(False)
    return model


def init_residual_train_state(
    res_config: residual_actor.ResidualActorConfig,
    optimizer_config: _optimizer.OptimizerConfig,
    lr_schedule: _optimizer.LRScheduleConfig,
    init_rng: at.KeyArrayLike,
    mesh: jax.sharding.Mesh,
):
    """Initialize training state for the residual actor."""
    tx = _optimizer.create_optimizer(optimizer_config, lr_schedule, weight_decay_mask=None)

    def init(rng):
        model = residual_actor.ResidualActor(res_config, rngs=nnx.Rngs(rng))
        params = nnx.state(model)
        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params),
            ema_decay=None,
            ema_params=None,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    train_state = jax.jit(init, in_shardings=replicated_sharding, out_shardings=state_sharding)(init_rng)

    return train_state, state_sharding


@at.typecheck
def train_step(
    pi05_model: nnx.Module,
    res_config: residual_actor.ResidualActorConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
):
    """Single training step: frozen pi05 inference + residual actor gradient update."""
    observation, gt_actions = batch

    # 1. Forward pi05 (frozen, no grad) to get base actions and features
    train_rng = jax.random.fold_in(rng, state.step)
    infer_rng, res_rng = jax.random.split(train_rng)

    base_actions, features = pi05_model.sample_actions_with_features(
        infer_rng, observation, feature_layer_idx=res_config.feature_layer_idx
    )
    # Stop gradients through pi05 outputs
    base_actions = jax.lax.stop_gradient(base_actions)
    features = jax.lax.stop_gradient(features)

    # 2. Extract proprio from observation state
    proprio = observation.state[:, :res_config.proprio_dim]

    # 3. Residual actor forward pass with gradient
    res_model = nnx.merge(state.model_def, state.params)

    def loss_fn(model):
        res_action, _, _ = model.get_action(res_rng, proprio, base_actions, features)
        # Target: ground truth residual = gt_action - base_action
        target_residual = gt_actions[:, :, :res_config.action_dim] - base_actions[:, :, :res_config.action_dim]
        loss = jnp.mean(jnp.square(res_action - target_residual))
        return loss

    loss, grads = nnx.value_and_grad(loss_fn)(res_model)

    # 4. Optimizer step
    params = state.params
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    nnx.update(res_model, new_params)
    new_params = nnx.state(res_model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)

    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(new_params),
    }
    return new_state, info


def main():
    parser = argparse.ArgumentParser(description="Train residual action correction on warmup TFDS dataset")
    # Data
    parser.add_argument("--rlds_data_dir", type=str, required=True)
    parser.add_argument("--norm_stats_dir", type=str, required=True)
    # Pi05 backbone
    parser.add_argument("--pi05_checkpoint_path", type=str, required=True, help="Path to pi05 base checkpoint")
    parser.add_argument("--lora", choices=["full", "action_expert_only"], default="full")
    # Experiment
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_train_steps", type=int, default=20_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    # Residual model config
    parser.add_argument("--res_scale", type=float, default=0.2, help="Residual scaling factor (alpha)")
    parser.add_argument("--res_encoded_dim", type=int, default=256)
    parser.add_argument("--res_hidden_dims", type=str, default="256,256,256", help="Comma-separated hidden dims")
    parser.add_argument("--feature_layer_idx", type=int, default=12, help="Which action expert layer to extract features from")
    parser.add_argument("--proprio_dim", type=int, default=7)
    parser.add_argument("--action_dim", type=int, default=7, help="Actual robot DOF")
    # Checkpointing / logging
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--keep_period", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--fsdp_devices", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--checkpoint_base_dir", type=str, default="./checkpoints")
    parser.add_argument("--project_name", type=str, default="openpi_residual")
    parser.add_argument("--use_relative_state", action="store_true")
    parser.add_argument("--action_type", choices=["raw_actions", "skeleton_actions", "residual_actions"], default="raw_actions")
    args = parser.parse_args()

    if args.resume and args.overwrite:
        raise ValueError("Cannot resume and overwrite at the same time.")

    init_logging()
    logging.info(f"Running on: {platform.node()}")

    # Parse hidden dims
    hidden_dims = tuple(int(x) for x in args.res_hidden_dims.split(","))

    # Pi05 model config
    paligemma_variant = "gemma_2b"
    action_expert_variant = "gemma_300m"
    if args.lora == "full":
        paligemma_variant = "gemma_2b_lora"
        action_expert_variant = "gemma_300m_lora"
    elif args.lora == "action_expert_only":
        action_expert_variant = "gemma_300m_lora"

    pi05_config_obj = pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=10,
        paligemma_variant=paligemma_variant,
        action_expert_variant=action_expert_variant,
    )

    # Residual actor config
    res_config = residual_actor.ResidualActorConfig(
        proprio_dim=args.proprio_dim,
        base_action_horizon=pi05_config_obj.action_horizon,
        base_action_dim=pi05_config_obj.action_dim,
        feature_dim=1024,  # gemma_300m width
        feature_horizon=pi05_config_obj.action_horizon,
        encoded_dim=args.res_encoded_dim,
        hidden_dims=hidden_dims,
        action_horizon=pi05_config_obj.action_horizon,
        action_dim=args.action_dim,
        feature_layer_idx=args.feature_layer_idx,
    )

    # Build checkpoint path
    from openpi.training.config import TrainConfig

    config = TrainConfig(
        name="pi05_residual",
        project_name=args.project_name,
        exp_name=args.exp_name,
        model=pi05_config_obj,
        weight_loader=_weight_loaders.CheckpointWeightLoader(args.pi05_checkpoint_path),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=args.lr,
            decay_steps=args.num_train_steps,
            decay_lr=args.lr * 0.1,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        freeze_filter=nnx.All(nnx.Param),
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
    train_rng, pi05_init_rng, res_init_rng = jax.random.split(rng, 3)

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
                config={**dataclasses.asdict(config), "res_config": dataclasses.asdict(res_config)},
                project=config.project_name,
            )
            config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            (config.checkpoint_dir / "wandb_id.txt").write_text(wandb.run.id)

    # Data loader (reuse warmup pipeline)
    data_loader = create_warmup_data_loader(
        rlds_data_dir=args.rlds_data_dir,
        norm_stats_dir=args.norm_stats_dir,
        batch_size=config.batch_size,
        action_horizon=pi05_config_obj.action_horizon,
        model_config=pi05_config_obj,
        action_type=args.action_type,
        use_relative_state=args.use_relative_state,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Init frozen pi05 model
    pi05_model = init_pi05_model(pi05_config_obj, config.weight_loader, pi05_init_rng)
    jax.block_until_ready(nnx.state(pi05_model))
    logging.info("Loaded frozen pi05 model")

    # Init residual actor train state
    train_state, train_state_sharding = init_residual_train_state(
        res_config, config.optimizer, config.lr_schedule, res_init_rng, mesh
    )
    jax.block_until_ready(train_state)
    logging.info(f"Initialized residual actor:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    # JIT compile train step (pi05_model is a closed-over constant)
    ptrain_step = jax.jit(
        functools.partial(train_step, pi05_model, res_config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(0,),
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
