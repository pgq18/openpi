"""Open-loop evaluation of pi05 + residual actor on warmup dataset episodes.

Compares three trajectories:
  1. Base only (pi05 predictions)
  2. Base + residual (pi05 + residual actor)
  3. Ground truth

Usage:
    cd openpi
    CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
      uv run scripts/eval_residual_warmup.py \
        --pi05_checkpoint_dir checkpoints/pi05_warmup/.../20000 \
        --residual_checkpoint_dir checkpoints/pi05_residual/.../10000 \
        --episode_idx 0
"""

import argparse
import dataclasses
import pathlib
import time

from eval_warmup_openloop import load_episode
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.residual_actor as residual_actor
import openpi.shared.normalize as _normalize
import openpi.training.checkpoints as _checkpoints

ACTION_TYPE_TO_STATS_KEY = {
    "raw_actions": "actions",
    "skeleton_actions": "skeleton_actions",
    "residual_actions": "residual_actions",
}


def load_action_norm_stats(checkpoint_dir: str, action_type: str, *, use_relative_state: bool):
    """Load norm stats and remap the selected action stats to the model's `actions` key."""
    norm_stats = _checkpoints.load_norm_stats(pathlib.Path(checkpoint_dir) / "assets", "warmup")
    action_key = ACTION_TYPE_TO_STATS_KEY[action_type]

    state_key = "relative_state" if use_relative_state and "relative_state" in norm_stats else "state"
    policy_norm_stats = {
        "state": norm_stats[state_key],
        "actions": norm_stats[action_key],
    }

    if "skeleton_actions" not in norm_stats or "residual_actions" not in norm_stats:
        raise ValueError("Residual eval requires skeleton_actions and residual_actions norm stats.")
    return policy_norm_stats, norm_stats[state_key], norm_stats["skeleton_actions"], norm_stats["residual_actions"]


def resolve_step_checkpoint_dir(checkpoint_dir: str) -> str:
    """Accept either a step checkpoint dir or one of its item dirs such as params/."""
    path = pathlib.Path(checkpoint_dir).expanduser()
    if path.name in {"params", "train_state"}:
        return str(path.parent)
    return checkpoint_dir


def unnormalize_quantile_actions(actions: np.ndarray, stats: _normalize.NormStats) -> np.ndarray:
    """Undo quantile normalization for action arrays using the provided action stats."""
    q01 = np.asarray(stats.q01, dtype=np.float32)[..., : actions.shape[-1]]
    q99 = np.asarray(stats.q99, dtype=np.float32)[..., : actions.shape[-1]]
    return (actions + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


def normalize_quantile_actions(actions: np.ndarray, stats: _normalize.NormStats) -> np.ndarray:
    """Apply quantile normalization for action/state arrays using the provided stats."""
    q01 = np.asarray(stats.q01, dtype=np.float32)[..., : actions.shape[-1]]
    q99 = np.asarray(stats.q99, dtype=np.float32)[..., : actions.shape[-1]]
    return (actions - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


def make_remaining_chunk(action_chunk: np.ndarray, offset: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    remaining = np.zeros((horizon, action_chunk.shape[-1]), dtype=action_chunk.dtype)
    valid = horizon - offset
    remaining[:valid] = action_chunk[offset:horizon]
    mask = np.zeros((horizon,), dtype=np.float32)
    mask[:valid] = 1.0
    return remaining, mask


def update_proprio_with_action(proprio: np.ndarray, action: np.ndarray, action_dim: int) -> np.ndarray:
    updated = np.array(proprio, copy=True)
    updated[:action_dim] = updated[:action_dim] + action[:action_dim]
    if updated.shape[0] > action_dim and action.shape[0] > action_dim:
        updated[action_dim] = action[action_dim]
    return updated


def load_pi05_model(
    checkpoint_dir: str,
    lora: str = "full",
    *,
    norm_stats_checkpoint_dir: str | None = None,
    action_type: str = "skeleton_actions",
    use_relative_state: bool = False,
):
    """Load the frozen pi05 model from a checkpoint directory."""
    import openpi.policies.policy_config as policy_config
    from openpi.training.config import DataConfig
    from openpi.training.config import DataConfigFactory
    from openpi.training.config import ModelTransformFactory
    from openpi.training.config import TrainConfig
    from openpi.transforms import Group

    use_relative_state_arg = use_relative_state

    paligemma_variant = "gemma_2b"
    action_expert_variant = "gemma_300m"
    if lora == "full":
        paligemma_variant = "gemma_2b_lora"
        action_expert_variant = "gemma_300m_lora"
    elif lora == "action_expert_only":
        action_expert_variant = "gemma_300m_lora"

    model_config = pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=10,
        paligemma_variant=paligemma_variant,
        action_expert_variant=action_expert_variant,
    )

    # Use the standard policy loading mechanism
    @dataclasses.dataclass(frozen=True)
    class WarmupDataConfig(DataConfigFactory):
        repo_id: str = "warmup"
        use_relative_state: bool = use_relative_state_arg

        def create(self, assets_dirs, model_config):
            import openpi.policies.warmup_policy as warmup_policy
            import openpi.shared.download as download
            import openpi.shared.normalize as normalize

            norm_stats = normalize.load(download.maybe_download(str(assets_dirs / "warmup")))
            if self.use_relative_state and "relative_state" in norm_stats:
                norm_stats = dict(norm_stats)
                norm_stats["state"] = norm_stats.pop("relative_state")
            return DataConfig(
                repo_id=self.repo_id,
                asset_id="warmup",
                norm_stats=norm_stats,
                data_transforms=Group(
                    inputs=[warmup_policy.WarmupInputs(model_type=_model.ModelType.PI05)],
                    outputs=[warmup_policy.WarmupOutputs()],
                ),
                model_transforms=ModelTransformFactory()(model_config),
                use_quantile_norm=True,
            )

    train_config = TrainConfig(
        name="pi05_warmup",
        model=model_config,
        data=WarmupDataConfig(),
        freeze_filter=model_config.get_freeze_filter() if lora else nnx.Nothing,
        ema_decay=None if lora else 0.99,
    )

    stats_checkpoint_dir = norm_stats_checkpoint_dir or checkpoint_dir
    policy_norm_stats, state_stats, skeleton_stats, residual_stats = load_action_norm_stats(
        stats_checkpoint_dir,
        action_type,
        use_relative_state=use_relative_state,
    )
    policy = policy_config.create_trained_policy(train_config, checkpoint_dir, norm_stats=policy_norm_stats)
    return policy, model_config, state_stats, skeleton_stats, residual_stats


def load_residual_actor(checkpoint_path: str, res_config: residual_actor.ResidualActorConfig):
    """Load the trained residual actor from a checkpoint."""
    import orbax.checkpoint as ocp

    model = residual_actor.ResidualActor(res_config, rngs=nnx.Rngs(jax.random.key(0)))
    graphdef, state = nnx.split(model)

    checkpointer = ocp.StandardCheckpointer()
    restored = checkpointer.restore(checkpoint_path, target={"params": state})
    restored_state = restored["params"]

    model = nnx.merge(graphdef, restored_state)
    model.eval()
    return model


def run_openloop_eval(
    policy,
    res_actor,
    res_config: residual_actor.ResidualActorConfig,
    state_stats: _normalize.NormStats,
    skeleton_stats: _normalize.NormStats,
    residual_stats: _normalize.NormStats,
    episode: dict,
    stride: int,
    action_horizon: int,
    action_type: str,
    *,
    res_scale: float = 0.2,
    use_relative_state: bool = False,
):
    """Run open-loop evaluation comparing base-only vs base+residual vs ground truth."""
    num_steps = episode["num_steps"]
    state_key = "relative_states" if (use_relative_state and "relative_states" in episode) else "states"

    positions = list(range(0, num_steps, stride))
    print(f"Open-loop eval: {len(positions)} inference points, stride={stride}, horizon={action_horizon}")

    all_base_actions = []
    all_final_actions = []
    all_gt_actions = []
    if "actions" not in episode:
        raise ValueError("Episode does not contain raw `actions`; cannot evaluate residual composition.")

    if getattr(policy, "_is_pytorch_model", False):
        raise ValueError("Feature extraction for residual eval currently requires a JAX pi05 policy.")

    for idx, pos in enumerate(positions):
        obs = {
            "observation/image": episode["images"][pos],
            "observation/wrist_image": episode["wrist_images"][pos],
            "observation/state": episode[state_key][pos],
            "prompt": episode["prompt"],
        }

        t0 = time.time()

        # Run the same input transforms used by policy.infer, then call pi05 directly
        # so we get both normalized skeleton actions and action-expert features.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = policy._input_transform(inputs)  # noqa: SLF001
        batched_inputs = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)
        observation = _model.Observation.from_dict(batched_inputs)
        policy._rng, infer_rng = jax.random.split(policy._rng)  # noqa: SLF001

        base_actions_norm, pi05_feature = policy._model.sample_actions_with_features(  # noqa: SLF001
            infer_rng,
            observation,
            feature_layer_idx=res_config.feature_layer_idx,
            feature_capture_step=res_config.feature_capture_step,
        )
        base_skeleton_norm = base_actions_norm[:, :, :res_config.base_action_dim]
        base_skeleton_norm_np = np.array(base_skeleton_norm[0])
        base_actions = unnormalize_quantile_actions(base_skeleton_norm_np, skeleton_stats)
        chunk_start_proprio_norm = np.array(batched_inputs["state"][0, :res_config.proprio_dim])
        current_proprio_phys = np.array(episode[state_key][pos][:res_config.proprio_dim], dtype=np.float32)
        final_actions = []

        for step_idx in range(action_horizon):
            remaining_skeleton, remaining_mask = make_remaining_chunk(
                base_skeleton_norm_np,
                step_idx,
                action_horizon,
            )
            current_proprio_norm = normalize_quantile_actions(current_proprio_phys[None], state_stats)[0]
            res_action = res_actor.get_eval_action(
                jnp.asarray(chunk_start_proprio_norm[None]),
                jnp.asarray(current_proprio_norm[None]),
                jnp.asarray(remaining_skeleton[None]),
                jnp.asarray(remaining_mask[None]),
                pi05_feature,
            )
            residual_action = unnormalize_quantile_actions(np.array(res_action[0]), residual_stats)
            final_action = np.array(base_actions[step_idx], copy=True)
            final_action[:res_config.action_dim] += res_scale * residual_action
            final_actions.append(final_action)
            current_proprio_phys = update_proprio_with_action(
                current_proprio_phys,
                final_action,
                res_config.action_dim,
            )

        final_actions = np.stack(final_actions, axis=0)

        end = min(pos + action_horizon, num_steps)
        gt_chunk = np.array(episode["actions"][pos:end])
        if len(gt_chunk) < action_horizon:
            gt_chunk = np.concatenate([gt_chunk, np.zeros((action_horizon - len(gt_chunk), 7))], axis=0)

        all_base_actions.append(base_actions[:, :7])
        all_final_actions.append(final_actions[:, :7])
        all_gt_actions.append(gt_chunk[:, :7])

        elapsed = (time.time() - t0) * 1000
        print(f"  [{idx + 1}/{len(positions)}] Step {pos}/{num_steps}: {elapsed:.0f}ms")

    base_actions_arr = np.concatenate(all_base_actions, axis=0)
    final_actions_arr = np.concatenate(all_final_actions, axis=0)
    gt_actions_arr = np.concatenate(all_gt_actions, axis=0)

    return base_actions_arr, final_actions_arr, gt_actions_arr


def plot_results(base_actions, final_actions, gt_actions, save_dir, episode_idx, res_scale):
    """Plot base-only vs base+residual vs ground truth."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.use("Agg")

    dim_labels = ["delta_x", "delta_y", "delta_z", "delta_rx", "delta_ry", "delta_rz", "gripper"]
    num_dims = gt_actions.shape[1]
    num_steps = gt_actions.shape[0]

    fig, axes = plt.subplots(num_dims, 1, figsize=(16, 3 * num_dims), sharex=True)
    if num_dims == 1:
        axes = [axes]

    for dim in range(num_dims):
        ax = axes[dim]
        x = np.arange(num_steps)
        ax.plot(x, gt_actions[:, dim], label="Ground Truth", color="orange", linewidth=1.2, alpha=0.8)
        ax.plot(x, base_actions[:, dim], label="Base Only", color="gray", linewidth=0.8, alpha=0.6)
        ax.plot(
            x,
            final_actions[:, dim],
            label=f"Base + Residual (alpha={res_scale})",
            color="steelblue",
            linewidth=1.0,
            alpha=0.8,
        )
        ax.set_ylabel(dim_labels[dim], fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(visible=True, alpha=0.3)

        base_mse = np.mean((base_actions[:, dim] - gt_actions[:, dim]) ** 2)
        final_mse = np.mean((final_actions[:, dim] - gt_actions[:, dim]) ** 2)
        ax.set_title(f"{dim_labels[dim]}  (base MSE={base_mse:.6f}, res MSE={final_mse:.6f})", fontsize=11)

    axes[-1].set_xlabel("Timestep", fontsize=10)
    fig.suptitle(f"Residual Evaluation — Episode {episode_idx}", fontsize=13, fontweight="bold")
    fig.tight_layout()

    save_path = pathlib.Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    out_file = save_path / f"residual_eval_ep{episode_idx}.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {out_file}")

    base_mse = np.mean((base_actions - gt_actions) ** 2)
    final_mse = np.mean((final_actions - gt_actions) ** 2)
    print(f"Overall MSE — Base: {base_mse:.6f}, Base+Residual: {final_mse:.6f}")
    improvement = (base_mse - final_mse) / base_mse * 100 if base_mse > 0 else 0
    print(f"Improvement: {improvement:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Open-loop eval of pi05 + residual actor")
    parser.add_argument("--pi05_checkpoint_dir", type=str, required=True, help="Pi05 checkpoint dir")
    parser.add_argument("--residual_checkpoint_dir", type=str, required=True, help="Residual actor checkpoint dir")
    parser.add_argument(
        "--norm_stats_checkpoint_dir",
        type=str,
        default=None,
        help="Checkpoint dir containing warmup norm stats. Defaults to the residual step checkpoint dir.",
    )
    parser.add_argument("--data_dir", type=str, default="../datasets", help="Path to warmup RLDS data")
    parser.add_argument("--episode_idx", type=int, default=0)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--lora", choices=["full", "action_expert_only"], default="full")
    parser.add_argument("--res_scale", type=float, default=0.2)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="./eval_results")
    parser.add_argument("--use_relative_state", action="store_true")
    parser.add_argument(
        "--action_type",
        choices=["raw_actions", "skeleton_actions", "residual_actions"],
        default="skeleton_actions",
        help="Which pi05 action space to evaluate; residual correction is intended for skeleton_actions.",
    )
    # Residual model config (must match training)
    parser.add_argument("--res_encoded_dim", type=int, default=256)
    parser.add_argument("--res_hidden_dims", type=str, default="256,256,256")
    parser.add_argument("--feature_layer_idx", type=int, default=12)
    parser.add_argument("--feature_capture_step", type=int, default=0)
    parser.add_argument("--proprio_dim", type=int, default=7)
    parser.add_argument("--action_dim", type=int, default=6)
    args = parser.parse_args()
    if args.action_type != "skeleton_actions":
        raise ValueError(
            "Residual eval is defined in normalized skeleton action space; use --action_type skeleton_actions."
        )

    # Load episode data first (before JAX GPU init)
    print(f"Loading episode {args.episode_idx}...")
    episode = load_episode(args.data_dir, args.episode_idx, args.split)

    # Load pi05 policy
    print("Loading pi05 policy...")
    norm_stats_checkpoint_dir = args.norm_stats_checkpoint_dir or resolve_step_checkpoint_dir(args.residual_checkpoint_dir)
    policy, pi05_config_obj, state_stats, skeleton_stats, residual_stats = load_pi05_model(
        args.pi05_checkpoint_dir,
        args.lora,
        norm_stats_checkpoint_dir=norm_stats_checkpoint_dir,
        action_type=args.action_type,
        use_relative_state=args.use_relative_state,
    )
    print("Pi05 policy loaded.")

    # Load residual actor
    hidden_dims = tuple(int(x) for x in args.res_hidden_dims.split(","))
    res_config = residual_actor.ResidualActorConfig(
        proprio_dim=args.proprio_dim,
        base_action_horizon=pi05_config_obj.action_horizon,
        base_action_dim=7,
        feature_dim=1024,
        feature_horizon=pi05_config_obj.action_horizon,
        encoded_dim=args.res_encoded_dim,
        hidden_dims=hidden_dims,
        action_horizon=pi05_config_obj.action_horizon,
        action_dim=args.action_dim,
        feature_layer_idx=args.feature_layer_idx,
        feature_capture_step=args.feature_capture_step,
    )
    print("Loading residual actor...")
    res_actor = load_residual_actor(args.residual_checkpoint_dir, res_config)
    print("Residual actor loaded.")

    # Run evaluation
    base_actions, final_actions, gt_actions = run_openloop_eval(
        policy, res_actor, res_config, state_stats, skeleton_stats, residual_stats, episode,
        stride=args.stride,
        action_horizon=pi05_config_obj.action_horizon,
        action_type=args.action_type,
        res_scale=args.res_scale,
        use_relative_state=args.use_relative_state,
    )

    # Plot results
    plot_results(base_actions, final_actions, gt_actions, args.save_dir, args.episode_idx, args.res_scale)


if __name__ == "__main__":
    main()
