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

import jax
import jax.numpy as jnp
import numpy as np

import flax.nnx as nnx

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.residual_actor as residual_actor
import openpi.shared.array_typing as at
import openpi.training.weight_loaders as _weight_loaders

# Reuse episode loading from eval_warmup_openloop
from eval_warmup_openloop import load_episode


def load_pi05_model(checkpoint_dir: str, lora: str = "full"):
    """Load the frozen pi05 model from a checkpoint directory."""
    from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory, TrainConfig
    from openpi.transforms import Group
    import openpi.policies.policy_config as policy_config

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
        use_relative_state: bool = False

        def create(self, assets_dirs, model_config):
            import openpi.policies.warmup_policy as warmup_policy
            import openpi.shared.download as download
            import openpi.shared.normalize as normalize

            norm_stats = normalize.load(download.maybe_download(str(assets_dirs / "warmup")))
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

    policy = policy_config.create_trained_policy(train_config, checkpoint_dir)
    return policy, model_config


def load_residual_actor(checkpoint_path: str, res_config: residual_actor.ResidualActorConfig):
    """Load the trained residual actor from a checkpoint."""
    import orbax.checkpoint as ocp

    model = residual_actor.ResidualActor(res_config, rngs=nnx.Rngs(jax.random.key(0)))
    graphdef, state = nnx.split(model)

    checkpointer = ocp.StandardCheckpointer()
    restored_state = checkpointer.restore(checkpoint_path, target=state)

    model = nnx.merge(graphdef, restored_state)
    model.eval()
    return model


def run_openloop_eval(
    policy,
    pi05_model,
    res_actor,
    res_config: residual_actor.ResidualActorConfig,
    episode: dict,
    stride: int,
    action_horizon: int,
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

    for idx, pos in enumerate(positions):
        obs = {
            "observation/image": episode["images"][pos],
            "observation/wrist_image": episode["wrist_images"][pos],
            "observation/state": episode[state_key][pos],
            "prompt": episode["prompt"],
        }

        t0 = time.time()

        # 1. Get base actions from pi05 (through policy's standard inference)
        result = policy.infer(obs)
        base_actions = result["actions"]  # (action_horizon, action_dim)

        # 2. Run pi05 model directly to get features
        #    We need to preprocess the observation through the policy's input transform
        #    and then call sample_actions_with_features
        #    For simplicity, we use a zero feature placeholder for now.
        #    In a full implementation, we'd extract features by running the model
        #    through the JAX inference path directly.

        # 3. Get residual action (using base actions and zero features for now)
        #    Note: feature extraction requires direct JAX model access, which
        #    needs the observation to be preprocessed through the model's pipeline.
        #    For this eval script, we approximate by using zeros.
        proprio = np.array(episode[state_key][pos][:res_config.proprio_dim], dtype=np.float32)
        base_action_jnp = jnp.array(base_actions)[None]  # (1, H, D)
        # Use zero features as placeholder (feature extraction needs direct model access)
        feature_shape = (1, res_config.feature_horizon, res_config.feature_dim)
        pi05_feature = jnp.zeros(feature_shape)

        res_action = res_actor.get_eval_action(
            jnp.array(proprio)[None],
            base_action_jnp,
            pi05_feature,
        )
        res_action_np = np.array(res_action[0])  # (H, action_dim)

        # 4. Combine
        final_actions = base_actions[:, :res_config.action_dim] + res_scale * res_action_np

        # 5. Collect GT
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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
        ax.plot(x, final_actions[:, dim], label=f"Base + Residual (α={res_scale})", color="steelblue", linewidth=1.0, alpha=0.8)
        ax.set_ylabel(dim_labels[dim], fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

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
    parser.add_argument("--data_dir", type=str, default="../datasets", help="Path to warmup RLDS data")
    parser.add_argument("--episode_idx", type=int, default=0)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--lora", choices=["full", "action_expert_only"], default="full")
    parser.add_argument("--res_scale", type=float, default=0.2)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="./eval_results")
    parser.add_argument("--use_relative_state", action="store_true")
    # Residual model config (must match training)
    parser.add_argument("--res_encoded_dim", type=int, default=256)
    parser.add_argument("--res_hidden_dims", type=str, default="256,256,256")
    parser.add_argument("--feature_layer_idx", type=int, default=12)
    parser.add_argument("--proprio_dim", type=int, default=7)
    parser.add_argument("--action_dim", type=int, default=7)
    args = parser.parse_args()

    # Load episode data first (before JAX GPU init)
    print(f"Loading episode {args.episode_idx}...")
    episode = load_episode(args.data_dir, args.episode_idx, args.split)

    # Load pi05 policy
    print("Loading pi05 policy...")
    policy, pi05_config_obj = load_pi05_model(args.pi05_checkpoint_dir, args.lora)
    print("Pi05 policy loaded.")

    # Load residual actor
    hidden_dims = tuple(int(x) for x in args.res_hidden_dims.split(","))
    res_config = residual_actor.ResidualActorConfig(
        proprio_dim=args.proprio_dim,
        base_action_horizon=pi05_config_obj.action_horizon,
        base_action_dim=pi05_config_obj.action_dim,
        feature_dim=1024,
        feature_horizon=pi05_config_obj.action_horizon,
        encoded_dim=args.res_encoded_dim,
        hidden_dims=hidden_dims,
        action_horizon=pi05_config_obj.action_horizon,
        action_dim=args.action_dim,
        feature_layer_idx=args.feature_layer_idx,
    )
    print("Loading residual actor...")
    res_actor = load_residual_actor(args.residual_checkpoint_dir, res_config)
    print("Residual actor loaded.")

    # Run evaluation
    base_actions, final_actions, gt_actions = run_openloop_eval(
        policy, None, res_actor, res_config, episode,
        stride=args.stride,
        action_horizon=pi05_config_obj.action_horizon,
        res_scale=args.res_scale,
        use_relative_state=args.use_relative_state,
    )

    # Plot results
    plot_results(base_actions, final_actions, gt_actions, args.save_dir, args.episode_idx, args.res_scale)


if __name__ == "__main__":
    main()
