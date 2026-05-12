"""
Open-loop evaluation: compare model predictions against ground truth actions
from a warmup dataset episode.

Usage:
    cd openpi
    CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    uv run scripts/eval_warmup_openloop.py \
        --checkpoint_dir checkpoints/pi05_warmup/warmup_lora_test_20260504_150746/12000 \
        --episode_idx 0
"""

import argparse
import dataclasses
import logging
import pathlib
import time

import numpy as np

from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory, TrainConfig
from openpi.transforms import Group


def load_episode(data_dir: str, episode_idx: int, split: str) -> dict:
    """Load a single episode from the warmup RLDS dataset into memory."""
    import dlimp as dl
    import tensorflow as tf
    import tensorflow_datasets as tfds

    tf.config.set_visible_devices([], "GPU")

    builder = tfds.builder("warmup", data_dir=data_dir, version="4.0.0")
    dataset = dl.DLataset.from_rlds(builder, split=split, shuffle=False)

    episode = None
    for i, ep in enumerate(dataset.as_numpy_iterator()):
        if i == episode_idx:
            episode = ep
            break

    if episode is None:
        raise ValueError(f"Episode {episode_idx} not found in {split} split (only {i + 1} episodes)")

    num_steps = len(episode["action"])
    print(f"Loaded episode {episode_idx}: {num_steps} steps")

    prompt_raw = episode["language_instruction"][0]
    prompt = prompt_raw.decode("utf-8") if isinstance(prompt_raw, bytes) else prompt_raw
    print(f"Instruction: {prompt}")

    # Decode JPEG images and convert everything to numpy arrays
    images, wrist_images, states, relative_states, actions = [], [], [], [], []
    for j in range(num_steps):
        img = tf.io.decode_image(
            episode["observation"]["image"][j], expand_animations=False, dtype=tf.uint8
        ).numpy()
        images.append(img)
        wrist = tf.io.decode_image(
            episode["observation"]["wrist_image"][j], expand_animations=False, dtype=tf.uint8
        ).numpy()
        wrist_images.append(wrist)
        states.append(np.asarray(episode["observation"]["state"][j], dtype=np.float32))
        if "relative_state" in episode["observation"]:
            relative_states.append(np.asarray(episode["observation"]["relative_state"][j], dtype=np.float32))
        actions.append(np.asarray(episode["action"][j], dtype=np.float32))

    result = {
        "images": images,
        "wrist_images": wrist_images,
        "states": states,
        "actions": actions,
        "prompt": prompt,
        "num_steps": num_steps,
    }
    if relative_states:
        result["relative_states"] = relative_states
    # Load skeleton/residual actions if available
    if "skeleton_action" in episode:
        result["skeleton_actions"] = [np.asarray(episode["skeleton_action"][j], dtype=np.float32) for j in range(num_steps)]
    if "residual_action" in episode:
        result["residual_actions"] = [np.asarray(episode["residual_action"][j], dtype=np.float32) for j in range(num_steps)]
    return result


@dataclasses.dataclass(frozen=True)
class WarmupDataConfig(DataConfigFactory):
    repo_id: str = "warmup"
    use_relative_state: bool = False

    def create(self, assets_dirs, model_config):
        import openpi.models.model as _model
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


def build_train_config(lora_mode: str, use_relative_state: bool = False) -> TrainConfig:
    import openpi.models.pi0_config as pi0_config

    paligemma_variant = "gemma_2b"
    action_expert_variant = "gemma_300m"

    if lora_mode == "full":
        paligemma_variant = "gemma_2b_lora"
        action_expert_variant = "gemma_300m_lora"
    elif lora_mode == "action_expert_only":
        action_expert_variant = "gemma_300m_lora"

    model_config = pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=10,
        paligemma_variant=paligemma_variant,
        action_expert_variant=action_expert_variant,
    )

    kwargs = {}
    if lora_mode is not None:
        kwargs["freeze_filter"] = model_config.get_freeze_filter()
        kwargs["ema_decay"] = None

    return TrainConfig(
        name="pi05_warmup",
        model=model_config,
        data=WarmupDataConfig(use_relative_state=use_relative_state),
        **kwargs,
    )


def run_openloop_eval(policy, episode: dict, stride: int, action_horizon: int, use_relative_state: bool = False, action_type: str = "raw_actions") -> tuple[np.ndarray, np.ndarray]:
    """Run open-loop evaluation: predict actions at each stride position."""
    num_steps = episode["num_steps"]
    all_pred_actions = []
    all_gt_actions = []

    state_key = "relative_states" if (use_relative_state and "relative_states" in episode) else "states"
    gt_key = "actions" if action_type == "raw_actions" else action_type

    positions = list(range(0, num_steps, stride))
    print(f"Open-loop eval: {len(positions)} inference points, stride={stride}, horizon={action_horizon}, gt_key={gt_key}")

    for idx, pos in enumerate(positions):
        obs = {
            "observation/image": episode["images"][pos],
            "observation/wrist_image": episode["wrist_images"][pos],
            "observation/state": episode[state_key][pos],
            "prompt": episode["prompt"],
        }

        t0 = time.time()
        result = policy.infer(obs)
        pred = result["actions"]  # (action_horizon, 7)

        # Collect GT for the corresponding horizon
        end = min(pos + action_horizon, num_steps)
        gt_chunk = np.array(episode[gt_key][pos:end])

        # Pad GT with zeros if shorter than horizon (matching training behavior)
        if len(gt_chunk) < action_horizon:
            gt_chunk = np.concatenate([gt_chunk, np.zeros((action_horizon - len(gt_chunk), 7))], axis=0)

        all_pred_actions.append(pred)
        all_gt_actions.append(gt_chunk)

        elapsed = (time.time() - t0) * 1000
        print(f"  [{idx + 1}/{len(positions)}] Step {pos}/{num_steps}: {elapsed:.0f}ms")

    pred_actions = np.concatenate(all_pred_actions, axis=0)
    gt_actions = np.concatenate(all_gt_actions, axis=0)
    return pred_actions, gt_actions


def plot_results(pred_actions: np.ndarray, gt_actions: np.ndarray, save_dir: str, episode_idx: int):
    """Plot predicted vs ground truth actions, one subplot per dimension."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dim_labels = ["delta_x", "delta_y", "delta_z", "delta_rx", "delta_ry", "delta_rz", "gripper"]
    num_dims = pred_actions.shape[1]
    num_steps = pred_actions.shape[0]

    fig, axes = plt.subplots(num_dims, 1, figsize=(16, 3 * num_dims), sharex=True)
    if num_dims == 1:
        axes = [axes]

    for dim in range(num_dims):
        ax = axes[dim]
        x = np.arange(num_steps)
        ax.plot(x, gt_actions[:, dim], label="Ground Truth", color="orange", linewidth=1.2, alpha=0.8)
        ax.plot(x, pred_actions[:, dim], label="Predicted", color="steelblue", linewidth=1.0, alpha=0.8)
        ax.set_ylabel(dim_labels[dim], fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        mse = np.mean((pred_actions[:, dim] - gt_actions[:, dim]) ** 2)
        ax.set_title(f"{dim_labels[dim]}  (MSE={mse:.6f})", fontsize=11)

    axes[-1].set_xlabel("Timestep", fontsize=10)
    fig.suptitle(f"Open-loop Evaluation — Episode {episode_idx}", fontsize=13, fontweight="bold")
    fig.tight_layout()

    save_path = pathlib.Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    out_file = save_path / f"openloop_ep{episode_idx}.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {out_file}")

    overall_mse = np.mean((pred_actions - gt_actions) ** 2)
    per_dim_mse = np.mean((pred_actions - gt_actions) ** 2, axis=0)
    print(f"Overall MSE: {overall_mse:.6f}")
    for dim in range(num_dims):
        print(f"  {dim_labels[dim]}: MSE={per_dim_mse[dim]:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Open-loop evaluation of warmup LoRA checkpoint")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to step checkpoint dir")
    parser.add_argument("--data_dir", type=str, default="../datasets", help="Path to warmup RLDS data")
    parser.add_argument("--episode_idx", type=int, default=0, help="Which episode to evaluate")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (train/val)")
    parser.add_argument("--lora", choices=["full", "action_expert_only"], default="full", help="LoRA mode")
    parser.add_argument("--save_dir", type=str, default="./eval_results", help="Output directory for plots")
    parser.add_argument("--stride", type=int, default=10, help="Inference stride (= action_horizon)")
    parser.add_argument("--use_relative_state", action="store_true", help="Use relative_state instead of state as model input")
    parser.add_argument(
        "--action_type",
        choices=["raw_actions", "skeleton_actions", "residual_actions"],
        default="raw_actions",
        help="Which action type was trained on (for GT comparison)",
    )
    args = parser.parse_args()

    # Step 1: Load episode data (TF on CPU, before JAX initializes GPU)
    print(f"Loading episode {args.episode_idx} from {args.split} split...")
    episode = load_episode(args.data_dir, args.episode_idx, args.split)

    # Step 2: Build config and load policy (JAX will use GPU)
    train_config = build_train_config(args.lora, use_relative_state=args.use_relative_state)
    action_horizon = train_config.model.action_horizon

    import openpi.policies.policy_config as policy_config

    print("Loading policy...")
    policy = policy_config.create_trained_policy(train_config, args.checkpoint_dir)
    print("Policy loaded.")

    # Step 3: Run open-loop evaluation
    pred_actions, gt_actions = run_openloop_eval(policy, episode, args.stride, action_horizon, use_relative_state=args.use_relative_state, action_type=args.action_type)

    # Step 4: Plot results
    plot_results(pred_actions, gt_actions, args.save_dir, args.episode_idx)


if __name__ == "__main__":
    main()
