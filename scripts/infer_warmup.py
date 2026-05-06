"""
Inference script for pi05 warmup LoRA checkpoints.

Usage:
    cd openpi
    CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    uv run scripts/infer_warmup.py \
        --checkpoint_dir checkpoints/pi05_warmup/warmup_lora_test_20260504_150746/12000
"""

import argparse
import dataclasses
import logging
import time

import numpy as np

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.policies.policy_config as policy_config
import openpi.policies.warmup_policy as warmup_policy
import openpi.shared.download as download
import openpi.shared.normalize as normalize
import openpi.training.config as config_mod
import openpi.training.weight_loaders as weight_loaders
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory, TrainConfig
from openpi.transforms import Group


@dataclasses.dataclass(frozen=True)
class WarmupDataConfig(DataConfigFactory):
    """Data config factory for warmup dataset inference."""

    repo_id: str = "warmup"

    def create(self, assets_dirs, model_config):
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


def build_train_config(lora_mode: str | None) -> TrainConfig:
    paligemma_variant = "gemma_2b"
    action_expert_variant = "gemma_300m"
    freeze_filter = None

    if lora_mode == "full":
        paligemma_variant = "gemma_2b_lora"
        action_expert_variant = "gemma_300m_lora"
        from flax import nnx
        from openpi.models.pi0_config import Pi0Config

        tmp_config = Pi0Config(
            pi05=True,
            paligemma_variant=paligemma_variant,
            action_expert_variant=action_expert_variant,
        )
        freeze_filter = tmp_config.get_freeze_filter()
    elif lora_mode == "action_expert_only":
        action_expert_variant = "gemma_300m_lora"
        from flax import nnx
        from openpi.models.pi0_config import Pi0Config

        tmp_config = Pi0Config(
            pi05=True,
            action_expert_variant=action_expert_variant,
        )
        freeze_filter = tmp_config.get_freeze_filter()

    model_config = pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=10,
        paligemma_variant=paligemma_variant,
        action_expert_variant=action_expert_variant,
    )

    kwargs = {}
    if freeze_filter is not None:
        from flax import nnx
        kwargs["freeze_filter"] = freeze_filter
        kwargs["ema_decay"] = None

    return TrainConfig(
        name="pi05_warmup",
        model=model_config,
        data=WarmupDataConfig(),
        **kwargs,
    )


def main():
    parser = argparse.ArgumentParser(description="Inference with pi05 warmup checkpoint")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to step checkpoint dir")
    parser.add_argument("--prompt", type=str, default="put carrot on plate", help="Task instruction")
    parser.add_argument("--image", type=str, default=None, help="Path to image file (optional)")
    parser.add_argument(
        "--state",
        type=float,
        nargs=7,
        default=None,
        help="7-dim state vector (ee_xyz + ee_euler + gripper)",
    )
    parser.add_argument(
        "--lora",
        choices=["full", "action_expert_only"],
        default="full",
        help="LoRA mode used during training",
    )
    parser.add_argument("--num_steps", type=int, default=1, help="Number of inference steps")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    train_config = build_train_config(args.lora)
    logging.info(f"Model config: pi05={train_config.model.pi05}, action_dim={train_config.model.action_dim}")

    logging.info(f"Loading checkpoint from {args.checkpoint_dir}...")
    policy = policy_config.create_trained_policy(train_config, args.checkpoint_dir)
    logging.info("Policy loaded successfully.")

    # Build observation
    obs = warmup_policy.make_warmup_example()
    obs["observation/state"] = np.zeros(7, dtype=np.float32)

    if args.image:
        from PIL import Image

        img = np.array(Image.open(args.image).convert("RGB").resize((224, 224)))
        obs["observation/image"] = img

    if args.state:
        obs["observation/state"] = np.array(args.state, dtype=np.float32)

    obs["prompt"] = args.prompt

    logging.info(f"Prompt: {obs['prompt']}")
    logging.info(f"Image shape: {obs['observation/image'].shape}")

    for i in range(args.num_steps):
        t0 = time.time()
        result = policy.infer(obs)
        elapsed = (time.time() - t0) * 1000

        actions = result["actions"]
        timing = result.get("policy_timing", {})
        print(f"\n--- Step {i + 1} ---")
        print(f"Inference time: {elapsed:.1f} ms (model: {timing.get('infer_ms', 0):.1f} ms)")
        print(f"Actions shape: {actions.shape}")
        print(f"Actions:\n{actions}")


if __name__ == "__main__":
    main()
