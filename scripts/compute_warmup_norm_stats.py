"""Compute normalization statistics for the warmup TFDS dataset.

Usage:
    cd openpi
    uv run scripts/compute_warmup_norm_stats.py \
        --rlds_data_dir ../datasets \
        --output_dir ./assets/pi05_warmup/warmup \
        --batch_size 32
"""

import argparse
import pathlib

import numpy as np
import tqdm

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.policies.warmup_policy as warmup_policy
import openpi.training.warmup_rlds_dataset as warmup_rlds_dataset
import openpi.transforms as _transforms


class RemoveStrings(_transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def main():
    parser = argparse.ArgumentParser(description="Compute normalization stats for warmup dataset")
    parser.add_argument("--rlds_data_dir", type=str, required=True, help="Path to dir containing warmup/1.0.0/")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for norm_stats.json")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_batches", type=int, default=None, help="Limit number of batches to process")
    args = parser.parse_args()

    # 1. Create dataset
    dataset = warmup_rlds_dataset.WarmupRldsDataset(
        data_dir=args.rlds_data_dir,
        batch_size=args.batch_size,
        shuffle=False,
        action_chunk_size=10,
        datasets=[
            warmup_rlds_dataset.WarmupRLDSDataset(name="warmup", version="5.0.0", split="train"),
        ],
    )

    # 2. Apply repack + data transforms (no normalization)
    from openpi.training.data_loader import IterableTransformedDataset
    from openpi.models.pi0_config import Pi0Config

    model_config = Pi0Config(pi05=True, action_dim=32, action_horizon=10)

    transform_pipeline = [
        _transforms.RepackTransform(
            {
                "observation/image": "observation/image",
                "observation/state": "observation/state",
                "observation/relative_state": "observation/relative_state",
                "actions": "actions",
                "skeleton_actions": "skeleton_actions",
                "residual_actions": "residual_actions",
                "prompt": "prompt",
            }
        ),
        warmup_policy.WarmupInputs(model_type=_model.ModelType.PI05),
        RemoveStrings(),
    ]

    transformed_dataset = IterableTransformedDataset(
        dataset,
        transform_pipeline,
        is_batched=True,
    )

    # 3. Compute running stats
    keys = ["state", "relative_state", "actions", "skeleton_actions", "residual_actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    num_batches = args.max_batches if args.max_batches is not None else (len(dataset) // args.batch_size)
    for i, batch in tqdm.tqdm(enumerate(transformed_dataset), total=num_batches, desc="Computing stats"):
        for key in keys:
            if key in batch:
                stats[key].update(np.asarray(batch[key]))
        if i + 1 >= num_batches:
            break

    norm_stats = {key: stats[key].get_statistics() for key in keys}

    # 4. Save
    output_path = pathlib.Path(args.output_dir)
    print(f"Saving norm stats to: {output_path}")
    for key, s in norm_stats.items():
        print(f"  {key}: mean={s.mean}, std={s.std}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    main()
