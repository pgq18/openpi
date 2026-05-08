"""
RLDS-based data loader for the warmup dataset (OpenVLA TFDS format).

The warmup dataset has a simpler structure than DROID:
- Flat `action` tensor (7-dim delta actions), not nested `action_dict`
- Single camera image, not multi-camera
- Single language instruction per episode, not multiple
- No joint/gripper proprioception
- Delta actions (zero-pad at trajectory boundaries instead of repeating last action)
"""

from collections.abc import Sequence
import dataclasses
import logging

import tqdm


@dataclasses.dataclass
class WarmupRLDSDataset:
    """Config for a single warmup TFDS dataset."""

    name: str  # TFDS builder name (e.g. "warmup")
    version: str  # TFDS version (e.g. "1.0.0")
    split: str = "train"


class WarmupRldsDataset:
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        datasets: Sequence[WarmupRLDSDataset],
        *,
        shuffle: bool = True,
        action_chunk_size: int = 10,
        shuffle_buffer_size: int = 100_000,
        num_parallel_reads: int = -1,
        num_parallel_calls: int = -1,
    ):
        import dlimp as dl
        import tensorflow as tf
        import tensorflow_datasets as tfds

        tf.config.set_visible_devices([], "GPU")

        def prepare_single_dataset(dataset_cfg: WarmupRLDSDataset):
            builder = tfds.builder(dataset_cfg.name, data_dir=data_dir, version=dataset_cfg.version)
            dataset = dl.DLataset.from_rlds(
                builder, split=dataset_cfg.split, shuffle=shuffle, num_parallel_reads=num_parallel_reads
            )

            dataset = dataset.repeat()

            def restructure(traj):
                obs = {
                    "image": traj["observation"]["image"],
                    "state": traj["observation"]["state"],
                }
                if "wrist_image" in traj["observation"]:
                    obs["wrist_image"] = traj["observation"]["wrist_image"]
                if "relative_state" in traj["observation"]:
                    obs["relative_state"] = traj["observation"]["relative_state"]
                out = {
                    "actions": traj["action"],
                    "observation": obs,
                    "prompt": traj["language_instruction"],
                }
                if "skeleton_action" in traj:
                    out["skeleton_actions"] = traj["skeleton_action"]
                if "residual_action" in traj:
                    out["residual_actions"] = traj["residual_action"]
                return out

            dataset = dataset.traj_map(restructure, num_parallel_calls)

            def chunk_actions(traj):
                """Split trajectory into action chunks with zero-padding for delta actions.

                For delta actions, repeating the last action at trajectory boundaries is
                semantically wrong (it would mean "keep moving by the same delta"). Instead,
                we pad with zeros ("no movement") beyond the trajectory end.
                """
                traj_len = tf.shape(traj["actions"])[0]
                action_dim = tf.shape(traj["actions"])[1]

                action_chunk_indices = tf.broadcast_to(
                    tf.range(action_chunk_size)[None],
                    [traj_len, action_chunk_size],
                ) + tf.broadcast_to(
                    tf.range(traj_len)[:, None],
                    [traj_len, action_chunk_size],
                )

                for key in ("actions", "skeleton_actions", "residual_actions"):
                    if key in traj:
                        padded = tf.concat(
                            [traj[key], tf.zeros((action_chunk_size, action_dim), dtype=tf.float32)],
                            axis=0,
                        )
                        traj[key] = tf.gather(padded, action_chunk_indices)

                return traj

            dataset = dataset.traj_map(chunk_actions, num_parallel_calls)

            dataset = dataset.flatten(num_parallel_calls=num_parallel_calls)

            def decode_images(frame):
                frame["observation"]["image"] = tf.io.decode_image(
                    frame["observation"]["image"], expand_animations=False, dtype=tf.uint8
                )
                if "wrist_image" in frame["observation"]:
                    frame["observation"]["wrist_image"] = tf.io.decode_image(
                        frame["observation"]["wrist_image"], expand_animations=False, dtype=tf.uint8
                    )
                return frame

            return dataset.frame_map(decode_images, num_parallel_calls)

        logging.info(f"Preparing {len(datasets)} warmup datasets...")
        for ds in datasets:
            logging.info(f"  {ds.name}:{ds.version} split={ds.split}")
        all_datasets = [prepare_single_dataset(ds) for ds in datasets]

        if len(all_datasets) == 1:
            final_dataset = all_datasets[0]
        else:
            final_dataset = dl.DLataset.interleave(all_datasets)

        if shuffle:
            final_dataset = final_dataset.shuffle(shuffle_buffer_size)
        final_dataset = final_dataset.batch(batch_size)
        final_dataset = final_dataset.with_ram_budget(1)

        self.dataset = final_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        yield from self.dataset.as_numpy_iterator()

    def __len__(self):
        return 20_000
