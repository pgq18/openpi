import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_warmup_example() -> dict:
    """Creates a random input example for the warmup policy."""
    return {
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "put carrot on plate",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class WarmupInputs(transforms.DataTransformFn):
    """Converts warmup dataset format to pi0 model inputs.

    The warmup dataset has a base camera and an optional wrist camera.
    If wrist_image is present, it is used as left_wrist_0_rgb.
    State is [ee_xyz(3), ee_euler(3), gripper(1)].
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        state = np.asarray(data["observation/state"], dtype=np.float32)

        # Check if wrist camera data is available
        wrist_image_raw = data.get("observation/wrist_image", None)
        if wrist_image_raw is not None:
            left_wrist = _parse_image(wrist_image_raw)
            left_wrist_mask = np.True_
        else:
            left_wrist = np.zeros_like(base_image)
            left_wrist_mask = np.False_

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": left_wrist_mask,
                "right_wrist_0_rgb": np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            if isinstance(prompt, np.ndarray):
                prompt = prompt.item()
                if isinstance(prompt, bytes):
                    prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class WarmupOutputs(transforms.DataTransformFn):
    """Converts model outputs back to warmup format (7-dim actions)."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :7])}
