"""Patching methods for inference."""

import logging

import torch
import torch.nn.functional as F  # noqa: N812

from darts_superresolution.data_processing.util import transform_augment_tensor

logger = logging.getLogger(__name__)


def create_patches_from_tile(input_tensor: torch.Tensor) -> torch.Tensor:
    """Generate patches from a tile.

    Args:
        input_tensor (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The patches.

    """
    logger.debug(f"Creating patched from tile of shape {input_tensor.shape}")
    _n, _c, size_x, size_y = input_tensor.size()
    pad_x = (60 - (size_x % 60)) % 60
    pad_y = (60 - (size_y % 60)) % 60

    logger.debug(f"Padding the input tensor with {pad_x} on the x-axis and {pad_y} on the y-axis")

    if pad_x % 2 == 0 and pad_y % 2 == 0:
        input_tensor = F.pad(input_tensor, (int(pad_y / 2), int(pad_y / 2), int(pad_x / 2), int(pad_x / 2)), "reflect")

    elif pad_x % 2 == 0 and pad_y % 2 != 0:
        input_tensor = F.pad(input_tensor, (pad_y, 0, int(pad_x / 2), int(pad_x / 2)), "reflect")

    elif pad_x % 2 != 0 and pad_y % 2 == 0:
        input_tensor = F.pad(input_tensor, (int(pad_y / 2), int(pad_y / 2), 0, pad_x), "reflect")

    else:
        input_tensor = F.pad(input_tensor, (0, pad_y, 0, pad_x), "reflect")

    logger.debug(f"Padded tensor shape: {input_tensor.shape}")

    patches = torch.stack(torch.chunk(input_tensor, int(input_tensor.shape[2] / 60), dim=2))
    patches = patches.view(patches.shape[0] * patches.shape[1], patches.shape[2], patches.shape[3], patches.shape[4])
    patches = torch.stack(torch.chunk(patches, int(patches.shape[3] / 60), dim=3))
    patches = patches.view(patches.shape[0] * patches.shape[1], patches.shape[2], patches.shape[3], patches.shape[4])

    logger.debug(f"Generated patches of shape {patches.shape}")

    patches[:, 0, :, :] = (patches[:, 0, :, :] - 31) / (1583 - 31)
    patches[:, 1, :, :] = (patches[:, 1, :, :] - 43) / (1973 - 43)
    patches[:, 2, :, :] = (patches[:, 2, :, :] - 25) / (2225 - 25)
    patches[:, 3, :, :] = (patches[:, 3, :, :] - 83) / (4553 - 83)

    upsample_patches = torch.nn.Upsample(size=(192, 192), mode="bicubic")

    patches_up = upsample_patches(patches)

    logger.debug(f"Upsampled patches of shape {patches_up.shape}")

    [patches_up] = transform_augment_tensor([patches_up], split="val", min_max=(-1, 1))

    logger.debug(f"Transformed patches of shape {patches_up.shape}")

    return patches_up


# Inference

# Stitching back together
