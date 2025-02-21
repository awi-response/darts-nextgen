"""Helper functions for boundary metrics."""

from math import sqrt
from typing import Literal, get_args

import torch
import torchvision

MatchingMetric = Literal["iou", "boundary"]


def _boundary_arg_validation(
    matching_threshold: float = 0.5,
    matching_metric: MatchingMetric = "iou",
    boundary_dilation: float | int = 0.02,
):
    if not (isinstance(matching_threshold, float) and (0 <= matching_threshold <= 1)):
        raise ValueError(
            f"Expected arg `matching_threshold` to be a float in the [0,1] range, but got {matching_threshold}."
        )
    if not isinstance(matching_metric, MatchingMetric):
        raise ValueError(
            f'Expected argument `matching_metric` to be either "iou" or "boundary", but got {matching_metric}.'
        )
    if matching_metric not in get_args(MatchingMetric):
        raise ValueError(
            f'Expected argument `matching_metric` to be either "iou" or "boundary", but got {matching_metric}.'
        )
    if not isinstance(boundary_dilation, float | int) and matching_metric == "boundary":
        raise ValueError(f"Expected argument `boundary_dilation` to be a float or int, but got {boundary_dilation}.")


@torch.no_grad()
def erode_pytorch(mask: torch.Tensor, iterations: int = 1, validate_args: bool = False) -> torch.Tensor:
    """Erodes a binary mask using a square kernel in PyTorch.

    Args:
        mask (torch.Tensor): The binary mask.
            Shape: (batch_size, height, width) or (batch_size, channels, height, width), dtype: torch.uint8
        iterations (int, optional): The size of the erosion. Defaults to 1.
        validate_args (bool, optional): Whether to validate the input arguments. Defaults to False.

    Returns:
        torch.Tensor: The eroded mask. Shape: (batch_size, height, width), dtype: torch.uint8

    """
    if validate_args:
        assert mask.dim() not in [3, 4], f"Expected 3 or 4 dimensions, got {mask.dim()}"
        assert mask.dtype == torch.uint8, f"Expected torch.uint8, got {mask.dtype}"
        assert mask.min() >= 0 and mask.max() <= 1, f"Expected binary mask, got {mask.min()} and {mask.max()}"

    isbatched = mask.dim() == 4
    if not isbatched:
        mask = mask.unsqueeze(1)

    _n, c, _h, _w = mask.shape

    kernel = torch.ones(c, 1, 3, 3, device=mask.device)
    erode = torch.nn.functional.conv2d(mask.float(), kernel, padding=1, stride=1, groups=c)

    for _ in range(iterations - 1):
        erode = torch.nn.functional.conv2d(erode, kernel, padding=1, stride=1, groups=c)

    if isbatched:
        eroded = (erode == erode.max()).to(torch.uint8)
    else:
        eroded = (erode == erode.max()).to(torch.uint8).squeeze(1)
    return eroded


@torch.no_grad()
def get_boundary(
    binary_instances: torch.Tensor,
    dilation: float | int = 0.02,
    validate_args: bool = False,
):
    """Convert instance masks to instance boundaries.

    Args:
        binary_instances (torch.Tensor): Target instance masks. Must be binary.
            Can be batched, one-hot encoded or both. (3 or 4 dimensions).
            The last two dimensions must be height and width.
        dilation (float | int, optional): The dilation (factor) / width of the boundary.
            Dilation in pixels if int, else ratio to calculate `dilation = dilation_ratio * image_diagonal`.
            Default: 0.02
        validate_args (bool, optional): Weather arguments should be validated. Defaults to False.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The boundaries of the instances.

    """
    if validate_args:
        assert binary_instances.dim() in [3, 4], f"Expected 3 or 4 dimensions, got {binary_instances.dim()}"
        assert binary_instances.dtype == torch.uint8, f"Expected torch.uint8, got {binary_instances.dtype}"
        assert (
            binary_instances.min() >= 0 and binary_instances.max() <= 1
        ), f"Expected binary mask, got range between {binary_instances.min()} and {binary_instances.max()}"
        assert isinstance(dilation, float | int), f"Expected float or int, got {type(dilation)}"
        assert dilation >= 0, f"Expected dilation >= 0, got {dilation}"

    if binary_instances.dim() == 3:
        _n, h, w = binary_instances.shape
    else:
        _n, _c, h, w = binary_instances.shape

    if isinstance(dilation, float):
        img_diag = sqrt(h**2 + w**2)
        dilation = round(dilation * img_diag)
        if dilation < 1:
            dilation = 1

    # Pad the instances to avoid boundary issues
    pad = torchvision.transforms.Pad(1)
    binary_instances_padded = pad(binary_instances)

    # Erode the instances to get the boundaries
    eroded = erode_pytorch(binary_instances_padded, iterations=dilation, validate_args=validate_args)

    # Remove the padding
    if binary_instances.dim() == 3:
        eroded = eroded[:, 1:-1, 1:-1]
    else:
        eroded = eroded[:, :, 1:-1, 1:-1]
    # Calculate the boundary of the instances
    boundaries = binary_instances - eroded

    return boundaries


@torch.no_grad()
def instance_boundary_iou(
    instances_target_onehot: torch.Tensor,
    instances_preds_onehot: torch.Tensor,
    dilation: float | int = 0.02,
    validate_args: bool = False,
) -> torch.Tensor:
    """Calculate the IoU of the boundaries of instances.

    Expects non-batched, one-hot encoded input from skimage.measure.label

    Args:
        instances_target_onehot (torch.Tensor): The instance mask of the target.
            Shape: (num_instances, height, width), dtype: torch.uint8
        instances_preds_onehot (torch.Tensor): The instance mask of the prediction.
            Shape: (num_instances, height, width), dtype: torch.uint8
        dilation (float | int, optional): The dilation (factor) / width of the boundary.
            Dilation in pixels if int, else ratio to calculate `dilation = dilation_ratio * image_diagonal`.
            Default: 0.02
        validate_args (bool, optional): Whether to validate the input arguments. Defaults to False.

    Returns:
        torch.Tensor: The IoU of the boundaries. Shape: (num_instances,)

    """
    # Calculate the boundary of the instances
    boundaries_target = get_boundary(instances_target_onehot, dilation, validate_args)
    boundaries_preds = get_boundary(instances_preds_onehot, dilation, validate_args)

    # Calculate the IoU of the boundaries (broadcast because of the different number of instances)
    intersection = (boundaries_target.unsqueeze(1) & boundaries_preds.unsqueeze(0)).sum(
        dim=(2, 3)
    )  # Shape: (num_instances_target, num_instances_preds)
    union = (boundaries_target.unsqueeze(1) | boundaries_preds.unsqueeze(0)).sum(
        dim=(2, 3)
    )  # Shape: (num_instances_target, num_instances_preds)
    iou = intersection / union  # Shape: (num_instances_target, num_instances_preds)

    return iou
