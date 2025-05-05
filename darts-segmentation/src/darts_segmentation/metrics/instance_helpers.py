"""Helper functions for instance segmentation metrics."""

import torch

try:
    import cupy as cp  # type: ignore
    from cucim.skimage.measure import label as label_gpu  # type: ignore

    CUCIM_AVAILABLE = True
except ImportError:
    CUCIM_AVAILABLE = False
    from skimage.measure import label


@torch.no_grad()
def mask_to_instances(x: torch.Tensor, validate_args: bool = False) -> list[torch.Tensor]:
    """Convert a binary segmentation mask into multiple instance masks. Expects a batched version of the input.

    Currently only supports uint8 tensors, hence a maximum number of 255 instances per mask.

    Args:
        x (torch.Tensor): The binary segmentation mask. Shape: (batch_size, height, width), dtype: torch.uint8
        validate_args (bool, optional): Whether to validate the input arguments. Defaults to False.

    Returns:
        list[torch.Tensor]: The instance masks. Length of list: batch_size.
            Shape of a tensor: (height, width), dtype: torch.uint8

    """
    if validate_args:
        assert x.dim() == 3, f"Expected 3 dimensions, got {x.dim()}"
        assert x.dtype == torch.uint8, f"Expected torch.uint8, got {x.dtype}"
        assert x.min() >= 0 and x.max() <= 1, f"Expected binary mask, got {x.min()} and {x.max()}"

    # A note on using lists as separation between instances instead of using a batched tensor:
    # Using a batched tensor with instance numbers (1, 2, 3, ...) would indicate that the instances of the samples
    # are identical. Using a list clearly separates the instances of the samples.

    if CUCIM_AVAILABLE:
        # Check if device is cuda
        assert x.device.type == "cuda", f"Expected device to be cuda, got {x.device.type}"
        x = cp.asarray(x).astype(cp.uint8)

        instances = []
        for x_i in x:
            instances_i = label_gpu(x_i)
            instances_i = torch.tensor(instances_i, dtype=torch.uint8)
            instances.append(instances_i)
        return instances

    else:
        instances = []
        for x_i in x:
            x_i = x_i.cpu().numpy()
            instances_i = label(x_i)
            instances_i = torch.tensor(instances_i, dtype=torch.uint8)
            instances.append(instances_i)
        return instances


@torch.no_grad()
def match_instances(
    instances_target: torch.Tensor,
    instances_preds: torch.Tensor,
    match_threshold: float = 0.5,
    validate_args: bool = False,
) -> tuple[int, int, int]:
    """Match instances between target and prediction masks. Expects non-batched input from skimage.measure.label.

    Args:
        instances_target (torch.Tensor): The instance mask of the target. Shape: (height, width), dtype: torch.uint8
        instances_preds (torch.Tensor): The instance mask of the prediction. Shape: (height, width), dtype: torch.uint8
        match_threshold (float, optional): The threshold for matching instances. Defaults to 0.5.
        validate_args (bool, optional): Whether to validate the input arguments. Defaults to False.

    Returns:
        tuple[int, int, int]: True positives, false positives, false negatives

    """
    if validate_args:
        assert instances_target.dim() == 2, f"Expected 2 dimensions, got {instances_target.dim()}"
        assert instances_preds.dim() == 2, f"Expected 2 dimensions, got {instances_preds.dim()}"
        assert instances_target.dtype == torch.uint8, f"Expected torch.uint8, got {instances_target.dtype}"
        assert instances_preds.dtype == torch.uint8, f"Expected torch.uint8, got {instances_preds.dtype}"
        assert instances_target.shape == instances_preds.shape, (
            f"Shapes do not match: {instances_target.shape} and {instances_preds.shape}"
        )

    height, width = instances_target.shape
    ntargets = instances_target.max().item()
    npreds = instances_preds.max().item()
    # If target or predictions has no instances, return 0 for their respective metrics.
    # If none of them has instances, return 0 for all metrics. (This is implied)
    if ntargets == 0:
        return 0, npreds, 0
    if npreds == 0:
        return 0, 0, ntargets

    # TODO: These are old edge case filter that need revision.
    # They are probably not necessary, since the instance metrics are meaningless for noisy predictions.
    # If there are too many predictions, return all as false positives (this happens when the model is very noisy)
    # if npreds > ntargets * 5:
    #     return 0, npreds, ntargets
    # If there is only one prediction, return all as false negatives (this happens when the model is very noisy)
    # if npreds == 1 and ntargets > 1:
    #     return 0, 1, ntargets

    # Create one-hot encoding of instances, so that each instance is a channel
    instances_target_onehot = torch.zeros((ntargets, height, width), dtype=torch.uint8, device=instances_target.device)
    instances_preds_onehot = torch.zeros((npreds, height, width), dtype=torch.uint8, device=instances_target.device)
    for i in range(ntargets):
        instances_target_onehot[i, :, :] = instances_target == (i + 1)
    for i in range(npreds):
        instances_preds_onehot[i, :, :] = instances_preds == (i + 1)

    # Now the instances are channels, hence tensors of shape (num_instances, height, width)

    # Calculate IoU (we need to do a n-m intersection and union, therefore we need to broadcast)
    intersection = (instances_target_onehot.unsqueeze(1) & instances_preds_onehot.unsqueeze(0)).sum(
        dim=(2, 3)
    )  # Shape: (num_instances_target, num_instances_preds)
    union = (instances_target_onehot.unsqueeze(1) | instances_preds_onehot.unsqueeze(0)).sum(
        dim=(2, 3)
    )  # Shape: (num_instances_target, num_instances_preds)
    iou = intersection / union  # Shape: (num_instances_target, num_instances_preds)

    # Match instances based on IoU
    tp = (iou >= match_threshold).sum().item()
    fp = npreds - tp
    fn = ntargets - tp

    return tp, fp, fn
