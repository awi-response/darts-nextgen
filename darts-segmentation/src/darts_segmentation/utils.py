"""Shared utilities for the inference modules."""

import math
from collections.abc import Callable, Generator

import torch


def patch_coords(h: int, w: int, patch_size: int, overlap: int) -> Generator[tuple[int, int, int, int], None, None]:
    """Yield patch coordinates based on height, width, patch size and margin size.

    Args:
        h (int): Height of the image.
        w (int): Width of the image.
        patch_size (int): Patch size.
        overlap (int): Margin size.

    Yields:
        tuple[int, int, int, int]: The patch coordinates y, x, patch_idx_y and patch_idx_x.

    """
    step_size = patch_size - overlap
    # Substract the overlap from h and w so that an exact match of the last patch won't create a duplicate
    for patch_idx_y, y in enumerate(range(0, h - overlap, step_size)):
        for patch_idx_x, x in enumerate(range(0, w - overlap, step_size)):
            if y + patch_size > h:
                y = h - patch_size
            if x + patch_size > w:
                x = w - patch_size
            yield y, x, patch_idx_y, patch_idx_x


@torch.no_grad()
def create_patches(
    tensor_tiles: torch.Tensor, patch_size: int, overlap: int, return_coords: bool = False
) -> torch.Tensor:
    """Create patches from a tensor.

    Args:
        tensor_tiles (torch.Tensor): The input tensor. Shape: (BS, C, H, W).
        patch_size (int, optional): The size of the patches.
        overlap (int, optional): The size of the overlap.
        return_coords (bool, optional): Whether to return the coordinates of the patches.
            Can be used for debugging. Defaults to False.

    Returns:
        torch.Tensor: The patches. Shape: (BS, N_h, N_w, C, patch_size, patch_size).

    """
    assert tensor_tiles.dim() == 4, f"Expects tensor_tiles to has shape (BS, C, H, W), got {tensor_tiles.shape}"
    bs, c, h, w = tensor_tiles.shape
    assert h > patch_size > overlap
    assert w > patch_size > overlap

    step_size = patch_size - overlap

    # The problem with unfold is that is cuts off the last patch if it doesn't fit exactly
    # Padding could help, but then the next problem is that the view needs to get reshaped (copied in memory)
    # to fit the model input shape. Such a complex view can't be inserted into the model.
    # Since we need, doing it manually is currently our best choice, since be can avoid the padding.
    # patches = (
    #     tensor_tiles.unfold(2, patch_size, step_size).unfold(3, patch_size, step_size).transpose(1, 2).transpose(2, 3)
    # )
    # return patches

    nh, nw = math.ceil((h - overlap) / step_size), math.ceil((w - overlap) / step_size)
    # Create Patches of size (BS, N_h, N_w, C, patch_size, patch_size)
    patches = torch.zeros((bs, nh, nw, c, patch_size, patch_size), device=tensor_tiles.device)
    coords = torch.zeros((nh, nw, 5))
    for i, (y, x, patch_idx_h, patch_idx_w) in enumerate(patch_coords(h, w, patch_size, overlap)):
        patches[:, patch_idx_h, patch_idx_w, :] = tensor_tiles[:, :, y : y + patch_size, x : x + patch_size]
        coords[patch_idx_h, patch_idx_w, :] = torch.tensor([i, y, x, patch_idx_h, patch_idx_w])
    if return_coords:
        return patches, coords
    else:
        return patches


@torch.no_grad()
def predict_in_patches(
    model: Callable, tensor_tiles: torch.Tensor, patch_size: int = 1024, overlap: int = 16, batch_size: int = 8
) -> torch.Tensor:
    """Predict on a tensor.

    Args:
        model: The model to use for prediction.
        tensor_tiles: The input tensor. Shape: (BS, C, H, W).
        patch_size (int): The size of the patches. Defaults to 1024.
        overlap (int): The size of the overlap. Defaults to 16.
        batch_size (int): The batch size for the prediction, NOT the batch_size of input tiles.
            Tensor will be sliced into patches and these again will be infered in batches. Defaults to 8.

    Returns:
        The predicted tensor.

    """
    assert tensor_tiles.dim() == 4, f"Expects tensor_tiles to has shape (BS, C, H, W), got {tensor_tiles.shape}"
    # Add a 1px border to avoid pixel loss when applying the soft margin
    tensor_tiles = torch.nn.functional.pad(tensor_tiles, (1, 1, 1, 1), mode="reflect")
    bs, c, h, w = tensor_tiles.shape
    step_size = patch_size - overlap
    nh, nw = math.ceil((h - overlap) / step_size), math.ceil((w - overlap) / step_size)

    # Create Patches of size (BS, N_h, N_w, C, patch_size, patch_size)
    patches = create_patches(tensor_tiles, patch_size=patch_size, overlap=overlap)

    # Flatten the patches so they fit to the model
    # (BS, N_h, N_w, C, patch_size, patch_size) -> (BS * N_h * N_w, C, patch_size, patch_size)
    patches = patches.view(bs * nh * nw, c, patch_size, patch_size)

    # Create a soft margin for the patches
    margin_ramp = torch.cat(
        [
            torch.linspace(0, 1, overlap),
            torch.ones(patch_size - 2 * overlap),
            torch.linspace(1, 0, overlap),
        ]
    )
    soft_margin = margin_ramp.reshape(1, 1, patch_size) * margin_ramp.reshape(1, patch_size, 1)

    # Infer logits with model and turn into probabilities with sigmoid in a batched manner
    patched_probabilities = torch.zeros_like(patches[:, 0, :, :])
    # Create batches of patches
    patches = patches.split(batch_size)
    for i, batch in enumerate(patches):
        patched_probabilities[i * batch_size : (i + 1) * batch_size] = torch.sigmoid(model(batch)).squeeze(1)

    patched_probabilities = patched_probabilities.view(bs, nh, nw, patch_size, patch_size)

    # Reconstruct the image from the patches
    prediction = torch.zeros(bs, h, w, device=tensor_tiles.device)
    weights = torch.zeros(bs, h, w, device=tensor_tiles.device)

    for y, x, patch_idx_h, patch_idx_w in patch_coords(h, w, patch_size, overlap):
        patch = patched_probabilities[:, patch_idx_h, patch_idx_w]
        prediction[:, y : y + patch_size, x : x + patch_size] += patch * soft_margin
        weights[:, y : y + patch_size, x : x + patch_size] += soft_margin

    # Avoid division by zero
    weights = torch.where(weights == 0, torch.ones_like(weights), weights)
    prediction = prediction / weights

    # Remove the 1px border and the padding
    prediction = prediction[:, 1:-1, 1:-1]
    return prediction