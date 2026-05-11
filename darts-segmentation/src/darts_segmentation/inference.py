"""Shared utilities for the inference modules."""

import logging
import math
from collections.abc import Generator
from dataclasses import dataclass
from typing import Literal, NamedTuple, overload
from warnings import deprecated

import torch
import torch.nn as nn

# from rich.progress import track

logger = logging.getLogger(__name__.replace("darts_", "darts."))


@dataclass
class PatchCoordinate:
    """Class to hold the coordinates of a patch."""

    y: int
    """Y coordinate in the original image."""
    x: int
    """X coordinate in the original image."""
    v: int
    """Index of the patch in the y-direction."""
    u: int
    """Index of the patch in the x-direction."""


@dataclass
class PatchedTile:
    """Class to hold the dimensions of the patched tile and the patches itself."""

    bs: int
    """Batch size of the input tensor."""
    c: int
    """Number of channels of the input tensor."""
    height: int
    """Height of the input tensor."""
    width: int
    """Width of the input tensor."""
    nh: int
    """Number of patches in the y-direction."""
    nw: int
    """Number of patches in the x-direction."""
    patches_coordinates: list[PatchCoordinate]
    """List of the coordinates of the patches."""
    patches: torch.Tensor
    """The patches. Shape: (BS, N_h, N_w, C, patch_size, patch_size)."""
    weights: torch.Tensor
    """The weights for the patches. Shape: (BS, H, W)."""

    @property
    def n(self) -> int:
        """The total number of patches."""
        return self.bs * self.nh * self.nw


# TODO: Validate against dtypes
class Patcher:
    """Class to hold the parameters for patching and prediction."""

    @torch.no_grad()
    def __init__(self, patch_size: int, overlap: int):
        """Initialize the patcher.

        Args:
            patch_size (int): The size of the patches.
            overlap (int): The size of the overlap.

        """
        assert patch_size > overlap, f"patch_size must be larger than overlap, got {patch_size=} and {overlap=}"
        self.patch_size = patch_size
        self.overlap = overlap
        self.step_size = patch_size - overlap

        margin_ramp = torch.cat(
            [
                torch.linspace(0, 1, self.overlap),
                torch.ones(self.patch_size - 2 * self.overlap),
                torch.linspace(1, 0, self.overlap),
            ]
        )
        self.soft_margin = margin_ramp.reshape(1, 1, self.patch_size) * margin_ramp.reshape(1, self.patch_size, 1)

    @torch.no_grad()
    def deconstruct(self, tensor_tiles: torch.Tensor) -> PatchedTile:
        """Create patches from a tensor.

        Args:
            tensor_tiles (torch.Tensor): The input tensor. Shape: (BS, C, H, W).
                H and W must be larger than patcher.patch_size and patcher.overlap.

        Returns:
            PatchedTile: The patched tile containing the dimensions and the patches itself.

        """
        assert tensor_tiles.dim() == 4, f"Expects tensor_tiles to has shape (BS, C, H, W), got {tensor_tiles.shape}"
        bs, c, height, width = tensor_tiles.shape
        assert height > self.patch_size, (
            f"Height of the input tensor ({height=}) must be larger than {self.patch_size=}"
        )
        assert width > self.patch_size, f"Width of the input tensor ({width=}) must be larger than {self.patch_size=}"

        if self.soft_margin.device != tensor_tiles.device:
            self.soft_margin = self.soft_margin.to(tensor_tiles.device)

        nh = math.ceil((height - self.overlap) / self.step_size)
        nw = math.ceil((width - self.overlap) / self.step_size)
        patches_coordinates: list[PatchCoordinate] = []
        for patch_idx_y, y in enumerate(range(0, height - self.overlap, self.step_size)):
            for patch_idx_x, x in enumerate(range(0, width - self.overlap, self.step_size)):
                if y + self.patch_size > height:
                    y = height - self.patch_size
                if x + self.patch_size > width:
                    x = width - self.patch_size
                patches_coordinates.append(PatchCoordinate(y, x, patch_idx_y, patch_idx_x))

        # Create Patches of size (BS, N_h, N_w, C, patch_size, patch_size)
        patches = torch.zeros((bs, nh, nw, c, self.patch_size, self.patch_size), device=tensor_tiles.device)
        weights = torch.zeros((bs, height, width), device=tensor_tiles.device)
        # The problem with unfold is that is cuts off the last patch if it doesn't fit exactly
        # Padding could help, but then the next problem is that the view needs to get reshaped (copied in memory)
        # to fit the model input shape. Such a complex view can't be inserted into the model.
        # Since we need a perfect fit, doing it manually is currently our best choice, since be can avoid the padding.
        for pc in patches_coordinates:
            # Assign a view of the original tensor to the patch tensor to avoid copying data in memory
            yslice, xslice = slice(pc.y, pc.y + self.patch_size), slice(pc.x, pc.x + self.patch_size)
            patch = tensor_tiles[:, :, yslice, xslice]
            patches[:, pc.v, pc.u, :] = patch
            weights[:, yslice, xslice] += self.soft_margin
        # Avoid division by zero
        weights = torch.where(weights == 0, torch.ones_like(weights), weights)
        return PatchedTile(bs, c, height, width, nh, nw, patches_coordinates, patches, weights)

    @torch.no_grad()
    def reconstruct(self, probability_patches: torch.Tensor, patched_tile: PatchedTile) -> torch.Tensor:
        """Reconstruct the image from the patches.

        Args:
            probability_patches (torch.Tensor): The predicted patches. Shape: (BS, N_h, N_w, patch_size, patch_size).
            patched_tile (PatchedTile): The patched tile containing the dimensions and the patches itself.

        Returns:
            torch.Tensor: The reconstructed image. Shape: (BS, H, W).

        """
        if self.soft_margin.device != probability_patches.device:
            self.soft_margin = self.soft_margin.to(probability_patches.device)

        reconstruction_shape = (patched_tile.bs, patched_tile.height, patched_tile.width)
        predictions = torch.zeros(reconstruction_shape, device=probability_patches.device)
        for pc in patched_tile.patches_coordinates:
            patch = probability_patches[:, pc.v, pc.u]
            yslice, xslice = slice(pc.y, pc.y + self.patch_size), slice(pc.x, pc.x + self.patch_size)
            predictions[:, yslice, xslice] += patch * self.soft_margin
        predictions = predictions / patched_tile.weights
        return predictions


class _BatchWithSlice(NamedTuple):
    data: torch.Tensor
    bslice: slice


@torch.no_grad()
def _gen_batches(patches: torch.Tensor, batch_size: int) -> Generator[_BatchWithSlice]:
    # Infer logits with model and turn into probabilities with sigmoid in a batched manner
    # Split the stack of patches into batches
    # (BS * N_h * N_w, C, patch_size, patch_size) -> tuple of n=batch_size [C, patch_size, patch_size]
    n_skipped = 0
    for i, batch in enumerate(torch.split(patches, batch_size)):
        bslice = slice(i * batch_size, (i + 1) * batch_size)
        batch_is_nan = torch.isnan(batch)
        # If batch contains only nans, skip it
        if batch_is_nan.all(dim=0).any():
            n_skipped += 1
            continue
        # If batch contains some nans, replace them with zeros
        if batch_is_nan.any():
            # Clone it to not alter the original, as this may be a view of the orginal dataset
            batch = batch.clone()
            batch[batch_is_nan] = 0
        yield _BatchWithSlice(data=batch, bslice=bslice)

    if n_skipped > 0:
        logger.debug(f"Skipping {n_skipped} batches because they only contained NaNs")


@torch.no_grad()
def _forward(patches: torch.Tensor, model: nn.Module, batch_size: int) -> torch.Tensor:
    # ?: This function assumes that the patches are already on the correct device.
    # This reduces the overhead of moving the patches to the device in batches,
    # but it also means that the caller needs to handle the device management.
    assert patches.ndim == 4, f"Expects patches to have shape (N, C, patch_size, patch_size), got {patches.shape}"
    n, _, ps, _ = patches.shape
    probability_patches = torch.zeros((n, ps, ps), device=patches.device, dtype=torch.float32)

    for batch, bslice in _gen_batches(patches, batch_size):
        probability_patches[bslice] = torch.sigmoid(model(batch)).squeeze(1)

    return probability_patches


@torch.no_grad()
def _forward_on_device(patches: torch.Tensor, model: nn.Module, batch_size: int, device: torch.device) -> torch.Tensor:
    # ?: This is the original implementation of our inference function
    # It can handle different devices for the patches and the model,
    # which results in a lot of device transferse and overheads
    assert patches.ndim == 4, f"Expects patches to have shape (N, C, patch_size, patch_size), got {patches.shape}"
    n, _, ps, _ = patches.shape
    probability_patches = torch.zeros((n, ps, ps), device=patches.device, dtype=torch.float32)

    for batch, bslice in _gen_batches(patches, batch_size):
        batch = batch.to(device)
        # logger.debug(f"Predicting on batch {i + 1}/{len(patches)}")
        probability_patches[bslice] = torch.sigmoid(model(batch)).squeeze(1).to(probability_patches.device)
        batch = batch.to(probability_patches)  # Transfer back to the original device to avoid memory leaks

    return probability_patches


@torch.no_grad()
def _forward_streaming(patches: torch.Tensor, model: nn.Module, batch_size: int, device: torch.device) -> torch.Tensor:
    assert device.type == "cuda", "Streaming inference is only implemented for CUDA devices"
    assert patches.ndim == 4, f"Expects patches to have shape (N, C, patch_size, patch_size), got {patches.shape}"

    n, _, ps, _ = patches.shape
    probability_patches = torch.empty((n, ps, ps), device=patches.device, dtype=torch.float32, pin_memory=True)

    batches = list(_gen_batches(patches, batch_size))
    assert len(batches) > 0, "No valid batches to process, please check your input data for NaNs"

    copy_stream = torch.cuda.Stream(device=device)
    compute_stream = torch.cuda.current_stream(device=device)

    moving_batch = None
    with torch.cuda.stream(copy_stream):
        moving_batch = batches[0].data.to(device, non_blocking=True)

    for i, batch in enumerate(batches):
        compute_stream.wait_stream(copy_stream)
        batch_on_device = moving_batch
        assert batch_on_device is not None
        probabilities = torch.sigmoid(model(batch_on_device)).squeeze(1)
        probability_patches[batch.bslice].copy_(probabilities, non_blocking=True)
        batch_on_device.record_stream(compute_stream)

        if i + 1 < len(batches):
            next_prepared_batch = batches[i + 1]
            if next_prepared_batch is not None:
                with torch.cuda.stream(copy_stream):
                    moving_batch = next_prepared_batch.data.to(device, non_blocking=True)
            else:
                moving_batch = None

    torch.cuda.synchronize(device)

    return probability_patches


@torch.no_grad()
def forward(
    tensor_tiles: torch.Tensor,
    model: nn.Module,
    patcher: Patcher,
    batch_size: int,
    device: torch.device,
    reflection: int = 0,
) -> torch.Tensor:
    """Predict on a tensor.

    Args:
        tensor_tiles (torch.Tensor): The input tensor. Shape: (BS, C, H, W).
        model (nn.Module): The model to use for prediction.
        patcher (Patcher): The patcher to use for patching and reconstructing the image.
        batch_size (int): The batch size for the prediction, NOT the batch_size of input tiles.
            Tensor will be sliced into patches and these again will be infered in batches.
        device (torch.device): The device to use for the prediction.
        reflection (int): Reflection-Padding which will be applied to the edges of the tensor.

    Returns:
        torch.Tensor: The predicted tensor.

    """
    logger.debug(
        f"Predicting on a tensor with shape {tensor_tiles.shape} with patch_size {patcher.patch_size},"
        f" overlap {patcher.overlap} and {batch_size=} on {device=}"
    )
    p = 1 + reflection
    tensor_tiles = torch.nn.functional.pad(tensor_tiles, (p, p, p, p), mode="reflect")
    patched_tile = patcher.deconstruct(tensor_tiles)

    # Flatten the patches so they fit to the model
    # (BS, N_h, N_w, C, patch_size, patch_size) -> (BS * N_h * N_w, C, patch_size, patch_size)
    patches = patched_tile.patches.view(patched_tile.n, patched_tile.c, patcher.patch_size, patcher.patch_size)

    # Inference approach depending on the used device
    if device == patches.device:
        probability_patches = _forward(patches, model, batch_size)
    elif device.type == "cuda" and patches.device.type == "cpu":
        if not patches.is_pinned():
            patches = patches.pin_memory()
        probability_patches = _forward_streaming(patches, model, batch_size, device)
    else:
        probability_patches = _forward_on_device(patches, model, batch_size, device)

    # Reshape the probability patches back to (BS, N_h, N_w, patch_size, patch_size)
    output_shape = (patched_tile.bs, patched_tile.nh, patched_tile.nw, patcher.patch_size, patcher.patch_size)
    probability_patches = probability_patches.view(output_shape)

    predictions = patcher.reconstruct(probability_patches, patched_tile)
    # Remove the 1px border and the padding
    predictions = predictions[:, p:-p, p:-p]
    return predictions


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


@overload
def create_patches(
    tensor_tiles: torch.Tensor, patch_size: int, overlap: int, return_coords: Literal[True]
) -> tuple[torch.Tensor, torch.Tensor]: ...


@overload
def create_patches(
    tensor_tiles: torch.Tensor, patch_size: int, overlap: int, return_coords: Literal[False]
) -> torch.Tensor: ...


@deprecated(
    "This function is not used anymore and will be removed in the future. Use the ImagePatchesDimensions class instead."
)
@torch.no_grad()
def create_patches(
    tensor_tiles: torch.Tensor, patch_size: int, overlap: int, return_coords: bool = False
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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
    logger.debug(
        f"Creating patches from a tensor with shape {tensor_tiles.shape} "
        f"with patch_size {patch_size} and overlap {overlap}"
    )
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


@overload
def predict_in_patches(
    model: nn.Module,
    tensor_tiles: torch.Tensor,
    patch_size: int,
    overlap: int,
    batch_size: int,
    reflection: int,
    device: torch.device,
    return_weights: Literal[True],
) -> tuple[torch.Tensor, torch.Tensor]: ...


@overload
def predict_in_patches(
    model: nn.Module,
    tensor_tiles: torch.Tensor,
    patch_size: int,
    overlap: int,
    batch_size: int,
    reflection: int,
    device: torch.device,
    return_weights: Literal[False],
) -> torch.Tensor: ...


@deprecated("This function is not used anymore and will be removed in the future. Use the Patcher class instead.")
@torch.no_grad()
def predict_in_patches(
    model: nn.Module,
    tensor_tiles: torch.Tensor,
    patch_size: int,
    overlap: int,
    batch_size: int,
    reflection: int,
    device: torch.device,
    return_weights: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Predict on a tensor.

    Args:
        model: The model to use for prediction.
        tensor_tiles: The input tensor. Shape: (BS, C, H, W).
        patch_size (int): The size of the patches.
        overlap (int): The size of the overlap.
        batch_size (int): The batch size for the prediction, NOT the batch_size of input tiles.
            Tensor will be sliced into patches and these again will be infered in batches.
        reflection (int): Reflection-Padding which will be applied to the edges of the tensor.
        device (torch.device): The device to use for the prediction.
        return_weights (bool, optional): Whether to return the weights. Can be used for debugging. Defaults to False.

    Returns:
        The predicted tensor.

    """
    logger.debug(
        f"Predicting on a tensor with shape {tensor_tiles.shape} "
        f"with patch_size {patch_size}, overlap {overlap} and batch_size {batch_size} on device {device}"
    )
    assert tensor_tiles.dim() == 4, f"Expects tensor_tiles to has shape (BS, C, H, W), got {tensor_tiles.shape}"
    # Add a 1px + reflection border to avoid pixel loss when applying the soft margin and to reduce edge-artefacts
    p = 1 + reflection
    tensor_tiles = torch.nn.functional.pad(tensor_tiles, (p, p, p, p), mode="reflect")
    bs, c, h, w = tensor_tiles.shape
    step_size = patch_size - overlap
    nh, nw = math.ceil((h - overlap) / step_size), math.ceil((w - overlap) / step_size)

    # Create Patches of size (BS, N_h, N_w, C, patch_size, patch_size)
    patches = create_patches(tensor_tiles, patch_size=patch_size, overlap=overlap, return_coords=False)

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
    soft_margin = soft_margin.to(patches.device)

    # Infer logits with model and turn into probabilities with sigmoid in a batched manner
    # TODO: check with ingmar and jonas if moving all patches to the device at the same time is a good idea
    patched_probabilities = torch.zeros_like(patches[:, 0, :, :])
    patches = patches.split(batch_size)
    n_skipped = 0
    for i, batch in enumerate(patches):
        # If batch contains only nans, skip it
        if torch.isnan(batch).all(dim=0).any():
            patched_probabilities[i * batch_size : (i + 1) * batch_size] = 0
            n_skipped += 1
            continue
        # If batch contains some nans, replace them with zeros
        batch[torch.isnan(batch)] = 0

        batch = batch.to(device)
        # logger.debug(f"Predicting on batch {i + 1}/{len(patches)}")
        patched_probabilities[i * batch_size : (i + 1) * batch_size] = (
            torch.sigmoid(model(batch)).squeeze(1).to(patched_probabilities.device)
        )
        batch = batch.to(patched_probabilities.device)  # Transfer back to the original device to avoid memory leaks

    if n_skipped > 0:
        logger.debug(f"Skipped {n_skipped} batches because they only contained NaNs")

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
    prediction = prediction[:, p:-p, p:-p]

    if return_weights:
        return prediction, weights
    else:
        return prediction
