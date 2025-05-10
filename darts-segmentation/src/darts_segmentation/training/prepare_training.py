"""Functions to prepare the training data for the segmentation model training."""

import logging
from collections import defaultdict
from collections.abc import Generator
from dataclasses import dataclass
from typing import Literal

import geopandas as gpd
import lovely_tensors
import torch
import xarray as xr

# TODO: move erode_mask to darts_utils, since uasge is not limited to prepare_export
from darts_postprocessing.postprocess import erode_mask
from geocube.api.core import make_geocube

from darts_segmentation.utils import create_patches

logger = logging.getLogger(__name__.replace("darts_", "darts."))


@dataclass
class PatchCoords:
    """Wrapper which stores the coordinate information of a patch in the original image."""

    i: int
    patch_idx_y: int
    patch_idx_x: int
    y: slice
    x: slice

    @classmethod
    def from_tensor(cls, coords: torch.Tensor, patch_size: int) -> "PatchCoords":
        """Create a PatchCoords object from the returned coord tensor of `create_patches`.

        Args:
            coords (torch.Tensor): The coordinates of the patch in the original image, from `create_patches`.
            patch_size (int): The size of the patch.

        Returns:
            PatchCoords: The coordinates of the patch in the original image.

        """
        i, y, x, h, w = coords.int().numpy()
        return cls(
            i=i,
            patch_idx_y=h.item(),
            patch_idx_x=w.item(),
            y=slice(y.item(), y.item() + patch_size),
            x=slice(x.item(), x.item() + patch_size),
        )


# TODO: Redo the "create_patches" functionality so that it works with numpy, xarray and torch.
# Make is more useful and generic, trying to also keep the coordinate information as long as possible.


def create_training_patches(  # noqa: C901
    tile: xr.Dataset,
    labels: gpd.GeoDataFrame,
    bands: list[str],
    norm_factors: dict[str, float],
    patch_size: int,
    overlap: int,
    exclude_nopositive: bool,
    exclude_nan: bool,
    device: Literal["cuda", "cpu"] | int,
    mask_erosion_size: int,
) -> Generator[tuple[torch.tensor, torch.tensor, PatchCoords]]:
    """Create training patches from a tile and labels.

    Args:
        tile (xr.Dataset): The input tile, containing preprocessed, harmonized data.
        labels (gpd.GeoDataFrame): The labels to be used for training.
        bands (list[str]): The bands to be used for training. Must be present in the tile.
        norm_factors (dict[str, float]): The normalization factors for the bands.
        patch_size (int): The size of the patches.
        overlap (int): The size of the overlap.
        exclude_nopositive (bool): Whether to exclude patches where the labels do not contain positives.
        exclude_nan (bool): Whether to exclude patches where the input data has nan values.
        device (Literal["cuda", "cpu"] | int): The device to use for the erosion.
        mask_erosion_size (int): The size of the disk to use for erosion.

    Yields:
        Generator[tuple[torch.tensor, torch.tensor]]: A tuple containing the input and the labels as pytorch tensors.
            The input has the format (C, H, W), the labels (H, W).

    Raises:
        ValueError: If a band is not found in the preprocessed data.

    """
    if len(labels) == 0 and exclude_nopositive:
        logger.warning("No labels found in the labels GeoDataFrame. Skipping.")
        return

    # Rasterize the labels
    if len(labels) > 0:
        labels_rasterized = 1 - make_geocube(labels, measurements=["id"], like=tile).id.isnull()
    else:
        labels_rasterized = xr.zeros_like(tile["quality_data_mask"])

    # Filter out the nodata values (class 2 -> invalid data)
    quality_mask = erode_mask(tile["quality_data_mask"] == 2, mask_erosion_size, device)
    labels_rasterized = xr.where(quality_mask, labels_rasterized, 2)

    # Transpose to (H, W)
    tile = tile.transpose("y", "x")

    tensor_labels = torch.tensor(labels_rasterized.values).float()
    tensor_tile = torch.zeros((len(bands), tile.dims["y"], tile.dims["x"]), device=device)
    invalid_mask = (tile["quality_data_mask"] == 0).values
    # This will also order the data into the correct order of bands
    for i, band in enumerate(bands):
        if band not in tile:
            raise ValueError(f"Band '{band}' not found in the preprocessed data.")
        band_data = torch.tensor(tile[band].values, device=device).float()
        # Normalize the bands and clip the values
        band_data = band_data * norm_factors[band]
        band_data = band_data.clip(0, 1)
        # Apply the quality mask
        band_data[invalid_mask] = float("nan")
        # Merge with the tile and move back to cpu
        tensor_tile[i] = band_data

    assert tensor_tile.dim() == 3, f"Expects tensor_tile to has shape (C, H, W), got {tensor_tile.shape}"
    assert tensor_labels.dim() == 2, f"Expects tensor_labels to has shape (H, W), got {tensor_labels.shape}"

    # Create patches
    tensor_patches = create_patches(tensor_tile.unsqueeze(0), patch_size, overlap)
    tensor_patches = tensor_patches.reshape(-1, len(bands), patch_size, patch_size)
    tensor_labels, tensor_coords = create_patches(
        tensor_labels.unsqueeze(0).unsqueeze(0), patch_size, overlap, return_coords=True
    )
    tensor_labels = tensor_labels.reshape(-1, patch_size, patch_size)
    tensor_coords = tensor_coords.reshape(-1, 5)

    # Turn the patches into a list of tuples
    n_patches = tensor_patches.shape[0]
    n_skipped = defaultdict(int)
    for i in range(n_patches):
        x = tensor_patches[i]
        y = tensor_labels[i]
        coords = PatchCoords.from_tensor(tensor_coords[i], patch_size)

        if exclude_nopositive and not (y == 1).any():
            n_skipped["nopositive"] += 1
            continue

        if exclude_nan and torch.isnan(x).any():
            n_skipped["nan"] += 1
            continue

        # Skip where there are less than 10% visible pixel
        if ((y != 2).sum() / y.numel()) < 0.1:
            n_skipped["visible"] += 1
            continue

        # Skip patches where everything is nan
        if torch.isnan(x).all():
            n_skipped["allnan"] += 1
            continue

        # Convert all nan values to 0
        x[torch.isnan(x)] = 0

        xlvly = lovely_tensors.lovely(x, color=False)
        ylvly = lovely_tensors.lovely(y, color=False)
        logger.debug(f"Yielding patch {i} with\n\tx={xlvly}\n\ty={ylvly}")
        yield x.cpu(), y.cpu(), coords

    if exclude_nopositive:
        logger.debug(f"Skipped {n_skipped['nopositive']} patches with no positive labels")
    if exclude_nan:
        logger.debug(f"Skipped {n_skipped['nan']} patches with nan values")
    logger.debug(f"Skipped {n_skipped['visible']} patches with less than 10% visible pixels")
    logger.debug(f"Skipped {n_skipped['allnan']} patches where everything is nan")
    logger.debug(f"Yielded {n_patches - sum(n_skipped.values())} patches")
    logger.debug(f"Total patches: {n_patches}")
