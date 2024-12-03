"""Functions to prepare the training data for the segmentation model training."""

import logging
from collections.abc import Generator
from typing import Literal

import geopandas as gpd
import torch
import xarray as xr

# TODO: move erode_mask to darts_utils, since uasge is not limited to prepare_export
from darts_postprocessing.prepare_export import erode_mask
from geocube.api.core import make_geocube

from darts_segmentation.utils import create_patches

logger = logging.getLogger(__name__.replace("darts_", "darts."))


def create_training_patches(
    tile: xr.Dataset,
    labels: gpd.GeoDataFrame,
    bands: list[str],
    norm_factors: dict[str, float],
    patch_size: int,
    overlap: int,
    include_allzero: bool,  # TODO: rename to include_nopositive
    include_nan_edges: bool,
    device: Literal["cuda", "cpu"] | int,
    mask_erosion_size: int,
) -> Generator[tuple[torch.tensor, torch.tensor]]:
    """Create training patches from a tile and labels.

    Args:
        tile (xr.Dataset): The input tile, containing preprocessed, harmonized data.
        labels (gpd.GeoDataFrame): The labels to be used for training.
        bands (list[str]): The bands to be used for training. Must be present in the tile.
        norm_factors (dict[str, float]): The normalization factors for the bands.
        patch_size (int): The size of the patches.
        overlap (int): The size of the overlap.
        include_allzero (bool): Whether to include patches where the labels are all zero.
        include_nan_edges (bool): Whether to include patches where the input data has nan values at the edges.
        device (Literal["cuda", "cpu"] | int): The device to use for the erosion.
        mask_erosion_size (int): The size of the disk to use for erosion.

    Yields:
        Generator[tuple[torch.tensor, torch.tensor]]: A tuple containing the input and the labels as pytorch tensors.
            The input has the format (C, H, W), the labels (H, W).

    Raises:
        ValueError: If a band is not found in the preprocessed data.

    """
    # Rasterize the labels
    labels_rasterized = 1 - make_geocube(labels, measurements=["id"], like=tile).id.isnull()  # noqa: PD003

    # Filter out the nodata values (class 2 -> invalid data)
    mask = erode_mask(tile["valid_data_mask"], mask_erosion_size, device)
    mask = tile["valid_data_mask"]
    labels_rasterized = xr.where(mask, labels_rasterized, 2)

    # Normalize the bands and clip the values
    for band in bands:
        if band not in tile:
            raise ValueError(f"Band '{band}' not found in the preprocessed data.")
        with xr.set_options(keep_attrs=True):
            tile[band] = tile[band] * norm_factors[band]
            tile[band] = tile[band].clip(0, 1)

    # Replace invalid values with nan (used for nan check later on)
    tile = xr.where(tile["valid_data_mask"], tile, float("nan"))

    # Convert to dataaray and select the bands (bands are now in specified order)
    tile = tile.to_dataarray(dim="band").sel(band=bands)

    # Transpose to (C, H, W)
    tile = tile.transpose("band", "y", "x")
    labels_rasterized = labels_rasterized.transpose("y", "x")

    # Convert to tensor
    tensor_tile = torch.tensor(tile.values).float()
    tensor_labels = torch.tensor(labels_rasterized.values).float()

    assert tensor_tile.dim() == 3, f"Expects tensor_tile to has shape (C, H, W), got {tensor_tile.shape}"
    assert tensor_labels.dim() == 2, f"Expects tensor_labels to has shape (H, W), got {tensor_labels.shape}"

    # Create patches
    tensor_patches = create_patches(tensor_tile.unsqueeze(0), patch_size, overlap)
    tensor_patches = tensor_patches.reshape(-1, len(bands), patch_size, patch_size)
    tensor_labels = create_patches(tensor_labels.unsqueeze(0).unsqueeze(0), patch_size, overlap)
    tensor_labels = tensor_labels.reshape(-1, patch_size, patch_size)

    # Turn the patches into a list of tuples
    n_patches = tensor_patches.shape[0]
    for i in range(n_patches):
        x = tensor_patches[i]
        y = tensor_labels[i]

        if not include_allzero and not (y == 1).any():
            continue

        if not include_nan_edges and torch.isnan(x).any():
            continue

        # Skip where there are less than 10% visible pixel
        if ((y != 2).sum() / y.numel()) < 0.1:
            continue

        # Skip patches where everything is nan
        if torch.isnan(x).all():
            continue

        # Convert all nan values to 0
        x[torch.isnan(x)] = 0

        logger.debug(f"Yielding patch {i} with\n\t{x=}\n\t{y=}")
        yield x, y
