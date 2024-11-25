"""Prepare the export, e.g. binarizes the data and convert the float probabilities to uint8."""

import logging
from typing import Literal

import numpy as np
import xarray as xr
from skimage.morphology import binary_erosion, disk, label, remove_small_objects

from darts_postprocessing.utils import free_cuda

logger = logging.getLogger(__name__.replace("darts_", "darts."))

try:
    import cupy as cp
    from cucim.skimage.morphology import binary_erosion as binary_erosion_gpu
    from cucim.skimage.morphology import disk as disk_gpu
    from cucim.skimage.morphology import remove_small_objects as remove_small_objects_gpu

    CUCIM_AVAILABLE = True
    DEFAULT_DEVICE = "cuda"
    logger.debug("GPU-accelerated xrspatial functions are available.")
except ImportError:
    CUCIM_AVAILABLE = False
    DEFAULT_DEVICE = "cpu"
    logger.debug("GPU-accelerated xrspatial functions are not available.")


def erode_mask(mask: xr.DataArray, size: int, device: Literal["cuda", "cpu"] | int) -> xr.DataArray:
    """Erode the mask, also set the edges to invalid.

    Args:
        mask (xr.DataArray): The mask to erode.
        size (int): The size of the disk to use for erosion and the edge-cropping.
        device (Literal["cuda", "cpu"] | int): The device to use for erosion.

    Returns:
        xr.DataArray: The dilated and inverted mask.

    """
    # Clone mask to avoid in-place operations
    mask = mask.copy()

    # Change to dtype uint8 for faster skimage operations
    mask = mask.astype("uint8")

    use_gpu = device == "cuda" or isinstance(device, int)

    # Warn user if use_gpu is set but no GPU is available
    if use_gpu and not CUCIM_AVAILABLE:
        logger.warning(
            f"Device was set to {device}, but GPU acceleration is not available. Calculating TPI and slope on CPU."
        )
        use_gpu = False

    # Dilate the mask with GPU
    if use_gpu:
        device_nr = device if isinstance(device, int) else 0
        logger.debug(f"Moving mask to GPU:{device}.")
        # Check if mask is dask, if not persist it, since dilation can't be calculated from cupy-dask arrays
        if mask.chunks is not None:
            mask = mask.persist()
        with cp.cuda.Device(device_nr):
            mask = mask.cupy.as_cupy()
            mask.values = binary_erosion_gpu(mask.data, disk_gpu(size))
            mask = mask.cupy.as_numpy()
            free_cuda()
    else:
        mask.values = binary_erosion(mask.values, disk(size))

    # Mask edges
    mask[:size, :] = 0
    mask[-size:, :] = 0
    mask[:, :size] = 0
    mask[:, -size:] = 0

    return mask


def binarize(
    probs: xr.DataArray,
    threshold: float,
    min_object_size: int,
    mask: xr.DataArray,
    device: Literal["cuda", "cpu"] | int,
) -> xr.DataArray:
    """Binarize the probabilities based on a threshold and a mask.

    Steps for binarization:
        1. Dilate the mask. This will dilate the edges of holes in the mask as well as the edges of the tile.
        2. Binarize the probabilities based on the threshold.
        3. Remove objects at which overlap with either the edge of the tile or the noData mask.
        4. Remove small objects.

    Args:
        probs (xr.DataArray): Probabilities to binarize.
        threshold (float): Threshold to binarize the probabilities.
        min_object_size (int): Minimum object size to keep.
        mask (xr.DataArray): Mask to apply to the binarized probabilities. Expects 0=negative, 1=postitive.
        device (Literal["cuda", "cpu"] | int): The device to use for removing small objects.

    Returns:
        xr.DataArray: Binarized probabilities.

    """
    use_gpu = device == "cuda" or isinstance(device, int)

    # Warn user if use_gpu is set but no GPU is available
    if use_gpu and not CUCIM_AVAILABLE:
        logger.warning(
            f"Device was set to {device}, but GPU acceleration is not available. Calculating TPI and slope on CPU."
        )
        use_gpu = False

    # Where the output from the ensemble / segmentation is nan turn it into 0, else threshold it
    # Also, where there was no valid input data, turn it into 0
    binarized = (probs.fillna(0) > threshold).astype("uint8")

    # Remove objects at which overlap with either the edge of the tile or the noData mask
    labels = binarized.copy(data=label(binarized, connectivity=2))
    edge_label_ids = np.unique(xr.where(~mask, labels, 0))
    binarized = ~labels.isin(edge_label_ids) & binarized

    # Remove small objects with GPU
    if use_gpu:
        device_nr = device if isinstance(device, int) else 0
        logger.debug(f"Moving binarized to GPU:{device}.")
        # Check if binarized is dask, if not persist it, since remove_small_objects_gpu can't be calculated from
        # cupy-dask arrays
        if binarized.chunks is not None:
            binarized = binarized.persist()
        with cp.cuda.Device(device_nr):
            binarized = binarized.cupy.as_cupy()
            binarized.values = remove_small_objects_gpu(
                binarized.astype(bool).expand_dims("batch", 0).data, min_size=min_object_size
            )[0]
            binarized = binarized.cupy.as_numpy()
            free_cuda()
    else:
        binarized.values = remove_small_objects(
            binarized.astype(bool).expand_dims("batch", 0).values, min_size=min_object_size
        )[0]

    # Convert back to int8
    binarized = binarized.astype("uint8")

    return binarized


def prepare_export(
    tile: xr.Dataset,
    bin_threshold: float = 0.5,
    mask_erosion_size: int = 10,
    min_object_size: int = 32,
    use_quality_mask: bool = False,
    device: Literal["cuda", "cpu"] | int = DEFAULT_DEVICE,
) -> xr.Dataset:
    """Prepare the export, e.g. binarizes the data and convert the float probabilities to uint8.

    Args:
        tile (xr.Dataset): Input tile from inference and / or an ensemble.
        bin_threshold (float, optional): The threshold to binarize the probabilities. Defaults to 0.5.
        mask_erosion_size (int, optional): The size of the disk to use for mask erosion and the edge-cropping.
            Defaults to 10.
        min_object_size (int, optional): The minimum object size to keep in pixel. Defaults to 32.
        use_quality_mask (bool, optional): Whether to use the "quality" mask instead of the
            "valid" mask to mask the output. Defaults to False.
        device (Literal["cuda", "cpu"] | int, optional): The device to use for dilation.
            Defaults to "cuda" if cuda for cucim is available, else "cpu".

    Returns:
        xr.Dataset: Output tile.

    """
    mask_name = "quality_data_mask" if use_quality_mask else "valid_data_mask"
    tile[mask_name] = erode_mask(tile[mask_name], mask_erosion_size, device)  # 0=positive, 1=negative

    def _prep_layer(tile, layername, binarized_layer_name):
        # Binarize the segmentation
        tile[binarized_layer_name] = binarize(tile[layername], bin_threshold, min_object_size, tile[mask_name], device)
        tile[binarized_layer_name].attrs = {
            "long_name": "Binarized Segmentation",
        }

        # Convert the probabilities to uint8
        # Same but this time with 255 as no-data
        # But first check if this step was already run
        if tile[layername].max() > 1:
            return tile

        intprobs = (tile[layername] * 100).fillna(255).astype("uint8")
        tile[layername] = xr.where(tile[mask_name], intprobs, 255)
        tile[layername].attrs = {
            "long_name": "Probabilities",
            "units": "%",
        }
        tile[layername] = tile[layername].rio.write_nodata(255)
        return tile

    tile = _prep_layer(tile, "probabilities", "binarized_segmentation")
    if "probabilities-tcvis" in tile:
        tile = _prep_layer(tile, "probabilities-tcvis", "binarized_segmentation-tcvis")
    if "probabilities-notcvis" in tile:
        tile = _prep_layer(tile, "probabilities-notcvis", "binarized_segmentation-notcvis")

    return tile
