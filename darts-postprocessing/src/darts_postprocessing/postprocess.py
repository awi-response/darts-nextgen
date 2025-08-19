"""Prepare the export, e.g. binarizes the data and convert the float probabilities to uint8."""

import logging
from typing import Literal

import numpy as np
import xarray as xr
from darts_utils.cuda import free_cupy
from skimage.morphology import binary_erosion, disk, label, remove_small_objects
from stopuhr import stopwatch

logger = logging.getLogger(__name__.replace("darts_", "darts."))

try:
    import cupy as cp  # type: ignore
    import cupy_xarray  # noqa: F401
    from cucim.skimage.morphology import binary_erosion as binary_erosion_gpu  # type: ignore
    from cucim.skimage.morphology import disk as disk_gpu  # type: ignore
    from cucim.skimage.morphology import remove_small_objects as remove_small_objects_gpu  # type: ignore

    CUCIM_AVAILABLE = True
    DEFAULT_DEVICE = "cuda"
    logger.debug("GPU-accelerated cucim functions are available.")
except ImportError:
    CUCIM_AVAILABLE = False
    DEFAULT_DEVICE = "cpu"
    logger.debug("GPU-accelerated cucim functions are not available.")


@stopwatch.f("Eroding mask", printer=logger.debug, print_kwargs=["size"])
def erode_mask(
    mask: xr.DataArray, size: int, device: Literal["cuda", "cpu"] | int, edge_size: int | None = None
) -> xr.DataArray:
    """Erode the mask, also set the edges to invalid.

    Args:
        mask (xr.DataArray): The mask to erode.
        size (int): The size of the disk to use for erosion and the edge-cropping.
        device (Literal["cuda", "cpu"] | int): The device to use for erosion.
        edge_size (int, optional): Define a different edge erosion width, will use size parameter if None.

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
            free_cupy()
    else:
        mask.values = binary_erosion(mask.values, disk(size))

    if edge_size is None:
        edge_size = size

    # Mask edges
    mask[:edge_size, :] = 0
    mask[-edge_size:, :] = 0
    mask[:, :edge_size] = 0
    mask[:, -edge_size:] = 0

    return mask


@stopwatch.f("Binarizing probabilities", printer=logger.debug, print_kwargs=["threshold", "min_object_size"])
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
            free_cupy()
    else:
        binarized.values = remove_small_objects(
            binarized.astype(bool).expand_dims("batch", 0).values, min_size=min_object_size
        )[0]

    # Convert back to int8
    binarized = binarized.astype("uint8")

    return binarized


@stopwatch.f(
    "Preparing export",
    printer=logger.debug,
    print_kwargs=[
        "bin_threshold",
        "mask_erosion_size",
        "edge_erosion_size",
        "min_object_size",
        "quality_level",
        "ensemble_subsets",
    ],
)
def prepare_export(
    tile: xr.Dataset,
    bin_threshold: float = 0.5,
    mask_erosion_size: int = 10,
    min_object_size: int = 32,
    quality_level: int | Literal["high_quality", "low_quality", "none"] = 0,
    ensemble_subsets: list[str] = [],
    device: Literal["cuda", "cpu"] | int = DEFAULT_DEVICE,
    edge_erosion_size: int | None = None,
) -> xr.Dataset:
    """Prepare the export, e.g. binarizes the data and convert the float probabilities to uint8.

    Args:
        tile (xr.Dataset): Input tile from inference and / or an ensemble.
        bin_threshold (float, optional): The threshold to binarize the probabilities. Defaults to 0.5.
        mask_erosion_size (int, optional): The size of the disk to use for mask erosion and the edge-cropping.
            Defaults to 10.
        min_object_size (int, optional): The minimum object size to keep in pixel. Defaults to 32.
        quality_level (int | str, optional): The quality level to use for the mask. If a string maps to int.
            high_quality -> 2, low_quality=1, none=0 (apply no masking). Defaults to 0.
        ensemble_subsets (list[str], optional): The ensemble subsets to use for the binarization.
            Defaults to [].
        device (Literal["cuda", "cpu"] | int, optional): The device to use for dilation.
            Defaults to "cuda" if cuda for cucim is available, else "cpu".
        edge_erosion_size (int, optional): If the edge-cropping should have a different witdth, than the (inner) mask
            erosion, set it here. Defaults to `mask_erosion_size`.

    Returns:
        xr.Dataset: Output tile.

    """
    quality_level = (
        quality_level
        if isinstance(quality_level, int)
        else {"high_quality": 2, "low_quality": 1, "none": 0}[quality_level]
    )
    mask = tile["quality_data_mask"] >= quality_level
    if quality_level > 0:
        mask = erode_mask(mask, mask_erosion_size, device, edge_size=edge_erosion_size)  # 0=positive, 1=negative
    tile["extent"] = mask.copy()
    tile["extent"].attrs = {
        "long_name": "Extent of the segmentation",
    }

    # TODO: Refactor according to bands
    def _prep_layer(tile, layername, binarized_layer_name):
        # Binarize the segmentation
        tile[binarized_layer_name] = binarize(tile[layername], bin_threshold, min_object_size, mask, device)
        tile[binarized_layer_name].attrs = {
            "long_name": "Binarized Segmentation",
        }

        # Convert the probabilities to uint8
        # Same but this time with 255 as no-data
        # But first check if this step was already run
        if tile[layername].max() > 1:
            return tile

        intprobs = (tile[layername] * 100).fillna(255).astype("uint8")
        tile[layername] = xr.where(mask, intprobs, 255)
        tile[layername].attrs = {
            "long_name": "Probabilities",
            "units": "%",
        }
        tile[layername] = tile[layername].rio.write_nodata(255)
        return tile

    tile = _prep_layer(tile, "probabilities", "binarized_segmentation")

    # get the names of the model probabilities if available
    # for example 'tcvis' from 'probabilities-tcvis'
    for ensemble_subset in ensemble_subsets:
        tile = _prep_layer(tile, f"probabilities-{ensemble_subset}", f"binarized_segmentation-{ensemble_subset}")

    return tile
