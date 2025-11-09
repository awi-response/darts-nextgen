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
    import cupy_xarray
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
    """Erode a binary mask and invalidate edge regions.

    This function applies morphological erosion to shrink valid regions in a mask and
    additionally sets a border region around the entire mask to invalid. This is useful
    for removing unreliable predictions near tile edges and data boundaries.

    Args:
        mask (xr.DataArray): Binary mask to erode (1=valid, 0=invalid). Will be converted to uint8.
        size (int): Radius of the disk structuring element for erosion in pixels.
            Also used as the width of the edge region to invalidate (unless edge_size is specified).
        device (Literal["cuda", "cpu"] | int): Device for processing. Use "cuda" for GPU acceleration,
            "cpu" for CPU processing, or an integer to specify a GPU device number.
        edge_size (int | None, optional): Width of the edge region to set to invalid in pixels.
            If None, uses the `size` parameter. Defaults to None.

    Returns:
        xr.DataArray: Eroded mask (uint8, 1=valid, 0=invalid) with edges invalidated.

    Note:
        GPU acceleration (requires cucim):
        - Significantly faster for large masks
        - Automatically falls back to CPU if cucim is not available
        - Handles both in-memory and dask arrays

        The erosion operation shrinks valid regions by removing pixels within `size` distance
        from invalid regions. Edge invalidation then sets the outermost `edge_size` pixels
        on all sides to 0.

    Example:
        Erode mask to remove edge effects:

        ```python
        from darts_postprocessing import erode_mask

        # Erode by 10 pixels and invalidate 10-pixel edges
        eroded = erode_mask(
            mask=quality_mask,
            size=10,
            device="cuda"
        )

        # Erode by 5 pixels but invalidate 20-pixel edges
        eroded_custom = erode_mask(
            mask=quality_mask,
            size=5,
            edge_size=20,
            device="cpu"
        )
        ```

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
    """Binarize segmentation probabilities with quality-based filtering.

    This function converts continuous probability predictions to binary segmentation masks
    by applying thresholding, removing edge artifacts, and filtering small objects.

    Processing steps:
        1. Threshold probabilities (prob > threshold → 1, else → 0)
        2. Identify and remove objects touching invalid regions or tile edges
        3. Remove objects smaller than min_object_size

    Args:
        probs (xr.DataArray): Segmentation probabilities (float32, range [0-1]).
            NaN values are treated as 0 (no detection).
        threshold (float): Probability threshold for binarization. Typical values: 0.3-0.7.
        min_object_size (int): Minimum object size in pixels. Objects with fewer pixels are removed.
        mask (xr.DataArray): Quality mask (uint8, 1=valid, 0=invalid). Objects overlapping
            invalid regions are removed to avoid artifacts at data boundaries.
        device (Literal["cuda", "cpu"] | int): Device for processing. GPU acceleration
            recommended for large tiles.

    Returns:
        xr.DataArray: Binary segmentation mask (bool, True=object, False=background).

    Note:
        Edge and boundary handling:
        - Objects touching tile edges or invalid data regions (mask==0) are completely removed
        - This prevents partial objects at boundaries from being misclassified
        - Uses connected component analysis (8-connectivity) to identify touching objects

        GPU acceleration:
        - Object removal operations are significantly faster on GPU
        - Automatically handles dask arrays by persisting before GPU operations

    Example:
        Binarize with standard parameters:

        ```python
        from darts_postprocessing import binarize, erode_mask

        # Erode quality mask first
        eroded_mask = erode_mask(
            tile["quality_data_mask"] == 2,  # High quality only
            size=10,
            device="cuda"
        )

        # Binarize predictions
        binary_mask = binarize(
            probs=tile["probabilities"],
            threshold=0.5,
            min_object_size=32,  # Remove objects < 32 pixels
            mask=eroded_mask,
            device="cuda"
        )
        ```

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
    # Using uint8 for further processing
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

    # Convert back to bool
    binarized = binarized.astype("bool")

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
    """Prepare segmentation results for export by applying quality filtering and binarization.

    This is a wrapper function that orchestrates the complete postprocessing pipeline:
    mask erosion, probability masking, and binarization. It processes both ensemble-averaged
    predictions and individual model outputs if present.

    Args:
        tile (xr.Dataset): Input tile from inference containing:
            - probabilities (float32): Segmentation probabilities [0-1]
            - quality_data_mask (uint8): Quality mask (0=invalid, 1=low quality, 2=high quality)
            - probabilities-{subset} (float32): Optional individual model predictions
        bin_threshold (float, optional): Probability threshold for binarization. Defaults to 0.5.
        mask_erosion_size (int, optional): Erosion radius for quality mask in pixels. Also used
            for edge invalidation unless edge_erosion_size is specified. Defaults to 10.
        min_object_size (int, optional): Minimum object size in pixels. Smaller objects are removed.
            Defaults to 32.
        quality_level (int | Literal["high_quality", "low_quality", "none"], optional):
            Quality threshold for masking. Maps to quality_data_mask values:
            - "high_quality" or 2: Only use high quality pixels
            - "low_quality" or 1: Use low and high quality pixels
            - "none" or 0: No quality masking
            Defaults to 0 (no masking).
        ensemble_subsets (list[str], optional): Names of individual models in the ensemble to
            process (e.g., ["with_tcvis", "without_tcvis"]). Defaults to [].
        device (Literal["cuda", "cpu"] | int, optional): Device for processing. Defaults to GPU if available.
        edge_erosion_size (int | None, optional): Separate erosion width for tile edges in pixels.
            If None, uses mask_erosion_size. Defaults to None.

    Returns:
        xr.Dataset: Input tile augmented with:

        Added data variables:
            - extent (uint8): Valid extent mask after erosion (1=valid, 0=invalid).
              Attributes: long_name="Extent of the segmentation"
            - binarized_segmentation (bool): Binary segmentation mask (True=object).
              Attributes: long_name="Binarized Segmentation"
            - binarized_segmentation-{subset} (bool): Binary masks for each ensemble subset
              (only if ensemble_subsets provided)

        Modified data variables:
            - probabilities (float32): Now masked to valid extent (NaN outside)
            - probabilities-{subset} (float32): Masked individual model predictions

    Note:
        Processing pipeline:
        1. Filter quality mask to specified quality_level
        2. Erode quality mask to remove boundary artifacts
        3. Create extent mask from eroded quality mask
        4. Mask probabilities to valid extent (NaN invalid regions)
        5. Binarize masked probabilities with threshold and min object size filter
        6. Repeat steps 4-5 for each ensemble subset if provided

        The extent variable defines the reliable prediction region and should be used to
        clip exported results.

    Example:
        Complete postprocessing workflow:

        ```python
        from darts_postprocessing import prepare_export

        # After ensemble inference
        processed_tile = prepare_export(
            tile=ensemble_result,
            bin_threshold=0.5,
            mask_erosion_size=10,
            min_object_size=32,
            quality_level="high_quality",  # Only high quality pixels
            ensemble_subsets=["with_tcvis", "without_tcvis"],
            device="cuda"
        )

        # Now ready for export with:
        # - processed_tile["binarized_segmentation"]: Main binary result
        # - processed_tile["extent"]: Valid data extent
        # - processed_tile["probabilities"]: Masked probabilities
        ```

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
    tile["extent"].attrs = {"long_name": "Extent of the segmentation"}

    def _prep_layer(tile: xr.Dataset, subset: str | None = None):
        layername = "probabilities" if subset is None else f"probabilities-{subset}"
        binarized_layername = "binarized_segmentation" if subset is None else f"binarized_segmentation-{subset}"

        # Mask the segmentation
        tile[layername] = tile[layername].where(mask)

        # Binarize the segmentation
        tile[binarized_layername] = binarize(tile[layername], bin_threshold, min_object_size, mask, device)
        tile[binarized_layername].attrs = {"long_name": "Binarized Segmentation"}
        return tile

    tile = _prep_layer(tile)

    # get the names of the model probabilities if available
    # for example 'tcvis' from 'probabilities-tcvis'
    for ensemble_subset in ensemble_subsets:
        tile = _prep_layer(tile, ensemble_subset)

    return tile
