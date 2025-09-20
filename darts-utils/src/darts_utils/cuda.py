"""Utility functions around cuda, e.g. memory management."""

import gc
import logging
from typing import Literal

import xarray as xr
from xrspatial.utils import has_cuda_and_cupy

logger = logging.getLogger(__name__.replace("darts_", "darts."))

if has_cuda_and_cupy():
    import cupy as cp  # type: ignore
    import cupy_xarray  # noqa: F401 # type: ignore

    DEFAULT_DEVICE = "cuda"
else:
    DEFAULT_DEVICE = "cpu"


def move_to_device(
    tile: xr.Dataset,
    device: Literal["cuda", "cpu"] | int,
):
    """Context manager to ensure a dataset is on the correct device.

    Args:
        tile: The xarray dataset to operate on.
        device: The device to use for calculations (either "cuda", "cpu", or a specific GPU index).

    Returns:
        xr.Dataset: The xarray dataset on the specified device.

    """
    use_gpu = device == "cuda" or isinstance(device, int)

    # Warn user if use_gpu is set but no GPU is available
    if use_gpu and not has_cuda_and_cupy():
        logger.warning(
            f"Device was set to {device}, but GPU acceleration is not available. Calculating optical indices on CPU."
        )
        use_gpu = False

    if use_gpu:
        device_nr = device if isinstance(device, int) else 0
        # Persist in case of dask - since cupy-dask is not supported
        if tile.chunks is not None:
            logger.debug("Persisting dask array before moving to GPU.")
            tile = tile.persist()
        # Move and calculate on specified device
        logger.debug(f"Moving tile to GPU:{device}.")
        with cp.cuda.Device(device_nr):
            tile = tile.cupy.as_cupy()
    return tile


def move_to_host(tile: xr.Dataset) -> xr.Dataset:
    """Move a dataset from GPU to CPU.

    Args:
        tile (xr.Dataset): The xarray dataset to move.

    Returns:
        xr.Dataset: _description_

    """
    if tile.cupy.is_cupy:
        tile = tile.cupy.as_numpy()
        free_cupy()
    return tile


def free_cupy():
    """Free the CUDA memory of cupy."""
    try:
        import cupy as cp  # type: ignore
    except ImportError:
        cp = None

    if cp is not None:
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()


def free_torch():
    """Free the CUDA memory of pytorch."""
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
