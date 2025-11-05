"""Utility functions around cuda, e.g. memory management."""

import gc
import logging
from typing import Literal

import xarray as xr
from xrspatial.utils import has_cuda_and_cupy

logger = logging.getLogger(__name__.replace("darts_", "darts."))

if has_cuda_and_cupy():
    import cupy as cp  # type: ignore
    import cupy_xarray  # type: ignore

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


def move_to_host(tile: xr.Dataset | xr.DataArray | "cp.ndarray") -> xr.Dataset | xr.DataArray | "np.ndarray":
    """Ensure data are moved from GPU (CuPy) memory to CPU (NumPy) memory.

    This function converts CuPy-backed arrays inside an xarray Dataset or DataArray
    into NumPy arrays, ensuring full CPU compatibility for serialization or
    downstream processing (e.g., Ray pipelines).

    Handles the following cases:
        1. **Raw CuPy array** → returns NumPy array via `cp.asnumpy`.
        2. **xarray.DataArray** backed by CuPy → returns a new DataArray
           with its data copied to NumPy.
        3. **xarray.Dataset** with CuPy-backed variables → returns a new Dataset
           where each variable is NumPy-backed.

    If the input is already CPU-backed or CuPy is unavailable, it is returned unchanged.

    Args:
        tile: The data object to move. Can be:
            - `cupy.ndarray`
            - `xarray.DataArray`
            - `xarray.Dataset`

    Returns:
        The same type of object, but backed by NumPy arrays on CPU.

    Raises:
        AttributeError: Only if unexpected object types or data attributes are missing.
    """
    if has_cuda_and_cupy():
        try:
            # Case 1: raw CuPy array
            if isinstance(tile, cp.ndarray):
                return cp.asnumpy(tile)

            # Case 2 & 3: DataArray or Dataset backed by CuPy
            if isinstance(tile, xr.DataArray):
                data = tile.data
                if hasattr(data, "__cuda_array_interface__"):
                    return tile.copy(data=cp.asnumpy(data))
                return tile

            # Case 3: Dataset containing CuPy-backed DataArrays
            if isinstance(tile, xr.Dataset):
                vars_cpu = {}
                for name, da in tile.data_vars.items():
                    data = da.data
                    if hasattr(data, "__cuda_array_interface__"):
                        data = cp.asnumpy(data)
                    vars_cpu[name] = (da.dims, data, da.attrs)
                return xr.Dataset(vars_cpu, attrs=tile.attrs)

        except AttributeError:
            # Dataset doesn't have cupy attribute, already on CPU
            pass
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
