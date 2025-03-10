"""PLANET scene based preprocessing."""

import logging
import time
from typing import Literal

import odc.geo.xr  # noqa: F401
import xarray as xr
from darts_utils.cuda import free_cupy
from xrspatial.utils import has_cuda_and_cupy

from darts_preprocessing.engineering.arcticdem import calculate_slope, calculate_topographic_position_index
from darts_preprocessing.engineering.indices import calculate_ndvi

logger = logging.getLogger(__name__.replace("darts_", "darts."))


if has_cuda_and_cupy():
    import cupy as cp
    import cupy_xarray  # noqa: F401

    DEFAULT_DEVICE = "cuda"
    logger.debug("GPU-accelerated xrspatial functions are available.")
else:
    DEFAULT_DEVICE = "cpu"
    logger.debug("GPU-accelerated xrspatial functions are not available.")


def preprocess_legacy(
    ds_optical: xr.Dataset,
    ds_arcticdem: xr.Dataset,
    ds_tcvis: xr.Dataset,
) -> xr.Dataset:
    """Preprocess optical data with legacy (DARTS v1) preprocessing steps.

    The processing steps are:
    - Calculate NDVI
    - Merge everything into a single ds.

    Args:
        ds_optical (xr.Dataset): The Planet scene optical data or Sentinel 2 scene optical data.
        ds_arcticdem (xr.Dataset): The ArcticDEM data.
        ds_tcvis (xr.Dataset): The TCVIS data.

    Returns:
        xr.Dataset: The preprocessed dataset.

    """
    # Calculate NDVI
    ds_ndvi = calculate_ndvi(ds_optical)

    # Reproject TCVIS to optical data
    ds_tcvis = ds_tcvis.odc.reproject(ds_optical.odc.geobox, resampling="cubic")

    # Since this function expects the arcticdem to be loaded from a VRT, which already contains slope and tpi,
    # we dont need to calculate them here

    # merge to final dataset
    ds_merged = xr.merge([ds_optical, ds_ndvi, ds_arcticdem, ds_tcvis])

    return ds_merged


def preprocess_legacy_arcticdem_fast(
    ds_arcticdem: xr.Dataset, tpi_outer_radius: int, tpi_inner_radius: int, device: Literal["cuda", "cpu"] | int
):
    """Preprocess the ArcticDEM data with legacy (DARTS v1) preprocessing steps.

    Args:
        ds_arcticdem (xr.Dataset): The ArcticDEM dataset.
        tpi_outer_radius (int): The outer radius of the annulus kernel for the tpi calculation in number of cells.
        tpi_inner_radius (int): The inner radius of the annulus kernel for the tpi calculation in number of cells.
        device (Literal["cuda", "cpu"] | int): The device to run the tpi and slope calculations on.
            If "cuda" take the first device (0), if int take the specified device.

    Returns:
        xr.Dataset: The preprocessed ArcticDEM dataset.

    """
    use_gpu = device == "cuda" or isinstance(device, int)

    # Warn user if use_gpu is set but no GPU is available
    if use_gpu and not has_cuda_and_cupy():
        logger.warning(
            f"Device was set to {device}, but GPU acceleration is not available. Calculating TPI and slope on CPU."
        )
        use_gpu = False

    # Calculate TPI and slope from ArcticDEM on GPU
    if use_gpu:
        device_nr = device if isinstance(device, int) else 0
        logger.debug(f"Moving arcticdem to GPU:{device}.")
        # Check if dem is dask, if not persist it, since tpi and slope can't be calculated from cupy-dask arrays
        if ds_arcticdem.chunks is not None:
            ds_arcticdem = ds_arcticdem.persist()
        # Move and calculate on specified device
        with cp.cuda.Device(device_nr):
            ds_arcticdem = ds_arcticdem.cupy.as_cupy()
            ds_arcticdem = calculate_topographic_position_index(ds_arcticdem, tpi_outer_radius, tpi_inner_radius)
            ds_arcticdem = calculate_slope(ds_arcticdem)
            ds_arcticdem = ds_arcticdem.cupy.as_numpy()
            free_cupy()

    # Calculate TPI and slope from ArcticDEM on CPU
    else:
        ds_arcticdem = calculate_topographic_position_index(ds_arcticdem, tpi_outer_radius, tpi_inner_radius)
        ds_arcticdem = calculate_slope(ds_arcticdem)

    # Apply legacy scaling to tpi
    with xr.set_options(keep_attrs=True):
        ds_arcticdem["tpi"] = (ds_arcticdem.tpi + 50) * 300
    return ds_arcticdem


def preprocess_legacy_fast(
    ds_merged: xr.Dataset,
    ds_arcticdem: xr.Dataset,
    ds_tcvis: xr.Dataset,
    tpi_outer_radius: int = 100,
    tpi_inner_radius: int = 0,
    device: Literal["cuda", "cpu"] | int = DEFAULT_DEVICE,
) -> xr.Dataset:
    """Preprocess optical data with legacy (DARTS v1) preprocessing steps, but with new data concepts.

    The processing steps are:
    - Calculate NDVI
    - Calculate slope and relative elevation from ArcticDEM
    - Merge everything into a single ds.

    The main difference to preprocess_legacy is the new data concept of the arcticdem.
    Instead of using already preprocessed arcticdem data which are loaded from a VRT, this step expects the raw
    arcticdem data and calculates slope and relative elevation on the fly.

    Args:
        ds_merged (xr.Dataset): The Planet scene optical data or Sentinel 2 scene optical dataset including data_masks.
        ds_arcticdem (xr.Dataset): The ArcticDEM dataset.
        ds_tcvis (xr.Dataset): The TCVIS dataset.
        tpi_outer_radius (int, optional): The outer radius of the annulus kernel for the tpi calculation
            in m. Defaults to 100m.
        tpi_inner_radius (int, optional): The inner radius of the annulus kernel for the tpi calculation
            in m. Defaults to 0.
        device (Literal["cuda", "cpu"] | int, optional): The device to run the tpi and slope calculations on.
            If "cuda" take the first device (0), if int take the specified device.
            Defaults to "cuda" if cuda is available, else "cpu".

    Returns:
        xr.Dataset: The preprocessed dataset.

    """
    tick_fstart = time.perf_counter()
    logger.info("Starting fast v1 preprocessing.")

    # Calculate NDVI
    ds_merged["ndvi"] = calculate_ndvi(ds_merged).ndvi

    # Reproject TCVIS to optical data
    tick_sproj = time.perf_counter()
    ds_tcvis = ds_tcvis.odc.reproject(ds_merged.odc.geobox, resampling="cubic")
    tick_eproj = time.perf_counter()
    logger.debug(f"Reprojection of TCVIS done in {tick_eproj - tick_sproj:.2f} seconds.")

    ds_merged["tc_brightness"] = ds_tcvis.tc_brightness
    ds_merged["tc_greenness"] = ds_tcvis.tc_greenness
    ds_merged["tc_wetness"] = ds_tcvis.tc_wetness

    # Calculate TPI and slope from ArcticDEM
    tick_sproj = time.perf_counter()
    ds_arcticdem = ds_arcticdem.odc.reproject(ds_merged.odc.geobox.buffered(tpi_outer_radius), resampling="cubic")
    tick_eproj = time.perf_counter()
    logger.debug(f"Reprojection of ArcticDEM done in {tick_eproj - tick_sproj:.2f} seconds.")

    ds_arcticdem = preprocess_legacy_arcticdem_fast(ds_arcticdem, tpi_outer_radius, tpi_inner_radius, device)
    ds_arcticdem = ds_arcticdem.odc.crop(ds_merged.odc.geobox.extent)
    ds_merged["dem"] = ds_arcticdem.dem
    ds_merged["relative_elevation"] = ds_arcticdem.tpi
    ds_merged["slope"] = ds_arcticdem.slope

    # Update datamask with arcticdem mask
    with xr.set_options(keep_attrs=True):
        ds_merged["quality_data_mask"] = ds_merged.quality_data_mask * ds_arcticdem.datamask
    ds_merged.quality_data_mask.attrs["data_source"] += " + ArcticDEM"

    tick_fend = time.perf_counter()
    logger.info(f"Preprocessing done in {tick_fend - tick_fstart:.2f} seconds.")
    return ds_merged
