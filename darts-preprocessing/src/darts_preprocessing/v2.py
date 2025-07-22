"""PLANET scene based preprocessing."""

import logging
from typing import Literal

import odc.geo.xr
import xarray as xr
from darts_utils.cuda import free_cupy
from stopuhr import stopwatch
from xrspatial.utils import has_cuda_and_cupy

from darts_preprocessing.engineering.arcticdem import (
    calculate_aspect,
    calculate_curvature,
    calculate_hillshade,
    calculate_slope,
    calculate_topographic_position_index,
)
from darts_preprocessing.engineering.indices import calculate_ndvi

logger = logging.getLogger(__name__.replace("darts_", "darts."))


if has_cuda_and_cupy():
    import cupy as cp  # type: ignore
    import cupy_xarray  # noqa: F401 # type: ignore

    DEFAULT_DEVICE = "cuda"
else:
    DEFAULT_DEVICE = "cpu"


@stopwatch.f("Preprocessing arcticdem", printer=logger.debug, print_kwargs=["tpi_outer_radius", "tpi_inner_radius"])
def preprocess_arcticdem(
    ds_arcticdem: xr.Dataset,
    tpi_outer_radius: int,
    tpi_inner_radius: int,
    device: Literal["cuda", "cpu"] | int,
    azimuth: int,
    angle_altitude: int,
) -> xr.Dataset:
    """Preprocess the ArcticDEM data with mdoern (DARTS v2) preprocessing steps.

    Args:
        ds_arcticdem (xr.Dataset): The ArcticDEM dataset.
        tpi_outer_radius (int): The outer radius of the annulus kernel for the tpi calculation in number of cells.
        tpi_inner_radius (int): The inner radius of the annulus kernel for the tpi calculation in number of cells.
        device (Literal["cuda", "cpu"] | int): The device to run the tpi and slope calculations on.
            If "cuda" take the first device (0), if int take the specified device.
        azimuth (int): The azimuth angle of the light source in degrees for hillshade calculation.
        angle_altitude (int): The altitude angle of the light source in degrees for hillshade

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
            ds_arcticdem = calculate_hillshade(ds_arcticdem, azimuth=azimuth, angle_altitude=angle_altitude)
            ds_arcticdem = calculate_aspect(ds_arcticdem)
            ds_arcticdem = calculate_curvature(ds_arcticdem)
            ds_arcticdem = ds_arcticdem.cupy.as_numpy()
            free_cupy()

    # Calculate TPI and slope from ArcticDEM on CPU
    else:
        ds_arcticdem = calculate_topographic_position_index(ds_arcticdem, tpi_outer_radius, tpi_inner_radius)
        ds_arcticdem = calculate_slope(ds_arcticdem)
        ds_arcticdem = calculate_hillshade(ds_arcticdem, azimuth=azimuth, angle_altitude=angle_altitude)
        ds_arcticdem = calculate_aspect(ds_arcticdem)
        ds_arcticdem = calculate_curvature(ds_arcticdem)

    # Apply legacy scaling to tpi
    with xr.set_options(keep_attrs=True):
        ds_arcticdem["tpi"] = (ds_arcticdem.tpi + 50) * 300
    return ds_arcticdem


@stopwatch("Preprocessing", printer=logger.debug)
def preprocess_v2(
    ds_merged: xr.Dataset,
    ds_arcticdem: xr.Dataset,
    ds_tcvis: xr.Dataset,
    tpi_outer_radius: int = 100,
    tpi_inner_radius: int = 0,
    device: Literal["cuda", "cpu"] | int = DEFAULT_DEVICE,
) -> xr.Dataset:
    """Preprocess optical data with modern (DARTS v2) preprocessing steps.

    The processing steps are:
    - Calculate NDVI
    - Calculate slope, hillshade, aspect, curvature and relative elevation from ArcticDEM
    - Merge everything into a single ds.

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
    # Calculate NDVI
    ds_merged["ndvi"] = calculate_ndvi(ds_merged).ndvi

    # Reproject TCVIS to optical data
    with stopwatch("Reprojecting TCVIS", printer=logger.debug):
        # *: Reprojecting this way will not alter the datatype of the data!
        # Should be uint8 before and after reprojection.
        ds_tcvis = ds_tcvis.odc.reproject(ds_merged.odc.geobox, resampling="cubic")

    ds_merged["tc_brightness"] = ds_tcvis.tc_brightness
    ds_merged["tc_greenness"] = ds_tcvis.tc_greenness
    ds_merged["tc_wetness"] = ds_tcvis.tc_wetness

    # Calculate TPI and slope from ArcticDEM
    with stopwatch("Reprojecting ArcticDEM", printer=logger.debug):
        ds_arcticdem = ds_arcticdem.odc.reproject(ds_merged.odc.geobox.buffered(tpi_outer_radius), resampling="cubic")

    azimuth = 225  # Default azimuth for hillshade calculation
    angle_altitude = 25  # Default angle altitude for hillshade calculation
    if "view:azimuth" in ds_arcticdem.attrs:
        azimuth = round(ds_arcticdem.attrs["view:azimuth"])
    if "view:sun_elevation" in ds_arcticdem.attrs:
        angle_altitude = round(ds_arcticdem.attrs["view:sun_elevation"])
    ds_arcticdem = preprocess_arcticdem(
        ds_arcticdem,
        tpi_outer_radius,
        tpi_inner_radius,
        device,
        azimuth,
        angle_altitude,
    )
    ds_arcticdem = ds_arcticdem.odc.crop(ds_merged.odc.geobox.extent)
    # For some reason, we need to reindex, because the reproject + crop of the arcticdem sometimes results
    # in floating point errors. These error are at the order of 1e-10, hence, way below millimeter precision.
    ds_arcticdem = ds_arcticdem.reindex_like(ds_merged)

    ds_merged["dem"] = ds_arcticdem.dem
    ds_merged["relative_elevation"] = ds_arcticdem.tpi
    ds_merged["slope"] = ds_arcticdem.slope
    ds_merged["hillshade"] = ds_arcticdem.hillshade
    ds_merged["aspect"] = ds_arcticdem.aspect
    ds_merged["curvature"] = ds_arcticdem.curvature
    ds_merged["arcticdem_data_mask"] = ds_arcticdem.datamask

    # Update datamask with arcticdem mask
    # with xr.set_options(keep_attrs=True):
    #     ds_merged["quality_data_mask"] = ds_merged.quality_data_mask * ds_arcticdem.datamask
    # ds_merged.quality_data_mask.attrs["data_source"] += " + ArcticDEM"

    return ds_merged
