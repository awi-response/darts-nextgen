"""PLANET scene based preprocessing."""

import logging
from typing import Literal

import odc.geo.xr
import xarray as xr
from darts_utils.cuda import DEFAULT_DEVICE, move_to_device, move_to_host
from stopuhr import stopwatch

from darts_preprocessing.engineering.arcticdem import (
    calculate_slope,
    calculate_topographic_position_index,
)
from darts_preprocessing.engineering.indices import calculate_ndvi

logger = logging.getLogger(__name__.replace("darts_", "darts."))


@stopwatch.f("Preprocessing arcticdem", printer=logger.debug, print_kwargs=["tpi_outer_radius", "tpi_inner_radius"])
def preprocess_legacy_arcticdem_fast(
    ds_arcticdem: xr.Dataset,
    tpi_outer_radius: int,
    tpi_inner_radius: int,
) -> xr.Dataset:
    """Preprocess the ArcticDEM data with legacy (DARTS v1) preprocessing steps.

    Args:
        ds_arcticdem (xr.Dataset): The ArcticDEM dataset.
        tpi_outer_radius (int): The outer radius of the annulus kernel for the tpi calculation in number of cells.
        tpi_inner_radius (int): The inner radius of the annulus kernel for the tpi calculation in number of cells.

    Returns:
        xr.Dataset: The preprocessed ArcticDEM dataset.

    """
    ds_arcticdem = calculate_topographic_position_index(ds_arcticdem, tpi_outer_radius, tpi_inner_radius)
    ds_arcticdem = calculate_slope(ds_arcticdem)

    return ds_arcticdem


@stopwatch("Preprocessing", printer=logger.debug)
def preprocess_legacy_fast(
    ds_optical: xr.Dataset,
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
        ds_optical (xr.Dataset): The Planet scene optical data or Sentinel 2 scene optical dataset including data_masks.
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
    # Move to GPU for faster calculations
    ds_optical = move_to_device(ds_optical, device)
    # Calculate NDVI
    ds_optical["ndvi"] = calculate_ndvi(ds_optical)
    ds_optical = move_to_host(ds_optical)

    # Reproject TCVIS to optical data
    with stopwatch("Reprojecting TCVIS", printer=logger.debug):
        ds_tcvis = ds_tcvis.odc.reproject(ds_optical.odc.geobox, resampling="cubic")

    ds_optical["tc_brightness"] = ds_tcvis.tc_brightness
    ds_optical["tc_greenness"] = ds_tcvis.tc_greenness
    ds_optical["tc_wetness"] = ds_tcvis.tc_wetness

    # Calculate TPI and slope from ArcticDEM
    with stopwatch("Reprojecting ArcticDEM", printer=logger.debug):
        ds_arcticdem = ds_arcticdem.odc.reproject(ds_optical.odc.geobox.buffered(tpi_outer_radius), resampling="cubic")
    # Move to same device as optical
    ds_arcticdem = move_to_device(ds_arcticdem, device)
    ds_arcticdem = preprocess_legacy_arcticdem_fast(ds_arcticdem, tpi_outer_radius, tpi_inner_radius)
    ds_arcticdem = move_to_host(ds_arcticdem)

    ds_arcticdem = ds_arcticdem.odc.crop(ds_optical.odc.geobox.extent)
    # For some reason, we need to reindex, because the reproject + crop of the arcticdem sometimes results
    # in floating point errors. These error are at the order of 1e-10, hence, way below millimeter precision.
    ds_arcticdem = ds_arcticdem.reindex_like(ds_optical)

    ds_optical["dem"] = ds_arcticdem.dem
    ds_optical["relative_elevation"] = ds_arcticdem.tpi
    ds_optical["slope"] = ds_arcticdem.slope
    ds_optical["arcticdem_data_mask"] = ds_arcticdem.arcticdem_data_mask

    return ds_optical
