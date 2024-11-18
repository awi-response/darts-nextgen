"""PLANET scene based preprocessing."""

import logging
import time

import rasterio
import xarray as xr

from darts_preprocessing.engineering.arcticdem import calculate_slope, calculate_topographic_position_index
from darts_preprocessing.engineering.indices import calculate_ndvi

logger = logging.getLogger(__name__.replace("darts_", "darts."))


def preprocess_legacy(
    ds_optical: xr.Dataset,
    ds_arcticdem: xr.Dataset,
    ds_tcvis: xr.Dataset,
    ds_data_masks: xr.Dataset,
) -> xr.Dataset:
    """Preprocess optical data with legacy (DARTS v1) preprocessing steps.

    The processing steps are:
    - Calculate NDVI
    - Merge everything into a single ds.

    Args:
        ds_optical (xr.Dataset): The Planet scene optical data or Sentinel 2 scene optical data.
        ds_arcticdem (xr.Dataset): The ArcticDEM data.
        ds_tcvis (xr.Dataset): The TCVIS data.
        ds_data_masks (xr.Dataset): The data masks, based on the optical data.

    Returns:
        xr.Dataset: The preprocessed dataset.

    """
    ds_ndvi = calculate_ndvi(ds_optical)

    # merge to final dataset
    ds_merged = xr.merge([ds_optical, ds_ndvi, ds_arcticdem, ds_tcvis, ds_data_masks])

    return ds_merged


def preprocess_legacy_fast(
    ds_optical: xr.Dataset,
    ds_arcticdem: xr.Dataset,
    ds_tcvis: xr.Dataset,
    ds_data_masks: xr.Dataset,
    tpi_outer_radius: int = 30,
    tpi_inner_radius: int = 25,
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
        ds_optical (xr.Dataset): The Planet scene optical data or Sentinel 2 scene optical data.
        ds_arcticdem (xr.Dataset): The ArcticDEM data.
        ds_tcvis (xr.Dataset): The TCVIS data.
        ds_data_masks (xr.Dataset): The data masks, based on the optical data.
        tpi_outer_radius (int, optional): The outer radius of the annulus kernel for the tpi calculation
            in number of cells. Defaults to 30.
        tpi_inner_radius (int, optional): The inner radius of the annulus kernel for the tpi calculation
            in number of cells. Defaults to 25.

    Returns:
        xr.Dataset: The preprocessed dataset.

    """
    tick_fstart = time.perf_counter()
    logger.info("Starting fast v1 preprocessing.")

    # merge to final dataset
    ds_merged = xr.merge([ds_optical, ds_tcvis, ds_data_masks])

    # Calculate NDVI
    ds_merged["ndvi"] = calculate_ndvi(ds_merged).ndvi

    # Calculate TPI and slope from ArcticDEM
    # We need to calculate them before reprojecting, hence we cant merge the data yet
    ds_arcticdem = calculate_topographic_position_index(ds_arcticdem, tpi_outer_radius, tpi_inner_radius)
    ds_arcticdem = calculate_slope(ds_arcticdem)
    ds_arcticdem = ds_arcticdem.rio.reproject_match(ds_optical, resampling=rasterio.enums.Resampling.cubic)

    ds_merged["dem"] = ds_arcticdem.dem
    ds_merged["relative_elevation"] = ds_arcticdem.tpi
    ds_merged["slope"] = ds_arcticdem.slope

    ds_merged["valid_data_mask"] = ds_data_masks.valid_data_mask * ds_arcticdem.datamask
    ds_merged.valid_data_mask.attrs = {
        "long_name": "Valid Data Mask",
        "description": "A mask indicating where valid data is available.",
        "data_source": "planet + arcticdem",
    }

    tick_fend = time.perf_counter()
    logger.info(f"Preprocessing done in {tick_fend - tick_fstart:.2f} seconds.")
    return ds_merged
