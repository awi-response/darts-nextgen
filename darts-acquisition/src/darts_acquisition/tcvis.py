"""Landsat Trends related Data Loading. Should be used temporary and maybe moved to the acquisition package."""

import logging
import time
import warnings
from pathlib import Path

import ee
import numpy as np
import pyproj
import rasterio
import xarray as xr
import xee  # noqa: F401

logger = logging.getLogger(__name__.replace("darts_", "darts."))

EE_WARN_MSG = "Unable to retrieve 'system:time_start' values from an ImageCollection due to: No 'system:time_start' values found in the 'ImageCollection'."  # noqa: E501


def ee_geom_from_image_bounds(reference_dataset: xr.Dataset, buffer=1000) -> ee.Geometry.Rectangle:
    """Create an Earth Engine geometry from the bounds of a xarray dataset.

    Args:
        reference_dataset (xr.Dataset): The reference dataset.
        buffer (int, optional): A buffer in m. Defaults to 1000.

    Returns:
        ee.Geometry.Rectangle: The Earth Engine geometry.

    """
    # get bounds
    xmin, ymin, xmax, ymax = reference_dataset.rio.bounds()  # left, bottom, right, top
    region_rect = [[xmin - buffer, ymin - buffer], [xmax + buffer, ymax + buffer]]

    # make polygon
    transformer = pyproj.Transformer.from_crs(reference_dataset.rio.crs, "epsg:4326")
    region_transformed = [transformer.transform(*vertex)[::-1] for vertex in region_rect]
    return ee.Geometry.Rectangle(coords=region_transformed)


def load_tcvis(reference_dataset: xr.Dataset, cache_dir: Path | None = None) -> xr.Dataset:
    """Load the Landsat Trends (TCVIS) from Google Earth Engine.

    Args:
        reference_dataset (xr.Dataset): The reference dataset.
        cache_dir (Path | None): The cache directory. If None, no caching will be used. Defaults to None.

    Returns:
        xr.Dataset: The TCVIS dataset.

    """
    start_time = time.time()

    # Try to load from cache - else from Google Earth Engine
    cache_fname = f"tcvis_{reference_dataset.attrs['tile_id']}.nc"
    if cache_dir is not None and (cache_dir / cache_fname).exists():
        logger.debug(f"Loading cached TCVis from {(cache_dir / cache_fname).resolve()}")
        return xr.open_dataset(cache_dir / cache_fname, engine="h5netcdf")

    logger.debug("Loading TCVis from Google Earth Engine, since no cache was found")
    geom = ee_geom_from_image_bounds(reference_dataset)
    ee_image_tcvis = ee.ImageCollection("users/ingmarnitze/TCTrend_SR_2000-2019_TCVIS").mosaic().clip(geom)
    # Wrap into image collection again to be able to use the engine
    ee_image_tcvis = ee.ImageCollection(ee_image_tcvis)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message=EE_WARN_MSG)
        ds = xr.open_dataset(ee_image_tcvis, engine="ee", geometry=geom, crs=str(reference_dataset.rio.crs), scale=30)
    # Update dataset properties to fit our pipeline-api
    ds = ds.isel(time=0).rename({"X": "x", "Y": "y"}).transpose("y", "x")
    ds = ds.rename_vars(
        {
            "TCB_slope": "tc_brightness",
            "TCG_slope": "tc_greenness",
            "TCW_slope": "tc_wetness",
        }
    )
    for band in ds.data_vars:
        ds[band].attrs = {
            "data_source": "ee:ingmarnitze/TCTrend_SR_2000-2019_TCVIS",
            "long_name": f"Tasseled Cap {band.split('_')[1].capitalize()}",
        }

    ds.rio.write_crs(ds.attrs["crs"], inplace=True)
    ds.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    search_time = time.time()
    logger.debug(f"Found a dataset with shape {ds.sizes} in {search_time - start_time} seconds.")

    # Save original min-max values for each band for clipping later
    clip_values = {band: (ds[band].min().values.item(), ds[band].max().values.item()) for band in ds.data_vars}  # noqa: PD011

    # Interpolate missing values (there are very few, so we actually can interpolate them)
    for band in ds.data_vars:
        ds[band] = ds[band].rio.write_nodata(np.nan).rio.interpolate_na()

    logger.debug(f"Reproject dataset to match reference dataset {reference_dataset.sizes}")
    ds = ds.rio.reproject_match(reference_dataset, resampling=rasterio.enums.Resampling.cubic)
    logger.debug(f"Reshaped dataset in {time.time() - search_time} seconds")

    # Convert to uint8
    for band in ds.data_vars:
        band_min, band_max = clip_values[band]
        ds[band] = ds[band].clip(band_min, band_max, keep_attrs=True).astype("uint8").rio.write_nodata(None)

    # Save to cache
    if cache_dir is not None:
        logger.debug(f"Saving TCVis to cache to {(cache_dir / cache_fname).resolve()}")
        cache_dir.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(cache_dir / cache_fname, engine="h5netcdf")

    logger.debug(f"Loading TCVis took {time.time() - start_time} seconds")
    return ds
