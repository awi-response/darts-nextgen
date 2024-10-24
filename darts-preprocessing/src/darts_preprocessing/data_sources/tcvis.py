"""Landsat Trends related Data Loading. Should be used temporary and maybe moved to the acquisition package."""

import logging
import time
from pathlib import Path

import ee
import pyproj
import rioxarray  # noqa: F401
import xarray as xr

logger = logging.getLogger(__name__)


def init_ee():
    """Initialize Earth Engine. Authenticate if necessary."""
    logger.debug("Initializing Earth Engine")
    try:
        ee.Initialize()
    except Exception:
        logger.debug("Initializing Earth Engine failed, trying to authenticate before")
        ee.Authenticate()
        ee.Initialize()
    logger.debug("Earth Engine initialized")


def ee_geom_from_image_bounds(reference_dataset: xr.Dataset, buffer=1000) -> ee.Geometry.Rectangle:
    """Create an Earth Engine geometry from the bounds of a xarray dataset.

    Args:
        reference_dataset (xr.Dataset): The reference dataset.
        buffer (int, optional): A buffer in m. Defaults to 1000.

    Returns:
        ee.Geometry.Rectangle: The Earth Engine geometry.

    """
    # get bounds
    xmin, ymin, xmax, ymax = reference_dataset.rio.bounds()
    region_rect = [[xmin - buffer, ymin - buffer], [xmax + buffer, ymax + buffer]]

    # make polygon
    transformer = pyproj.Transformer.from_crs(reference_dataset.rio.crs(), "epsg:4326")
    region_transformed = [transformer.transform(*vertex)[::-1] for vertex in region_rect]
    return ee.Geometry.Rectangle(coords=region_transformed)


def load_tcvis_xee(cache_dir: Path, scene_id: str, reference_dataset: xr.Dataset) -> xr.Dataset:
    """Load the Landsat Trends (TCVIS) from Google Earth Engine.

    Args:
        cache_dir (Path): Path for the cache file. The file get's created if not exists.
        reference_dataset (xr.Dataset): The reference dataset.

    Returns:
        xr.Dataset: The TCVIS dataset.

    """
    start_time = time.time()

    cache_file = cache_dir / f"tcvis_{scene_id}.nc"
    if cache_file.exists():
        return xr.open_dataset(cache_file)

    ee_image_tcvis = ee.ImageCollection("users/ingmarnitze/TCTrend_SR_2000-2019_TCVIS").mosaic()
    geom = ee_geom_from_image_bounds(reference_dataset)
    ds = xr.open_dataset(ee_image_tcvis, engine="ee", projection=ee_image_tcvis.projection(), geometry=geom)
    ds = ds.rio.reproject_match(reference_dataset)

    logger.debug(f"Loading TCVis took {time.time() - start_time} seconds")
    return ds
