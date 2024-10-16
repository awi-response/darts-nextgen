"""ArcticDEM related data loading. Should be used temporary and maybe moved to the acquisition package."""

import logging
import time
from pathlib import Path

import rioxarray  # noqa: F401
import xarray as xr

logger = logging.getLogger(__name__)


def load_arcticdem(elevation_path: Path, slope_path: Path, reference_dataset: xr.Dataset) -> xr.Dataset:
    """Load ArcticDEM data and reproject it to match the reference dataset.

    Args:
        elevation_path (Path): The path to the ArcticDEM elevation data.
        slope_path (Path): The path to the ArcticDEM slope data.
        reference_dataset (xr.Dataset): The reference dataset to reproject, resampled and cropped the ArcticDEM data to.

    Returns:
        xr.Dataset: The ArcticDEM data reprojected, resampled and cropped to match the reference dataset.

    """
    start_time = time.time()
    logger.debug(f"Loading ArcticDEM data from {elevation_path} and {slope_path}")
    relative_elevation = xr.open_dataarray(elevation_path).isel(band=0).drop_vars("band")
    relative_elevation: xr.DataArray = relative_elevation.rio.reproject_match(reference_dataset)
    relative_elevation: xr.Dataset = relative_elevation.assign_attrs(
        {"data_source": "arcticdem", "long_name": "Relative Elevation"}
    ).to_dataset(name="relative_elevation")

    slope = xr.open_dataarray(slope_path).isel(band=0).drop_vars("band")
    slope: xr.DataArray = slope.rio.reproject_match(reference_dataset)
    slope: xr.Dataset = slope.assign_attrs({"data_source": "arcticdem", "long_name": "Slope"}).to_dataset(name="slope")

    articdem_ds = xr.merge([relative_elevation, slope])
    logger.debug(f"Loaded ArcticDEM data in {time.time() - start_time} seconds.")
    return articdem_ds
