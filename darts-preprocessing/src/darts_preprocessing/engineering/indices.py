"""Calculation of spectral indices from optical data."""

import logging
import time

import xarray as xr

logger = logging.getLogger(__name__)


def calculate_ndvi(planet_scene_dataset: xr.Dataset, nir_band: str = "nir", red_band: str = "red") -> xr.Dataset:
    """Calculate NDVI from an xarray Dataset containing spectral bands.

    Example:
        ```python
        ndvi_data = calculate_ndvi(planet_scene_dataset)
        ```

    Args:
        planet_scene_dataset (xr.Dataset): The xarray Dataset containing the spectral bands, where the bands are indexed
            along a dimension (e.g., 'band'). The Dataset should have dimensions including 'band', 'y', and 'x'.
        nir_band (str, optional): The name of the NIR band in the Dataset (default is "nir"). This name should
            correspond to the variable name for the NIR band in the 'band' dimension. Defaults to "nir".
        red_band (str, optional): The name of the Red band in the Dataset (default is "red"). This name should
            correspond to the variable name for the Red band in the 'band' dimension. Defaults to "red".

    Returns:
        xr.Dataset: A new Dataset containing the calculated NDVI values. The resulting Dataset will have
            dimensions (band: 1, y: ..., x: ...) and will be named "ndvi".


    Notes:
        NDVI (Normalized Difference Vegetation Index) is calculated using the formula:
            NDVI = (NIR - Red) / (NIR + Red)

        This index is commonly used in remote sensing to assess vegetation health and density.

    """
    start = time.time()
    logger.debug(f"Calculating NDVI from {nir_band=} and {red_band=}.")
    # Calculate NDVI using the formula
    nir = planet_scene_dataset[nir_band].astype("float32")
    r = planet_scene_dataset[red_band].astype("float32")
    ndvi = (nir - r) / (nir + r)
    ndvi = ndvi.assign_attrs({"data_source": "planet", "long_name": "NDVI"}).to_dataset(name="ndvi")
    logger.debug(f"NDVI calculated in {time.time() - start} seconds.")
    return ndvi
