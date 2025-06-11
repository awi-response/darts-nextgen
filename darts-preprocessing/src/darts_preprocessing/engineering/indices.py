"""Calculation of spectral indices from optical data."""

import logging

import xarray as xr
from stopuhr import stopwatch

logger = logging.getLogger(__name__.replace("darts_", "darts."))


@stopwatch.f("Calculating NDVI", printer=logger.debug, print_kwargs=["nir_band", "red_band"])
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
    # Calculate NDVI using the formula
    nir = planet_scene_dataset[nir_band].astype("float32")
    r = planet_scene_dataset[red_band].astype("float32")
    ndvi = (nir - r) / (nir + r)

    # TODO: For the bands rework: Change the scaling acording to the specs
    # TODO: Also do rio.write_nodata(0) AFTER the cast to uint16
    # Otherwise the _FillValue will be set to 0.0 (float) instead of 0 (int)
    # This will then cause the xarray loader to cast the data to float32
    # Scale to 0 - 20000 (for later conversion to uint16)
    ndvi = (ndvi.clip(-1, 1) + 1) * 1e4
    # Make nan to 0
    ndvi = ndvi.fillna(0).rio.write_nodata(0)
    # Convert to uint16
    ndvi = ndvi.astype("uint16")

    ndvi = ndvi.assign_attrs({"data_source": "planet", "long_name": "NDVI"}).to_dataset(name="ndvi")
    return ndvi
