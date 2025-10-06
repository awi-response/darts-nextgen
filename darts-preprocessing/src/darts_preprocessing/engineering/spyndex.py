"""Calculation of spectral indices from optical data with the spyndex library.

Note: Cupy acceleration is not tested yet.
"""

import logging

import spyndex
import xarray as xr
from stopuhr import stopwatch

logger = logging.getLogger(__name__.replace("darts_", "darts."))

spyndex_band_mapping = {
    "B": "blue",
    "G": "green",
    "R": "red",
    "N": "nir",
}


@stopwatch.f("Computing spectral index with spyndex", printer=logger.debug, print_kwargs=["index"])
def compute_spyndex(ds_optical: xr.Dataset, index: str, **kwargs) -> xr.DataArray:
    """Compute a spectral index using the spyndex library.

    Run `spyndex.indices` to see all available indices.

    Args:
        ds_optical (xr.Dataset): The optical dataset containing the required bands.
        index (str): The name of the spectral index to compute.
        **kwargs: Additional parameters required by the spectral index (e.g., L for SAVI).

    Raises:
        ValueError: If a required band is missing.
        ValueError: If all bands are provided as scalars.

    Returns:
        xr.DataArray: The computed spectral index.

    """
    index: spyndex.axioms.SpectralIndex = spyndex.indices[index]

    params = {}
    atleast_one_dataarray = False
    for band in index.bands:
        # Case 3: band is missing -> error
        if band not in kwargs and band not in spyndex_band_mapping:
            raise ValueError(f"Band '{band}' is required for index '{index.short_name}' but not provided in {kwargs=}.")
        # Case 2: band is in kwargs
        if band in kwargs:
            params[band] = kwargs[band]
            continue
        # Case 1: band is in optical
        band_name = spyndex_band_mapping[band]
        params[band] = ds_optical[band_name].clip(0, 1)
        atleast_one_dataarray = True

    if not atleast_one_dataarray:
        raise ValueError(f"At least one band must be a DataArray, got {params=}. Did you pass all bands as scalars?")

    idx = index.compute(params)
    assert isinstance(idx, xr.DataArray)

    idx = idx.assign_attrs(
        {
            "source": "spyndex",
            "long_name": index.long_name,
            "reference": index.reference,
            "formula": index.formula,
            "author": index.contributor,
        }
    ).rename(index.short_name.lower())
    return idx
