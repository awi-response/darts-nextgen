"""Calculation of spectral indices from optical data with the spyndex library."""

import logging

import spyndex
import xarray as xr
from stopuhr import stopwatch

logger = logging.getLogger(__name__.replace("darts_", "darts."))


@stopwatch.f("Computing spectral index with spyndex", printer=logger.debug, print_kwargs=["index"])
def compute_spyndex(ds_optical: xr.Dataset, index: str, **kwargs) -> xr.DataArray:
    """Compute a spectral index using the spyndex library.

    Run `spyndex.indices` to see all available indices.

    Will look for required bands in the provided dataset `ds_optical` and in the keyword arguments `kwargs`.
    Bands provided in `kwargs` will override bands from `ds_optical`.
    Constants not found in `kwargs` will be set to their default value.

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
        is_constant = band in spyndex.constants
        is_in_kwargs = band in kwargs

        is_in_optical = False
        if not is_constant:
            spyndex_band: spyndex.axioms.Band = spyndex.bands.get(band)
            is_in_optical = spyndex_band is not None and spyndex_band.common_name in ds_optical

        # Case 4: band is missing -> error
        if not (is_constant or is_in_kwargs or is_in_optical):
            raise ValueError(f"Band '{band}' is required for index '{index.short_name}' but not provided in {kwargs=}.")
        # Case 3: band is in kwargs
        if is_in_kwargs:
            params[band] = kwargs[band]
            continue
        # Case 2: band is a constant
        if is_constant:
            constant: spyndex.axioms.Constant = spyndex.constants[band]
            params[band] = constant.default
            continue
        # Case 1: band is in optical
        params[band] = ds_optical[spyndex_band.common_name].clip(0, 1)
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
