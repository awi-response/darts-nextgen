"""Calculation of spectral indices from optical data with the spyndex library."""

import logging
from typing import Any

import spyndex
import xarray as xr
from stopuhr import stopwatch

logger = logging.getLogger(__name__.replace("darts_", "darts."))


@stopwatch.f("Computing spectral index with spyndex", printer=logger.debug, print_kwargs=["index"])
def calculate_spyndex(ds_optical: xr.Dataset, index: str, **kwargs: dict[str, Any]) -> xr.DataArray:
    """Compute a spectral index using the spyndex library.

    This wrapper provides access to 200+ spectral indices from the spyndex library with
    automatic band mapping and parameter handling.

    Args:
        ds_optical (xr.Dataset): Dataset containing spectral bands. Band names should match
            spyndex common names (e.g., 'red', 'nir', 'blue', 'green').
        index (str): Name of the spectral index to compute. Run `spyndex.indices` to see
            all available indices (e.g., 'NDVI', 'EVI', 'SAVI').
        **kwargs: Additional parameters or band overrides:
            - Band values: Override bands from ds_optical with scalar or array values (e.g., red=0.2)
            - Constants: Override default values for index-specific constants (e.g., L=0.5 for SAVI)

    Returns:
        xr.DataArray: Computed spectral index with attributes:
            - source: "spyndex"
            - long_name: Full name of the index
            - reference: Citation for the index
            - formula: Mathematical formula
            - author: Index contributor

    Raises:
        ValueError: If a required band is missing from both ds_optical and kwargs.
        ValueError: If all bands are provided as scalar values.

    Note:
        Band resolution priority:

        1. Bands in ds_optical (with common_name matching spyndex.bands)
        2. Values in kwargs (override ds_optical bands)
        3. Default values for constants (from spyndex.constants)

        All optical bands are automatically clipped to [0, 1] before calculation.

        At least one band must come from ds_optical as a DataArray (not all scalars).

    Example:
        Compute various indices with spyndex:

        ```python
        from darts_preprocessing import calculate_spyndex
        import spyndex

        # List all available indices
        print(list(spyndex.indices.keys()))

        # Basic NDVI
        ndvi = calculate_spyndex(ds_optical, "NDVI")

        # SAVI with custom soil adjustment factor
        savi = calculate_spyndex(ds_optical, "SAVI", L=0.5)

        # EVI with custom parameters
        evi = calculate_spyndex(ds_optical, "EVI", g=2.5, C1=6, C2=7.5, L=1)
        ```

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
