# ruff: noqa
import numpy as np
import rioxarray
import xarray as xr
from affine import Affine

from .hardcoded_stuff import BAND_MAPPING


def mock_source(name, n_bands, dtype, meta) -> xr.DataArray:
    """Mock a single source band."""
    da = xr.DataArray(
        np.ones([n_bands, 1000, 1000], dtype=dtype),
        coords={
            f"{name}_band": BAND_MAPPING[name],
            "y": meta["y"],
            "x": meta["x"],
        },
        dims=[f"{name}_band", "y", "x"],
    )
    da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    da.rio.write_crs(meta["crs"], inplace=True)
    da.rio.write_transform(meta["transform"], inplace=True)
    return da


def mock_tile(primary_source):
    # mock coordinates
    meta = {
        "y": np.arange(1000),
        "x": np.arange(1000),
        "transform": Affine(0.5, 0.0, 0.0, 0.0, -0.5, 100.0),  # mock transform
        "crs": "EPSG:4326",  # mock CRS
    }

    primary = mock_source(primary_source, 4, np.uint16, meta)
    slope = mock_source("slope", 1, np.float32, meta)
    relative_elevation = mock_source("relative_elevation", 1, np.float32, meta)
    ndvi = mock_source("ndvi", 1, np.float32, meta)
    tcvis = mock_source("tcvis", 3, np.uint8, meta)

    return xr.Dataset(
        {
            primary_source: primary,
            "slope": slope,
            "relative_elevation": relative_elevation,
            "ndvi": ndvi,
            "tcvis": tcvis,
        }
    )
