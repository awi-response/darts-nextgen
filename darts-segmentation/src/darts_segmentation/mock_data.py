# ruff: noqa
import numpy as np
import rioxarray  # noqa
import xarray as xr
from affine import Affine

H, W = 2_000, 2_000


def mock_source(name, dtype, meta, source) -> xr.DataArray:
    """Mock a single source band."""
    da = xr.DataArray(
        np.ones([H, W], dtype=dtype),
        coords={
            "y": meta["y"],
            "x": meta["x"],
        },
        dims=["y", "x"],
        attrs={"source": source},
        name=name,
    )
    da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    da.rio.write_crs(meta["crs"], inplace=True)
    da.rio.write_transform(meta["transform"], inplace=True)
    return da


def mock_tile(primary_source):
    # mock coordinates
    meta = {
        "y": np.arange(H),
        "x": np.arange(W),
        "transform": Affine(0.5, 0.0, 0.0, 0.0, -0.5, 100.0),  # mock transform
        "crs": "EPSG:4326",  # mock CRS
    }

    blue = mock_source("blue", np.uint16, meta, primary_source)
    green = mock_source("green", np.uint16, meta, primary_source)
    red = mock_source("red", np.uint16, meta, primary_source)
    nir = mock_source("nir", np.uint16, meta, primary_source)

    relative_elevation = mock_source("relative_elevation", np.float32, meta, "artic-dem")
    slope = mock_source("slope", np.float32, meta, "artic-dem")

    ndvi = ((nir - red) / (nir + red)).astype(np.float32)

    tcvis_brightness = mock_source("tc_brightness", np.uint8, meta, "tcvis")
    tcvis_greenness = mock_source("tc_greenness", np.uint8, meta, "tcvis")
    tcvis_wetness = mock_source("tc_wetness", np.uint8, meta, "tcvis")

    return xr.Dataset(
        {
            "blue": blue,
            "green": green,
            "red": red,
            "nir": nir,
            "relative_elevation": relative_elevation,
            "slope": slope,
            "ndvi": ndvi,
            "tc_brightness": tcvis_brightness,
            "tc_greenness": tcvis_greenness,
            "tc_wetness": tcvis_wetness,
        },
    )
