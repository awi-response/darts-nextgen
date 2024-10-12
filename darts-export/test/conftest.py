import numpy as np
import pytest
import xarray


@pytest.fixture
def probabilities():
    xds = xarray.Dataset(
        {"probabilities": (["y", "x"], np.random.randint(0, 100, [16, 16]))},  # noqa: NPY002
        coords={"x": range(500000, 500016), "y": range(4500000, 4500016)},
    )
    xds.rio.write_crs("EPSG:32601", inplace=True)
    xds.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    xds.rio.write_transform(inplace=True)
    return xds
