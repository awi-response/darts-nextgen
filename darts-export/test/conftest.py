import numpy as np
import pytest
import rioxarray  # noqa: F401
import xarray
from scipy import ndimage


@pytest.fixture
def probabilities():
    # create a 16x16 probabilities array:
    nd_prob_cluster = np.zeros((16, 16), dtype=np.int8)
    # have a cluster at 100 probability
    nd_prob_cluster[5:9, 8:10] = 100
    # blur that cluster
    nd_prob_blurred = ndimage.gaussian_filter(nd_prob_cluster, sigma=2)
    # and normalize the values back to 100
    nd_probabilites = (nd_prob_blurred * (100 / nd_prob_blurred.max())).astype("int8")

    xds = xarray.Dataset(
        {"probabilities": (["y", "x"], nd_probabilites)},
        coords={"x": range(500000, 500016), "y": range(4500000, 4500016)},
    )
    xds.rio.write_crs("EPSG:32601", inplace=True)
    xds.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    xds.rio.write_transform(inplace=True)
    return xds
