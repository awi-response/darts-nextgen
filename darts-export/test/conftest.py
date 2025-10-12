import numpy as np
import pytest
import rioxarray
import xarray
from scipy import ndimage

BINARIZATION_THRESHOLD = 50


def _create_dataset(ary: np.ndarray, binarization_threshold: int):
    nd_prob_blurred = ndimage.gaussian_filter(ary, sigma=2)
    # and normalize the values back to 100
    nd_probabilites = (nd_prob_blurred * (100 / nd_prob_blurred.max())).astype("int8")

    # create the binarized data:
    binarized_segs = np.zeros(nd_probabilites.shape, dtype="uint8")
    binarized_segs[nd_probabilites > binarization_threshold] = 1

    xds = xarray.Dataset(
        {
            "probabilities": (["y", "x"], nd_probabilites),
            "binarized_segmentation": (["y", "x"], binarized_segs),
        },
        coords={"x": range(500000, 500000 + ary.shape[0]), "y": range(4500000, 4500000 + ary.shape[1])},
    )
    xds.rio.write_crs("EPSG:32601", inplace=True)
    xds.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    xds.rio.write_transform(inplace=True)
    return xds


@pytest.fixture
def probabilities_1():
    # create a 16x16 probabilities array:
    nd_prob_cluster = np.zeros((16, 16), dtype=np.int8)
    # have a cluster at 100 probability
    nd_prob_cluster[5:9, 8:10] = 100

    return _create_dataset(nd_prob_cluster, binarization_threshold=50)


@pytest.fixture
def probabilities_2():
    size = 128

    nd_prob_cluster = np.zeros((size, size), dtype=np.int8)

    # create four single clusters
    nd_prob_cluster[110:115, 10:15] = 100  # top left mini
    nd_prob_cluster[16:44, 11:17] = 100  # bottom left
    nd_prob_cluster[87:92, 56:78] = 100  # horizontal mid topish
    nd_prob_cluster[22:35, 110:117] = 100  # right

    # big blob middle bottom :
    nd_prob_cluster[20:31, 55:78] = 100
    nd_prob_cluster[5:23, 75:85] = 100
    nd_prob_cluster[12:15, 84:87] = 110

    return _create_dataset(nd_prob_cluster, binarization_threshold=50)


@pytest.fixture
def ensemble_submodel_dataset(probabilities_2):
    ensemble_ds = probabilities_2

    ensemble_ds["probabilities-tcvis"] = ensemble_ds["probabilities"].copy()
    ensemble_ds["probabilities-notcvis"] = ensemble_ds["probabilities"].copy()

    ensemble_ds["binarized_segmentation-tcvis"] = ensemble_ds["binarized_segmentation"].copy()
    ensemble_ds["binarized_segmentation-notcvis"] = ensemble_ds["binarized_segmentation"].copy()

    return ensemble_ds
