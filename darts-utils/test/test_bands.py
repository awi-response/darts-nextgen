import os

import numpy as np
import xarray as xr

from darts_utils.bands import BandCodec, BandLoader

loader = BandLoader(
    {
        "ndvi": BandCodec.ndi(),
        "probability": BandCodec.percentage(),
        "probability_*": BandCodec.percentage(),
        "nir": BandCodec(
            disk_dtype="int16",
            memory_dtype="float32",
            valid_range=(0.0, 10000.0),
            scale_factor=1.0,
            offset=0.0,
            fill_value=-1,
        ),
        "tcvis": BandCodec(
            disk_dtype="uint8",
            memory_dtype="uint8",
            valid_range=(0, 255),
        ),
        "qmask": BandCodec(
            disk_dtype="uint8",
            memory_dtype="uint8",
            valid_range=(0, 2),
        ),
        "mask": BandCodec.bool(),
    }
)


def _create_dataset():
    # TODO: Add Georeferencing / Generally more complex dataset
    ndvi = xr.DataArray([[-0.1, 0.2], [-0.3, np.nan]], dims=["x", "y"], coords={"x": [1, 2], "y": [3, 4]}).astype(
        "float32"
    )
    probability = xr.DataArray([[0.5, 0.6], [0.7, np.nan]], dims=["x", "y"], coords={"x": [1, 2], "y": [3, 4]}).astype(
        "float32"
    )
    nir = xr.DataArray([[1000.0, 1200.0], [3000.0, np.nan]], dims=["x", "y"], coords={"x": [1, 2], "y": [3, 4]}).astype(
        "float32"
    )
    tcvis = xr.DataArray([[100, 150], [200, 0]], dims=["x", "y"], coords={"x": [1, 2], "y": [3, 4]}).astype("uint8")
    qmask = xr.DataArray([[1, 2], [2, 0]], dims=["x", "y"], coords={"x": [1, 2], "y": [3, 4]}).astype("uint8")
    mask = qmask == 2

    dataset = xr.Dataset(
        {
            "ndvi": ndvi,
            "probability": probability,
            "probability_firstmodel": probability.copy(deep=True),
            "nir": nir,
            "tcvis": tcvis,
            "qmask": qmask,
            "mask": mask,
        }
    )
    return dataset


def test_band_validate():
    loader.validate()  # This will raise an error if validation fails


def test_band_roundtrip():
    dataset = _create_dataset()
    try:
        loader.store(dataset, "test_bands.nc")
        loaded = loader.load("test_bands.nc")
        xr.testing.assert_identical(dataset, loaded)
    finally:
        if os.path.exists("test_bands.nc"):
            os.remove("test_bands.nc")


def test_band_normalization():
    dataset = _create_dataset()
    normalized = loader.normalize(dataset[["ndvi", "nir", "tcvis", "qmask", "mask"]])
    for band in normalized:
        assert normalized[band].dtype == "float32"
        assert (normalized[band] >= 0.0).all()
        assert (normalized[band] <= 1.0).all()
