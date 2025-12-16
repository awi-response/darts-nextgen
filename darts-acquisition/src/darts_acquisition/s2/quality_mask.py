"""Sentinel-2 related data loading."""

import logging

import xarray as xr
from stopuhr import stopwatch
from xrspatial.utils import has_cuda_and_cupy

if has_cuda_and_cupy():
    import cupy as cp  # type: ignore
    import cupy_xarray  # type: ignore

logger = logging.getLogger(__name__.replace("darts_", "darts."))


@stopwatch("Converting Sentinel-2 masks", printer=logger.debug)
def convert_masks(ds_s2: xr.Dataset) -> xr.Dataset:
    """Convert the Sentinel-2 scl mask into our own mask format inplace.

    https://sentiwiki.copernicus.eu/web/s2-processing#S2Processing-ClassificationMaskGeneration

    Invalid: S2 SCL → 0,1
    Low Quality S2: S2 SCL != 0,1 → 3,8,9,11
    High Quality: S2 SCL != 0,1,3,8,9,11 → Alles andere (2,4,5,6,7,10)

    Args:
        ds_s2 (xr.Dataset): The Sentinel-2 dataset containing the SCL band.

    Returns:
        xr.Dataset: The modified dataset.

    """
    assert "s2_scl" in ds_s2.data_vars, "The dataset does not contain the SCL band."

    if has_cuda_and_cupy() and ds_s2.cupy.is_cupy:
        invalids = ds_s2["s2_scl"].fillna(0).isin(cp.array([0, 1]))
        high_quality = ds_s2["s2_scl"].isin(cp.array([2, 4, 5, 6, 7, 10]))
    else:
        invalids = ds_s2["s2_scl"].fillna(0).isin([0, 1])
        high_quality = ds_s2["s2_scl"].isin([2, 4, 5, 6, 7, 10])
    ds_s2["quality_data_mask"] = (
        (~invalids).astype("uint8")  # 0 for invalid, 1 for valid
        + (high_quality).astype("uint8")  # +1 for high quality
    )

    ds_s2["quality_data_mask"].attrs["data_source"] = "s2"
    ds_s2["quality_data_mask"].attrs["long_name"] = "Quality Data Mask"
    ds_s2["quality_data_mask"].attrs["description"] = "0 = Invalid, 1 = Low Quality, 2 = High Quality"

    return ds_s2


@stopwatch("Creating Sentinel-2 mask from observations", printer=logger.debug)
def create_quality_mask_from_observations(ds_s2: xr.Dataset) -> xr.Dataset:
    """Create a quality mask from the observations band.

    Quality mask derivation from observations:
    - Invalid (0): observations == 0
    - Low quality (1): 1 <= observations <= 3
    - High quality (2): observations > 3

    Args:
        ds_s2 (xr.Dataset): The Sentinel-2 dataset containing the observations band.

    Returns:
        xr.Dataset: The modified dataset with the quality_data_mask variable.

    """
    assert "s2_observations" in ds_s2.data_vars, "The dataset does not contain the s2_observations band."

    observations = ds_s2["s2_observations"].fillna(0)

    ds_s2["quality_data_mask"] = (
        (observations > 0).astype("uint8")  # 0 if invalid, 1 otherwise
        + (observations > 3).astype("uint8")  # +1 if high quality (>3)
    ).astype("uint8")

    ds_s2["quality_data_mask"].attrs["data_source"] = "s2"
    ds_s2["quality_data_mask"].attrs["long_name"] = "Quality Data Mask"
    ds_s2["quality_data_mask"].attrs["description"] = "0 = Invalid, 1 = Low Quality, 2 = High Quality"

    return ds_s2
