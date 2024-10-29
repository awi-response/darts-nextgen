"""ArcticDEM related data loading. Should be used temporary and maybe moved to the acquisition package."""

import logging
import time
from pathlib import Path

import rasterio
import rasterio.mask
import rioxarray  # noqa: F401
import xarray as xr

logger = logging.getLogger(__name__.replace("darts_", "darts."))


def load_vrt(vrt_path: Path, reference_dataset: xr.Dataset) -> xr.DataArray:
    """Load a VRT file and reproject it to match the reference dataset.

    Args:
        vrt_path (Path): Path to the vrt file.
        reference_dataset (xr.Dataset): The reference dataset.

    Raises:
        FileNotFoundError: If the VRT file is not found.

    Returns:
        xr.DataArray: The VRT data reprojected to match the reference dataarray.

    """
    if not vrt_path.exists():
        raise FileNotFoundError(f"Could not find the VRT file at {vrt_path}")

    start_time = time.time()

    with rasterio.open(vrt_path) as src:
        with rasterio.vrt.WarpedVRT(
            src, crs=reference_dataset.rio.crs, resampling=rasterio.enums.Resampling.cubic
        ) as vrt:
            bounds = reference_dataset.rio.bounds()
            windows = vrt.window(*bounds)
            shape = (1, len(reference_dataset.y), len(reference_dataset.x))
            data = vrt.read(window=windows, out_shape=shape)[0]  # This is the most time consuming part of the function
            da = xr.DataArray(data, dims=["y", "x"], coords={"y": reference_dataset.y, "x": reference_dataset.x})
            da.rio.write_crs(reference_dataset.rio.crs, inplace=True)
            da.rio.write_transform(reference_dataset.rio.transform(), inplace=True)

    logger.debug(f"Loaded VRT data from {vrt_path} in {time.time() - start_time} seconds.")
    return da


def load_arcticdem(fpath: Path, reference_dataset: xr.Dataset) -> xr.Dataset:
    """Load ArcticDEM data and reproject it to match the reference dataset.

    Args:
        fpath (Path): The path to the ArcticDEM data.
        reference_dataset (xr.Dataset): The reference dataset to reproject, resampled and cropped the ArcticDEM data to.

    Returns:
        xr.Dataset: The ArcticDEM data reprojected, resampled and cropped to match the reference dataset.


    """
    start_time = time.time()
    logger.debug(f"Loading ArcticDEM data from {fpath}")

    slope_vrt = fpath / "slope.vrt"
    elevation_vrt = fpath / "elevation.vrt"

    slope = load_vrt(slope_vrt, reference_dataset)
    slope: xr.Dataset = (
        slope.assign_attrs({"data_source": "arcticdem", "long_name": "Slope"})
        .rio.write_nodata(float("nan"))
        .astype("float32")
        .to_dataset(name="slope")
    )

    relative_elevation = load_vrt(elevation_vrt, reference_dataset)
    relative_elevation: xr.Dataset = (
        relative_elevation.assign_attrs({"data_source": "arcticdem", "long_name": "Relative Elevation", "units": "m"})
        .fillna(0)
        .rio.write_nodata(0)
        .astype("int16")
        .to_dataset(name="relative_elevation")
    )

    articdem_ds = xr.merge([relative_elevation, slope])
    logger.debug(f"Loaded ArcticDEM data in {time.time() - start_time} seconds.")
    return articdem_ds
