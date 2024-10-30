"""Sentinel 2 related data loading. Should be used temporary and maybe moved to the acquisition package."""

import logging
import time
from pathlib import Path

import rioxarray  # noqa: F401
import xarray as xr

logger = logging.getLogger(__name__.replace("darts_", "darts."))


def load_s2_scene(fpath: str | Path) -> xr.Dataset:
    """Load a Sentinel 2 satellite GeoTIFF file and return it as an xarray datset.

    Args:
        fpath (str | Path): The path to the directory containing the TIFF files.

    Returns:
        xr.Dataset: The loaded dataset

    Raises:
        FileNotFoundError: If no matching TIFF file is found in the specified path.

    """
    start_time = time.time()

    # Convert to Path object if a string is provided
    fpath = fpath if isinstance(fpath, str) else Path(fpath)

    logger.debug(f"Loading Sentinel 2 scene from {fpath.resolve()}")

    # Get imagepath
    try:
        s2_image = next(fpath.glob("*_SR_clip.tif"))
    except StopIteration:
        raise FileNotFoundError(f"No matching TIFF files found in {fpath.resolve()} (.glob('*_SR_clip.tif'))")

    # Define band names and corresponding indices
    s2_da = xr.open_dataarray(s2_image)

    bands = {1: "blue", 2: "green", 3: "red", 4: "nir"}

    # Create a list to hold datasets
    datasets = [
        s2_da.sel(band=index)
        .assign_attrs({"data_source": "s2", "long_name": f"Sentinel 2 {name.capitalize()}", "units": "Reflectance"})
        .fillna(0)
        .rio.write_nodata(0)
        .astype("uint16")
        .to_dataset(name=name)
        .drop_vars("band")
        for index, name in bands.items()
    ]

    ds_s2 = xr.merge(datasets)
    ds_s2.attrs["tile_id"] = fpath.stem
    logger.debug(f"Loaded Sentinel 2 scene in {time.time() - start_time} seconds.")
    return ds_s2


def load_s2_masks(fpath: str | Path):
    start_time = time.time()

    # Convert to Path object if a string is provided
    fpath = fpath if isinstance(fpath, str) else Path(fpath)

    logger.debug(f"Loading data masks from {fpath.resolve()}")

    scl_path = next(fpath.glob("*_SCL_clip.tif"))
    if not scl_path:
        raise FileNotFoundError(f"No matching TIFF files found in {fpath.resolve()} (.glob('*_SCL.tif'))")

    # See scene classes here: https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/scene-classification/
    da_scl = xr.open_dataarray(scl_path)

    # valid data mask: valid data = 1, no data = 0
    valid_data_mask = (
        (1 - da_scl.sel(band=1).fillna(0).isin([0, 1]))
        .assign_attrs({"data_source": "s2", "long_name": "Valid Data Mask"})
        .to_dataset(name="valid_data_mask")
        .drop_vars("band")
    )

    # quality data mask: high quality = 1, low quality = 0
    quality_data_mask = (
        da_scl.sel(band=1)
        .isin([4, 5, 6])
        .assign_attrs({"data_source": "s2", "long_name": "Quality Data Mask"})
        .to_dataset(name="quality_data_mask")
        .drop_vars("band")
    )

    qa_ds = xr.merge([valid_data_mask, quality_data_mask])
    logger.debug(f"Loaded data masks in {time.time() - start_time} seconds.")
    return qa_ds
