"""Sentinel 2 related data loading. Should be used temporary and maybe moved to the acquisition package."""

import logging
import time
from pathlib import Path

import rioxarray  # noqa: F401
import xarray as xr

logger = logging.getLogger(__name__)


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
    logger.debug(f"Loading Sentinel 2 scene from {fpath}")
    # Convert to Path object if a string is provided
    fpath = fpath if isinstance(fpath, str) else Path(fpath)

    # Get imagepath
    try:
        s2_image = next(fpath.glob("*_SR_clip.tif"))
    except StopIteration:
        raise FileNotFoundError(f"No matching TIFF files found in {fpath} (.glob('*_SR_clip.tif'))")

    # Define band names and corresponding indices
    s2_da = xr.open_dataarray(s2_image)

    bands = {1: "blue", 2: "green", 3: "red", 4: "nir"}

    # Create a list to hold datasets
    datasets = [
        s2_da.sel(band=index)
        .assign_attrs({"data_source": "s2", "long_name": f"Sentinel 2 {name.capitalize()}"})
        .to_dataset(name=name)
        .drop_vars("band")
        for index, name in bands.items()
    ]

    ds_s2 = xr.merge(datasets)
    logger.debug(f"Loaded Sentinel 2 scene in {time.time() - start_time} seconds.")
    return ds_s2
