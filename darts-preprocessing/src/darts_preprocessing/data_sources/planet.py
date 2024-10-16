"""PLANET related data loading. Should be used temporary and maybe moved to the acquisition package."""

import logging
import time
from pathlib import Path

import rioxarray  # noqa: F401
import xarray as xr

logger = logging.getLogger(__name__)


def load_planet_scene(fpath: str | Path) -> xr.Dataset:
    """Load a PlanetScope satellite GeoTIFF file and return it as an xarray datset.

    Args:
        fpath (str | Path): The path to the directory containing the TIFF files or a specific path to the TIFF file.

    Returns:
        xr.Dataset: The loaded dataset

    Raises:
        FileNotFoundError: If no matching TIFF file is found in the specified path.

    """
    start_time = time.time()
    logger.debug(f"Loading Planet scene from {fpath}")
    # Convert to Path object if a string is provided
    fpath = fpath if isinstance(fpath, str) else Path(fpath)

    # Get imagepath
    ps_image = next(fpath.glob("*_SR.tif"))
    if not ps_image:
        raise FileNotFoundError(f"No matching TIFF files found in {fpath} (.glob('*_SR.tif'))")

    # Define band names and corresponding indices
    planet_da = xr.open_dataarray(ps_image)

    bands = {1: "blue", 2: "green", 3: "red", 4: "nir"}

    # Create a list to hold datasets
    datasets = [
        planet_da.sel(band=index)
        .assign_attrs({"data_source": "planet", "long_name": f"PLANET {name.capitalize()}"})
        .to_dataset(name=name)
        .drop_vars("band")
        for index, name in bands.items()
    ]

    # Merge all datasets into one
    ds_planet = xr.merge(datasets)
    logger.debug(f"Loaded Planet scene in {time.time() - start_time} seconds.")
    return ds_planet
