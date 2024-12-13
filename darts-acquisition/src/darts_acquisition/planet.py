"""PLANET related data loading. Should be used temporary and maybe moved to the acquisition package."""

import logging
import time
from pathlib import Path
from typing import Literal

import rioxarray  # noqa: F401
import xarray as xr

logger = logging.getLogger(__name__.replace("darts_", "darts."))


def parse_planet_type(fpath: Path) -> Literal["orthotile", "scene"]:
    """Parse the type of Planet data from the file path.

    Args:
        fpath (Path): The file path to the Planet data.

    Returns:
        Literal["orthotile", "scene"]: The type of Planet data.

    Raises:
        FileNotFoundError: If no matching TIFF file is found in the specified path.
        ValueError: If the Planet data type cannot be parsed from the file path.

    """
    # Check if the directory parents contains a "PSOrthoTile" or "PSScene"
    if "PSOrthoTile" == fpath.parent.parent.stem:
        return "orthotile"
    elif "PSScene" == fpath.parent.stem:
        return "scene"

    # If not suceeds, check if the directory contains a file with id of the parent directory

    # Get imagepath
    try:
        ps_image_name_parts = next(fpath.glob("*_SR.tif")).split("_")
    except StopIteration:
        raise FileNotFoundError(f"No matching TIFF files found in {fpath.resolve()} (.glob('*_SR.tif'))")

    if len(ps_image_name_parts) == 6:  # PSOrthoTile
        _, tile_id, _, _, _, _ = ps_image_name_parts
        if tile_id == fpath.parent.stem:
            return "orthotile"
        else:
            raise ValueError(f"Could not parse Planet data type from {fpath}")
    elif len(ps_image_name_parts) == 7:  # PSScene
        return "scene"
    else:
        raise ValueError(f"Could not parse Planet data type from {fpath}")


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

    # Convert to Path object if a string is provided
    fpath = fpath if isinstance(fpath, Path) else Path(fpath)

    # Check if the directory contains a PSOrthoTile or PSScene
    planet_type = parse_planet_type(fpath)
    logger.debug(f"Loading Planet PS {planet_type.capitalize()} from {fpath.resolve()}")

    # Get imagepath
    try:
        ps_image = next(fpath.glob("*_SR.tif"))
    except StopIteration:
        raise FileNotFoundError(f"No matching TIFF files found in {fpath.resolve()} (.glob('*_SR.tif'))")

    # Define band names and corresponding indices
    planet_da = xr.open_dataarray(ps_image)

    # Create a dataset with the bands
    bands = ["blue", "green", "red", "nir"]
    ds_planet = (
        planet_da.fillna(0).rio.write_nodata(0).astype("uint16").assign_coords({"band": bands}).to_dataset(dim="band")
    )
    for var in ds_planet.variables:
        ds_planet[var].assign_attrs(
            {
                "long_name": f"PLANET {var.capitalize()}",
                "data_source": "planet",
                "planet_type": planet_type,
                "units": "Reflectance",
            }
        )
    ds_planet.attrs = {"tile_id": fpath.parent.stem if planet_type == "orthotile" else fpath.stem}
    logger.debug(f"Loaded Planet scene in {time.time() - start_time} seconds.")
    return ds_planet


def load_planet_masks(fpath: str | Path) -> xr.Dataset:
    """Load the valid and quality data masks from a Planet scene.

    Args:
        fpath (str | Path): The file path to the Planet scene from which to derive the masks.

    Raises:
        FileNotFoundError: If no matching UDM-2 TIFF file is found in the specified path.

    Returns:
        xr.Dataset: A merged xarray Dataset containing two data masks:
            - 'valid_data_mask': A mask indicating valid (1) and no data (0).
            - 'quality_data_mask': A mask indicating high quality (1) and low quality (0).

    """
    start_time = time.time()

    # Convert to Path object if a string is provided
    fpath = fpath if isinstance(fpath, Path) else Path(fpath)

    logger.debug(f"Loading data masks from {fpath.resolve()}")

    # Get imagepath
    udm_path = next(fpath.glob("*_udm2.tif"))
    if not udm_path:
        raise FileNotFoundError(f"No matching UDM-2 TIFF files found in {fpath.resolve()} (.glob('*_udm2.tif'))")

    # See udm classes here: https://developers.planet.com/docs/data/udm-2/
    da_udm = xr.open_dataarray(udm_path)

    # valid data mask: valid data = 1, no data = 0
    valid_data_mask = (
        (da_udm.sel(band=8) == 0)
        .assign_attrs({"data_source": "planet", "long_name": "Valid data mask"})
        .to_dataset(name="valid_data_mask")
        .drop_vars("band")
    )

    # quality data mask: high quality = 1, low quality = 0
    quality_data_mask = (
        (da_udm.sel(band=[2, 3, 4, 5, 6]).max(axis=0) != 1)
        .assign_attrs({"data_source": "planet", "long_name": "Quality data mask"})
        .to_dataset(name="quality_data_mask")
    )

    qa_ds = xr.merge([valid_data_mask, quality_data_mask])
    logger.debug(f"Loaded data masks in {time.time() - start_time} seconds.")
    return qa_ds
