"""PLANET related data loading. Should be used temporary and maybe moved to the acquisition package."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Literal

import rioxarray  # noqa: F401
import xarray as xr
from stopuhr import stopwatch

logger = logging.getLogger(__name__.replace("darts_", "darts."))


def _is_valid_date(date_str: str, format: str) -> bool:
    try:
        datetime.strptime(date_str, format)
        return True
    except ValueError:
        return False


def parse_planet_type(fpath: Path) -> Literal["orthotile", "scene"]:
    """Parse the type of Planet data from the directory path.

    Args:
        fpath (Path): The directory path to the Planet data.

    Returns:
        Literal["orthotile", "scene"]: The type of Planet data.

    Raises:
        ValueError: If the Planet data type cannot be parsed from the file path.

    """
    # Cases for Scenes:
    # - YYYYMMDD_HHMMSS_NN_XXXX
    # - YYYYMMDD_HHMMSS_XXXX

    # Cases for Orthotiles:
    # NNNNNNN/NNNNNNN_NNNNNNN_YYYY-MM-DD_XXXX
    # NNNNNNN_NNNNNNN_YYYY-MM-DD_XXXX

    assert fpath.is_dir(), "fpath must be the parent directory!"

    ps_name_parts = fpath.stem.split("_")

    if len(ps_name_parts) == 3:
        # Must be scene or invalid
        date, time, ident = ps_name_parts
        if _is_valid_date(date, "%Y%m%d") and _is_valid_date(time, "%H%M%S") and len(ident) == 4:
            return "scene"

    if len(ps_name_parts) == 4:
        # Assume scene
        date, time, n, ident = ps_name_parts
        if _is_valid_date(date, "%Y%m%d") and _is_valid_date(time, "%H%M%S") and n.isdigit() and len(ident) == 4:
            return "scene"
        # Is not scene, assume orthotile
        chunkid, tileid, date, ident = ps_name_parts
        if chunkid.isdigit() and tileid.isdigit() and _is_valid_date(date, "%Y-%m-%d") and len(ident) == 4:
            return "orthotile"

    raise ValueError(
        f"Could not parse Planet data type from {fpath}."
        f"Expected a format of YYYYMMDD_HHMMSS_NN_XXXX or YYYYMMDD_HHMMSS_XXXX for scene, "
        "or NNNNNNN/NNNNNNN_NNNNNNN_YYYY-MM-DD_XXXX or NNNNNNN_NNNNNNN_YYYY-MM-DD_XXXX for orthotile."
        f"Got {fpath.stem} instead."
        "Please ensure that the parent directory of the file is used, instead of the file itself."
    )


@stopwatch.f("Loading Planet scene", printer=logger.debug)
def load_planet_scene(fpath: str | Path) -> xr.Dataset:
    """Load a PlanetScope satellite GeoTIFF file and return it as an xarray datset.

    Args:
        fpath (str | Path): The path to the directory containing the TIFF files or a specific path to the TIFF file.

    Returns:
        xr.Dataset: The loaded dataset

    Raises:
        FileNotFoundError: If no matching TIFF file is found in the specified path.

    """
    # Convert to Path object if a string is provided
    fpath = fpath if isinstance(fpath, Path) else Path(fpath)

    # Check if the directory contains a PSOrthoTile or PSScene
    planet_type = parse_planet_type(fpath)
    logger.debug(f"Loading Planet PS {planet_type.capitalize()} from {fpath.resolve()}")

    # Get imagepath
    ps_image = next(fpath.glob("*_SR.tif"), None)
    if not ps_image:
        ps_image = next(fpath.glob("*_SR_clip.tif"), None)
    if not ps_image:
        raise FileNotFoundError(f"No matching TIFF files found in {fpath.resolve()} (.glob('*_SR.tif'))")

    # Define band names and corresponding indices
    planet_da = xr.open_dataarray(ps_image)

    # Divide by 10000 to get reflectance between 0 and 1
    planet_da = planet_da.astype("float32") / 10000.0

    # Create a dataset with the bands
    bands = ["blue", "green", "red", "nir"]
    ds_planet = planet_da.assign_coords({"band": bands}).to_dataset(dim="band")
    for var in bands:
        ds_planet[var].attrs["long_name"] = f"PLANET {var.capitalize()}"
        ds_planet[var].attrs["units"] = "Reflectance"

    for var in ds_planet.data_vars:
        ds_planet[var].attrs["data_source"] = "planet"
        ds_planet[var].attrs["planet_type"] = planet_type

    if planet_type == "scene":
        ds_planet.attrs["tile_id"] = fpath.stem
        ds_planet.attrs["planet_scene_id"] = fpath.stem
    elif planet_type == "orthotile":
        ds_planet.attrs["tile_id"] = f"{fpath.parent.stem}-{fpath.stem}"
        ds_planet.attrs["planet_orthotile_id"] = fpath.parent.stem
        ds_planet.attrs["planet_scene_id"] = fpath.stem

    return ds_planet


@stopwatch.f("Loading Planet masks", printer=logger.debug)
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
    # Convert to Path object if a string is provided
    fpath = fpath if isinstance(fpath, Path) else Path(fpath)

    logger.debug(f"Loading data masks from {fpath.resolve()}")

    # Get imagepath
    udm_path = next(fpath.glob("*_udm2.tif"), None)
    if not udm_path:
        udm_path = next(fpath.glob("*_udm2_clip.tif"), None)
    if not udm_path:
        raise FileNotFoundError(f"No matching UDM-2 TIFF files found in {fpath.resolve()} (.glob('*_udm2.tif'))")

    # See udm classes here: https://developers.planet.com/docs/data/udm-2/
    da_udm = xr.open_dataarray(udm_path).astype("uint8")
    invalids = da_udm.sel(band=8).fillna(0) != 0
    low_quality = da_udm.sel(band=[2, 3, 4, 5, 6]).max(axis=0) == 1
    high_quality = ~low_quality & ~invalids
    qa_ds = (
        xr.where(high_quality, 2, 0)
        .where(~low_quality, 1)
        .where(~invalids, 0)
        .astype("uint8")
        .to_dataset(name="quality_data_mask")
        .drop_vars("band")
    )
    qa_ds["planet_udm"] = da_udm

    qa_ds["quality_data_mask"].attrs = {
        "data_source": "planet",
        "long_name": "Quality data mask",
        "description": "0 = Invalid, 1 = Low Quality, 2 = High Quality",
    }
    qa_ds["planet_udm"].attrs = {
        "data_source": "planet",
        "long_name": "Planet UDM",
        "description": "Usable Data Mask",
    }

    return qa_ds
