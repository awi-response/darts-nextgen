"""PLANET related data loading. Should be used temporary and maybe moved to the acquisition package."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Literal

import odc.geo
import rasterio
import rioxarray
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
    """Load a PlanetScope satellite scene from GeoTIFF files.

    This function loads PlanetScope surface reflectance data (PSScene or PSOrthoTile) from
    a directory containing TIFF files and metadata. The scene type is automatically detected
    from the directory name format.

    Args:
        fpath (str | Path): Path to the directory containing the PlanetScope scene data.
            The directory must follow PlanetScope naming conventions:
            - Scene: YYYYMMDD_HHMMSS_NN_XXXX or YYYYMMDD_HHMMSS_XXXX
            - Orthotile: NNNNNNN_NNNNNNN_YYYY-MM-DD_XXXX
            Must contain *_SR.tif (or *_SR_clip.tif) and *_metadata.json files.

    Returns:
        xr.Dataset: The loaded PlanetScope dataset with the following data variables:
            - blue (float32): Blue band surface reflectance [0-1]
            - green (float32): Green band surface reflectance [0-1]
            - red (float32): Red band surface reflectance [0-1]
            - nir (float32): Near-infrared band surface reflectance [0-1]

            Each variable has attributes:
            - long_name: "PLANET {Band}"
            - units: "Reflectance"
            - data_source: "planet"
            - planet_type: "scene" or "orthotile"

            Dataset-level attributes:
            - azimuth (float): Solar azimuth angle in degrees
            - elevation (float): Solar elevation angle in degrees
            - tile_id (str): Unique identifier for the scene
            - planet_scene_id (str): Scene identifier (for scenes) or scene portion (for orthotiles)
            - planet_orthotile_id (str): Orthotile identifier (only for orthotiles)

    Raises:
        FileNotFoundError: If required TIFF or metadata files are not found in the directory.

    Note:
        - Input DN values are divided by 10000 to convert to reflectance [0-1].
        - The scene type (PSScene vs PSOrthoTile) is automatically detected from the directory name.
        - Solar geometry is extracted from the metadata JSON file.

    Example:
        Load a PlanetScope scene:

        ```python
        from darts_acquisition import load_planet_scene

        # Load scene data
        planet_ds = load_planet_scene("/data/planet/20230615_123045_1234")

        # Access bands
        ndvi = (planet_ds.nir - planet_ds.red) / (planet_ds.nir + planet_ds.red)

        # Check solar geometry
        print(f"Solar azimuth: {planet_ds.azimuth}")
        print(f"Solar elevation: {planet_ds.elevation}")
        ```

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

    ps_meta = next(fpath.glob("*_metadata.json"), None)
    if not ps_meta:
        raise FileNotFoundError(
            f"No matching metadata JSON files found in {fpath.resolve()} (.glob('*_metadata.json'))"
        )
    metadata = json.load(ps_meta.open())

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

    # Add sun and elevation from metadata
    ds_planet.attrs["azimuth"] = metadata.get("sun_azimuth", float("nan"))
    ds_planet.attrs["elevation"] = metadata.get("sun_elevation", float("nan"))

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
    """Load quality and validity masks from a PlanetScope scene's UDM-2 data.

    This function extracts data quality information from the PlanetScope Usable Data Mask
    (UDM-2) to create simplified quality masks for filtering and analysis.

    Args:
        fpath (str | Path): Path to the directory containing the PlanetScope scene data.
            Must contain *_udm2.tif (or *_udm2_clip.tif) file.

    Returns:
        xr.Dataset: Dataset containing quality mask information with the following data variables:
            - quality_data_mask (uint8): Combined quality indicator
                * 0 = Invalid (no data)
                * 1 = Low quality (clouds, shadows, haze, snow, or other artifacts)
                * 2 = High quality (clear, usable data)
              Attributes: data_source="planet", long_name="Quality data mask",
              description="0 = Invalid, 1 = Low Quality, 2 = High Quality"
            - planet_udm (uint8): Raw UDM-2 bands (8 bands)
              Attributes: data_source="planet", long_name="Planet UDM",
              description="Usable Data Mask"

    Raises:
        FileNotFoundError: If the UDM-2 TIFF file is not found in the directory.

    Note:
        Quality mask derivation logic:
        - Invalid: UDM band 8 (no data) is set
        - Low quality: Any of UDM bands 2-6 (clouds, shadows, haze, snow, or artifacts) is set
        - High quality: Neither invalid nor low quality

        UDM-2 band definitions:
        1. Clear - 2. Snow - 3. Shadow - 4. Light Haze - 5. Heavy Haze
        6. Cloud - 7. Confidence - 8. No Data

    Example:
        Load and apply quality masks:

        ```python
        from darts_acquisition import load_planet_scene, load_planet_masks

        # Load scene and masks
        scene = load_planet_scene("/data/planet/20230615_123045_1234")
        masks = load_planet_masks("/data/planet/20230615_123045_1234")

        # Filter to high quality pixels only
        scene_filtered = scene.where(masks.quality_data_mask == 2)

        # Count quality distribution
        import numpy as np
        unique, counts = np.unique(
            masks.quality_data_mask.values,
            return_counts=True
        )
        print(dict(zip(unique, counts)))
        ```

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


def get_planet_geometry(fpath: str | Path) -> odc.geo.Geometry:
    """Get the geometry of a Planet scene.

    Args:
        fpath (str | Path): The file path to the Planet scene from which to derive the geometry.

    Returns:
        odc.geo.Geometry: The geometry of the Planet scene.

    Raises:
        FileNotFoundError: If no matching TIFF file is found in the specified path.

    """
    # Convert to Path object if a string is provided
    fpath = fpath if isinstance(fpath, Path) else Path(fpath)
    # Get imagepath
    ps_image = next(fpath.glob("*_SR.tif"), None)
    if not ps_image:
        ps_image = next(fpath.glob("*_SR_clip.tif"), None)
    if not ps_image:
        raise FileNotFoundError(f"No matching TIFF files found in {fpath.resolve()} (.glob('*_SR.tif'))")

    planet_raster = rasterio.open(ps_image)
    return odc.geo.BoundingBox(*planet_raster.bounds, crs=planet_raster.crs)
