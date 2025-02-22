"""Sentinel 2 related data loading. Should be used temporary and maybe moved to the acquisition package."""

import logging
import time
from pathlib import Path

import odc.geo.xr
import rioxarray  # noqa: F401
import xarray as xr
from odc.geo.geobox import GeoBox

logger = logging.getLogger(__name__.replace("darts_", "darts."))


def parse_s2_tile_id(fpath: str | Path) -> tuple[str, str, str]:
    """Parse the Sentinel 2 tile ID from a file path.

    Args:
        fpath (str | Path): The path to the directory containing the TIFF files.

    Returns:
        tuple[str, str, str]: A tuple containing the Planet crop ID, the Sentinel 2 tile ID and the combined tile ID.

    Raises:
        FileNotFoundError: If no matching TIFF file is found in the specified path.

    """
    fpath = fpath if isinstance(fpath, Path) else Path(fpath)
    try:
        s2_image = next(fpath.glob("*_SR*.tif"))
    except StopIteration:
        raise FileNotFoundError(f"No matching TIFF files found in {fpath.resolve()} (.glob('*_SR*.tif'))")
    planet_crop_id = fpath.stem
    s2_tile_id = "_".join(s2_image.stem.split("_")[:3])
    tile_id = f"{planet_crop_id}_{s2_tile_id}"
    return planet_crop_id, s2_tile_id, tile_id


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
    fpath = fpath if isinstance(fpath, Path) else Path(fpath)

    logger.debug(f"Loading Sentinel 2 scene from {fpath.resolve()}")

    # Get imagepath
    try:
        s2_image = next(fpath.glob("*_SR*.tif"))
    except StopIteration:
        raise FileNotFoundError(f"No matching TIFF files found in {fpath.resolve()} (.glob('*_SR*.tif'))")

    # Define band names and corresponding indices
    s2_da = xr.open_dataarray(s2_image)

    # Create a dataset with the bands
    bands = ["blue", "green", "red", "nir"]
    ds_s2 = s2_da.fillna(0).rio.write_nodata(0).astype("uint16").assign_coords({"band": bands}).to_dataset(dim="band")

    for var in ds_s2.data_vars:
        ds_s2[var].assign_attrs(
            {"data_source": "s2", "long_name": f"Sentinel 2 {var.capitalize()}", "units": "Reflectance"}
        )

    planet_crop_id, s2_tile_id, tile_id = parse_s2_tile_id(fpath)
    ds_s2.attrs["planet_crop_id"] = planet_crop_id
    ds_s2.attrs["s2_tile_id"] = s2_tile_id
    ds_s2.attrs["tile_id"] = tile_id
    logger.debug(f"Loaded Sentinel 2 scene in {time.time() - start_time} seconds.")
    return ds_s2


def load_s2_masks(fpath: str | Path, reference_geobox: GeoBox) -> xr.Dataset:
    """Load the valid and quality data masks from a Sentinel 2 scene.

    Args:
        fpath (str | Path): The path to the directory containing the TIFF files.
        reference_geobox (GeoBox): The reference geobox to reproject, resample and crop the masks data to.


    Returns:
        xr.Dataset: A merged xarray Dataset containing two data masks:
            - 'valid_data_mask': A mask indicating valid (1) and no data (0).
            - 'quality_data_mask': A mask indicating high quality (1) and low quality (0).

    """
    start_time = time.time()

    # Convert to Path object if a string is provided
    fpath = fpath if isinstance(fpath, Path) else Path(fpath)

    logger.debug(f"Loading data masks from {fpath.resolve()}")

    # TODO: SCL band in SR file
    try:
        scl_path = next(fpath.glob("*_SCL*.tif"))
    except StopIteration:
        logger.warning("Found no data quality mask (SCL). No masking will occur.")
        valid_data_mask = (odc.geo.xr.xr_zeros(reference_geobox, dtype="uint8") + 1).to_dataset(name="valid_data_mask")
        valid_data_mask.attrs = {"data_source": "s2", "long_name": "Valid Data Mask"}
        quality_data_mask = odc.geo.xr.xr_zeros(reference_geobox, dtype="uint8").to_dataset(name="quality_data_mask")
        quality_data_mask.attrs = {"data_source": "s2", "long_name": "Quality Data Mask"}
        qa_ds = xr.merge([valid_data_mask, quality_data_mask])
        return qa_ds

    # See scene classes here: https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/scene-classification/
    da_scl = xr.open_dataarray(scl_path)

    da_scl = da_scl.odc.reproject(reference_geobox, sampling="nearest")

    # Match crs
    da_scl = da_scl.rio.write_crs(reference_geobox.crs)

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
