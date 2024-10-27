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
    try:
        ps_image = next(fpath.glob("*_SR.tif"))
    except StopIteration:
        raise FileNotFoundError(f"No matching TIFF files found in {fpath} (.glob('*_SR.tif'))")

    # Define band names and corresponding indices
    planet_da = xr.open_dataarray(ps_image)

    bands = {1: "blue", 2: "green", 3: "red", 4: "nir"}

    # Create a list to hold datasets
    datasets = [
        planet_da.sel(band=index)
        .assign_attrs({"data_source": "planet", "long_name": f"PLANET {name.capitalize()}", "units": "Reflectance"})
        .astype("uint16")
        .to_dataset(name=name)
        .drop_vars("band")
        for index, name in bands.items()
    ]

    # Merge all datasets into one
    ds_planet = xr.merge(datasets)
    ds_planet.attrs["scene_id"] = fpath.stem
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

    Notes:
        - The function utilizes the `get_planet_udm2path_from_planet_scene_path`
          to obtain the path to the UDM (User Data Model) file.
        - It uses `rasterio` to read the UDM file and `xarray` for handling

    """
    start_time = time.time()
    logger.debug(f"Loading data masks from {fpath}")
    # Convert to Path object if a string is provided
    fpath = fpath if isinstance(fpath, str) else Path(fpath)

    # Get imagepath
    udm_path = next(fpath.glob("*_udm2.tif"))
    if not udm_path:
        raise FileNotFoundError(f"No matching UDM-2 TIFF files found in {fpath} (.glob('*_udm2.tif'))")

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
