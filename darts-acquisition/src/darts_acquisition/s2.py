"""Sentinel 2 related data loading. Should be used temporary and maybe moved to the acquisition package."""

import logging
import time
from pathlib import Path

import ee
import geopandas as gpd
import odc.geo.xr
import rioxarray  # noqa: F401
import xarray as xr
from odc.geo.geobox import GeoBox
from pystac_client import Client

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


def load_s2_from_gee(
    img: str | ee.Image, bands_mapping: dict = {"B2": "blue", "B3": "green", "B4": "red", "B8": "nir"}
) -> xr.Dataset:
    """Load a Sentinel 2 scene from Google Earth Engine and return it as an xarray dataset.

    Args:
        img (str | ee.Image): The Sentinel 2 image ID or the ee image object.
        bands_mapping (dict[str, str], optional): A mapping from bands to obtain.
            Will be renamed to the corresponding band names.
            Defaults to {"B2": "blue", "B3": "green", "B4": "red", "B8": "nir"}.

    Returns:
        xr.Dataset: The loaded dataset

    """
    tick_fstart = time.perf_counter()

    if isinstance(img, str):
        s2id = img
        img = ee.Image(f"COPERNICUS/S2_SR_HARMONIZED/{s2id}")
    else:
        s2id = img.id().getInfo().split("/")[-1]
    logger.debug(f"Loading Sentinel 2 scene from GEE with ID {s2id}")

    if "SCL" not in bands_mapping.keys():
        bands_mapping["SCL"] = "scl"
    img = img.select(list(bands_mapping.keys()))
    ds_s2 = xr.open_dataset(
        img,
        engine="ee",
        geometry=img.geometry(),
        crs=img.select(0).projection().crs().getInfo(),
        scale=10,
    )
    ds_s2 = ds_s2.isel(time=0).drop_vars("time").rename({"X": "x", "Y": "y"}).transpose("y", "x")
    ds_s2 = ds_s2.rename_vars(bands_mapping)
    tick_send = time.perf_counter()
    logger.debug(f"Found a dataset with shape {ds_s2.sizes} for tile {s2id} in {tick_send - tick_fstart} seconds.")

    tick_dstart = time.perf_counter()
    ds_s2 = ds_s2.compute()
    tick_dend = time.perf_counter()
    logger.debug(f"Downloaded the data for tile {s2id} in {tick_dend - tick_dstart} seconds.")

    # valid data mask: valid data = 1, no data = 0
    ds_s2["valid_data_mask"] = (1 - ds_s2["scl"].fillna(0).isin([0, 1])).assign_attrs(
        {"data_source": "s2", "long_name": "Valid Data Mask"}
    )

    # quality data mask: high quality = 1, low quality = 0
    ds_s2["quality_data_mask"] = (
        ds_s2["scl"].isin([4, 5, 6]).assign_attrs({"data_source": "s2", "long_name": "Quality Data Mask"})
    )

    ds_s2 = ds_s2.drop_vars("scl")

    return ds_s2


def load_s2_from_stac(
    s2id: str,
    bands_mapping: dict = {"B02_10m": "blue", "B03_10m": "green", "B04_10m": "red", "B08_10m": "nir"},
) -> xr.Dataset:
    """Load a Sentinel 2 scene from the Copernicus STAC API and return it as an xarray dataset.

    Args:
        s2id (str): The Sentinel 2 image ID.
        bands_mapping (dict[str, str], optional): A mapping from bands to obtain.
            Will be renamed to the corresponding band names.
            Defaults to {"B2": "blue", "B3": "green", "B4": "red", "B8": "nir"}.

    Returns:
        xr.Dataset: The loaded dataset

    """
    tick_fstart = time.perf_counter()

    if "SCL_20m" not in bands_mapping.keys():
        bands_mapping["SCL_20m"] = "scl"

    catalog = Client.open("https://stac.dataspace.copernicus.eu/v1/")
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        ids=[s2id],
    )
    ds_s2 = xr.open_dataset(
        search,
        engine="stac",
        backend_kwargs={"crs": "utm", "resolution": 10, "bands": list(bands_mapping.keys())},
    )

    ds_s2 = ds_s2.max("time")
    ds_s2 = ds_s2.rename_vars(bands_mapping)
    tick_send = time.perf_counter()
    logger.debug(f"Found a dataset with shape {ds_s2.sizes} for tile {s2id} in {tick_send - tick_fstart} seconds.")

    tick_dstart = time.perf_counter()
    ds_s2.load().load()
    tick_dend = time.perf_counter()
    logger.debug(f"Downloaded the data for tile {s2id} in {tick_dend - tick_dstart} seconds.")

    # valid data mask: valid data = 1, no data = 0
    ds_s2["valid_data_mask"] = (1 - ds_s2["scl"].fillna(0).isin([0, 1])).assign_attrs(
        {"data_source": "s2", "long_name": "Valid Data Mask"}
    )

    # quality data mask: high quality = 1, low quality = 0
    ds_s2["quality_data_mask"] = (
        ds_s2["scl"].isin([4, 5, 6]).assign_attrs({"data_source": "s2", "long_name": "Quality Data Mask"})
    )

    ds_s2 = ds_s2.drop_vars("scl")
    return ds_s2


def get_s2ids_from_shape_ee(
    aoi_shapefile: Path,
    start_date: str,
    end_date: str,
    max_cloud_cover: int = 100,
) -> set[str]:
    """Search for Sentinel 2 tiles via Earth Engine based on an aoi shapefile.

    Args:
        aoi_shapefile (Path): AOI shapefile path. Can be anything readable by geopandas.
        start_date (str): Starting date in a format readable by ee.
        end_date (str): Ending date in a format readable by ee.
        max_cloud_cover (int, optional): Maximum percentage of cloud cover. Defaults to 100.

    Returns:
        set[str]: Unique Sentinel 2 tile IDs.

    """
    logger.debug(
        f"Searching for Sentinel 2 tiles via Earth Engine: "
        f"{aoi_shapefile=} {start_date=} {end_date=} {max_cloud_cover=}."
    )
    tick_fstart = time.perf_counter()
    aoi = gpd.read_file(aoi_shapefile)
    aoi = aoi.to_crs("EPSG:4326")
    s2ids = set()
    for i, row in aoi.iterrows():
        geom = ee.Geometry.Polygon(list(row.geometry.exterior.coords))
        ic = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(geom)
            .filterDate(start_date, end_date)
            .filterMetadata("CLOUDY_PIXEL_PERCENTAGE", "less_than", max_cloud_cover)
        )
        s2ids.update(ic.aggregate_array("system:index").getInfo())
    tick_fend = time.perf_counter()
    logger.debug(f"Found {len(s2ids)} Sentinel 2 tiles via ee in {tick_fend - tick_fstart} seconds.")
    return s2ids


def get_s2ids_from_shape_stac(
    aoi_shapefile: Path,
    start_date: str,
    end_date: str,
    max_cloud_cover: int = 100,
) -> set[str]:
    """Search for Sentinel 2 tiles via Earth Engine based on an aoi shapefile.

    Note:
        `start_date` and `end_date` will be concatted with a `/` to form a date range.
        Read more about the date format here: https://pystac-client.readthedocs.io/en/stable/api.html#pystac_client.Client.search

    Args:
        aoi_shapefile (Path): AOI shapefile path. Can be anything readable by geopandas.
        start_date (str): Starting date in a format readable by pystac_client.
        end_date (str): Ending date in a format readable by pystac_client.
        max_cloud_cover (int, optional): Maximum percentage of cloud cover. Defaults to 100.

    Returns:
        set[str]: Unique Sentinel 2 tile IDs.

    """
    logger.debug(
        f"Searching for Sentinel 2 tiles via STAC: " f"{aoi_shapefile=} {start_date=} {end_date=} {max_cloud_cover=}."
    )
    tick_fstart = time.perf_counter()
    aoi = gpd.read_file(aoi_shapefile)
    catalog = Client.open("https://stac.dataspace.copernicus.eu/v1/")
    s2ids = set()
    for i, row in aoi.iterrows():
        geom = ee.Geometry.Polygon(list(row.geometry.exterior.coords))
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=geom.bounds,
            datetime=f"{start_date}/{end_date}",
            query=[f"eo:cloud_cover<={max_cloud_cover}"],
        )
        s2ids.update(search.get_all_items())
    tick_fend = time.perf_counter()
    logger.debug(f"Found {len(s2ids)} Sentinel 2 tiles via stac in {tick_fend - tick_fstart} seconds.")
    return s2ids
