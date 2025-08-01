"""Sentinel-2 related data loading. Should be used temporary and maybe moved to the acquisition package."""

import logging
from collections.abc import MutableMapping
from pathlib import Path
from typing import Literal

import ee
import geopandas as gpd
import odc.geo.xr
import pandas as pd
import rioxarray  # noqa: F401
import xarray as xr
from darts_utils.tilecache import XarrayCacheManager
from odc.geo.geobox import GeoBox
from odc.stac import stac_load
from pystac import Item
from pystac_client import Client
from stopuhr import stopwatch

from darts_acquisition.utils.copernicus import init_copernicus

logger = logging.getLogger(__name__.replace("darts_", "darts."))


def _flatten_dict(d: MutableMapping, parent_key: str = "", sep: str = ".") -> MutableMapping:
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


@stopwatch("Converting Sentinel-2 masks", printer=logger.debug)
def convert_masks(ds_s2: xr.Dataset) -> xr.Dataset:
    """Convert the Sentinel-2 scl mask into our own mask format inplace.

    https://sentiwiki.copernicus.eu/web/s2-processing#S2Processing-ClassificationMaskGeneration

    Invalid: S2 SCL → 0,1
    Low Quality S2: S2 SCL != 0,1 → 3,8,9,11
    High Quality: S2 SCL != 0,1,3,8,9,11 → Alles andere (2,4,4,5,6,7,10)

    Args:
        ds_s2 (xr.Dataset): The Sentinel-2 dataset containing the SCL band.

    Returns:
        xr.Dataset: The modified dataset.

    """
    assert "scl" in ds_s2.data_vars, "The dataset does not contain the SCL band."

    ds_s2["quality_data_mask"] = xr.zeros_like(ds_s2["scl"], dtype="uint8")
    # TODO: What about nan values?
    invalids = ds_s2["scl"].fillna(0).isin([0, 1])
    low_quality = ds_s2["scl"].isin([3, 8, 9, 11])
    high_quality = ~invalids & ~low_quality
    # ds_s2["quality_data_mask"] = ds_s2["quality_data_mask"].where(invalids, 0)
    ds_s2["quality_data_mask"] = xr.where(low_quality, 1, ds_s2["quality_data_mask"])
    ds_s2["quality_data_mask"] = xr.where(high_quality, 2, ds_s2["quality_data_mask"])

    ds_s2["quality_data_mask"].attrs["data_source"] = "s2"
    ds_s2["quality_data_mask"].attrs["long_name"] = "Quality Data Mask"
    ds_s2["quality_data_mask"].attrs["description"] = "0 = Invalid, 1 = Low Quality, 2 = High Quality"

    return ds_s2


def parse_s2_tile_id(fpath: str | Path) -> tuple[str, str, str]:
    """Parse the Sentinel-2 tile ID from a file path.

    Args:
        fpath (str | Path): The path to the directory containing the TIFF files.

    Returns:
        tuple[str, str, str]: A tuple containing the Planet crop ID, the Sentinel-2 tile ID and the combined tile ID.

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


@stopwatch.f("Loading Sentinel-2 scene from file", printer=logger.debug)
def load_s2_scene(fpath: str | Path) -> xr.Dataset:
    """Load a Sentinel-2 satellite GeoTIFF file and return it as an xarray datset.

    Args:
        fpath (str | Path): The path to the directory containing the TIFF files.

    Returns:
        xr.Dataset: The loaded dataset

    Raises:
        FileNotFoundError: If no matching TIFF file is found in the specified path.

    """
    # Convert to Path object if a string is provided
    fpath = fpath if isinstance(fpath, Path) else Path(fpath)

    logger.debug(f"Loading Sentinel-2 scene from {fpath.resolve()}")

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
        ds_s2[var].attrs["data_source"] = "s2"
        ds_s2[var].attrs["long_name"] = f"Sentinel 2 {var.capitalize()}"
        ds_s2[var].attrs["units"] = "Reflectance"

    planet_crop_id, s2_tile_id, tile_id = parse_s2_tile_id(fpath)
    ds_s2.attrs["planet_crop_id"] = planet_crop_id
    ds_s2.attrs["s2_tile_id"] = s2_tile_id
    ds_s2.attrs["tile_id"] = tile_id
    return ds_s2


@stopwatch.f("Loading Sentinel-2 masks", printer=logger.debug, print_kwargs=["fpath"])
def load_s2_masks(fpath: str | Path, reference_geobox: GeoBox) -> xr.Dataset:
    """Load the valid and quality data masks from a Sentinel-2 scene.

    Args:
        fpath (str | Path): The path to the directory containing the TIFF files.
        reference_geobox (GeoBox): The reference geobox to reproject, resample and crop the masks data to.


    Returns:
        xr.Dataset: A merged xarray Dataset containing two data masks:
            - 'valid_data_mask': A mask indicating valid (1) and no data (0).
            - 'quality_data_mask': A mask indicating high quality (1) and low quality (0).

    """
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

    da_scl = xr.Dataset({"scl": da_scl.sel(band=1).fillna(0).drop_vars("band").astype("uint8")})
    da_scl = convert_masks(da_scl)

    return da_scl


@stopwatch.f("Loading Sentinel-2 scene from GEE", printer=logger.debug, print_kwargs=["img"])
def load_s2_from_gee(
    img: str | ee.Image,
    bands_mapping: dict = {"B2": "blue", "B3": "green", "B4": "red", "B8": "nir"},
    scale_and_offset: bool | tuple[float, float] = True,
    cache: Path | None = None,
) -> xr.Dataset:
    """Load a Sentinel-2 scene from Google Earth Engine and return it as an xarray dataset.

    Args:
        img (str | ee.Image): The Sentinel-2 image ID or the ee image object.
        bands_mapping (dict[str, str], optional): A mapping from bands to obtain.
            Will be renamed to the corresponding band names.
            Defaults to {"B2": "blue", "B3": "green", "B4": "red", "B8": "nir"}.
        scale_and_offset (bool | tuple[float, float], optional): Whether to apply the scale and offset to the bands.
            If a tuple is provided, it will be used as the (`scale`, `offset`) values with `band * scale + offset`.
            If True, use the default values of `scale` = 0.0001 and `offset` = 0, taken from ee_extra.
            Defaults to True.
        cache (Path | None, optional): The path to the cache directory. If None, no caching will be done.
            Defaults to None.

    Returns:
        xr.Dataset: The loaded dataset

    """
    if isinstance(img, str):
        s2id = img
        img = ee.Image(f"COPERNICUS/S2_SR_HARMONIZED/{s2id}")
    else:
        s2id = img.id().getInfo().split("/")[-1]
    logger.debug(f"Loading Sentinel-2 tile {s2id=} from GEE")

    if "SCL" not in bands_mapping.keys():
        bands_mapping["SCL"] = "scl"

    img = img.select(list(bands_mapping.keys()))

    def _get_tile():
        ds_s2 = xr.open_dataset(
            img,
            engine="ee",
            geometry=img.geometry(),
            crs=img.select(0).projection().crs().getInfo(),
            scale=10,
        )
        ds_s2.attrs["time"] = str(ds_s2.time.values[0])
        ds_s2 = ds_s2.isel(time=0).drop_vars("time").rename({"X": "x", "Y": "y"}).transpose("y", "x")
        ds_s2 = ds_s2.odc.assign_crs(ds_s2.attrs["crs"])
        with stopwatch(f"Downloading data from GEE for {s2id=}", printer=logger.debug):
            ds_s2.load()
        return ds_s2

    ds_s2 = XarrayCacheManager(cache).get_or_create(
        identifier=f"gee-s2srh-{s2id}-{''.join(bands_mapping.keys())}",
        creation_func=_get_tile,
        force=False,
    )

    ds_s2 = ds_s2.rename_vars(bands_mapping)

    for var in ds_s2.data_vars:
        ds_s2[var].attrs["data_source"] = "s2-gee"
        ds_s2[var].attrs["long_name"] = f"Sentinel 2 {var.capitalize()}"
        ds_s2[var].attrs["units"] = "Reflectance"

    ds_s2 = convert_masks(ds_s2)

    # For some reason, there are some spatially random nan values in the data, not only at the borders
    # To workaround this, set all nan values to 0 and add this information to the quality_data_mask
    # This workaround is quite computational expensive, but it works for now
    # TODO: Find other solutions for this problem!
    with stopwatch(f"Fixing nan values in {s2id=}", printer=logger.debug):
        for band in set(bands_mapping.values()) - {"scl"}:
            ds_s2["quality_data_mask"] = xr.where(ds_s2[band].isnull(), 0, ds_s2["quality_data_mask"])
            ds_s2[band] = ds_s2[band].fillna(0)
            # Turn real nan values (scl is nan) into invalid data
            ds_s2[band] = ds_s2[band].where(~ds_s2["scl"].isnull())

    if scale_and_offset:
        if isinstance(scale_and_offset, tuple):
            scale, offset = scale_and_offset
        else:
            scale, offset = 0.0001, 0
        logger.debug(f"Applying {scale=} and {offset=} to {s2id=} optical data")
        for band in set(bands_mapping.values()) - {"scl"}:
            ds_s2[band] = ds_s2[band] * scale + offset

    ds_s2.attrs["s2_tile_id"] = s2id
    ds_s2.attrs["tile_id"] = s2id

    return ds_s2


@stopwatch.f("Loading Sentinel-2 scene from STAC", printer=logger.debug, print_kwargs=["s2item"])
def load_s2_from_stac(
    s2item: str | Item,
    bands_mapping: dict = {"B02_10m": "blue", "B03_10m": "green", "B04_10m": "red", "B08_10m": "nir"},
    scale_and_offset: bool | tuple[float, float] = True,
    cache: Path | None = None,
    aws_profile_name: str = "default",
) -> xr.Dataset:
    """Load a Sentinel-2 scene from the Copernicus STAC API and return it as an xarray dataset.

    Note:
        Must use the `darts.utils.copernicus.init_copernicus` function to setup authentification
        with the Copernicus AWS S3 bucket before using this function.

    Args:
        s2item (str | Item): The Sentinel-2 image ID or the corresponing STAC Item.
        bands_mapping (dict[str, str], optional): A mapping from bands to obtain.
            Will be renamed to the corresponding band names.
            Defaults to {"B02_10m": "blue", "B03_10m": "green", "B04_10m": "red", "B08_10m": "nir"}.
        scale_and_offset (bool | tuple[float, float], optional): Whether to apply the scale and offset to the bands.
            If a tuple is provided, it will be used as the (`scale`, `offset`) values with `band * scale + offset`.
            If True, use the default values of `scale` = 0.0001 and `offset` = 0, taken from ee_extra.
            Defaults to True.
        cache (Path | None, optional): The path to the cache directory. If None, no caching will be done.
            Defaults to None.

    Returns:
        xr.Dataset: The loaded dataset

    """
    s2id = s2item.id if isinstance(s2item, Item) else s2item

    if "SCL_20m" not in bands_mapping.keys():
        bands_mapping["SCL_20m"] = "scl"

    def _get_tile():
        nonlocal s2item

        bands = list(bands_mapping.keys())

        if isinstance(s2item, str):
            catalog = Client.open("https://stac.dataspace.copernicus.eu/v1/")
            search = catalog.search(
                collections=["sentinel-2-l2a"],
                ids=[s2id],
            )
            s2item = next(search.items())

        with stopwatch(f"Downloading data from STAC for {s2id=}", printer=logger.debug):
            # We can't use xpystac here, because they enforce chunking of 1024x1024, which results in long loading times
            # and a potential AWS limit error.
            init_copernicus(profile_name=aws_profile_name)
            ds_s2 = stac_load(
                [s2item],
                bands=bands,
                crs="utm",
                resolution=10,
            )

        ds_s2.attrs = _flatten_dict(s2item.properties)
        # Convert boolean values to int, since they are not supported in netcdf
        for key, value in ds_s2.attrs.items():
            if isinstance(value, bool):
                ds_s2.attrs[key] = int(value)
        ds_s2.attrs["time"] = str(ds_s2.time.values[0])
        ds_s2 = ds_s2.isel(time=0).drop_vars("time")
        return ds_s2

    ds_s2 = XarrayCacheManager(cache).get_or_create(
        identifier=f"stac-s2l2a-{s2id}-{''.join(bands_mapping.keys())}", creation_func=_get_tile, force=False
    )

    ds_s2 = ds_s2.rename_vars(bands_mapping)
    for var in ds_s2.data_vars:
        ds_s2[var].attrs["data_source"] = "s2-stac"
        ds_s2[var].attrs["long_name"] = f"Sentinel-2 {var.capitalize()}"
        ds_s2[var].attrs["units"] = "Reflectance"

    ds_s2 = convert_masks(ds_s2)

    if scale_and_offset:
        if isinstance(scale_and_offset, tuple):
            scale, offset = scale_and_offset
        else:
            scale, offset = 0.0001, 0
        logger.debug(f"Applying {scale=} and {offset=} to {s2id=} optical data")
        for band in set(bands_mapping.values()) - {"scl"}:
            ds_s2[band] = ds_s2[band] * scale + offset

    ds_s2.attrs["s2_tile_id"] = s2id
    ds_s2.attrs["tile_id"] = s2id

    return ds_s2


@stopwatch("Searching for Sentinel-2 tiles via Earth Engine", printer=logger.debug)
def get_s2ids_from_geodataframe_ee(
    aoi: gpd.GeoDataFrame | Path | str,
    start_date: str,
    end_date: str,
    max_cloud_cover: int = 100,
) -> set[str]:
    """Search for Sentinel-2 tiles via Earth Engine based on an aoi shapefile.

    Args:
        aoi (gpd.GeoDataFrame | Path | str): AOI as a GeoDataFrame or path to a shapefile.
            If a path is provided, it will be read using geopandas.
        start_date (str): Starting date in a format readable by ee.
        end_date (str): Ending date in a format readable by ee.
        max_cloud_cover (int, optional): Maximum percentage of cloud cover. Defaults to 100.

    Returns:
        set[str]: Unique Sentinel-2 tile IDs.

    """
    if isinstance(aoi, Path | str):
        aoi = gpd.read_file(aoi)
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
    logger.debug(f"Found {len(s2ids)} Sentinel-2 tiles via ee.")
    return s2ids


@stopwatch("Searching for Sentinel-2 tiles via STAC", printer=logger.debug)
def search_s2_stac(
    intersects,
    start_date: str,
    end_date: str,
    max_cloud_cover: int = 100,
) -> dict[str, Item]:
    """Search for Sentinel-2 tiles via STAC based on an area of interest (intersects) and date range.

    Note:
        `start_date` and `end_date` will be concatted with a `/` to form a date range.
        Read more about the date format here: https://pystac-client.readthedocs.io/en/stable/api.html#pystac_client.Client.search

    Args:
        intersects (any): The geometry object to search for Sentinel-2 tiles.
            Can be anything implementing the `__geo_interface__` protocol, such as a GeoDataFrame or a shapely geometry.
        start_date (str): Starting date in a format readable by pystac_client.
        end_date (str): Ending date in a format readable by pystac_client.
        max_cloud_cover (int, optional): Maximum percentage of cloud cover. Defaults to 100.

    Returns:
        dict[str, Item]: A dictionary of found Sentinel-2 items as values and the s2id as keys.

    """
    catalog = Client.open("https://stac.dataspace.copernicus.eu/v1/")
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects=intersects,
        datetime=f"{start_date}/{end_date}",
        query=[f"eo:cloud_cover<={max_cloud_cover}"],
    )
    found_items = list(search.items())
    if len(found_items) == 0:
        logger.debug(
            "No Sentinel-2 items found for the given parameters:"
            f" {intersects=}, {start_date=}, {end_date=}, {max_cloud_cover=}"
        )
        return {}
    logger.debug(f"Found {len(found_items)} Sentinel-2 items via STAC.")
    return {item.id: item for item in found_items}


@stopwatch("Searching for Sentinel-2 tiles via STAC from AOI", printer=logger.debug)
def get_s2ids_from_geodataframe_stac(
    aoi: gpd.GeoDataFrame | Path | str,
    start_date: str,
    end_date: str,
    max_cloud_cover: int = 100,
    simplify_geometry: float | Literal[False] = False,
) -> dict[str, Item]:
    """Search for Sentinel-2 tiles via STAC based on an area of interest (aoi) and date range.

    Args:
        aoi (gpd.GeoDataFrame | Path | str): AOI as a GeoDataFrame or path to a shapefile.
            If a path is provided, it will be read using geopandas.
        start_date (str): Starting date in a format readable by pystac_client.
        end_date (str): Ending date in a format readable by pystac_client.
        max_cloud_cover (int, optional): Maximum percentage of cloud cover. Defaults to 100.
        simplify_geometry (float | Literal[False], optional): If a float is provided, the geometry will be simplified
            using the `simplify` method of geopandas. If False, no simplification will be done.
            This may become useful for large / weird AOIs which are too large for the STAC API.
            Defaults to False.

    Returns:
        dict[str, Item]: A dictionary of found Sentinel-2 items.

    """
    if isinstance(aoi, Path | str):
        aoi = gpd.read_file(aoi)
    s2items: dict[str, Item] = {}
    if simplify_geometry:
        aoi = aoi.copy()
        aoi["geometry"] = aoi.geometry.simplify(simplify_geometry)
    for i, row in aoi.iterrows():
        s2items.update(
            search_s2_stac(
                intersects=row.geometry,
                start_date=start_date,
                end_date=end_date,
                max_cloud_cover=max_cloud_cover,
            )
        )
    return s2items


@stopwatch("Searching for Sentinel-2 tiles via STAC from AOI", printer=logger.debug)
def match_s2ids_from_geodataframe_stac(
    aoi: gpd.GeoDataFrame,
    day_range: int = 60,
    max_cloud_cover: int = 20,
    min_intersects: float = 0.7,
    simplify_geometry: float | Literal[False] = False,
    save_scores: Path | None = None,
) -> dict[int, Item]:
    """Match items from a GeoDataFrame with Sentinel-2 items from the STAC API based on a date range.

    Args:
        aoi (gpd.GeoDataFrame): The area of interest as a GeoDataFrame.
        day_range (int): The number of days before and after the date to search for.
            Defaults to 60.
        max_cloud_cover (int, optional): The maximum cloud cover percentage. Defaults to 20.
        min_intersects (float, optional): The minimum intersection area ratio to consider a match. Defaults to 0.7.
        simplify_geometry (float | Literal[False], optional): If a float is provided, the geometry will be simplified
            using the `simplify` method of geopandas. If False, no simplification will be done.
            This may become useful for large / weird AOIs which are too large for the STAC API.
            Defaults to False.
        save_scores (Path | None, optional): If provided, the scores will be saved to this path as a Parquet file.

    Raises:
        ValueError: If the 'date' column is not present or not of type datetime.

    Returns:
        dict[int, Item]: A dictionary mapping each row to its best matching Sentinel-2 item.

    """
    # Check weather the "date" column is present and of type datetime
    if "date" not in aoi.columns or not pd.api.types.is_datetime64_any_dtype(aoi["date"]):
        raise ValueError("The 'date' column must be present and of type datetime in the GeoDataFrame.")

    if simplify_geometry:
        aoi = aoi.copy()
        aoi["geometry"] = aoi.geometry.simplify(simplify_geometry)

    matches = {}
    scores = []
    for i, row in aoi.iterrows():
        intersects = row.geometry.__geo_interface__
        start_date = (row["date"] - pd.Timedelta(days=day_range)).strftime("%Y-%m-%d")
        end_date = (row["date"] + pd.Timedelta(days=day_range)).strftime("%Y-%m-%d")
        intersecting_items = search_s2_stac(
            intersects=intersects,
            start_date=start_date,
            end_date=end_date,
            max_cloud_cover=max_cloud_cover,
        )
        if not intersecting_items:
            logger.info(f"No Sentinel-2 items found for {i} in the date range {start_date} to {end_date}.")
            matches[i] = None
            continue
        intersecting_items_gdf = gpd.GeoDataFrame.from_features(
            [item.to_dict() for item in intersecting_items.values()],
            crs="EPSG:4326",
        )
        intersecting_items_gdf["footprint_index"] = i
        intersecting_items_gdf["s2id"] = list(intersecting_items.keys())
        # Some item geometries might be invalid (probably because of the arctic circle)
        # We will drop those items, since they cannot be used for intersection calculations
        intersecting_items_gdf = intersecting_items_gdf[intersecting_items_gdf.geometry.is_valid]
        if intersecting_items_gdf.empty:
            logger.info(f"No valid Sentinel-2 items found for {i} in the date range {start_date} to {end_date}.")
            matches[i] = None
            continue
        # Get to UTM zone for better area calculations
        utm_zone = intersecting_items_gdf.estimate_utm_crs()
        # Calculate intersection area ratio
        intersecting_items_gdf["intersection_area"] = (
            intersecting_items_gdf.intersection(row.geometry).to_crs(utm_zone).area
        )
        # We need a geodataframe containing only our wanted row, since to_crs is not available for a single row
        intersecting_items_gdf["aoi_area"] = aoi.loc[[i]].to_crs(utm_zone).iloc[0].geometry.area
        intersecting_items_gdf["intersection_ratio"] = (
            intersecting_items_gdf["intersection_area"] / intersecting_items_gdf["aoi_area"]
        )
        # Filter items based on the minimum intersection ratio
        max_intersection = intersecting_items_gdf["intersection_ratio"].max()
        intersecting_items_gdf = intersecting_items_gdf[intersecting_items_gdf["intersection_ratio"] >= min_intersects]
        if intersecting_items_gdf.empty:
            logger.info(
                f"No Sentinel-2 items found for {i} with sufficient intersection ratio "
                f"({min_intersects}, maximum was {max_intersection:.4f})"
                f" in the date range {start_date} to {end_date}."
            )
            matches[i] = None
            continue
        intersecting_items_gdf["datetime"] = pd.to_datetime(intersecting_items_gdf["datetime"])
        intersecting_items_gdf["time_diff"] = abs(intersecting_items_gdf["datetime"] - row["date"])
        intersecting_items_gdf["score_cloud"] = ((100.0 - intersecting_items_gdf["eo:cloud_cover"]) / 5) ** 2
        intersecting_items_gdf["score_fill"] = ((100.0 - intersecting_items_gdf["intersection_ratio"] * 100) / 5) ** 2
        intersecting_items_gdf["score_time_diff"] = (
            intersecting_items_gdf["time_diff"].dt.total_seconds() / (2 * 24 * 3600)
        ) ** 2

        intersecting_items_gdf["score"] = (
            intersecting_items_gdf["score_cloud"]
            + intersecting_items_gdf["score_fill"]
            + intersecting_items_gdf["score_time_diff"]
        )

        # Debug the scoring
        score_msg = f"Scores for {i}:\n"
        for j, match in intersecting_items_gdf.iterrows():
            score_msg += (
                f"\n- Match with {j}: "
                f"Cloud Cover={match['eo:cloud_cover']}, "
                f"Intersection Ratio={match['intersection_ratio']:.2f}, "
                f"Time Diff={match['time_diff']}, "
                f"Score Cloud={match['score_cloud']:.2f}, "
                f"Score Fill={match['score_fill']:.2f}, "
                f"Score Time Diff={match['score_time_diff']:.2f}, "
                f"-> Score={match['score']:.2f}"
            )
        logger.debug(score_msg)

        # Get the s2id with the lowest score
        best_item = intersecting_items_gdf.loc[intersecting_items_gdf["score"].idxmin()]
        matches[i] = intersecting_items[best_item["s2id"]]
        scores.append(intersecting_items_gdf)

    if save_scores:
        scores_df = gpd.GeoDataFrame(pd.concat(scores))
        scores_df.to_parquet(save_scores)

    return matches
