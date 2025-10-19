"""Sentinel-2 related data loading. Should be used temporary and maybe moved to the acquisition package."""

import logging
from collections.abc import MutableMapping
from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
import odc.geo.xr
import pandas as pd
import xarray as xr
from odc.stac import stac_load
from pystac import Item
from pystac_client import Client
from stopuhr import stopwatch
from zarr.codecs import BloscCodec

from darts_acquisition.s2.quality_mask import convert_masks
from darts_acquisition.s2.raw_data_store import StoreManager
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


def _get_band_mapping(bands_mapping: dict[str, str] | Literal["all"]) -> dict[str, str]:
    if bands_mapping == "all":
        # Mapping according to spyndex band common names:
        # for key, band in spyndex.bands.items():
        #     if not hasattr(band, "sentinel2a"): continue
        #     print(f"{band.sentinel2a.band}: {band.common_name}")
        bands_mapping = {
            "B01_20m": "coastal",
            "B02_10m": "blue",
            "B03_10m": "green",
            "B04_10m": "red",
            "B05_20m": "rededge071",
            "B06_20m": "rededge075",
            "B07_20m": "rededge078",
            "B08_10m": "nir",
            "B8A_20m": "nir08",
            "B09_60m": "nir09",
            "B11_20m": "swir16",
            "B12_20m": "swir22",
        }

    if "SCL_20m" not in bands_mapping.keys():
        bands_mapping["SCL_20m"] = "s2_scl"
    return bands_mapping


class CDSEStoreManager(StoreManager[Item]):
    """Raw Data Store manager for CDSE."""

    def __init__(self, store: Path | str | None, bands_mapping: dict[str, str], aws_profile_name: str):
        """Initialize the store manager.

        Args:
            store (str | Path | None): Directory path for storing raw sentinel 2 data
            bands_mapping (dict[str, str]): A mapping from bands to obtain.
            aws_profile_name (str): AWS profile name for authentication

        """
        bands = list(bands_mapping.keys())
        super().__init__(bands, store)
        self.aws_profile_name = aws_profile_name

    def identifier(self, s2item: str | Item) -> str:  # noqa: D102
        s2id = s2item.id if isinstance(s2item, Item) else s2item
        return f"cdse-s2-sr-scene-{s2id}"

    def encodings(self, bands: list[str]) -> dict[str, dict[str, str]]:  # noqa: D102
        encodings = {
            band: {
                "dtype": "uint16",
                "compressors": BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle"),
                "chunks": (4096, 4096),
            }
            for band in bands
        }
        for band in set(bands) - {"SCL_20m"}:
            encodings[band]["_FillValue"] = 0
        encodings["SCL_20m"]["dtype"] = "uint8"
        return encodings

    def download_scene_from_source(self, s2item: str | Item, bands: list[str]) -> xr.Dataset:
        """Download a Sentinel-2 scene from CDSE via STAC API.

        Args:
            s2item (str | Item): The Sentinel-2 image ID or the corresponing STAC Item.
            bands (list[str]): List of bands to download.

        Returns:
            xr.Dataset: The downloaded scene as xarray Dataset.

        """
        s2id = s2item.id if isinstance(s2item, Item) else s2item

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
            init_copernicus(profile_name=self.aws_profile_name)
            ds_s2 = stac_load(
                [s2item],
                bands=bands,
                crs="utm",
                resolution=10,
                resampling="nearest",  # is used as default, but lets be sure
            )

        ds_s2.attrs = _flatten_dict(s2item.properties)
        # Convert boolean values to int, since they are not supported in netcdf
        # Also convert array, dicts and np types to str
        for key, value in ds_s2.attrs.items():
            if isinstance(value, bool):
                ds_s2.attrs[key] = int(value)
            elif isinstance(value, (list, dict, np.ndarray)):
                ds_s2.attrs[key] = str(value)
        ds_s2.attrs["time"] = str(ds_s2.time.values[0])  # noqa: PD011
        ds_s2 = ds_s2.isel(time=0).drop_vars("time")

        # Because of the resampling to 10m, the SCL is a float -> fix it
        if "SCL_20m" in ds_s2.data_vars:
            ds_s2["SCL_20m"] = ds_s2["SCL_20m"].astype("uint8")

        return ds_s2


@stopwatch.f("Downloading Sentinel-2 scene from CDSE if missing", printer=logger.debug, print_kwargs=["s2item"])
def download_cdse_s2_sr_scene(
    s2item: str | Item,
    store: Path,
    bands_mapping: dict | Literal["all"] = {"B02_10m": "blue", "B03_10m": "green", "B04_10m": "red", "B08_10m": "nir"},
    aws_profile_name: str = "default",
):
    """Download a Sentinel-2 scene from CDSE via STAC API and stores it in the specified raw data store.

    Note:
        Must use the `darts.utils.copernicus.init_copernicus` function to setup authentification
        with the Copernicus AWS S3 bucket before using this function.

    Args:
        s2item (str | Item): The Sentinel-2 image ID or the corresponing STAC Item.
        store (Path): The path to the raw data store.
        bands_mapping (dict[str, str], optional): A mapping from bands to obtain.
            Will be renamed to the corresponding band names.
            Defaults to {"B02_10m": "blue", "B03_10m": "green", "B04_10m": "red", "B08_10m": "nir"}.
        aws_profile_name (str, optional): The name of the AWS profile to use for authentication.
            Defaults to "default".

    """
    bands_mapping = _get_band_mapping(bands_mapping)
    store_manager = CDSEStoreManager(
        store=store,
        bands_mapping=bands_mapping,
        aws_profile_name=aws_profile_name,
    )

    store_manager.download_and_store(s2item)


@stopwatch.f("Loading Sentinel-2 scene from STAC", printer=logger.debug, print_kwargs=["s2item"])
def load_cdse_s2_sr_scene(
    s2item: str | Item,
    bands_mapping: dict | Literal["all"] = {"B02_10m": "blue", "B03_10m": "green", "B04_10m": "red", "B08_10m": "nir"},
    store: Path | None = None,
    aws_profile_name: str = "default",
    offline: bool = False,
    # TODO: debug-data flag
) -> xr.Dataset:
    """Load a Sentinel-2 scene from CDSE via STAC API and return it as an xarray dataset.

    Args:
        s2item (str | Item): The Sentinel-2 image ID or the corresponing STAC Item.
        bands_mapping (dict[str, str], optional): A mapping from bands to obtain.
            Will be renamed to the corresponding band names.
            Defaults to {"B02_10m": "blue", "B03_10m": "green", "B04_10m": "red", "B08_10m": "nir"}.
        store (Path | None, optional): The path to the raw data store. If None, data will not be stored locally.
            Defaults to None.
        aws_profile_name (str, optional): The name of the AWS profile to use for authentication.
            Defaults to "default".
        offline (bool, optional): If True, will not attempt to download any missing data. Defaults to False.

    Returns:
        xr.Dataset: The loaded dataset

    """
    s2id = s2item.id if isinstance(s2item, Item) else s2item

    bands_mapping = _get_band_mapping(bands_mapping)
    store_manager = CDSEStoreManager(
        store=store,
        bands_mapping=bands_mapping,
        aws_profile_name=aws_profile_name,
    )

    if not offline:
        ds_s2 = store_manager.load(s2item)
    else:
        assert store is not None, "Store must be provided in offline mode!"
        ds_s2 = store_manager.open(s2item)

    ds_s2 = ds_s2.rename_vars(bands_mapping)
    optical_bands = [band for name, band in bands_mapping.items() if name.startswith("B")]
    for band in optical_bands:
        # Set values where SCL_20m == 0 to NaN in all other bands
        # This way the data is similar to data from gee or planet data
        # We need to filter out 0 values, since they are not valid reflectance values
        # But also not reflected in the SCL for some reason
        ds_s2[band] = ds_s2[band].where(ds_s2.s2_scl != 0).where(ds_s2[band].astype("float32") != 0) / 10000.0 - 0.1
        ds_s2[band].attrs["long_name"] = f"Sentinel-2 {band.capitalize()}"
        ds_s2[band].attrs["units"] = "Reflectance"
    ds_s2["s2_scl"].attrs = {
        "long_name": "Sentinel-2 Scene Classification Layer",
        "description": (
            "0: NO_DATA - 1: SATURATED_OR_DEFECTIVE - 2: CAST_SHADOWS - 3: CLOUD_SHADOWS - 4: VEGETATION"
            " - 5: NOT_VEGETATED - 6: WATER - 7: UNCLASSIFIED - 8: CLOUD_MEDIUM_PROBABILITY - 9: CLOUD_HIGH_PROBABILITY"
            " - 10: THIN_CIRRUS - 11: SNOW or ICE"
        ),
    }
    for band in ds_s2.data_vars:
        ds_s2[band].attrs["data_source"] = "Sentinel-2 L2A via Copernicus STAC API (sentinel-2-l2a)"

    ds_s2 = convert_masks(ds_s2)

    # Convert sun elevation and azimuth to match our naming
    ds_s2.attrs["azimuth"] = ds_s2.attrs.get("view:azimuth", float("nan"))
    ds_s2.attrs["elevation"] = ds_s2.attrs.get("view:sun_elevation", float("nan"))

    ds_s2.attrs["s2_tile_id"] = s2id
    ds_s2.attrs["tile_id"] = s2id

    return ds_s2


def _build_cql2_filter(
    tiles: list[str] | None = None,
    max_cloud_cover: int | None = 10,
    max_snow_cover: int | None = 10,
) -> dict:
    # Disable max xxx cover if set to 100
    if max_cloud_cover is not None and max_cloud_cover == 100:
        max_cloud_cover = None
    if max_snow_cover is not None and max_snow_cover == 100:
        max_snow_cover = None

    if tiles is None and max_cloud_cover is None and max_snow_cover is None:
        return None

    filter = {}
    filter["op"] = "and"
    filter["args"] = []

    if tiles is not None:
        tiles = [f"MGRS-{tile.lstrip('T')}" for tile in tiles]
        filter["args"].append({"op": "in", "args": [{"property": "grid:code"}, tiles]})
    if max_cloud_cover is not None:
        filter["args"].append({"op": "lte", "args": [{"property": "eo:cloud_cover"}, max_cloud_cover]})
    if max_snow_cover is not None:
        filter["args"].append({"op": "lte", "args": [{"property": "eo:snow_cover"}, max_snow_cover]})
    return filter


@stopwatch("Searching for Sentinel-2 scenes in CDSE", printer=logger.debug)
def search_cdse_s2_sr(
    intersects=None,
    tiles: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    max_cloud_cover: int | None = 10,
    max_snow_cover: int | None = 10,
    months: list[int] | None = None,
    years: list[int] | None = None,
) -> dict[str, Item]:
    """Search for Sentinel-2 scenes via STAC based on an area of interest (intersects) and date range.

    Note:
        `start_date` and `end_date` will be concatted with a `/` to form a date range.
        Read more about the date format here: https://pystac-client.readthedocs.io/en/stable/api.html#pystac_client.Client.search

    Args:
        intersects (any): The geometry object to search for Sentinel-2 tiles.
            Can be anything implementing the `__geo_interface__` protocol, such as a GeoDataFrame or a shapely geometry.
            If None, and tiles is also None, the search will be performed globally.
            If set and tiles is also set, will be ignored.
        tiles (list[str] | None, optional): List of MGRS tile IDs to filter the search.
            If set, ignores intersects parameter.
            Defaults to None.
        start_date (str): Starting date in a format readable by pystac_client.
            If None, months and years parameters will be used for filtering if set.
            Defaults to None.
        end_date (str): Ending date in a format readable by pystac_client.
            If None, months and years parameters will be used for filtering if set.
            Defaults to None.
        max_cloud_cover (int, optional): Maximum percentage of cloud cover. Defaults to 10.
        max_snow_cover (int, optional): Maximum percentage of snow cover. Defaults to 10.
        months (list[int] | None, optional): List of months (1-12) to filter the search.
            Only used if start_date and end_date are None.
            Defaults to None.
        years (list[int] | None, optional): List of years to filter the search.
            Only used if start_date and end_date are None.
            Defaults to None.

    Returns:
        dict[str, Item]: A dictionary of found Sentinel-2 items as values and the s2id as keys.

    """
    catalog = Client.open("https://stac.dataspace.copernicus.eu/v1/")

    if tiles is not None and intersects is not None:
        logger.warning("Both tile and intersects provided. Ignoring intersects parameter.")
        intersects = None

    cql2_filter = _build_cql2_filter(tiles, max_cloud_cover, max_snow_cover)

    if start_date is not None and end_date is not None:
        if months is not None or years is not None:
            logger.warning("Both date range and months/years filtering provided. Ignoring months/years filter.")
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            intersects=intersects,
            datetime=f"{start_date}/{end_date}",
            filter=cql2_filter,
        )
        found_items = list(search.items())
    elif months is not None or years is not None:
        if months is None:
            months = list(range(1, 13))
        if years is None:
            years = list(range(2017, 2026))
        found_items = set()
        for year in years:
            for month in months:
                search = catalog.search(
                    collections=["sentinel-2-l2a"],
                    intersects=intersects,
                    datetime=f"{year}-{month:02d}",
                    filter=cql2_filter,
                )
                found_items.update(list(search.items()))
    else:
        logger.warning("No valid date filtering provided. This may result in a too large number of scenes for CDSE.")
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            intersects=intersects,
            filter=cql2_filter,
        )
        found_items = list(search.items())

    if len(found_items) == 0:
        logger.debug(
            "No Sentinel-2 items found for the given parameters:"
            f" {intersects=}, {start_date=}, {end_date=}, {max_cloud_cover=}"
        )
        return {}
    logger.debug(f"Found {len(found_items)} Sentinel-2 items in CDSE.")
    return {item.id: item for item in found_items}


@stopwatch("Searching for Sentinel-2 scenes in CDSE from Tile-IDs", printer=logger.debug)
def get_cdse_s2_sr_scene_ids_from_tile_ids(
    tile_ids: list[str],
    start_date: str | None = None,
    end_date: str | None = None,
    max_cloud_cover: int | None = 10,
    max_snow_cover: int | None = 10,
    months: list[int] | None = None,
    years: list[int] | None = None,
) -> dict[str, Item]:
    """Search for Sentinel-2 scenes via STAC based on a list of tile IDs.

    Args:
        tile_ids (list[str]): List of MGRS tile IDs to search for.
        start_date (str): Starting date in a format readable by pystac_client.
            If None, months and years parameters will be used for filtering if set.
            Defaults to None.
        end_date (str): Ending date in a format readable by pystac_client.
            If None, months and years parameters will be used for filtering if set.
            Defaults to None.
        max_cloud_cover (int, optional): Maximum percentage of cloud cover. Defaults to 10.
        max_snow_cover (int, optional): Maximum percentage of snow cover. Defaults to 10.
        months (list[int] | None, optional): List of months (1-12) to filter the search.
            Only used if start_date and end_date are None.
            Defaults to None.
        years (list[int] | None, optional): List of years to filter the search.
            Only used if start_date and end_date are None.
            Defaults to None.

    Returns:
        dict[str, Item]: A dictionary of found Sentinel-2 items.

    """
    s2ids = {}
    for tile in tile_ids:
        s2ids.update(
            search_cdse_s2_sr(
                tile=tile,
                start_date=start_date,
                end_date=end_date,
                max_cloud_cover=max_cloud_cover,
                max_snow_cover=max_snow_cover,
                months=months,
                years=years,
            )
        )
    return s2ids


@stopwatch("Searching for Sentinel-2 scenes in CDSE from AOI", printer=logger.debug)
def get_cdse_s2_sr_scene_ids_from_geodataframe(
    aoi: gpd.GeoDataFrame | Path | str,
    start_date: str | None = None,
    end_date: str | None = None,
    max_cloud_cover: int | None = 10,
    max_snow_cover: int | None = 10,
    months: list[int] | None = None,
    years: list[int] | None = None,
    simplify_geometry: float | Literal[False] = False,
) -> dict[str, Item]:
    """Search for Sentinel-2 scenes via STAC based on an area of interest (aoi).

    Args:
        aoi (gpd.GeoDataFrame | Path | str): AOI as a GeoDataFrame or path to a shapefile.
            If a path is provided, it will be read using geopandas.
        start_date (str): Starting date in a format readable by pystac_client.
            If None, months and years parameters will be used for filtering if set.
            Defaults to None.
        end_date (str): Ending date in a format readable by pystac_client.
            If None, months and years parameters will be used for filtering if set.
            Defaults to None.
        max_cloud_cover (int, optional): Maximum percentage of cloud cover. Defaults to 10.
        max_snow_cover (int, optional): Maximum percentage of snow cover. Defaults to 10.
        months (list[int] | None, optional): List of months (1-12) to filter the search.
            Only used if start_date and end_date are None.
            Defaults to None.
        years (list[int] | None, optional): List of years to filter the search.
            Only used if start_date and end_date are None.
            Defaults to None.
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
            search_cdse_s2_sr(
                intersects=row.geometry,
                start_date=start_date,
                end_date=end_date,
                max_cloud_cover=max_cloud_cover,
                max_snow_cover=max_snow_cover,
                months=months,
                years=years,
            )
        )
    return s2items


def get_aoi_from_cdse_scene_ids(
    scene_ids: list[str],
) -> gpd.GeoDataFrame:
    """Get the area of interest (AOI) as a GeoDataFrame from a list of Sentinel-2 scene IDs.

    Args:
        scene_ids (list[str]): List of Sentinel-2 scene IDs.

    Returns:
        gpd.GeoDataFrame: The AOI as a GeoDataFrame.

    Raises:
        ValueError: If no Sentinel-2 items are found for the given scene IDs.

    """
    catalog = Client.open("https://stac.dataspace.copernicus.eu/v1/")
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        ids=scene_ids,
    )
    items = list(search.items())
    if not items:
        raise ValueError("No Sentinel-2 items found for the given scene IDs.")
    gdf = gpd.GeoDataFrame.from_features(
        [item.to_dict() for item in items],
        crs="EPSG:4326",
    )
    return gdf


@stopwatch("Matching Sentinel-2 scenes in CDSE from AOI", printer=logger.debug)
def match_cdse_s2_sr_scene_ids_from_geodataframe(
    aoi: gpd.GeoDataFrame,
    day_range: int = 60,
    max_cloud_cover: int = 20,
    min_intersects: float = 0.7,
    simplify_geometry: float | Literal[False] = False,
    save_scores: Path | None = None,
) -> dict[int, Item | None]:
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
        dict[int, Item | None]: A dictionary mapping each row to its best matching Sentinel-2 item.
            The keys are the indices of the rows in the GeoDataFrame, and the values are the matching Sentinel-2 items.
            If no matching item is found, the value will be None.

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
        intersecting_items = search_cdse_s2_sr(
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
