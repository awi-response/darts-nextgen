"""Sentinel-2 related data loading. Should be used temporary and maybe moved to the acquisition package."""

import logging
import random
import time
from collections.abc import MutableMapping
from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
import odc.geo.xr
import pandas as pd
import rioxarray
import xarray as xr
from darts_utils.cuda import DEFAULT_DEVICE, move_to_device, move_to_host
from odc.stac import stac_load
from pystac import Item
from pystac_client import Client, ItemSearch
from pystac_client.exceptions import APIError
from stopuhr import stopwatch
from zarr.codecs import BloscCodec

from darts_acquisition.exceptions import DartsAcquisitionError
from darts_acquisition.s2.debug_export import save_debug_geotiff
from darts_acquisition.s2.quality_mask import create_quality_mask_from_observations
from darts_acquisition.s2.raw_data_store import StoreManager
from darts_acquisition.utils.copernicus import init_copernicus

logger = logging.getLogger(__name__.replace("darts_", "darts."))

CDSE_MAX_RETRIES = 30


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
            "B02": "blue",
            "B03": "green",
            "B04": "red",
            "B08": "nir",
        }

    if "observations" not in bands_mapping.keys():
        bands_mapping["observations"] = "s2_observations"
    return bands_mapping


def _cdse_query_controlled(search: ItemSearch) -> list[Item]:
    """Query the CDSE STAC catalogue reacting to rate limitations.

    If the CDSE server returns a "rate limit exceeded" error, wait
    a random amount of seconds untile querying again. With each round,
    we may wait a longer time.

    Args:
        search (ItemSearch): The PySTAC-Client search object

    Returns:
        list[Item]: the resulting items.

    Raises:
        APIError: if a non-429 status code error occurs during the STAC query.
        DartsAcquisitionError: if the maximum queries are exhausted.

    """
    query_ctr = 0
    random.seed("".join([str(p) for p in search.get_parameters().values()]) + str(time.time()))
    while query_ctr < CDSE_MAX_RETRIES:
        try:
            return list(search.items())
        except APIError as e:
            if hasattr(e, "status_code") and (e.status_code == 429):  # rate limit exceeded
                query_ctr += 1
                sleep_time = random.randint(query_ctr, 10 + query_ctr * 5)

                logger.info(f"CDSE query rate limit exceeded, retrying after {sleep_time} seconds.")
                time.sleep(sleep_time)
            else:
                raise

    raise DartsAcquisitionError(f"CDSE request failed after {query_ctr} tries")


class CDSEMosaicStoreManager(StoreManager[Item]):
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
        return f"cdse-s2-mosaic-{s2id}"

    def encodings(self, bands: list[str]) -> dict[str, dict[str, str]]:  # noqa: D102
        encodings = {
            band: {
                "dtype": "uint16",
                "compressors": BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle"),
                "chunks": (4096, 4096),
                "shards": (4096 * 4, 4096 * 4),
            }
            for band in bands
        }
        for band in set(bands) - {"observations"}:
            encodings[band]["_FillValue"] = 0
        encodings["observations"]["dtype"] = "uint16"
        return encodings

    def download_scene_from_source(self, s2item: str | Item, bands: list[str]) -> xr.Dataset:
        """Download a Sentinel-2 mosaic tile (the "scene") from CDSE via STAC API.

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
                collections=["sentinel-2-global-mosaics"],
                ids=[s2id],
            )
            s2item = next(search.items())

        with stopwatch("Downloading data from CDSE", printer=logger.debug):
            # We can't use xpystac here, because they enforce chunking of 1024x1024, which results in long loading times
            # and a potential AWS limit error.
            init_copernicus(profile_name=self.aws_profile_name)

            # to download the data without reprojection even if the CDSE STAC metadata
            # is incomplete, we have to explicitly tell odc what CRS the data is in.
            # so we make lot of assertions to make sure we pass the correct data, because
            # otherwise odc may silently reproject
            assert s2item.properties["proj:code"][-2:] == s2item.id[-9:-7]  # e.g. 32610 == 10WGS
            assert s2item.properties["gsd"] == 10  # 10m ground resolution

            logger.debug(f"start download of '{s2item.id}'")
            ds_s2 = stac_load(
                [s2item],
                bands=bands,
                crs=s2item.properties["proj:code"],  # explicit in case of missing STAC metadata
                resolution=s2item.properties["gsd"],
            )
            assert (  # check if downloaded dataset is really in advertised EPSG code
                str(ds_s2.odc.crs.to_epsg()) == s2item.properties["proj:code"][5:]  # e.g. 36210
            )

        ds_s2.attrs = _flatten_dict(s2item.properties)
        # Convert boolean values to int, since they are not supported in netcdf
        # Also convert array, dicts and np types to str
        for key, value in ds_s2.attrs.items():
            if isinstance(value, bool):
                ds_s2.attrs[key] = int(value)
            elif isinstance(value, (list, dict, np.ndarray)):
                ds_s2.attrs[key] = str(value)
        ds_s2.attrs["time"] = str(ds_s2.time.values[0])
        ds_s2 = ds_s2.isel(time=0).drop_vars("time")

        # Because of the resampling, the observations is a float -> fix it
        if "observations" in ds_s2.data_vars:
            ds_s2["observations"] = ds_s2["observations"].astype("uint16")

        return ds_s2


@stopwatch.f("Downloading Sentinel-2 mosaic from CDSE if missing", printer=logger.debug, print_kwargs=["s2item"])
def download_cdse_s2_mosaic(
    s2item: str | Item,
    store: Path,
    bands_mapping: dict | Literal["all"] = "all",
    aws_profile_name: str = "default",
):
    """Download a Sentinel-2 mosaic from CDSE via STAC API and store it in the local data store.

    This function downloads Sentinel-2 mosaics from the Copernicus
    Data Space Ecosystem (CDSE) and stores it locally in a compressed zarr store for efficient
    repeated access.

    Args:
        s2item (str | Item): Sentinel-2 mosaic identifier (e.g., "Sentinel-2_mosaic_2025_Q3_60WWS_0_0...") or
            a PySTAC Item object from a STAC search.
        store (Path): Path to the local zarr store directory where the mosaic will be saved.
        bands_mapping (dict | Literal["all"], optional): Mapping of Sentinel-2 band names to
            custom band names. Keys should be CDSE band names (e.g., "B02", "B03", "B04", "B08"),
            values are the desired output names. Use "all" to load all optical bands and SCL.
            Defaults to "all".
        aws_profile_name (str, optional): AWS profile name for authentication with the
            Copernicus S3 bucket. Defaults to "default".

    Note:
        - Requires Copernicus Data Space authentication. Use `darts_utils.copernicus.init_copernicus()`
          to set up credentials before calling this function.
        - All bands are resampled to 10m resolution during download.
        - Data is stored with zstd compression for efficient storage.
        - The "observations" band is automatically included if not specified.

    Example:
        Download Sentinel-2 mosaic for a project:

        ```python
        from pathlib import Path
        from darts_acquisition import download_cdse_s2_mosaic
        from darts_utils.copernicus import init_copernicus

        # Setup authentication
        init_copernicus(profile_name="default")

        # Download scene with all bands
        download_cdse_s2_mosaic(
            s2item="Sentinel-2_mosaic_2025_Q3_60WWS_0_0",
            store=Path("/data/s2_store"),
            bands_mapping="all",
            aws_profile_name="default"
        )
        ```

    """
    bands_mapping = _get_band_mapping(bands_mapping)
    store_manager = CDSEMosaicStoreManager(
        store=store,
        bands_mapping=bands_mapping,
        aws_profile_name=aws_profile_name,
    )

    store_manager.download_and_store(s2item)


@stopwatch.f("Loading Sentinel-2 mosaic from CDSE", printer=logger.debug, print_kwargs=["s2item"])
def load_cdse_s2_mosaic(
    s2item: str | Item,
    bands_mapping: dict | Literal["all"] = "all",
    store: Path | None = None,
    aws_profile_name: str = "default",
    offline: bool = False,
    output_dir_for_debug_geotiff: Path | None = None,
    device: Literal["cuda", "cpu"] | int = DEFAULT_DEVICE,
) -> xr.Dataset:
    """Load a Sentinel-2 mosaic from CDSE, downloading from STAC API if necessary.

    This function loads Sentinel-2 mosaic data from the Copernicus
    Data Space Ecosystem (CDSE). If a local store is provided, the data is cached for
    efficient repeated access. The function handles quality masking, reflectance scaling,
    and optional GPU acceleration.

    The download logic is basically as follows:

    ```
    IF flag:raw-data-store THEN
        IF exist_local THEN
            open -> memory
        ELIF online THEN
            download -> memory
            save
        ELIF offline THEN
            RAISE ERROR
        ENDIF
    ELIF online THEN
        download -> memory
    ELIF offline THEN
        RAISE ERROR
    ENDIF
    ```

    Args:
        s2item (str | Item): Sentinel-2 mosaic identifier or PySTAC Item object.
        bands_mapping (dict | Literal["all"], optional): Mapping of Sentinel-2 band names to
            custom band names. Keys should be CDSE band names (e.g., "B02"), values are
            output names. Use "all" to load all optical bands and the observations band.
            Defaults to "all".
        store (Path | None, optional): Path to local zarr store for caching. If None, data is
            loaded directly without caching. Defaults to None.
        aws_profile_name (str, optional): AWS profile name for Copernicus S3 authentication.
            Defaults to "default".
        offline (bool, optional): If True, only loads from local store without downloading.
            Requires `store` to be provided. If False, missing data is downloaded.
            Defaults to False.
        output_dir_for_debug_geotiff (Path | None, optional): If provided, writes raw data as
            GeoTIFF files for debugging. Defaults to None.
        device (Literal["cuda", "cpu"] | int, optional): Device for processing (GPU or CPU).
            Defaults to DEFAULT_DEVICE.

    Returns:
        xr.Dataset: Sentinel-2 dataset with the following data variables based on bands_mapping:
            - Optical bands (float32): Surface reflectance values [~-0.1 to ~1.0]
              Default bands: blue, green, red, nir
              Each has attributes:
              - long_name: "Sentinel-2 {Band}"
              - units: "Reflectance"
              - data_source: "Sentinel-2 Global Mosaics via Copernicus STAC API (sentinel-2-global-mosaics)"
            - observations (uint16): Layer with the number of observations per pixel
              Attributes: long_name
            - quality_data_mask (uint8): Derived quality mask
              - 0 = Invalid (observations == 0)
              - 1 = Low quality (1 <= observations <= 3)
              - 2 = High quality (observations > 3)
            - valid_data_mask (uint8): Binary validity mask (1=valid, 0=invalid)

            Dataset attributes:
            - azimuth (float): Solar azimuth angle from view:azimuth
            - elevation (float): Solar elevation angle from view:sun_elevation
            - s2_tile_id (str): Mosaic identifier
            - tile_id (str): Mosaic identifier (same as s2_tile_id)
            - Plus additional STAC metadata fields

    Note:
        The `offline` parameter controls data fetching:
        - When `offline=False`: Automatically downloads missing data from CDSE and stores it
          in the local zarr store (if store is provided).
        - When `offline=True`: Only reads from the local store. Raises an error if data is
          missing or if store is None.

        Reflectance processing:
        - Raw DN values are scaled: (DN / 10000.0) (see https://documentation.dataspace.copernicus.eu/Data/SentinelMissions/Sentinel2.html#sentinel-2-level-3-quarterly-mosaics)
        - Pixels where observations == 0 are masked as NaN
        - This matches the data format from GEE and Planet loaders

        Quality mask derivation from observations:
        - Invalid (0): observations == 0
        - Low quality (1): 1 <= observations <= 3
        - High quality (2): observations > 3

    Example:
        Load mosaic with local caching:

        ```python
        from pathlib import Path
        from darts_acquisition import load_cdse_s2_mosaic
        from darts_utils.copernicus import init_copernicus

        # Setup authentication
        init_copernicus(profile_name="default")

        # Load with caching
        s2_ds = load_cdse_s2_mosaic(
            s2item="Sentinel-2_mosaic_2025_Q3_60WWS_0_0",
            bands_mapping="all",
            store=Path("/data/s2_store"),
            offline=False  # Download if not cached
        )

        # Compute NDVI
        ndvi = (s2_ds.nir - s2_ds.red) / (s2_ds.nir + s2_ds.red)

        # Filter to high quality pixels
        s2_filtered = s2_ds.where(s2_ds.quality_data_mask == 2)
        ```

    """
    s2id = s2item.id if isinstance(s2item, Item) else s2item

    bands_mapping = _get_band_mapping(bands_mapping)
    store_manager = CDSEMosaicStoreManager(
        store=store,
        bands_mapping=bands_mapping,
        aws_profile_name=aws_profile_name,
    )

    with stopwatch("Load Sentinel-2 mosaic from store", printer=logger.debug):
        if not offline:
            ds_s2 = store_manager.load(s2item)
        else:
            assert store is not None, "Store must be provided in offline mode!"
            ds_s2 = store_manager.open(s2item)

    if output_dir_for_debug_geotiff is not None:
        save_debug_geotiff(
            dataset=ds_s2,
            output_path=output_dir_for_debug_geotiff,
            optical_bands=[band for band in bands_mapping.keys() if band.startswith("B")],
            mask_bands=["observations"] if "observations" in bands_mapping.keys() else None,
        )

    # ? The following part takes ~2.5s on CPU and ~0.1 on GPU,
    # however moving to GPU and back takes ~2.2s
    ds_s2 = move_to_device(ds_s2, device)
    ds_s2 = ds_s2.rename_vars(bands_mapping)
    optical_bands = [band for name, band in bands_mapping.items() if name.startswith("B")]
    for band in optical_bands:
        # Set values where SCL_20m == 0 to NaN in all other bands
        # This way the data is similar to data from gee or planet data
        # We need to filter out 0 values, since they are not valid reflectance values
        # But also not reflected in the SCL for some reason
        ds_s2[band] = (
            ds_s2[band].where(ds_s2["s2_observations"] != 0).where(ds_s2[band].astype("float32") != 0) / 10000.0
        )
        ds_s2[band].attrs["long_name"] = f"Sentinel-2 {band.capitalize()}"
        ds_s2[band].attrs["units"] = "Reflectance"

    for band in ds_s2.data_vars:
        ds_s2[band].attrs["data_source"] = (
            "Sentinel-2 Global Mosaics via Copernicus STAC API (sentinel-2-global-mosaics)"
        )

    # ? This takes approx. 1.5s on CPU
    # For some reason, this takes ~1.2s on the GPU
    ds_s2 = create_quality_mask_from_observations(ds_s2)

    ds_s2 = move_to_host(ds_s2)

    # ? The mosaics do not have a sun azimuth/elevation attribute

    ds_s2.attrs["s2_tile_id"] = s2id
    ds_s2.attrs["tile_id"] = s2id

    return ds_s2


def _build_cql2_filter(
    tiles: list[str] | None = None,
) -> dict:
    # Disable max xxx cover if set to 100

    if tiles is None:
        return None

    filter = {"op": "and", "args": [{"op": "in", "args": [{"property": "grid:code"}, tiles]}]}
    filter["op"] = "and"
    filter["args"] = []

    tiles = [f"MGRS-{tile.lstrip('T')}" for tile in tiles]
    filter["args"].append({"op": "in", "args": [{"property": "grid:code"}, tiles]})
    return filter


@stopwatch("Searching for Sentinel-2 mosaics in CDSE", printer=logger.debug)
def search_cdse_s2_mosaic(
    intersects=None,
    tiles: list[str] | None = None,
    quarters: list[int] | None = None,
    years: list[int] | None = None,
) -> dict[str, Item]:
    """Search for Sentinel-2 mosaics via STAC based on an area of interest (intersects) and date range.

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
        quarters (list[int] | None, optional): List of quarters (1-4) to filter the search.
            Defaults to None.
        years (list[int] | None, optional): List of years to filter the search.
            Defaults to None.

    Returns:
        dict[str, Item]: A dictionary of found Sentinel-2 items as values and the s2id as keys.

    """
    catalog = Client.open("https://stac.dataspace.copernicus.eu/v1/")

    if tiles is not None and intersects is not None:
        logger.warning("Both tile and intersects provided. Ignoring intersects parameter.")
        intersects = None

    cql2_filter = _build_cql2_filter(tiles)

    if quarters is not None or years is not None:
        if quarters is None:
            quarters = list(range(1, 5))
        if years is None:
            years = list(range(2017, 2026))
        found_items = set()
        for year in years:
            for quarter in quarters:
                month = (quarter - 1) * 3 + 1
                search = catalog.search(
                    collections=["sentinel-2-global-mosaics"],
                    intersects=intersects,
                    datetime=f"{year}-{month:02d}",
                    filter=cql2_filter,
                )

                found_items.update(_cdse_query_controlled(search))

    else:
        search = catalog.search(
            collections=["sentinel-2-global-mosaics"],
            intersects=intersects,
            filter=cql2_filter,
        )
        found_items = list(search.items())

    if len(found_items) == 0:
        logger.debug(
            f"No Sentinel-2 items found for the given parameters: {intersects=}, {tiles=}, {quarters=}, {years=}"
        )
        return {}
    logger.debug(f"Found {len(found_items)} Sentinel-2 items in CDSE.")
    return {item.id: item for item in found_items}


@stopwatch("Searching for Sentinel-2 mosaics in CDSE from Tile-IDs", printer=logger.debug)
def get_cdse_s2_mosaic_ids_from_tile_ids(
    tile_ids: list[str],
    quarters: list[int] | None = None,
    years: list[int] | None = None,
) -> dict[str, Item]:
    """Search for Sentinel-2 scenes via STAC based on a list of tile IDs.

    Args:
        tile_ids (list[str]): List of MGRS tile IDs to search for.
        quarters (list[int] | None, optional): List of quarters (1-4) to filter the search.
            Defaults to None.
        years (list[int] | None, optional): List of years to filter the search.
            Defaults to None.

    Returns:
        dict[str, Item]: A dictionary of found Sentinel-2 items.

    """
    return search_cdse_s2_mosaic(
        tiles=tile_ids,
        quarters=quarters,
        years=years,
    )


@stopwatch("Searching for Sentinel-2 mosaics in CDSE from AOI", printer=logger.debug)
def get_cdse_s2_mosaic_ids_from_geodataframe(
    aoi: gpd.GeoDataFrame | Path | str,
    quarters: list[int] | None = None,
    years: list[int] | None = None,
    simplify_geometry: float | Literal[False] = False,
) -> dict[str, Item]:
    """Search for Sentinel-2 mosaics via STAC based on an area of interest (aoi).

    Args:
        aoi (gpd.GeoDataFrame | Path | str): AOI as a GeoDataFrame or path to a shapefile.
            If a path is provided, it will be read using geopandas.
        quarters (list[int] | None, optional): List of quarters (1-4) to filter the search.
            Defaults to None.
        years (list[int] | None, optional): List of years to filter the search.
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
            search_cdse_s2_mosaic(
                intersects=row.geometry,
                quarters=quarters,
                years=years,
            )
        )
    return s2items


@stopwatch("Getting AOI from Sentinel-2 CDSE mosaic IDs", printer=logger.debug)
def get_aoi_from_cdse_s2_mosaic_ids(
    mosaic_ids: list[str],
) -> gpd.GeoDataFrame:
    """Get the area of interest (AOI) as a GeoDataFrame from a list of Sentinel-2 mosaic IDs.

    Args:
        mosaic_ids (list[str]): List of Sentinel-2 mosaic IDs.

    Returns:
        gpd.GeoDataFrame: The AOI as a GeoDataFrame.

    Raises:
        ValueError: If no Sentinel-2 items are found for the given scene IDs.

    """
    catalog = Client.open("https://stac.dataspace.copernicus.eu/v1/")
    search = catalog.search(
        collections=["sentinel-2-global-mosaics"],
        ids=mosaic_ids,
    )
    items = list(search.items())
    if not items:
        raise ValueError("No Sentinel-2 items found for the given mosaic IDs.")
    gdf = gpd.GeoDataFrame.from_features(
        [item.to_dict() for item in items],
        crs="EPSG:4326",
    )
    return gdf


@stopwatch("Matching Sentinel-2 mosaics in CDSE from AOI", printer=logger.debug)
def match_cdse_s2_mosaic_ids_from_geodataframe(
    aoi: gpd.GeoDataFrame,
    min_intersects: float = 0.7,
    simplify_geometry: float | Literal[False] = False,
    save_scores: Path | None = None,
) -> dict[int, Item | None]:
    """Match items from a GeoDataFrame with Sentinel-2 items from the STAC API based on a date range.

    Args:
        aoi (gpd.GeoDataFrame): The area of interest as a GeoDataFrame.
        min_intersects (float, optional): The minimum intersection area ratio to consider a match. Defaults to 0.7.
        simplify_geometry (float | Literal[False], optional): If a float is provided, the geometry will be simplified
            using the `simplify` method of geopandas. If False, no simplification will be done.
            This may become useful for large / weird AOIs which are too large for the STAC API.
            Defaults to False.
        save_scores (Path | None, optional): If provided, the scores will be saved to this path as a Parquet file.

    Returns:
        dict[int, Item | None]: A dictionary mapping each row to its best matching Sentinel-2 item.
            The keys are the indices of the rows in the GeoDataFrame, and the values are the matching Sentinel-2 items.
            If no matching item is found, the value will be None.

    Raises:
        ValueError: If the 'date' column is not present or not of type datetime.

    """
    # Check weather the "date" column is present and of type datetime
    if "date" not in aoi.columns or not pd.api.types.is_datetime64_any_dtype(aoi["date"]):
        raise ValueError("The 'date' column must be present and of type datetime in the GeoDataFrame.")

    if simplify_geometry:
        aoi = aoi.copy()
        aoi["geometry"] = aoi.geometry.simplify(simplify_geometry)

    matches = {}
    scores = []

    last_request_time = time.time()
    for i, row in aoi.iterrows():
        intersects = row.geometry.__geo_interface__
        # Get the year and quarter from the date
        quarter = (row["date"].month - 1) // 3 + 1
        year = row["date"].year

        if time.time() - last_request_time < 3.0:
            time.sleep(3.0 - (time.time() - last_request_time))
        intersecting_items = search_cdse_s2_mosaic(
            intersects=intersects,
            quarters=[quarter],
            years=[year],
        )
        last_request_time = time.time()
        if not intersecting_items:
            logger.info(f"No Sentinel-2 items found for footprint #{i} in the {quarter=} of {year=}.")
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
            logger.info(f"No valid Sentinel-2 items found for footprint #{i} in the {quarter=} of {year=}.")
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
                f" in the {quarter=} of {year=}."
            )
            matches[i] = None
            continue
        intersecting_items_gdf["score"] = ((100.0 - intersecting_items_gdf["intersection_ratio"] * 100) / 5) ** 2

        # Debug the scoring
        score_msg = f"Scores for {i}:\n"
        for j, match in intersecting_items_gdf.iterrows():
            score_msg += (
                f"\n- Match with {j}: "
                f"Intersection Ratio={match['intersection_ratio']:.2f}, "
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
