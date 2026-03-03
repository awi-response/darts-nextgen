"""Landsat related data loading. Should be used temporary and maybe moved to the acquisition package."""

import logging
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
from odc.geo import CRS
from odc.stac import stac_load
from pystac import Item
from pystac_client import Client
from stopuhr import stopwatch
from zarr.codecs import BloscCodec

from darts_acquisition.s2.debug_export import save_debug_geotiff
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
        # Mapping according to STAC and compared against spyndex band common names
        bands_mapping = {
            "B01": "blue",
            "B02": "green",
            "B03": "red",
            "B04": "nir08",
            "B05": "swir16",
            "B06": "swir22",
            "B07": "lwir12",
        }

    if "clear_sky_mask" not in bands_mapping.keys():
        bands_mapping["clear_sky_mask"] = "landsat_clear_sky_mask"
    return bands_mapping


def create_quality_mask_from_clear_sky_mask(ds_landsat: xr.Dataset) -> xr.Dataset:
    """Create a quality mask from the clear_sky_mask band.

    Quality mask derivation from clear_sky_mask:
    - 0 = Invalid (clear_sky_mask == 0)
    - 1 = Low quality (0 < clear_sky_mask < 1)
    - 2 = High quality (clear_sky_mask == 1)

    Args:
        ds_landsat (xr.Dataset): The Landsat dataset containing the clear_sky_mask band.

    Returns:
        xr.Dataset: The modified dataset with the quality_data_mask variable.

    """
    assert "landsat_clear_sky_mask" in ds_landsat.data_vars, (
        "The dataset does not contain the landsat_clear_sky_mask band."
    )

    clear_sky_mask = ds_landsat["landsat_clear_sky_mask"].fillna(0)

    ds_landsat["quality_data_mask"] = (
        (clear_sky_mask > 0).astype("uint8")  # 0 if invalid, 1 otherwise
        + (clear_sky_mask == 1).astype("uint8")  # +1 if high quality (>3)
    ).astype("uint8")

    ds_landsat["quality_data_mask"].attrs["data_source"] = "s2"
    ds_landsat["quality_data_mask"].attrs["long_name"] = "Quality Data Mask"
    ds_landsat["quality_data_mask"].attrs["description"] = "0 = Invalid, 1 = Low Quality, 2 = High Quality"

    return ds_landsat


class LandsatStoreManager(StoreManager[Item]):
    """Raw Data Store manager for Landsat."""

    def __init__(self, store: Path | str | None, bands_mapping: dict[str, str], aws_profile_name: str):
        """Initialize the store manager.

        Args:
            store (str | Path | None): Directory path for storing raw landsat data
            bands_mapping (dict[str, str]): A mapping from bands to obtain.
            aws_profile_name (str): AWS profile name for authentication

        """
        bands = list(bands_mapping.keys())
        super().__init__(bands, store)
        self.aws_profile_name = aws_profile_name

    def identifier(self, landsat_item: str | Item) -> str:  # noqa: D102
        landsat_id = landsat_item.id if isinstance(landsat_item, Item) else landsat_item
        return f"landsat-mosaic-{landsat_id}"

    def encodings(self, bands: list[str]) -> dict[str, dict[str, str]]:  # noqa: D102
        encodings = {
            band: {
                "dtype": "uint8",
                "compressors": BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle"),
                "chunks": (4000, 4000),
            }
            for band in bands
        }
        for band in set(bands) - {"clear_sky_mask"}:
            encodings[band]["_FillValue"] = 0
        encodings["clear_sky_mask"]["dtype"] = "uint8"
        return encodings

    def download_scene_from_source(self, landsat_item: str | Item, bands: list[str]) -> xr.Dataset:
        """Download a Landsat mosaic tile (the "scene") from CDSE via STAC API.

        Args:
            landsat_item (str | Item): The Landsat image ID or the corresponing STAC Item.
            bands (list[str]): List of bands to download.

        Returns:
            xr.Dataset: The downloaded scene as xarray Dataset.

        """
        landsat_id = landsat_item.id if isinstance(landsat_item, Item) else landsat_item

        if isinstance(landsat_item, str):
            catalog = Client.open("https://stac.dataspace.copernicus.eu/v1/")
            search = catalog.search(
                collections=["opengeohub-landsat-bimonthly-mosaic-v1.0.1"],
                ids=[landsat_id],
            )
            landsat_item = next(search.items())

        # Proj fix: the landsat mosaics are not properly georeferenced, so we need to specify the bbox manually
        assert landsat_item.bbox is not None, "Landsat item does not have a bbox, cannot download with stac_load"
        minx, miny, maxx, maxy = landsat_item.bbox

        with stopwatch("Downloading data from CDSE", printer=logger.debug):
            # We can't use xpystac here, because they enforce chunking of 1024x1024, which results in long loading times
            # and a potential AWS limit error.
            init_copernicus(profile_name=self.aws_profile_name)
            ds_landsat = stac_load(
                [landsat_item], bands=bands, bbox=(minx, miny, maxx, maxy), resolution=0.00025, chunks={}
            ).load()

        ds_landsat.attrs = _flatten_dict(landsat_item.properties)
        # Convert boolean values to int, since they are not supported in netcdf
        # Also convert array, dicts and np types to str
        for key, value in ds_landsat.attrs.items():
            if isinstance(value, bool):
                ds_landsat.attrs[key] = int(value)
            elif isinstance(value, (list, dict, np.ndarray)):
                ds_landsat.attrs[key] = str(value)
        ds_landsat.attrs["time"] = str(ds_landsat.time.values[0])
        ds_landsat = ds_landsat.isel(time=0).drop_vars("time")

        # Because of the resampling, the clear_sky_mask is a float -> fix it
        if "clear_sky_mask" in ds_landsat.data_vars:
            ds_landsat["clear_sky_mask"] = ds_landsat["clear_sky_mask"].astype("uint8")

        return ds_landsat


@stopwatch.f("Downloading Landsat mosaic from CDSE if missing", printer=logger.debug, print_kwargs=["landsat_item"])
def download_cdse_landsat_mosaic(
    landsat_item: str | Item,
    store: Path,
    bands_mapping: dict | Literal["all"] = "all",
    aws_profile_name: str = "default",
):
    """Download a Landsat mosaic from CDSE via STAC API and store it in the local data store.

    This function downloads Landsat mosaics from the Copernicus
    Data Space Ecosystem (CDSE) and stores it locally in a compressed zarr store for efficient
    repeated access.

    Args:
        landsat_item (str | Item): Landsat mosaic identifier (e.g., "Landsat_mosaic_2024_11-12_83N068W_V1.0") or
            a PySTAC Item object from a STAC search.
        store (Path): Path to the local zarr store directory where the mosaic will be saved.
        bands_mapping (dict | Literal["all"], optional): Mapping of Landsat band names to
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
        - The "clear_sky_mask" band is automatically included if not specified.

    Example:
        Download Landsat mosaic for a project:

        ```python
        from pathlib import Path
        from darts_acquisition import download_cdse_landsat_mosaic
        from darts_utils.copernicus import init_copernicus

        # Setup authentication
        init_copernicus(profile_name="default")

        # Download scene with all bands
        download_cdse_landsat_mosaic(
            landsat_item="Landsat_mosaic_2025_Q3_60WWS_0_0",
            store=Path("/data/landsat_store"),
            bands_mapping="all",
            aws_profile_name="default"
        )
        ```

    """
    bands_mapping = _get_band_mapping(bands_mapping)
    store_manager = LandsatStoreManager(
        store=store,
        bands_mapping=bands_mapping,
        aws_profile_name=aws_profile_name,
    )

    store_manager.download_and_store(landsat_item)


@stopwatch.f("Loading Landsat mosaic from CDSE", printer=logger.debug, print_kwargs=["landsat_item"])
def load_cdse_landsat_mosaic(
    landsat_item: str | Item,
    bands_mapping: dict | Literal["all"] = "all",
    store: Path | None = None,
    aws_profile_name: str = "default",
    offline: bool = False,
    output_dir_for_debug_geotiff: Path | None = None,
    device: Literal["cuda", "cpu"] | int = DEFAULT_DEVICE,
) -> xr.Dataset:
    """Load a Landsat mosaic from CDSE, downloading from STAC API if necessary.

    This function loads Landsat mosaic data from the Copernicus
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
        landsat_item (str | Item): Landsat mosaic identifier or PySTAC Item object.
        bands_mapping (dict | Literal["all"], optional): Mapping of Landsat band names to
            custom band names. Keys should be CDSE band names (e.g., "B02"), values are
            output names. Use "all" to load all optical bands and the clear_sky_mask band.
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
        xr.Dataset: Landsat dataset with the following data variables based on bands_mapping:
            - Optical bands (uint8): Surface reflectance values [0 to 1 after scaling]
              Default bands: blue, green, red, nir
              Each has attributes:
              - long_name: "Landsat {Band}"
              - units: "Reflectance"
              - data_source:
                "Landsat Global Mosaics via Copernicus STAC API (opengeohub-landsat-bimonthly-mosaic-v1.0.1)"
            - clear_sky_mask (uint8): Layer with the relative number of clear sky observations per pixel (0-1)
              Attributes: long_name
            - quality_data_mask (uint8): Derived quality mask
              - 0 = Invalid (clear_sky_mask == 0)
              - 1 = Low quality (0 < clear_sky_mask < 1)
              - 2 = High quality (clear_sky_mask == 1)
            - valid_data_mask (uint8): Binary validity mask (1=valid, 0=invalid)

            Dataset attributes:
            - landsat_tile_id (str): Mosaic identifier
            - tile_id (str): Mosaic identifier (same as landsat_tile_id)
            - Plus additional STAC metadata fields

    Note:
        The `offline` parameter controls data fetching:
        - When `offline=False`: Automatically downloads missing data from CDSE and stores it
          in the local zarr store (if store is provided).
        - When `offline=True`: Only reads from the local store. Raises an error if data is
          missing or if store is None.

        Reflectance processing:
        - Raw DN values are scaled: (DN / 250) (see https://peerj.com/articles/18585/)
        - Pixels where clear_sky_mask == 0 are masked as NaN
        - This matches the data format from GEE and Planet loaders

        Quality mask derivation from clear_sky_mask:
        - Invalid (0): clear_sky_mask == 0
        - Low quality (1): 0 < clear_sky_mask < 1
        - High quality (2): clear_sky_mask == 1

    Example:
        Load mosaic with local caching:

        ```python
        from pathlib import Path
        from darts_acquisition import load_cdse_s2_mosaic
        from darts_utils.copernicus import init_copernicus

        # Setup authentication
        init_copernicus(profile_name="default")

        # Load with caching
        landsat_ds = load_cdse_s2_mosaic(
            landsat_item="Sentinel-2_mosaic_2025_Q3_60WWS_0_0",
            bands_mapping="all",
            store=Path("/data/s2_store"),
            offline=False  # Download if not cached
        )

        # Compute NDVI
        ndvi = (landsat_ds.nir - landsat_ds.red) / (landsat_ds.nir + landsat_ds.red)

        # Filter to high quality pixels
        s2_filtered = landsat_ds.where(landsat_ds.quality_data_mask == 2)
        ```

    """
    landsat_id = landsat_item.id if isinstance(landsat_item, Item) else landsat_item

    bands_mapping = _get_band_mapping(bands_mapping)
    store_manager = LandsatStoreManager(
        store=store,
        bands_mapping=bands_mapping,
        aws_profile_name=aws_profile_name,
    )

    with stopwatch("Load Landsat mosaic from store", printer=logger.debug):
        if not offline:
            ds_landsat = store_manager.load(landsat_item)
        else:
            assert store is not None, "Store must be provided in offline mode!"
            ds_landsat = store_manager.open(landsat_item)

    if output_dir_for_debug_geotiff is not None:
        save_debug_geotiff(
            dataset=ds_landsat,
            output_path=output_dir_for_debug_geotiff,
            optical_bands=[band for band in bands_mapping.keys() if band.startswith("B")],
            mask_bands=["clear_sky_"] if "observations" in bands_mapping.keys() else None,
        )

    # ? The following part takes ~2.5s on CPU and ~0.1 on GPU,
    # however moving to GPU and back takes ~2.2s
    ds_landsat = move_to_device(ds_landsat, device)
    ds_landsat = ds_landsat.rename_vars(bands_mapping)
    optical_bands = [band for name, band in bands_mapping.items() if name.startswith("B")]
    for band in optical_bands:
        # Set values where SCL_20m == 0 to NaN in all other bands
        # This way the data is similar to data from gee or planet data
        # We need to filter out 0 values, since they are not valid reflectance values
        # But also not reflected in the SCL for some reason
        ds_landsat[band] = (
            ds_landsat[band]
            .where(ds_landsat["landsat_clear_sky_mask"] != 0)
            .where(ds_landsat[band].astype("float32") != 0)
            / 250
        )
        ds_landsat[band].attrs["long_name"] = f"Sentinel-2 {band.capitalize()}"
        ds_landsat[band].attrs["units"] = "Reflectance"

    for band in ds_landsat.data_vars:
        ds_landsat[band].attrs["data_source"] = (
            "Landsat Global Mosaics via Copernicus STAC API (opengeohub-landsat-bimonthly-mosaic-v1.0.1)"
        )

    # ? This takes approx. 1.5s on CPU
    # For some reason, this takes ~1.2s on the GPU
    ds_landsat = create_quality_mask_from_clear_sky_mask(ds_landsat)

    ds_landsat = move_to_host(ds_landsat)

    # ? The mosaics do not have a sun azimuth/elevation attribute

    # Reproject to UTM Zone
    lat, lon = (ds_landsat.latitude.mean().item(), ds_landsat.longitude.mean().item())
    utm_zone = CRS.utm(lon, lat)
    ds_landsat = ds_landsat.odc.reproject(utm_zone)
    ds_landsat = ds_landsat.odc.assign_crs(utm_zone)

    ds_landsat.attrs["landsat_tile_id"] = landsat_id
    ds_landsat.attrs["tile_id"] = landsat_id

    return ds_landsat


def _build_cql2_filter(
    tiles: list[str] | None = None,
) -> dict | None:
    # Disable max xxx cover if set to 100

    if tiles is None:
        return None

    tiles = [f"CDEM-{tile}" for tile in tiles]
    filter = {"op": "and", "args": [{"op": "in", "args": [{"property": "grid:code"}, tiles]}]}
    return filter


Period = Literal["01-02", "03-04", "05-06", "07-08", "09-10", "11-12"]


@stopwatch("Searching for Landsat mosaics in CDSE", printer=logger.debug)
def search_cdse_landsat_mosaic(
    intersects=None,
    tiles: list[str] | None = None,
    periods: list[Period] | None = None,
    years: list[int] | None = None,
) -> dict[str, Item]:
    """Search for Landsat mosaics via STAC based on an area of interest (intersects) and date range.

    Note:
        `start_date` and `end_date` will be concatted with a `/` to form a date range.
        Read more about the date format here: https://pystac-client.readthedocs.io/en/stable/api.html#pystac_client.Client.search

    Args:
        intersects (any): The geometry object to search for Landsat tiles.
            Can be anything implementing the `__geo_interface__` protocol, such as a GeoDataFrame or a shapely geometry.
            If None, and tiles is also None, the search will be performed globally.
            If set and tiles is also set, will be ignored.
        tiles (list[str] | None, optional): List of CDEM tile IDs to filter the search.
            If set, ignores intersects parameter.
            Defaults to None.
        periods (list[Literal["01-02", "03-04", "05-06", "07-08", "09-10", "11-12"]] | None, optional):
            List of periods to filter the search.
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

    all_periods: list[Period] = ["01-02", "03-04", "05-06", "07-08", "09-10", "11-12"]
    if periods is not None or years is not None:
        if periods is None:
            periods = all_periods
        if years is None:
            years = list(range(2017, 2026))
        found_items = set()
        for year in years:
            for period in periods:
                month = {"01-02": 1, "03-04": 3, "05-06": 5, "07-08": 7, "09-10": 9, "11-12": 11}[period]
                search = catalog.search(
                    collections=["opengeohub-landsat-bimonthly-mosaic-v1.0.1"],
                    intersects=intersects,
                    datetime=f"{year}-{month:02d}",
                    filter=cql2_filter,
                )
                found_items.update(list(search.items()))
    else:
        search = catalog.search(
            collections=["opengeohub-landsat-bimonthly-mosaic-v1.0.1"],
            intersects=intersects,
            filter=cql2_filter,
        )
        found_items = list(search.items())

    if len(found_items) == 0:
        logger.debug(f"No Landsat items found for the given parameters: {intersects=}, {tiles=}, {periods=}, {years=}")
        return {}
    logger.debug(f"Found {len(found_items)} Landsat items in CDSE.")
    return {item.id: item for item in found_items}


@stopwatch("Searching for Landsat mosaics in CDSE from Tile-IDs", printer=logger.debug)
def get_cdse_landsat_mosaic_ids_from_tile_ids(
    tile_ids: list[str],
    periods: list[Period] | None = None,
    years: list[int] | None = None,
) -> dict[str, Item]:
    """Search for Landsat scenes via STAC based on a list of tile IDs.

    Args:
        tile_ids (list[str]): List of MGRS tile IDs to search for.
        periods (list[Literal["01-02", "03-04", "05-06", "07-08", "09-10", "11-12"]] | None, optional):
            List of quarters to filter the search. Defaults to None.
        years (list[int] | None, optional): List of years to filter the search.
            Defaults to None.

    Returns:
        dict[str, Item]: A dictionary of found Landsat items.

    """
    return search_cdse_landsat_mosaic(
        tiles=tile_ids,
        periods=periods,
        years=years,
    )


@stopwatch("Searching for Landas mosaics in CDSE from AOI", printer=logger.debug)
def get_cdse_landsat_mosaic_ids_from_geodataframe(
    aoi: gpd.GeoDataFrame | Path | str,
    periods: list[Period] | None = None,
    years: list[int] | None = None,
    simplify_geometry: float | Literal[False] = False,
) -> dict[str, Item]:
    """Search for Landsat mosaics via STAC based on an area of interest (aoi).

    Args:
        aoi (gpd.GeoDataFrame | Path | str): AOI as a GeoDataFrame or path to a shapefile.
            If a path is provided, it will be read using geopandas.
        periods (list[Literal["01-02", "03-04", "05-06", "07-08", "09-10", "11-12"]] | None, optional):
            List of quarters to filter the search. Defaults to None.
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
    landsat_items: dict[str, Item] = {}
    if simplify_geometry:
        aoi = aoi.copy()
        aoi["geometry"] = aoi.geometry.simplify(simplify_geometry)
    for i, row in aoi.iterrows():
        landsat_items.update(
            search_cdse_landsat_mosaic(
                intersects=row.geometry,
                periods=periods,
                years=years,
            )
        )
    return landsat_items


@stopwatch("Getting AOI from CDSE mosaic IDs", printer=logger.debug)
def get_aoi_from_cdse_mosaic_ids(
    mosaic_ids: list[str],
) -> gpd.GeoDataFrame:
    """Get the area of interest (AOI) as a GeoDataFrame from a list of Landsat mosaic IDs.

    Args:
        mosaic_ids (list[str]): List of Landsat mosaic IDs.

    Returns:
        gpd.GeoDataFrame: The AOI as a GeoDataFrame.

    Raises:
        ValueError: If no Landsat items are found for the given scene IDs.

    """
    catalog = Client.open("https://stac.dataspace.copernicus.eu/v1/")
    search = catalog.search(
        collections=["opengeohub-landsat-bimonthly-mosaic-v1.0.1"],
        ids=mosaic_ids,
    )
    items = list(search.items())
    if not items:
        raise ValueError("No Landsat items found for the given mosaic IDs.")
    gdf = gpd.GeoDataFrame.from_features(
        [item.to_dict() for item in items],
        crs="EPSG:4326",
    )
    return gdf


@stopwatch("Matching Landsat mosaics in CDSE from AOI", printer=logger.debug)
def match_cdse_landsat_mosaic_ids_from_geodataframe(
    aoi: gpd.GeoDataFrame,
    min_intersects: float = 0.7,
    simplify_geometry: float | Literal[False] = False,
    save_scores: Path | None = None,
) -> dict[int, Item | None]:
    """Match items from a GeoDataFrame with Landsat items from the STAC API based on a date range.

    Args:
        aoi (gpd.GeoDataFrame): The area of interest as a GeoDataFrame.
        min_intersects (float, optional): The minimum intersection area ratio to consider a match. Defaults to 0.7.
        simplify_geometry (float | Literal[False], optional): If a float is provided, the geometry will be simplified
            using the `simplify` method of geopandas. If False, no simplification will be done.
            This may become useful for large / weird AOIs which are too large for the STAC API.
            Defaults to False.
        save_scores (Path | None, optional): If provided, the scores will be saved to this path as a Parquet file.

    Raises:
        ValueError: If the 'date' column is not present or not of type datetime.

    Returns:
        dict[int, Item | None]: A dictionary mapping each row to its best matching Landsat item.
            The keys are the indices of the rows in the GeoDataFrame, and the values are the matching Landsat items.
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

    last_request_time = time.time()
    for i, row in aoi.iterrows():
        intersects = row.geometry.__geo_interface__
        # Get the year and period from the date
        match row["date"].month:
            case 1 | 2:
                period = "01-02"
            case 3 | 4:
                period = "03-04"
            case 5 | 6:
                period = "05-06"
            case 7 | 8:
                period = "07-08"
            case 9 | 10:
                period = "09-10"
            case 11 | 12:
                period = "11-12"
        year = row["date"].year

        if time.time() - last_request_time < 3.0:
            time.sleep(3.0 - (time.time() - last_request_time))
        intersecting_items = search_cdse_landsat_mosaic(
            intersects=intersects,
            periods=[period],
            years=[year],
        )
        last_request_time = time.time()
        if not intersecting_items:
            logger.info(f"No Landsat items found for footprint #{i} in the {period=} of {year=}.")
            matches[i] = None
            continue
        intersecting_items_gdf = gpd.GeoDataFrame.from_features(
            [item.to_dict() for item in intersecting_items.values()],
            crs="EPSG:4326",
        )
        intersecting_items_gdf["footprint_index"] = i
        intersecting_items_gdf["landsat_id"] = list(intersecting_items.keys())
        # Some item geometries might be invalid (probably because of the arctic circle)
        # We will drop those items, since they cannot be used for intersection calculations
        intersecting_items_gdf = intersecting_items_gdf[intersecting_items_gdf.geometry.is_valid]
        if intersecting_items_gdf.empty:
            logger.info(f"No valid Landsat items found for footprint #{i} in the {period=} of {year=}.")
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
                f"No Landsat items found for {i} with sufficient intersection ratio "
                f"({min_intersects}, maximum was {max_intersection:.4f})"
                f" in the {period=} of {year=}."
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
        matches[i] = intersecting_items[best_item["landsat_id"]]
        scores.append(intersecting_items_gdf)

    if save_scores:
        scores_df = gpd.GeoDataFrame(pd.concat(scores))
        scores_df.to_parquet(save_scores)

    return matches
