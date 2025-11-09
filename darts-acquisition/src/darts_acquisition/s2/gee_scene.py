"""Sentinel-2 related data loading. Should be used temporary and maybe moved to the acquisition package."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Literal

import ee
import geopandas as gpd
import odc.geo.xr
import rioxarray
import xarray as xr
from darts_utils.cuda import DEFAULT_DEVICE, move_to_device, move_to_host
from stopuhr import stopwatch
from zarr.codecs import BloscCodec

from darts_acquisition.s2.debug_export import save_debug_geotiff
from darts_acquisition.s2.quality_mask import convert_masks
from darts_acquisition.s2.raw_data_store import StoreManager

logger = logging.getLogger(__name__.replace("darts_", "darts."))


def _get_band_mapping(bands_mapping: dict[str, str] | Literal["all"]) -> dict[str, str]:
    if bands_mapping == "all":
        # Mapping according to spyndex band common names:
        # for key, band in spyndex.bands.items():
        #     if not hasattr(band, "sentinel2a"): continue
        #     print(f"{band.sentinel2a.band}: {band.common_name}")
        bands_mapping = {
            "B1": "coastal",
            "B2": "blue",
            "B3": "green",
            "B4": "red",
            "B5": "rededge071",
            "B6": "rededge075",
            "B7": "rededge078",
            "B8": "nir",
            "B8A": "nir08",
            "B9": "nir09",
            "B11": "swir16",
            "B12": "swir22",
        }

    if "SCL" not in bands_mapping.keys():
        bands_mapping["SCL"] = "s2_scl"
    return bands_mapping


class GEEStoreManager(StoreManager[ee.Image]):
    """Raw Data Store manager for GEE."""

    def __init__(self, store: Path | str | None, bands_mapping: dict[str, str]):
        """Initialize the store manager.

        Args:
            store (str | Path | None): Directory path for storing raw sentinel 2 data
            bands_mapping (dict[str, str]): A mapping from bands to obtain.

        """
        bands = list(bands_mapping.keys())
        super().__init__(bands, store)

    def identifier(self, s2item: str | ee.Image) -> str:  # noqa: D102
        s2id = s2item.id().getInfo().split("/")[-1] if isinstance(s2item, ee.Image) else s2item
        return f"gee-s2-sr-scene-{s2id}"

    def encodings(self, bands: list[str]) -> dict[str, dict[str, str]]:  # noqa: D102
        encodings = {
            band: {
                "dtype": "uint16",
                "compressors": BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle"),
                "chunks": (4096, 4096),
            }
            for band in bands
        }
        for band in set(bands) - {"SCL"}:
            encodings[band]["_FillValue"] = 0
        encodings["SCL"]["dtype"] = "uint8"
        return encodings

    def download_scene_from_source(self, s2item: str | ee.Image, bands: list[str]) -> xr.Dataset:
        """Download a Sentinel-2 scene from GEE.

        Args:
            s2item (str | ee.Image): The Sentinel-2 image ID or the corresponding ee.Image.
            bands (list[str]): List of bands to download.

        Returns:
            xr.Dataset: The downloaded scene as xarray Dataset.

        """
        if isinstance(s2item, str):
            s2id = s2item
            s2item = ee.Image(f"COPERNICUS/S2_SR/{s2id}")
        else:
            s2id = s2item.id().getInfo().split("/")[-1]

        s2item = s2item.select(bands)

        ds_s2 = xr.open_dataset(
            s2item,
            engine="ee",
            geometry=s2item.geometry(),
            crs=s2item.select(0).projection().crs().getInfo(),
            scale=10,
        )
        props = s2item.getInfo()["properties"]
        ds_s2.attrs["azimuth"] = props.get("MEAN_SOLAR_AZIMUTH_ANGLE", float("nan"))
        ds_s2.attrs["elevation"] = props.get("MEAN_SOLAR_ZENITH_ANGLE", float("nan"))

        ds_s2.attrs["time"] = str(ds_s2.time.values[0])
        ds_s2 = ds_s2.isel(time=0).drop_vars("time").rename({"X": "x", "Y": "y"}).transpose("y", "x")
        ds_s2 = ds_s2.odc.assign_crs(ds_s2.attrs["crs"])
        with stopwatch("Downloading data from GEE", printer=logger.debug):
            ds_s2.load()
        return ds_s2


@stopwatch.f("Downloading Sentinel-2 scene from GEE if missing", printer=logger.debug, print_kwargs=["s2item"])
def download_gee_s2_sr_scene(
    s2item: str | ee.Image,
    store: Path,
    bands_mapping: dict | Literal["all"] = {"B2": "blue", "B3": "green", "B4": "red", "B8": "nir"},
):
    """Download a Sentinel-2 scene from Google Earth Engine and store it in the local data store.

    This function downloads Sentinel-2 Level-2A surface reflectance data from Google Earth
    Engine (GEE) and stores it locally in a compressed zarr store for efficient repeated access.

    Args:
        s2item (str | ee.Image): Sentinel-2 scene identifier (e.g., "20230615T123456_20230615T123659_T33UUP")
            or an ee.Image object from the COPERNICUS/S2_SR collection.
        store (Path): Path to the local zarr store directory where the scene will be saved.
        bands_mapping (dict | Literal["all"], optional): Mapping of Sentinel-2 band names to
            custom band names. Keys should be GEE band names (e.g., "B2", "B3"), values are
            the desired output names. Use "all" to load all optical bands and SCL.
            Defaults to {"B2": "blue", "B3": "green", "B4": "red", "B8": "nir"}.

    Note:
        - Requires Google Earth Engine authentication. Use `ee.Initialize()` before calling.
        - All bands are downloaded at 10m resolution.
        - Data is stored with zstd compression for efficient storage.
        - The SCL (Scene Classification Layer) band is automatically included if not specified.

    Example:
        Download Sentinel-2 scenes from GEE:

        ```python
        import ee
        from pathlib import Path
        from darts_acquisition import download_gee_s2_sr_scene

        # Initialize Earth Engine
        ee.Initialize()

        # Download scene with all bands
        download_gee_s2_sr_scene(
            s2item="20230615T123456_20230615T123659_T33UUP",
            store=Path("/data/s2_store"),
            bands_mapping="all"
        )
        ```

    """
    bands_mapping = _get_band_mapping(bands_mapping)
    store_manager = GEEStoreManager(
        store=store,
        bands_mapping=bands_mapping,
    )

    store_manager.download_and_store(s2item)


@stopwatch.f("Loading Sentinel-2 scene from GEE", printer=logger.debug, print_kwargs=["s2item"])
def load_gee_s2_sr_scene(
    s2item: str | ee.Image,
    bands_mapping: dict | Literal["all"] = {"B2": "blue", "B3": "green", "B4": "red", "B8": "nir"},
    store: Path | None = None,
    offline: bool = False,
    output_dir_for_debug_geotiff: Path | None = None,
    device: Literal["cuda", "cpu"] | int = DEFAULT_DEVICE,
) -> xr.Dataset:
    """Load a Sentinel-2 scene from Google Earth Engine, downloading if necessary.

    This function loads Sentinel-2 Level-2A surface reflectance data from Google Earth Engine.
    If a local store is provided, the data is cached for efficient repeated access. The function
    handles quality masking, reflectance scaling with time-dependent offsets, and optional GPU
    acceleration. It also handles NaN values in the data by masking them as invalid.

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
        s2item (str | ee.Image): Sentinel-2 scene identifier or ee.Image object from COPERNICUS/S2_SR.
        bands_mapping (dict | Literal["all"], optional): Mapping of Sentinel-2 band names to
            custom band names. Keys should be GEE band names (e.g., "B2", "B3"), values are
            output names. Use "all" to load all optical bands and SCL.
            Defaults to {"B2": "blue", "B3": "green", "B4": "red", "B8": "nir"}.
        store (Path | None, optional): Path to local zarr store for caching. If None, data is
            loaded directly without caching. Defaults to None.
        offline (bool, optional): If True, only loads from local store without downloading.
            Requires `store` to be provided. If False, missing data is downloaded.
            Defaults to False.
        output_dir_for_debug_geotiff (Path | None, optional): If provided, writes raw data as
            GeoTIFF files for debugging. Defaults to None.
        device (Literal["cuda", "cpu"] | int, optional): Device for processing (GPU or CPU).
            Defaults to DEFAULT_DEVICE.

    Returns:
        xr.Dataset: Sentinel-2 dataset with the following data variables based on bands_mapping:
            - Optical bands (float32): Surface reflectance values [~-0.1 to ~1.0 for newer scenes,
              ~0.0 to ~1.0 for scenes before 2022-01-25]
              Default bands: blue, green, red, nir
              Additional bands available: coastal, rededge071, rededge075, rededge078,
              nir08, nir09, swir16, swir22
              Each has attributes:
              - long_name: "Sentinel 2 {Band}"
              - units: "Reflectance"
              - data_source: "Sentinel-2 L2A via Google Earth Engine (COPERNICUS/S2_SR)"
            - s2_scl (uint8): Scene Classification Layer
              Attributes: long_name, description of class values (0=NO_DATA, 1=SATURATED, etc.)
            - quality_data_mask (uint8): Derived quality mask
              - 0 = Invalid (no data, saturated, defective, or NaN values)
              - 1 = Low quality (shadows, clouds, cirrus, snow/ice, water)
              - 2 = High quality (clear vegetation or non-vegetated land)
            - valid_data_mask (uint8): Binary validity mask (1=valid, 0=invalid)

            Dataset attributes:
            - azimuth (float): Solar azimuth angle from MEAN_SOLAR_AZIMUTH_ANGLE
            - elevation (float): Solar elevation angle from MEAN_SOLAR_ZENITH_ANGLE
            - s2_tile_id (str): Full PRODUCT_ID from GEE
            - tile_id (str): Scene identifier
            - time (str): Acquisition timestamp

    Note:
        The `offline` parameter controls data fetching:
        - When `offline=False`: Automatically downloads missing data from GEE and stores it
          in the local zarr store (if store is provided).
        - When `offline=True`: Only reads from the local store. Raises an error if data is
          missing or if store is None.

        Reflectance processing:
        - For scenes >= 2022-01-25: (DN / 10000.0) - 0.1 (processing baseline 04.00+)
        - For scenes < 2022-01-25: DN / 10000.0 (older processing baseline)
        - NaN values are filled with 0 and marked as invalid in quality_data_mask
        - Pixels where SCL is NaN are also masked as invalid

        This function handles spatially random NaN values that can occur in GEE data by
        marking them as invalid and filling with 0 to prevent propagation in calculations.

        Quality mask derivation from SCL:
        - Invalid (0): NO_DATA, SATURATED_OR_DEFECTIVE, or NaN values
        - Low quality (1): CAST_SHADOWS, CLOUD_SHADOWS, CLOUD_*, THIN_CIRRUS, SNOW/ICE, WATER
        - High quality (2): VEGETATION, NOT_VEGETATED

    Example:
        Load scene with local caching:

        ```python
        import ee
        from pathlib import Path
        from darts_acquisition import load_gee_s2_sr_scene

        # Initialize Earth Engine
        ee.Initialize()

        # Load with caching
        s2_ds = load_gee_s2_sr_scene(
            s2item="20230615T123456_20230615T123659_T33UUP",
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
    if isinstance(s2item, str):
        s2id = s2item
        s2item = ee.Image(f"COPERNICUS/S2_SR/{s2id}")
    else:
        s2id = s2item.id().getInfo().split("/")[-1]
    logger.debug(f"Loading Sentinel-2 tile {s2id=} from GEE")

    bands_mapping = _get_band_mapping(bands_mapping)
    store_manager = GEEStoreManager(
        store=store,
        bands_mapping=bands_mapping,
    )

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
            mask_bands=["SCL"],
        )

    ds_s2 = ds_s2.rename_vars(bands_mapping)

    optical_bands = [band for name, band in bands_mapping.items() if name.startswith("B")]

    # Fix new preprocessing offset -> See docs about bands
    dt = datetime.strptime(ds_s2.attrs["time"], "%Y-%m-%dT%H:%M:%S.%f000")
    offset = 0.1 if dt >= datetime(2022, 1, 25) else 0.0

    ds_s2 = move_to_device(ds_s2, device)
    for band in optical_bands:
        # Apply scale and offset
        ds_s2[band] = ds_s2[band].astype("float32") / 10000.0 - offset
        ds_s2[band].attrs["long_name"] = f"Sentinel 2 {band.capitalize()}"
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
        ds_s2[band].attrs["data_source"] = "Sentinel-2 L2A via Google Earth Engine (COPERNICUS/S2_SR)"

    ds_s2 = convert_masks(ds_s2)
    qdm_attrs = ds_s2["quality_data_mask"].attrs.copy()

    # For some reason, there are some spatially random nan values in the data, not only at the borders
    # To workaround this, set all nan values to 0 and add this information to the quality_data_mask
    # This workaround is quite computational expensive, but it works for now
    # TODO: Find other solutions for this problem!
    with stopwatch(f"Fixing nan values in {s2id=}", printer=logger.debug):
        for band in optical_bands:
            ds_s2["quality_data_mask"] = xr.where(ds_s2[band].isnull(), 0, ds_s2["quality_data_mask"])
            ds_s2[band] = ds_s2[band].fillna(0)
            # Turn real nan values (s2_scl is nan) into invalid data
            ds_s2[band] = ds_s2[band].where(~ds_s2["s2_scl"].isnull())
    ds_s2 = move_to_host(ds_s2)

    ds_s2["quality_data_mask"].attrs = qdm_attrs
    ds_s2.attrs["s2_tile_id"] = s2item.getInfo()["properties"]["PRODUCT_ID"]
    ds_s2.attrs["tile_id"] = s2id

    return ds_s2


def get_gee_s2_sr_scene_ids_from_tile_ids(
    tiles: list[str],
    start_date: str | None = None,
    end_date: str | None = None,
    max_cloud_cover: int | None = 10,
    max_snow_cover: int | None = 10,
) -> set[str]:
    """Search for Sentinel-2 scenes via Earth Engine based on a list of tile IDs.

    Args:
        tiles (list[str]): List of Sentinel-2 tile IDs.
        start_date (str): Starting date in a format readable by ee.
            If None, months and years parameters will be used for filtering if set.
            Defaults to None.
        end_date (str): Ending date in a format readable by ee.
            If None, months and years parameters will be used for filtering if set.
            Defaults to None.
        max_cloud_cover (int, optional): Maximum percentage of cloud cover. Defaults to 10.
        max_snow_cover (int, optional): Maximum percentage of snow cover. Defaults to 10.

    Returns:
        set[str]: Unique Sentinel-2 tile IDs.

    """
    # Disable max xxx cover if set to 100
    if max_cloud_cover is not None and max_cloud_cover == 100:
        max_cloud_cover = None
    if max_snow_cover is not None and max_snow_cover == 100:
        max_snow_cover = None

    s2ids = set()
    for tile in tiles:
        if start_date is not None and end_date is not None:
            ic = (
                ee.ImageCollection("COPERNICUS/S2_SR")
                .filterDate(start_date, end_date)
                .filterMetadata("MGRS_TILE", "equals", tile)
            )
            if max_cloud_cover:
                ic = ic.filterMetadata("CLOUDY_PIXEL_PERCENTAGE", "less_than", max_cloud_cover)
            if max_snow_cover:
                ic = ic.filterMetadata("SNOW_SNOW_ICE_PERCENTAGE", "less_than", max_snow_cover)
            s2ids.update(ic.aggregate_array("system:index").getInfo())
        else:
            logger.warning("No valid date filtering provided. This may result in a too large number of scenes for GEE.")
            ic = ee.ImageCollection("COPERNICUS/S2_SR").filterMetadata("MGRS_TILE", "equals", tile)
            if max_cloud_cover:
                ic = ic.filterMetadata("CLOUDY_PIXEL_PERCENTAGE", "less_than", max_cloud_cover)
            if max_snow_cover:
                ic = ic.filterMetadata("SNOW_SNOW_ICE_PERCENTAGE", "less_than", max_snow_cover)
            s2ids.update(ic.aggregate_array("system:index").getInfo())

    logger.debug(f"Found {len(s2ids)} Sentinel-2 tiles via ee.")
    return s2ids


@stopwatch("Searching for Sentinel-2 scenes in Earth Engine from AOI", printer=logger.debug)
def get_gee_s2_sr_scene_ids_from_geodataframe(
    aoi: gpd.GeoDataFrame | Path | str,
    start_date: str | None = None,
    end_date: str | None = None,
    max_cloud_cover: int | None = 10,
    max_snow_cover: int | None = 10,
) -> set[str]:
    """Search for Sentinel-2 scenes via Earth Engine based on an aoi shapefile.

    Args:
        aoi (gpd.GeoDataFrame | Path | str): AOI as a GeoDataFrame or path to a shapefile.
            If a path is provided, it will be read using geopandas.
        start_date (str): Starting date in a format readable by ee.
            If None, months and years parameters will be used for filtering if set.
            Defaults to None.
        end_date (str): Ending date in a format readable by ee.
            If None, months and years parameters will be used for filtering if set.
            Defaults to None.
        max_cloud_cover (int, optional): Maximum percentage of cloud cover. Defaults to 10.
        max_snow_cover (int, optional): Maximum percentage of snow cover. Defaults to 10.

    Returns:
        set[str]: Unique Sentinel-2 tile IDs.

    """
    # Disable max xxx cover if set to 100
    if max_cloud_cover is not None and max_cloud_cover == 100:
        max_cloud_cover = None
    if max_snow_cover is not None and max_snow_cover == 100:
        max_snow_cover = None

    if isinstance(aoi, Path | str):
        aoi = gpd.read_file(aoi)
    aoi = aoi.to_crs("EPSG:4326")
    s2ids = set()
    for i, row in aoi.iterrows():
        geom = ee.Geometry.Polygon(list(row.geometry.exterior.coords))
        if start_date is not None and end_date is not None:
            ic = ee.ImageCollection("COPERNICUS/S2_SR").filterBounds(geom).filterDate(start_date, end_date)
            if max_cloud_cover:
                ic = ic.filterMetadata("CLOUDY_PIXEL_PERCENTAGE", "less_than", max_cloud_cover)
            if max_snow_cover:
                ic = ic.filterMetadata("SNOW_SNOW_ICE_PERCENTAGE", "less_than", max_snow_cover)
            s2ids.update(ic.aggregate_array("system:index").getInfo())
        else:
            logger.warning("No valid date filtering provided. This may result in a too large number of scenes for GEE.")
            ic = ee.ImageCollection("COPERNICUS/S2_SR").filterBounds(geom)
            if max_cloud_cover:
                ic = ic.filterMetadata("CLOUDY_PIXEL_PERCENTAGE", "less_than", max_cloud_cover)
            if max_snow_cover:
                ic = ic.filterMetadata("SNOW_SNOW_ICE_PERCENTAGE", "less_than", max_snow_cover)
            s2ids.update(ic.aggregate_array("system:index").getInfo())

    logger.debug(f"Found {len(s2ids)} Sentinel-2 tiles via ee.")
    return s2ids


def get_aoi_from_gee_scene_ids(
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
    geoms = []
    for s2id in scene_ids:
        s2item = ee.Image(f"COPERNICUS/S2_SR/{s2id}")
        geom = s2item.geometry().getInfo()
        geoms.append(geom)

    if not geoms:
        raise ValueError("No Sentinel-2 items found for the given scene IDs.")

    features = [{"type": "Feature", "geometry": geom, "properties": {}} for geom in geoms]
    feature_collection = {"type": "FeatureCollection", "features": features}
    gdf = gpd.GeoDataFrame.from_features(feature_collection, crs="EPSG:4326")
    return gdf
