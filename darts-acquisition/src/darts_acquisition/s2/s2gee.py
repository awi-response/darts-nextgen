"""Sentinel-2 related data loading. Should be used temporary and maybe moved to the acquisition package."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Literal

import ee
import geopandas as gpd
import odc.geo.xr
import xarray as xr
from darts_utils.tilecache import XarrayCacheManager
from stopuhr import stopwatch

from darts_acquisition.s2.quality_mask import convert_masks

logger = logging.getLogger(__name__.replace("darts_", "darts."))


@stopwatch.f("Loading Sentinel-2 scene from GEE", printer=logger.debug, print_kwargs=["img"])
def load_gee_s2_sr_scene(
    img: str | ee.Image,
    bands_mapping: dict | Literal["all"] = {"B2": "blue", "B3": "green", "B4": "red", "B8": "nir"},
    cache: Path | None = None,
    offline: bool = False,
) -> xr.Dataset:
    """Load a Sentinel-2 scene from Google Earth Engine and return it as an xarray dataset.

    Args:
        img (str | ee.Image): The Sentinel-2 image ID or the ee image object.
        bands_mapping (dict[str, str] | Literal["all"], optional): A mapping from bands to obtain.
            Will be renamed to the corresponding band names.
            If "all" is provided, will load all optical bands and the SCL band.
            Defaults to {"B2": "blue", "B3": "green", "B4": "red", "B8": "nir"}.
        cache (Path | None, optional): The path to the cache directory. If None, no caching will be done.
            Defaults to None.
        offline (bool, optional): If True, will not attempt to download any missing data. Defaults to False.

    Returns:
        xr.Dataset: The loaded dataset

    """
    if isinstance(img, str):
        s2id = img
        img = ee.Image(f"COPERNICUS/S2_SR/{s2id}")
    else:
        s2id = img.id().getInfo().split("/")[-1]
    logger.debug(f"Loading Sentinel-2 tile {s2id=} from GEE")

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

    img = img.select(list(bands_mapping.keys()))

    def _get_tile():
        ds_s2 = xr.open_dataset(
            img,
            engine="ee",
            geometry=img.geometry(),
            crs=img.select(0).projection().crs().getInfo(),
            scale=10,
        )
        props = img.getInfo()["properties"]
        ds_s2.attrs["azimuth"] = props.get("MEAN_SOLAR_AZIMUTH_ANGLE", float("nan"))
        ds_s2.attrs["elevation"] = props.get("MEAN_SOLAR_ZENITH_ANGLE", float("nan"))

        ds_s2.attrs["time"] = str(ds_s2.time.values[0])
        ds_s2 = ds_s2.isel(time=0).drop_vars("time").rename({"X": "x", "Y": "y"}).transpose("y", "x")
        ds_s2 = ds_s2.odc.assign_crs(ds_s2.attrs["crs"])
        with stopwatch(f"Downloading data from GEE for {s2id=}", printer=logger.debug):
            ds_s2.load()
        return ds_s2

    cache_manager = XarrayCacheManager(cache)
    cache_id = f"gee-s2sr-{s2id}-{''.join(bands_mapping.keys())}"
    if not offline:
        ds_s2 = cache_manager.get_or_create(
            identifier=cache_id,
            creation_func=_get_tile,
            force=False,
            use_band_manager=False,
        )
    else:
        assert cache is not None, "Cache must be provided in offline mode!"
        ds_s2 = cache_manager.load_from_cache(identifier=cache_id)

    ds_s2 = ds_s2.rename_vars(bands_mapping)

    optical_bands = [band for name, band in bands_mapping.items() if name.startswith("B")]

    # Fix new preprocessing offset -> See docs about bands
    dt = datetime.strptime(ds_s2.attrs["time"], "%Y-%m-%dT%H:%M:%S.%f000")
    offset = 0.1 if dt >= datetime(2022, 1, 25) else 0.0

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
    print(ds_s2.quality_data_mask.attrs)
    qdm_attrs = ds_s2["quality_data_mask"].attrs.copy()
    print(qdm_attrs)

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

    print(qdm_attrs)
    ds_s2["quality_data_mask"].attrs = qdm_attrs
    print(ds_s2.quality_data_mask.attrs)
    ds_s2.attrs["s2_tile_id"] = img.getInfo()["properties"]["PRODUCT_ID"]
    ds_s2.attrs["tile_id"] = s2id

    return ds_s2


@stopwatch("Searching for Sentinel-2 scenes in Earth Engine from AOI", printer=logger.debug)
def get_gee_s2_sr_scene_ids_from_geodataframe(
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
