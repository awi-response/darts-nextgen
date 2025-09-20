"""Sentinel-2 related data loading. Should be used temporary and maybe moved to the acquisition package."""

import logging
from pathlib import Path

import ee
import geopandas as gpd
import odc.geo.xr
import rioxarray  # noqa: F401
import xarray as xr
from darts_utils.tilecache import XarrayCacheManager
from stopuhr import stopwatch

from darts_acquisition.s2.quality_mask import convert_masks

logger = logging.getLogger(__name__.replace("darts_", "darts."))


@stopwatch.f("Loading Sentinel-2 scene from GEE", printer=logger.debug, print_kwargs=["img"])
def load_s2_from_gee(
    img: str | ee.Image,
    bands_mapping: dict = {"B2": "blue", "B3": "green", "B4": "red", "B8": "nir"},
    cache: Path | None = None,
) -> xr.Dataset:
    """Load a Sentinel-2 scene from Google Earth Engine and return it as an xarray dataset.

    Args:
        img (str | ee.Image): The Sentinel-2 image ID or the ee image object.
        bands_mapping (dict[str, str], optional): A mapping from bands to obtain.
            Will be renamed to the corresponding band names.
            Defaults to {"B2": "blue", "B3": "green", "B4": "red", "B8": "nir"}.
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

    for band in set(bands_mapping.values()) - {"s2_scl"}:
        ds_s2[band].attrs["data_source"] = "s2-gee"
        ds_s2[band].attrs["long_name"] = f"Sentinel 2 {band.capitalize()}"
        ds_s2[band].attrs["units"] = "Reflectance"

    ds_s2 = convert_masks(ds_s2)

    # For some reason, there are some spatially random nan values in the data, not only at the borders
    # To workaround this, set all nan values to 0 and add this information to the quality_data_mask
    # This workaround is quite computational expensive, but it works for now
    # TODO: Find other solutions for this problem!
    with stopwatch(f"Fixing nan values in {s2id=}", printer=logger.debug):
        for band in set(bands_mapping.values()) - {"s2_scl"}:
            ds_s2["quality_data_mask"] = xr.where(ds_s2[band].isnull(), 0, ds_s2["quality_data_mask"])
            ds_s2[band] = ds_s2[band].fillna(0)
            # Turn real nan values (s2_scl is nan) into invalid data
            ds_s2[band] = ds_s2[band].where(~ds_s2["s2_scl"].isnull())

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
