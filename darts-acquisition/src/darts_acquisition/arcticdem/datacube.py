"""Downloading and loading related functions for the Zarr-Datacube approach."""

import io
import logging
import time
import zipfile
from pathlib import Path
from typing import Literal

import geopandas as gpd
import odc.geo.xr
import pystac
import rasterio
import requests
import shapely
import xarray as xr
import zarr
import zarr.storage
from lovely_numpy import lovely
from odc.geo.geobox import GeoBox

from darts_acquisition.utils.storage import optimize_coord_encoding

logger = logging.getLogger(__name__.replace("darts_", "darts."))


RESOLUTIONS = Literal[2, 10, 32]
DATA_VARS = ["dem", "count", "mad", "maxdate", "mindate", "datamask"]


def download_arcticdem_extend(dem_data_dir: Path):
    """Download the gdal ArcticDEM extend data from the provided URL and extracts it to the specified directory.

    Args:
        dem_data_dir (Path): The directory where the extracted data will be saved.

    """
    start = time.time()
    url = "https://data.pgc.umn.edu/elev/dem/setsm/ArcticDEM/indexes/ArcticDEM_Mosaic_Index_latest_gpqt.zip"
    logger.info(f"Downloading the gdal arcticdem extend from {url} to {dem_data_dir.resolve()}")
    response = requests.get(url)

    # Get the downloaded data as a byte string
    data = response.content

    logger.debug(f"Downloaded {len(data)} bytes")

    # Create a bytesIO object
    with io.BytesIO(data) as buffer:
        # Create a zipfile.ZipFile object and extract the files to a directory
        dem_data_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(buffer, "r") as zip_ref:
            # Get the name of the zipfile (the parent directory)
            zip_name = zip_ref.namelist()[0].split("/")[0]

            # Extract the files to the specified directory
            zip_ref.extractall(dem_data_dir)

    # Move the extracted files to the parent directory
    extracted_dir = dem_data_dir / zip_name
    for file in extracted_dir.iterdir():
        file.rename(dem_data_dir / file.name)

    # Remove the empty directory
    extracted_dir.rmdir()

    logger.info(f"Download completed in {time.time() - start:.2f} seconds")


def download_arcticdem_stac(stac_url: str) -> xr.Dataset:
    """Download ArcticDEM data from the provided STAC URL.

    This function utilizes pystac, xpystac and odc-stac to create a lazy dataset containing all assets.

    Assets should include:
        - dem
        - count
        - mad
        - maxdate
        - mindate
        - datamask

    Args:
        stac_url (str): The URL of the ArcticDEM STAC. Must be one of:
            - A stac-browser url, like it is provided in the mosaic-extend dataframe, e.g. https://polargeospatialcenter.github.io/stac-browser/#/external/pgc-opendata-dems.s3.us-west-2.amazonaws.com/arcticdem/mosaics/v4.1/32m/36_24/36_24_32m_v4.1.json
            - A direct link to the STAC file, e.g. https://pgc-opendata-dems.s3.us-west-2.amazonaws.com/arcticdem/mosaics/v4.1/32m/36_24/36_24_32m_v4.1.json

    Returns:
        xr.Dataset: The ArcticDEM data as an lazy xarray Dataset.
            Data downloaded on demand, NOT from within this function.

    """
    start = time.time()
    logger.info(f"Loading ArcticDEM data from {stac_url}")

    # Check weather the browser url is provided -> if so parse the right url from the string
    if "#" in stac_url:
        stac_url = "https://" + "/".join(stac_url.split("#")[1].split("/")[2:])

    resolution = int(stac_url.split("/")[-3].replace("m", ""))

    item = pystac.Item.from_file(stac_url)

    ds = xr.open_dataset(item, engine="stac", resolution=resolution, crs="3413").isel(time=0).drop_vars("time")

    logger.info(f"Metadata download completed in {time.time() - start:.2f} seconds")

    return ds


def create_empty_datacube(storage: zarr.storage.Store, resolution: int, chunk_size: int):
    """Create an empty datacube from a GeoBox covering the complete extend of the EPSG:3413.

    Args:
        storage (zarr.storage.Store): The zarr storage object where the datacube will be saved.
        resolution (int): The resolution of a single pixel in meters.
        chunk_size (int): The size of a single chunk in pixels.

    """
    data_vars = ["dem", "count", "mad", "maxdate", "mindate", "datamask"]

    geobox = GeoBox.from_bbox((-3314693.24, -3314693.24, 3314693.24, 3314693.24), "epsg:3413", resolution=resolution)

    ds = xr.Dataset(
        {name: odc.geo.xr.xr_zeros(geobox, chunks=-1, dtype="float32") for name in data_vars},
        attrs={"title": "ArcticDEM Data Cube", "loaded_scenes": []},
    )

    lon_encoding = optimize_coord_encoding(ds.x.values, resolution)
    lat_encoding = optimize_coord_encoding(ds.y.values, -resolution)
    var_encoding = {
        name: {
            "chunks": (chunk_size, chunk_size),
            "compressor": zarr.Blosc(cname="zstd"),
            # workaround to create a fill value for the underlying zarr array
            # since Xarray doesn't let us specify one explicitly
            "_FillValue": float("nan"),
            # "scale_factor": 0.1,
            # "add_offset": 0,
            "dtype": "float32",
        }
        for name in data_vars
    }
    encoding = {
        "x": {"chunks": ds.x.shape, **lon_encoding},
        "y": {"chunks": ds.y.shape, **lat_encoding},
        "spatial_ref": {"chunks": None, "dtype": "int32"},
        **var_encoding,
    }
    ds.to_zarr(
        storage,
        encoding=encoding,
        compute=False,
    )


def procedural_download_datacube(storage: zarr.storage.Store, scenes: gpd.GeoDataFrame):
    """Download the ArcticDEM data for the specified scenes and add it to the datacube.

    Args:
        storage (zarr.storage.Store): The zarr storage object where the datacube will be saved.
        scenes (gpd.GeoDataFrame): The GeoDataFrame containing the scene information from the mosaic extend dataframe.

    References:
        - https://earthmover.io/blog/serverless-datacube-pipeline

    """
    # Check if zarr data already contains the data via the attrs
    arcticdem_datacube = xr.open_zarr(storage)

    loaded_scenes = arcticdem_datacube.attrs.get("loaded_scenes", []).copy()
    # TODO: Add another attribute called "loading_scenes" to keep track of the scenes that are currently being loaded
    # This is necessary for parallel loading of the data
    # Maybe it would be better to turn this into a class which is meant to be used as singleton and can store
    # the loaded scenes as state

    for sceneinfo in scenes.itertuples():
        # Skip if the scene is already in the datacube
        if sceneinfo.dem_id in loaded_scenes:
            continue

        scene = download_arcticdem_stac(sceneinfo.s3url)

        x_start_idx = int((scene.x[0] - arcticdem_datacube.x[0]) // scene.x.attrs["resolution"])
        y_start_idx = int((scene.y[0] - arcticdem_datacube.y[0]) // scene.y.attrs["resolution"])
        target_slice = {
            "x": slice(x_start_idx, x_start_idx + scene.sizes["x"]),
            "y": slice(y_start_idx, y_start_idx + scene.sizes["y"]),
        }
        logger.debug(f"Target slice: {target_slice}")

        for var in scene.data_vars:
            if var not in arcticdem_datacube.data_vars:
                logger.warning(f"Variable '{var}' not in the datacube, skipping")
                continue
            raw_data = scene[var].values  # noqa: PD011
            logger.debug(f"Adding {var} to the datacube: {lovely(raw_data)}")
            arcticdem_datacube[var][target_slice] = raw_data
            arcticdem_datacube[var][target_slice].to_zarr(storage, region="auto", safe_chunks=False)
        loaded_scenes.append(sceneinfo.dem_id)

    # Update storage (with native zarr)
    za = zarr.open(storage)
    za.attrs["loaded_scenes"] = loaded_scenes
    # Xarray default behaviour is to read the consolidated metadata, hence, we must update it
    zarr.consolidate_metadata(storage)


def get_arcticdem_tile(
    reference_dataset: xr.Dataset,
    data_dir: Path,
    resolution: RESOLUTIONS | None = None,
    chunk_size: int = 4000,
    buffer: int = 256,
) -> xr.Dataset:
    """Get the corresponding ArcticDEM tile for the given reference dataset.

    Args:
        reference_dataset (xr.Dataset): The reference dataset.
        data_dir (Path): The directory where the ArcticDEM data is stored.
        resolution (Literal[2, 10, 32] | None, optional): The resolution of the ArcticDEM data in m.
            If None tries to automatically detect the lowest resolution possible for the reference. Defaults to None.
        chunk_size (int, optional): The chunk size for the datacube. Only relevant for the initial creation.
            Has no effect otherwise. Defaults to 4000.
        buffer (int, optional): The buffer around the reference dataset in pixels. Defaults to 256.

    Returns:
        xr.Dataset: The ArcticDEM tile.

    """
    # TODO: What is a good chunk size?
    # TODO: what happens if two processes try to download the same file at the same time?
    start = time.time()
    logger.info(f"Getting ArcticDEM tile from {data_dir.resolve()}")

    reference_resolution = abs(int(reference_dataset.x[1] - reference_dataset.x[0]))
    # Select the resolution based on the reference dataset
    if resolution is None:
        # TODO: Discuss about the resolution thresholds
        if reference_resolution < 8:
            resolution = 2
        elif reference_resolution < 25:
            resolution = 10
        else:
            resolution = 32

    datacube_fpath = data_dir / f"datacube_{resolution}m_v4.1.zarr"
    storage = zarr.storage.FSStore(datacube_fpath)

    # Check if the zarr data already exists
    if not datacube_fpath.exists():
        logger.debug(f"Creating a new zarr datacube at {datacube_fpath.resolve()} with {storage=}")
        create_empty_datacube(storage, resolution, chunk_size)

    # Get the adjacent arcticdem scenes
    # * Note: We could also use pystac here, but this would result in a slight performance decrease
    # * because of the network overhead

    # Load the extend, download if the file does not exist
    extend_fpath = data_dir / f"ArcticDEM_Mosaic_Index_v4_1_{resolution}m.parquet"
    if not extend_fpath.exists():
        download_arcticdem_extend(data_dir)
    extend = gpd.read_parquet(extend_fpath)

    # Add a buffer around the reference dataset to get the adjacent scenes
    buffer_m = buffer * reference_resolution  # 256 pixels * the resolution of the reference dataset
    reference_bbox = shapely.geometry.box(*reference_dataset.rio.transform_bounds("epsg:3413")).buffer(
        buffer_m, join_style="mitre"
    )
    adjacent_scenes = extend[extend.intersects(reference_bbox)]

    # Download the adjacent scenes (if necessary)
    procedural_download_datacube(storage, adjacent_scenes)
    logger.debug(f"Procedural download completed in {time.time() - start:.2f} seconds")

    # Load the datacube and set the spatial_ref since it is set as a coordinate within the zarr format
    arcticdem_datacube = xr.open_zarr(storage).set_coords("spatial_ref")

    # Get an AOI slice of the datacube, since rio.reproject_match would load the whole datacube
    xmin, ymin, xmax, ymax = reference_bbox.bounds
    aoi_slice = {
        "x": slice(xmin, xmax),
        "y": slice(ymax, ymin),
    }
    logger.debug(f"AOI slice: {aoi_slice}")
    arcticdem_aoi = arcticdem_datacube.sel(aoi_slice)
    logger.debug(f"ArcticDEM AOI: {arcticdem_aoi}")

    # TODO: I think the buffer gets lost here, because the reproject_match function crops to the reference dataset
    ds = arcticdem_aoi.rio.reproject_match(reference_dataset, resampling=rasterio.enums.Resampling.cubic)

    logger.info(f"ArcticDEM tile loaded in {time.time() - start:.2f} seconds")
    return ds
