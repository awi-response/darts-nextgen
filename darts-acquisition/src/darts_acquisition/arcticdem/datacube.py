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
import requests
import shapely
import xarray as xr
import zarr
import zarr.storage
from odc.geo.geobox import GeoBox

from darts_acquisition.utils.storage import optimize_coord_encoding

logger = logging.getLogger(__name__.replace("darts_", "darts."))


RESOLUTIONS = Literal[2, 10, 32]
# https://www.pgc.umn.edu/guides/stereo-derived-elevation-models/pgc-dem-products-arcticdem-rema-and-earthdem
DATA_VARS = ["dem", "datamask"]  # ["dem", "count", "mad", "maxdate", "mindate", "datamask"]
DATA_VARS_META = {
    "dem": {
        "long_name": "Digital Elevation Model",
        "source": "ArcticDEM",
        "units": "m",
        "description": "Digital Elevation Model, elevation resolution is cropped to ~1cm",
    },
    # "count": {"long_name": "Count", "source": "ArcticDEM", "description": "Number of contributing DEMs"},
    # "mad": {
    #     "long_name": "Median Absolute Deviation",
    #     "source": "ArcticDEM",
    #     "units": "m",
    #     "description": "Median absolute deviation of contributing DEMs",
    # },
    # "maxdate": {
    #     "long_name": "Max Date",
    #     "source": "ArcticDEM",
    #     "description": "The date of the most recent image in days since 01. Jan 2000",
    # },
    # "mindate": {
    #     "long_name": "Min Date",
    #     "source": "ArcticDEM",
    #     "description": "The date of the oldest image in days since 01. Jan 2000",
    # },
    "datamask": {"long_name": "Data Mask", "source": "ArcticDEM"},
}
DATA_VARS_ENCODING = {
    "dem": {"dtype": "float32"},
    # "dem": {"dtype": "float32", "_FillValue": float("nan")},
    # "count": {"dtype": "int16", "_FillValue": 0},
    # "mad": {"dtype": "float32", "_FillValue": float("nan")},
    # "maxdate": {"dtype": "int16", "_FillValue": -1},  # Storing the date as int16 is enough until 18. Sep 2089
    # "mindate": {"dtype": "int16", "_FillValue": -1},  # Storing the date as int16 is enough until 18. Sep 2089
    "datamask": {"dtype": "bool"},
    # "datamask": {"dtype": "bool", "_FillValue": False},
}


def download_arcticdem_extent(dem_data_dir: Path):
    """Download the ArcticDEM mosaic extent info from the provided URL and extracts it to the specified directory.

    Args:
        dem_data_dir (Path): The directory where the extracted data will be saved.

    Example:
        ```python
        from darts_acquisition.arcticdem.datacube import download_arcticdem_extent

        dem_data_dir = Path("data/arcticdem")
        download_arcticdem_extent(dem_data_dir)
        ```

        Resulting in the following directory structure:

        ```sh
        $ tree data/arcticdem
        data/arcticdem
        ├── ArcticDEM_Mosaic_Index_v4_1_2m.parquet
        ├── ArcticDEM_Mosaic_Index_v4_1_10m.parquet
        └── ArcticDEM_Mosaic_Index_v4_1_32m.parquet
        ```

    """
    tick_fstart = time.perf_counter()
    url = "https://data.pgc.umn.edu/elev/dem/setsm/ArcticDEM/indexes/ArcticDEM_Mosaic_Index_latest_gpqt.zip"
    logger.debug(f"Downloading the arcticdem mosaic extent from {url} to {dem_data_dir.resolve()}")
    response = requests.get(url)

    # Get the downloaded data as a byte string
    data = response.content

    tick_download = time.perf_counter()
    logger.debug(f"Downloaded {len(data)} bytes in {tick_download - tick_fstart:.2f} seconds")

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

    tick_extract = time.perf_counter()
    logger.debug(f"Extraction completed in {tick_extract - tick_download:.2f} seconds")
    logger.info(
        f"Download and extraction of the arcticdem mosiac extent from {url} to {dem_data_dir.resolve()}"
        f"completed in {tick_extract - tick_fstart:.2f} seconds"
    )


# ! unused - remove later
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

    Read more here: https://www.pgc.umn.edu/guides/stereo-derived-elevation-models/pgc-dem-products-arcticdem-rema-and-earthdem

    Args:
        stac_url (str): The URL of the ArcticDEM STAC. Must be one of:
            - A stac-browser url, like it is provided in the mosaic-extent dataframe, e.g. https://polargeospatialcenter.github.io/stac-browser/#/external/pgc-opendata-dems.s3.us-west-2.amazonaws.com/arcticdem/mosaics/v4.1/32m/36_24/36_24_32m_v4.1.json
            - A direct link to the STAC file, e.g. https://pgc-opendata-dems.s3.us-west-2.amazonaws.com/arcticdem/mosaics/v4.1/32m/36_24/36_24_32m_v4.1.json

    Returns:
        xr.Dataset: The ArcticDEM data as an lazy xarray Dataset.
            Data downloaded on demand, NOT from within this function.

    """
    tick_fstart = time.perf_counter()

    # Check weather the browser url is provided -> if so parse the right url from the string
    if "#" in stac_url:
        stac_url = "https://" + "/".join(stac_url.split("#")[1].split("/")[2:])

    resolution = int(stac_url.split("/")[-3].replace("m", ""))
    logger.debug(f"Downloading ArcticDEM data from {stac_url} with a resolution of {resolution}m")

    item = pystac.Item.from_file(stac_url)
    ds = xr.open_dataset(item, engine="stac", resolution=resolution, crs="3413").isel(time=0).drop_vars("time")

    tick_fend = time.perf_counter()
    logger.debug(f"Scene metadata download completed in {tick_fend - tick_fstart:.2f} seconds")

    return ds


def create_empty_datacube(storage: zarr.storage.Store, resolution: int, chunk_size: int):
    """Create an empty datacube from a GeoBox covering the complete extent of the EPSG:3413.

    Args:
        storage (zarr.storage.Store): The zarr storage object where the datacube will be saved.
        resolution (int): The resolution of a single pixel in meters.
        chunk_size (int): The size of a single chunk in pixels.

    """
    tick_fstart = time.perf_counter()
    logger.info(
        f"Creating an empty zarr datacube with the variables"
        f"{DATA_VARS} at a {resolution=}m and {chunk_size=} to {storage=}"
    )
    geobox = GeoBox.from_bbox((-3314693.24, -3314693.24, 3314693.24, 3314693.24), "epsg:3413", resolution=resolution)

    ds = xr.Dataset(
        {name: odc.geo.xr.xr_zeros(geobox, chunks=-1, dtype="float32") for name in DATA_VARS},
        attrs={"title": "ArcticDEM Data Cube", "loaded_scenes": []},
    )

    # Add metadata
    for name, meta in DATA_VARS_META.items():
        ds[name].attrs.update(meta)

    coords_encoding = {
        "x": {"chunks": ds.x.shape, **optimize_coord_encoding(ds.x.values, resolution)},
        "y": {"chunks": ds.y.shape, **optimize_coord_encoding(ds.y.values, -resolution)},
    }
    var_encoding = {
        name: {"chunks": (chunk_size, chunk_size), "compressor": zarr.Blosc(cname="zstd"), **DATA_VARS_ENCODING[name]}
        for name in DATA_VARS
    }
    encoding = {
        "spatial_ref": {"chunks": None, "dtype": "int32"},
        **coords_encoding,
        **var_encoding,
    }
    logger.debug(f"Datacube {encoding=}")

    ds.to_zarr(
        storage,
        encoding=encoding,
        compute=False,
    )
    tick_fend = time.perf_counter()
    logger.info(f"Empty datacube created in {tick_fend - tick_fstart:.2f} seconds")


def convert_s3url_to_stac(s3url: str) -> str:
    """Convert an ArcticDEM S3 URL to a STAC URL.

    The field in the extent dataframe is a STAC browser URL, hence we need to convert it to a STAC URL.

    Args:
        s3url (str): The S3 STAC-Browser URL of the ArcticDEM data.

    Returns:
        str: The STAC URL of the ArcticDEM data.

    """
    return "https://" + "/".join(s3url.split("#")[1].split("/")[2:])


def procedural_download_datacube(storage: zarr.storage.Store, scenes: gpd.GeoDataFrame):
    """Download the ArcticDEM data for the specified scenes and add it to the datacube.

    Args:
        storage (zarr.storage.Store): The zarr storage object where the datacube will be saved.
        scenes (gpd.GeoDataFrame): The GeoDataFrame containing the scene information from the mosaic extent dataframe.

    References:
        - https://earthmover.io/blog/serverless-datacube-pipeline

    Warning:
        This function is not thread-safe. Thread-safety might be added in the future.

    """
    tick_fstart = time.perf_counter()

    # Check if zarr data already contains the data via the attrs
    arcticdem_datacube = xr.open_zarr(storage, mask_and_scale=False)

    loaded_scenes: list[str] = arcticdem_datacube.attrs.get("loaded_scenes", []).copy()
    # TODO: Add another attribute called "loading_scenes" to keep track of the scenes that are currently being loaded
    # This is necessary for parallel loading of the data
    # Maybe it would be better to turn this into a class which is meant to be used as singleton and can store
    # the loaded scenes as state

    # Collect all scenes which should be downloaded
    new_scenes = scenes[~scenes.dem_id.isin(loaded_scenes)]

    if not len(new_scenes):
        logger.debug("No new scenes to download")
        return
    logger.debug(f"Found {len(new_scenes)} new scenes: {new_scenes.dem_id.to_list()}")

    # Collect the stac items
    new_scenes["stacurl"] = new_scenes.s3url.apply(convert_s3url_to_stac)
    items = pystac.ItemCollection([pystac.Item.from_file(scene.stacurl) for scene in new_scenes.itertuples()])

    # Parse the resolution from the first item
    resolution = int(new_scenes.stacurl.iloc[0].split("/")[-3].replace("m", ""))
    assert resolution in [2, 10, 32], f"Resolution {resolution} not supported, only 2m, 10m and 32m are supported"

    # Read the metadata and calculate the target slice
    # TODO: There should be a way to calculate the target slice without downloading the metadata
    # However, this is fine for now, since the overhead is very small and the resulting code very clear

    # This does not download the data into memory, since chunks=-1 will create a dask array
    # We need the coordinate information to calculate the target slice and the needed chunking for the real loading
    ds = xr.open_dataset(items, bands=DATA_VARS, engine="stac", resolution=resolution, crs="3413", chunks=-1)

    # Get the slice of the datacube where the scene will be written
    x_start_idx = int((ds.x[0] - arcticdem_datacube.x[0]) // ds.x.attrs["resolution"])
    y_start_idx = int((ds.y[0] - arcticdem_datacube.y[0]) // ds.y.attrs["resolution"])
    target_slice = {
        "x": slice(x_start_idx, x_start_idx + ds.sizes["x"]),
        "y": slice(y_start_idx, y_start_idx + ds.sizes["y"]),
    }

    arcticdem_datacube_aoi = arcticdem_datacube.isel(target_slice).drop_vars("spatial_ref")

    # Now open the data for real, but still as dask array, hence the download occurs later
    ds = (
        xr.open_dataset(
            items,
            bands=DATA_VARS,
            engine="stac",
            resolution=resolution,
            crs="3413",
            chunks=dict(arcticdem_datacube_aoi.chunks),
        )
        .max("time")
        .drop_vars("spatial_ref")
    )

    # Sometimes the data downloaded from stac has nan-borders, which would overwrite existing data
    # Replace these nan borders with existing data if there is any
    ds = ds.fillna(arcticdem_datacube_aoi)

    # Write the data to the datacube, we manually aligned the chunks, hence we can do safe_chunks=False
    tick_downloads = time.perf_counter()
    ds.to_zarr(storage, region=target_slice, safe_chunks=False)
    tick_downloade = time.perf_counter()
    logger.debug(f"Downloaded and written data to datacube in {tick_downloade - tick_downloads:.2f}s")

    # Update loaded_scenes (with native zarr, since xarray does not support this yet)
    loaded_scenes.extend(new_scenes.dem_id)
    za = zarr.open(storage)
    za.attrs["loaded_scenes"] = loaded_scenes
    # Xarray default behaviour is to read the consolidated metadata, hence, we must update it
    zarr.consolidate_metadata(storage)

    tick_fend = time.perf_counter()
    logger.info(f"Procedural download of {len(new_scenes)} scenes completed in {tick_fend - tick_fstart:.2f} seconds")


def get_arcticdem_tile(
    reference_dataset: xr.Dataset,
    data_dir: Path,
    resolution: RESOLUTIONS,
    chunk_size: int = 6000,
    buffer: int = 256,
) -> xr.Dataset:
    """Get the corresponding ArcticDEM tile for the given reference dataset.

    Args:
        reference_dataset (xr.Dataset): The reference dataset.
        data_dir (Path): The directory where the ArcticDEM data is stored.
        resolution (Literal[2, 10, 32]): The resolution of the ArcticDEM data in m.
        chunk_size (int, optional): The chunk size for the datacube. Only relevant for the initial creation.
            Has no effect otherwise. Defaults to 6000.
        buffer (int, optional): The buffer around the reference dataset in pixels. Defaults to 256.

    Returns:
        xr.Dataset: The ArcticDEM tile, with a buffer applied.
            Note: The buffer is applied in the arcticdem dataset's CRS, hence the orientation might be different.
            Final dataset is NOT matched to the reference CRS and resolution.

    Warning:
        1. This function is not thread-safe. Thread-safety might be added in the future.
        2. Reference dataset must be in a meter based CRS.

    """
    # TODO: What is a good chunk size?
    # TODO: Thread-safety concers:
    # - How can we ensure that the same arcticdem scene is not downloaded twice at the same time?
    # - How can we ensure that the extent is not downloaded twice at the same time?

    tick_fstart = time.perf_counter()

    datacube_fpath = data_dir / f"datacube_{resolution}m_v4.1.zarr"
    storage = zarr.storage.FSStore(datacube_fpath)
    logger.debug(f"Getting ArcticDEM tile from {datacube_fpath.resolve()}")

    # ! The reference dataset must be in a meter based CRS
    reference_resolution = abs(int(reference_dataset.x[1] - reference_dataset.x[0]))
    logger.debug(f"Found a reference resolution of {reference_resolution}m")

    # Check if the zarr data already exists
    if not datacube_fpath.exists():
        logger.debug(f"Creating a new zarr datacube at {datacube_fpath.resolve()} with {storage=}")
        create_empty_datacube(storage, resolution, chunk_size)

    # Get the adjacent arcticdem scenes
    # Note: We could also use pystac here, but this would result in a slight performance decrease
    # because of the network overhead, hence we use the extent file
    # Download the extent, download if the file does not exist
    extent_fpath = data_dir / f"ArcticDEM_Mosaic_Index_v4_1_{resolution}m.parquet"
    if not extent_fpath.exists():
        download_arcticdem_extent(data_dir)
    extent = gpd.read_parquet(extent_fpath)

    # Add a buffer around the reference dataset to get the adjacent scenes
    buffer_m = buffer * reference_resolution  # nbuffer pixels * the resolution of the reference dataset
    reference_bbox = shapely.geometry.box(*reference_dataset.rio.transform_bounds("epsg:3413")).buffer(
        buffer_m, join_style="mitre"
    )
    adjacent_scenes = extent[extent.intersects(reference_bbox)]

    # Download the adjacent scenes (if necessary)
    procedural_download_datacube(storage, adjacent_scenes)

    # Load the datacube and set the spatial_ref since it is set as a coordinate within the zarr format
    arcticdem_datacube = xr.open_zarr(storage, mask_and_scale=False).set_coords("spatial_ref")

    # Get an AOI slice of the datacube, since rio.reproject_match would load the whole datacube
    # Note: The bounds are not equal to the bbox orientation, because of the change of the CRS
    xmin, ymin, xmax, ymax = reference_bbox.bounds
    aoi_slice = {
        "x": slice(xmin, xmax),
        "y": slice(ymax, ymin),
    }
    logger.debug(f"AOI slice: {aoi_slice}")
    arcticdem_aoi = arcticdem_datacube.sel(aoi_slice)

    # The following code would load the data from disk, but we want to keep it lazy
    # tick_sload = time.perf_counter()
    # arcticdem_aoi = arcticdem_aoi.compute()
    # tick_eload = time.perf_counter()
    # logger.debug(f"ArcticDEM AOI loaded from disk in {tick_eload - tick_sload:.2f} seconds")

    logger.info(f"ArcticDEM tile loaded in {time.perf_counter() - tick_fstart:.2f} seconds")
    return arcticdem_aoi
