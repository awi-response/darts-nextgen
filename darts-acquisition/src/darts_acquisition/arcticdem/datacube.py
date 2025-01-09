"""Downloading and loading related functions for the Zarr-Datacube approach."""

import io
import logging
import multiprocessing as mp
import time
import zipfile
from pathlib import Path
from typing import Literal

import geopandas as gpd
import pystac
import requests
import xarray as xr
import zarr
import zarr.storage
from odc.geo.geobox import GeoBox

from darts_acquisition.utils.storage import create_empty_datacube

logger = logging.getLogger(__name__.replace("darts_", "darts."))


RESOLUTIONS = Literal[2, 10, 32]
CHUNK_SIZE = 3600
# https://www.pgc.umn.edu/guides/stereo-derived-elevation-models/pgc-dem-products-arcticdem-rema-and-earthdem
DATA_EXTENT = {
    2: GeoBox.from_bbox((-3314693.24, -3314693.24, 3314693.24, 3314693.24), "epsg:3413", resolution=2),
    10: GeoBox.from_bbox((-3314693.24, -3314693.24, 3314693.24, 3314693.24), "epsg:3413", resolution=10),
    32: GeoBox.from_bbox((-3314693.24, -3314693.24, 3314693.24, 3314693.24), "epsg:3413", resolution=32),
}
DATA_VARS = ["dem", "datamask"]  # ["dem", "count", "mad", "maxdate", "mindate", "datamask"]
DATA_VARS_META = {
    "dem": {
        "long_name": "Digital Elevation Model",
        "data_source": "ArcticDEM",
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

# Lock for downloading the data
# This will block other processes from downloading or processing the data, until the current download is finished.
# This may result in halting a tile-process for a while, even if it's data was already downloaded.
download_lock = mp.Lock()


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


def convert_s3url_to_stac(s3url: str) -> str:
    """Convert an ArcticDEM S3 URL to a STAC URL.

    The field in the extent dataframe is a STAC browser URL, hence we need to convert it to a STAC URL.

    Args:
        s3url (str): The S3 STAC-Browser URL of the ArcticDEM data.

    Returns:
        str: The STAC URL of the ArcticDEM data.

    Example:
        ```python
        s3url = "https://polargeospatialcenter.github.io/stac-browser/#/external/pgc-opendata-dems.s3.us-west-2.amazonaws.com/arcticdem/mosaics/v4.1/32m/36_24/36_24_32m_v4.1.json"
        stacurl = convert_s3url_to_stac(s3url)
        stacurl
        >>> "https://pgc-opendata-dems.s3.us-west-2.amazonaws.com/arcticdem/mosaics/v4.1/32m/36_24/36_24_32m_v4.1.json"
        ```

    """
    return "https://" + "/".join(s3url.split("#")[1].split("/")[2:])


def procedural_download_datacube(storage: zarr.storage.Store, tiles: gpd.GeoDataFrame):
    """Download the ArcticDEM data for the specified tiles and add it to the datacube.

    Args:
        storage (zarr.storage.Store): The zarr storage object where the datacube will be saved.
        tiles (gpd.GeoDataFrame): The GeoDataFrame containing the tile information from the mosaic extent dataframe.

    References:
        - https://earthmover.io/blog/serverless-datacube-pipeline

    Warning:
        This function is not thread-safe. Thread-safety might be added in the future.

    """
    tick_fstart = time.perf_counter()

    # Check if zarr data already contains the data via the attrs
    arcticdem_datacube = xr.open_zarr(storage, mask_and_scale=False)

    loaded_tiles: list[str] = arcticdem_datacube.attrs.get("loaded_tiles", []).copy()

    # Collect all tiles which should be downloaded
    new_tiles = tiles[~tiles.dem_id.isin(loaded_tiles)]

    if not len(new_tiles):
        logger.debug("No new tiles to download")
        return
    logger.debug(f"Found {len(new_tiles)} new tiles: {new_tiles.dem_id.to_list()}")

    # Collect the stac items
    new_tiles["stacurl"] = new_tiles.s3url.apply(convert_s3url_to_stac)
    items = pystac.ItemCollection([pystac.Item.from_file(tile.stacurl) for tile in new_tiles.itertuples()])

    # Parse the resolution from the first item
    resolution = int(new_tiles.stacurl.iloc[0].split("/")[-3].replace("m", ""))
    assert resolution in [2, 10, 32], f"Resolution {resolution} not supported, only 2m, 10m and 32m are supported"

    # Read the metadata and calculate the target slice
    # TODO: There should be a way to calculate the target slice without downloading the metadata
    # However, this is fine for now, since the overhead is very small and the resulting code very clear

    # This does not download the data into memory, since chunks=-1 will create a dask array
    # We need the coordinate information to calculate the target slice and the needed chunking for the real loading
    ds = xr.open_dataset(items, bands=DATA_VARS, engine="stac", resolution=resolution, crs="3413", chunks=-1)

    # Get the slice of the datacube where the tile will be written
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

    # Update loaded_tiles (with native zarr, since xarray does not support this yet)
    loaded_tiles.extend(new_tiles.dem_id)
    za = zarr.open(storage)
    za.attrs["loaded_tiles"] = loaded_tiles
    # Xarray default behaviour is to read the consolidated metadata, hence, we must update it
    zarr.consolidate_metadata(storage)

    tick_fend = time.perf_counter()
    logger.info(f"Procedural download of {len(new_tiles)} tiles completed in {tick_fend - tick_fstart:.2f} seconds")


def load_arcticdem(
    geobox: GeoBox,
    data_dir: Path | str,
    resolution: RESOLUTIONS,
    buffer: int = 0,
    persist: bool = True,
) -> xr.Dataset:
    """Load the ArcticDEM for the given geobox, fetch new data from the STAC server if necessary.

    Args:
        geobox (GeoBox): The geobox for which the tile should be loaded.
        data_dir (Path | str): The directory where the ArcticDEM data is stored.
        resolution (Literal[2, 10, 32]): The resolution of the ArcticDEM data in m.
        buffer (int, optional): The buffer around the projected (epsg:3413) geobox in pixels. Defaults to 0.
        persist (bool, optional): If the data should be persisted in memory.
            If not, this will return a Dask backed Dataset. Defaults to True.

    Returns:
        xr.Dataset: The ArcticDEM tile, with a buffer applied.
            Note: The buffer is applied in the arcticdem dataset's CRS, hence the orientation might be different.
            Final dataset is NOT matched to the reference CRS and resolution.

    Warning:
        Geobox must be in a meter based CRS.

    Usage:
        Since the API of the `load_arcticdem` is based on GeoBox, one can load a specific ROI based on an existing Xarray DataArray:

        ```python
        import xarray as xr
        import odc.geo.xr

        from darts_aquisition import load_arcticdem

        # Assume "optical" is an already loaded s2 based dataarray

        arcticdem = load_arcticdem(
            optical.odc.geobox,
            "/path/to/arcticdem-parent-directory",
            resolution=2,
            buffer=ceil(self.tpi_outer_radius / 2 * sqrt(2))
        )

        # Now we can for example match the resolution and extent of the optical data:
        arcticdem = arcticdem.odc.reproject(optical.odc.geobox, resampling="cubic")
        ```

        The `buffer` parameter is used to extend the region of interest by a certain amount of pixels.
        This comes handy when calculating e.g. the Topographic Position Index (TPI), which requires a buffer around the region of interest to remove edge effects.

    """  # noqa: E501
    tick_fstart = time.perf_counter()

    data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir

    datacube_fpath = data_dir / f"datacube_{resolution}m_v4.1.zarr"
    storage = zarr.storage.FSStore(datacube_fpath)
    logger.debug(f"Getting ArcticDEM tile from {datacube_fpath.resolve()}")

    # ! The geobox must be in a meter based CRS
    logger.debug(f"Found a reference resolution of {geobox.resolution.x}m")

    # Check if the zarr data already exists
    if not datacube_fpath.exists():
        logger.debug(f"Creating a new zarr datacube at {datacube_fpath.resolve()} with {storage=}")
        create_empty_datacube(
            "ArcticDEM Data Cube",
            storage,
            DATA_EXTENT[resolution],
            CHUNK_SIZE,
            DATA_VARS,
            DATA_VARS_META,
            DATA_VARS_ENCODING,
        )

    # Get the adjacent arcticdem tiles
    # Note: We could also use pystac here, but this would result in a slight performance decrease
    # because of the network overhead, hence we use the extent file
    # Download the extent, download if the file does not exist
    extent_fpath = data_dir / f"ArcticDEM_Mosaic_Index_v4_1_{resolution}m.parquet"
    with download_lock:
        if not extent_fpath.exists():
            download_arcticdem_extent(data_dir)
    extent = gpd.read_parquet(extent_fpath)

    # Add a buffer around the geobox to get the adjacent tiles
    reference_geobox = geobox.to_crs("epsg:3413", resolution=resolution).pad(buffer)
    adjacent_tiles = extent[extent.intersects(reference_geobox.extent.geom)]

    # Download the adjacent tiles (if necessary)
    with download_lock:
        procedural_download_datacube(storage, adjacent_tiles)

    # Load the datacube and set the spatial_ref since it is set as a coordinate within the zarr format
    chunks = None if persist else "auto"
    arcticdem_datacube = xr.open_zarr(storage, mask_and_scale=False, chunks=chunks).set_coords("spatial_ref")

    # Get an AOI slice of the datacube
    arcticdem_aoi = arcticdem_datacube.odc.crop(reference_geobox.extent, apply_mask=False)

    # The following code would load the lazy zarr data from disk into memory
    if persist:
        tick_sload = time.perf_counter()
        arcticdem_aoi = arcticdem_aoi.load()
        tick_eload = time.perf_counter()
        logger.debug(f"ArcticDEM AOI loaded from disk in {tick_eload - tick_sload:.2f} seconds")

    # Change dtype of the datamask to uint8 for later reproject_match
    arcticdem_aoi["datamask"] = arcticdem_aoi.datamask.astype("uint8")

    logger.info(
        f"ArcticDEM tile {'loaded' if persist else 'lazy-opened'} in {time.perf_counter() - tick_fstart:.2f} seconds"
    )
    return arcticdem_aoi
