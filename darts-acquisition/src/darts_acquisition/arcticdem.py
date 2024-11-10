"""ArcticDEM related data loading."""

import io
import logging
import os
import signal
import tarfile
import time
import zipfile
from pathlib import Path
from threading import Event
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
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from darts_acquisition.utils.storage import optimize_coord_encoding

logger = logging.getLogger(__name__.replace("darts_", "darts."))


RESOLUTIONS = Literal[2, 10, 32]


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


def download_arcticdem_scene(dem_data_dir: Path, tile_url: str, force: bool = False):
    """Download an ArcticDEM tile from the provided URL and extracts it to the specified directory.

    This saves the data on disk. For a memory-only solution, use `download_arcticdem_stac`.

    Assets include among others:
        - Hillshade
        - DEM
        - Count
        - Median Absolute Deviation

    Args:
        dem_data_dir (Path): The directory where the extracted data will be saved.
        tile_url (str): The URL of the ArcticDEM tile. Expects a tar.gz file.
        force (bool, optional): Weather to download the file, even if it already exists. Defaults to False.

    """
    start = time.time()
    fname = tile_url.split("/")[-1]
    tile_name = ".".join(fname.split(".")[:-2])

    if not force and (dem_data_dir / tile_name / f"{tile_name}_dem.tif").exists():
        (f"ArcticDEM tile '{tile_name}' already exists in {dem_data_dir.resolve()}")
        return

    logger.info(f"Downloading the arcticdem tile '{tile_name}' from {tile_url} to {dem_data_dir.resolve()}")

    response = requests.get(tile_url, stream=True)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))

    logger.debug(f"Downloading {fname} ({total} bytes)")

    done_event = Event()

    def handle_sigint(signum, frame):
        done_event.set()

    signal.signal(signal.SIGINT, handle_sigint)

    # Create an IO stream to write the data to
    with io.BytesIO() as buffer:
        # Create a progress bar
        download_progress = Progress(
            TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "•",
            DownloadColumn(),
            "•",
            TransferSpeedColumn(),
            "•",
            TimeRemainingColumn(),
        )
        with download_progress:
            task = download_progress.add_task("download", filename=fname, total=total)

            for chunk in response.iter_content(chunk_size=32768):
                download_progress.update(task, advance=len(chunk))
                if chunk:
                    buffer.write(chunk)
                if done_event.is_set():
                    break
            download_progress.update(task, completed=total)
            download_progress.refresh()
            download_progress.stop()

        logger.debug(f"Downloaded {len(buffer.getvalue())} bytes, starting extraction")

        # Create a tarfile object and extract the files to a directory
        (dem_data_dir / tile_name).mkdir(parents=True, exist_ok=True)
        buffer.seek(0)
        with tarfile.open(fileobj=buffer, mode="r:gz") as tar:
            tar.extractall(dem_data_dir / tile_name)

    tile_dir = dem_data_dir / tile_name
    tile_assets = [str(p.relative_to(tile_dir)) for p in tile_dir.glob("**/*") if p.is_file()]
    tile_assets_str = "\n\t- " + "\n\t- ".join(tile_assets)
    logger.debug(f"Extracted files:\t\n- {tile_assets_str}")
    total_gb = total / 1024 / 1024 / 1024
    logger.info(f"Download of '{fname}' ({total_gb:.1f} Gb) completed in {time.time() - start:.2f} seconds")


def download_arcticdem_stac(stac_url: str, crs: str = "3413", resolution: RESOLUTIONS = 2) -> xr.Dataset:
    """Download ArcticDEM data from the provided STAC URL.

    This function utilizes pystac, xpystac and odc-stac to create a lazy dataset containing all assets.

    Assets include among others:
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
        crs (str, optional): The coordinate reference system of the data. Defaults to "3413".
        resolution (Literal[2, 10, 32], optional): The resolution of the data. Defaults to 2 (m).

    Returns:
        xr.Dataset: The ArcticDEM data as an lazy xarray Dataset.
            Data downloaded on demand, NOT from within this function.

    """
    start = time.time()
    logger.info(f"Loading ArcticDEM data from {stac_url}")

    # Check weather the browser url is provided -> if so parse the right url from the string
    if "#" in stac_url:
        stac_url = "https://" + "/".join(stac_url.split("#")[1].split("/")[2:])

    item = pystac.Item.from_file(stac_url)

    ds = xr.open_dataset(item, engine="stac", resolution=resolution, crs=crs).isel(time=0).drop_vars("time")

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


def procedural_download_datacube(storage: zarr.storage.Store, scenes: gpd.GeoDataFrame, resolution: int):
    """Download the ArcticDEM data for the specified scenes and add it to the datacube.

    Args:
        storage (zarr.storage.Store): The zarr storage object where the datacube will be saved.
        scenes (gpd.GeoDataFrame): The GeoDataFrame containing the scene information from the mosaic extend dataframe.
        resolution (int): The resolution of a single pixel in meters.

    """
    # Check if zarr data already contains the data via the attrs
    arcticdem_datacube = xr.open_zarr(storage)

    loaded_scenes = arcticdem_datacube.attrs.get("loaded_scenes", []).copy()
    # TODO: Add another attribute called "loading_scenes" to keep track of the scenes that are currently being loaded
    # This is necessary for parallel loading of the data

    for sceneinfo in scenes.itertuples():
        # Skip if the scene is already in the datacube
        if sceneinfo.dem_id in loaded_scenes:
            continue

        scene = download_arcticdem_stac(sceneinfo.s3url, resolution=resolution)

        x_start_idx = int((scene.x[0] - arcticdem_datacube.x[0]) // resolution)
        y_start_idx = int((arcticdem_datacube.y[0] - scene.y[0]) // resolution)  # inverse because of the flipped y-axis
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


def get_arcticdem_tile(reference_dataset: xr.Dataset, data_dir: Path) -> xr.Dataset:
    """Get the corresponding ArcticDEM tile for the given reference dataset.

    Args:
        reference_dataset (xr.Dataset): The reference dataset.
        data_dir (Path): The directory where the ArcticDEM data is stored.

    Returns:
        xr.Dataset: The ArcticDEM tile.

    """
    # TODO: what happens if two processes try to download the same file at the same time?
    start = time.time()
    logger.info(f"Getting ArcticDEM tile from {data_dir.resolve()}")

    # TODO: Add these as parameters
    resolution = 32
    chunk_size = 4_000  # TODO: What is a good chunk size?
    buffer = 256 * 3.125  # 256 pixels * 3.125 m/pixel - the resolution of the reference dataset

    datacube_fpath = data_dir / "datacube.zarr"
    storage = zarr.storage.FSStore(datacube_fpath)

    # Check if the zarr data already exists
    if not datacube_fpath.exists():
        logger.debug(f"Creating a new zarr datacube at {datacube_fpath.resolve()} with {storage=}")
        create_empty_datacube(storage, resolution, chunk_size)

    # Get the adjacent arcticdem scenes
    # * Note: We could also use pystac here, but this would result in a slight performance decrease
    # * because of the network overhead
    extend_fpath = data_dir / f"ArcticDEM_Mosaic_Index_v4_1_{resolution}m.parquet"
    # Download the extend file if it does not exist
    if not extend_fpath.exists():
        download_arcticdem_extend(data_dir)
    extend = gpd.read_parquet(extend_fpath)

    # Add a buffer around the reference dataset to get the adjacent scenes
    reference_bbox = shapely.geometry.box(*reference_dataset.rio.transform_bounds("epsg:3413")).buffer(
        buffer, join_style="mitre"
    )
    adjacent_scenes = extend[extend.intersects(reference_bbox)]
    procedural_download_datacube(storage, adjacent_scenes, resolution)
    logger.debug(f"Procedural download completed in {time.time() - start:.2f} seconds")

    # Load the datacube and set the spatial_ref since it is set as a coordinate within the zarr format
    arcticdem_datacube = xr.open_zarr(storage).set_coords("spatial_ref")

    xmin, ymin, xmax, ymax = reference_bbox.bounds

    aoi_slice = {
        "x": slice(xmin, xmax),
        "y": slice(ymax, ymin),
    }
    logger.debug(f"AOI slice: {aoi_slice}")
    arcticdem_aoi = arcticdem_datacube.sel(aoi_slice)
    logger.debug(f"ArcticDEM AOI: {arcticdem_aoi}")

    ds = arcticdem_aoi.rio.reproject_match(reference_dataset, resampling=rasterio.enums.Resampling.cubic)

    logger.info(f"ArcticDEM tile loaded in {time.time() - start:.2f} seconds")
    return ds


def create_arcticdem_vrt(dem_data_dir: Path, vrt_target_dir: Path):
    """Create a VRT file from ArcticDEM data.

    Args:
        dem_data_dir (Path): The directory containing the ArcticDEM data (.tif).
        vrt_target_dir (Path): The output directory.

    Raises:
        OSError: If the target directory is not writable.

    """
    start_time = time.time()
    logger.debug(f"Creating ArcticDEM VRT file at {vrt_target_dir.resolve()} based on {dem_data_dir.resolve()}")

    try:
        from osgeo import gdal

        logger.debug(f"Found gdal bindings {gdal.__version__}.")
    except ModuleNotFoundError as e:
        logger.exception(
            "The python GDAL bindings where not found. Please install those which are appropriate for your platform."
        )
        raise e

    # decide on the exception behavior of GDAL to supress a warning if we dont
    # don't know if this is necessary in all GDAL versions
    try:
        gdal.UseExceptions()
        logger.debug("Enabled gdal exceptions")
    except AttributeError():
        pass

    # subdirs = {"elevation": "tiles_rel_el", "slope": "tiles_slope"}
    subdirs = {"elevation": "relative_elevation", "slope": "slope"}

    # check first if BOTH files are writable
    non_writable_files = []
    for name in subdirs.keys():
        output_file_path = vrt_target_dir / f"{name}.vrt"
        if not os.access(output_file_path, os.W_OK) and output_file_path.exists():
            non_writable_files.append(output_file_path)
    if len(non_writable_files) > 0:
        raise OSError(f"cannot write to {', '.join([f.name for f in non_writable_files])}")

    for name, subdir in subdirs.items():
        output_file_path = vrt_target_dir / f"{name}.vrt"
        # check the file first if we can write to it

        ds_path = dem_data_dir / subdir
        filelist = [str(f.resolve()) for f in ds_path.glob("*.tif")]
        logger.debug(f"Found {len(filelist)} files for {name} at {ds_path}.")
        logger.debug(f"Writing VRT to '{output_file_path.resolve()}'")
        src_nodata = "nan" if name == "slope" else 0
        opt = gdal.BuildVRTOptions(srcNodata=src_nodata, VRTNodata=0)
        gdal.BuildVRT(str(output_file_path.resolve()), filelist, options=opt)

    logger.debug(f"Creation of VRT took {time.time() - start_time:.2f}s")
