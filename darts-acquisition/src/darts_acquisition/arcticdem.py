"""ArcticDEM related data loading."""

import io
import logging
import os
import time
import zipfile
from pathlib import Path

import requests

logger = logging.getLogger(__name__.replace("darts_", "darts."))


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

    # Create a bytesIO object
    buffer = io.BytesIO(data)

    # Create a zipfile.ZipFile object and extract the files to a directory
    dem_data_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(buffer, "r") as zip_ref:
        zip_ref.extractall(dem_data_dir)

    logger.info(f"Download completed in {time.time() - start:.2f} seconds")


def create_arcticdem_vrt(dem_data_dir: Path, vrt_target_dir: Path):  # noqa: C901
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

    # memorize if any files weren't written
    file_written = dict.fromkeys(subdirs.keys(), False)

    for name, subdir in subdirs.items():
        output_file_path = vrt_target_dir / f"{name}.vrt"
        # check the file first if we can write to it

        ds_path = dem_data_dir / subdir
        if not ds_path.exists():
            logger.warning(f"{ds_path.absolute()} does NOT exist!")
            continue

        filelist = [str(f.resolve()) for f in ds_path.glob("*.tif")]
        if len(filelist) < 1:
            logger.warning(f"NO files found in {ds_path.absolute()}")
            continue

        logger.debug(f"Found {len(filelist)} files for {name} at {ds_path}.")
        logger.debug(f"Writing VRT to '{output_file_path.resolve()}'")
        src_nodata = "nan" if name == "slope" else 0
        opt = gdal.BuildVRTOptions(srcNodata=src_nodata, VRTNodata=0)
        gdal.BuildVRT(str(output_file_path.resolve()), filelist, options=opt)
        file_written[name] = True

    for name, is_written in file_written.items():
        if not is_written:
            logger.warning(f"VRT file for {name} was NOT created.")

    logger.debug(f"Creation of VRT took {time.time() - start_time:.2f}s")
