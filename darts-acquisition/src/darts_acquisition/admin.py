"""Download of admin level files for the regions."""

import io
import logging
import time
import zipfile
from pathlib import Path

import requests

logger = logging.getLogger(__name__.replace("darts_", "darts."))


def _download_zip(url: str, admin_dir: Path):
    tick_fstart = time.perf_counter()
    response = requests.get(url)

    # Get the downloaded data as a byte string
    data = response.content

    tick_download = time.perf_counter()
    logger.debug(f"Downloaded {len(data)} bytes in {tick_download - tick_fstart:.2f} seconds")

    # Create a bytesIO object
    with io.BytesIO(data) as buffer:
        # Create a zipfile.ZipFile object and extract the files to a directory
        admin_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(buffer, "r") as zip_ref:
            # Extract the files to the specified directory
            zip_ref.extractall(admin_dir)

    tick_extract = time.perf_counter()
    logger.debug(f"Extraction completed in {tick_extract - tick_download:.2f} seconds")


def download_admin_files(admin_dir: Path):
    """Download the admin files for the regions.

    Files will be stored under [admin_dir]/adm1.shp and [admin_dir]/adm2.shp.

    Args:
        admin_dir (Path): The path to the admin files.

    """
    tick_fstart = time.perf_counter()

    # Download the admin files
    admin_1_url = "https://github.com/wmgeolab/geoBoundaries/raw/main/releaseData/CGAZ/geoBoundariesCGAZ_ADM1.zip"
    admin_2_url = "https://github.com/wmgeolab/geoBoundaries/raw/main/releaseData/CGAZ/geoBoundariesCGAZ_ADM2.zip"

    admin_dir.mkdir(exist_ok=True, parents=True)

    logger.debug(f"Downloading {admin_1_url} to {admin_dir.resolve()}")
    _download_zip(admin_1_url, admin_dir)

    logger.debug(f"Downloading {admin_2_url} to {admin_dir.resolve()}")
    _download_zip(admin_2_url, admin_dir)

    tick_fend = time.perf_counter()
    logger.info(f"Downloaded admin files in {tick_fend - tick_fstart:.2f} seconds")
