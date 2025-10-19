"""Download of the s2 mgrs based grid."""

import io
import logging
import zipfile
from pathlib import Path

import requests
from stopuhr import stopwatch

logger = logging.getLogger(__name__.replace("darts_", "darts."))


@stopwatch.f("Downloading and extracting zip file", printer=logger.debug)
def _download_zip(url: str, grid_dir: Path):
    response = requests.get(url)

    # Get the downloaded data as a byte string
    data = response.content
    logger.debug(f"Downloaded {len(data)} bytes")

    # Create a bytesIO object
    with io.BytesIO(data) as buffer:
        # Create a zipfile.ZipFile object and extract the files to a directory
        grid_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(buffer, "r") as zip_ref:
            # Extract the files to the specified directory
            zip_ref.extractall(grid_dir)


@stopwatch.f("Downloading Sentinel 2 grid", printer=logger.debug)
def download_sentinel_2_grid(grid_dir: Path):
    """Download the Sentinel 2 grid files.

    Files will be stored under [grid_dir]/adm1.shp and [grid_dir]/...

    Args:
        grid_dir (Path): The path to the grid.

    """
    grid_dir.mkdir(exist_ok=True, parents=True)
    grid_url = "https://github.com/justinelliotmeyers/Sentinel-2-Shapefile-Index/archive/refs/heads/master.zip"
    logger.debug(f"Downloading {grid_url} to {grid_dir.resolve()}")
    _download_zip(grid_url, grid_dir)
