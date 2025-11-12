"""Downloading and loading related functions for the Zarr-Datacube approach."""

import logging
from pathlib import Path
from typing import Literal

import geopandas as gpd
import odc.stac
import smart_geocubes
import xarray as xr
from odc.geo.geobox import GeoBox
from stopuhr import stopwatch

logger = logging.getLogger(__name__.replace("darts_", "darts."))


RESOLUTIONS = Literal[2, 10, 32]


def _validate_and_get_accessor(
    data_dir: Path | str,
    resolution: RESOLUTIONS,
) -> smart_geocubes.ArcticDEM2m | smart_geocubes.ArcticDEM10m | smart_geocubes.ArcticDEM32m:
    data_dir = Path(data_dir)
    assert ".icechunk" == data_dir.suffix, f"Data directory {data_dir} must have an .icechunk suffix!"

    match resolution:
        case 2:
            assert "2m" in data_dir.stem and "32m" not in data_dir.stem, (
                f"Data directory {data_dir} must have a '2m' in the name!"
            )
            accessor = smart_geocubes.ArcticDEM2m(data_dir)
        case 10:
            assert "10m" in data_dir.stem, f"Data directory {data_dir} must have a '10m' in the name!"
            accessor = smart_geocubes.ArcticDEM10m(data_dir)
        case 32:
            assert "32m" in data_dir.stem, f"Data directory {data_dir} must have a '32m' in the name!"
            accessor = smart_geocubes.ArcticDEM32m(data_dir)
        case _:
            raise ValueError(f"Resolution {resolution} not supported, only 2m, 10m and 32m are supported")
    accessor.assert_created()
    return accessor


@stopwatch.f("Loading ArcticDEM", printer=logger.debug, print_kwargs=["data_dir", "resolution", "buffer", "offline"])
def load_arcticdem(
    geobox: GeoBox,
    data_dir: Path | str,
    resolution: RESOLUTIONS,
    buffer: int = 0,
    offline: bool = False,
) -> xr.Dataset:
    """Load the ArcticDEM for the given geobox, fetch new data from the STAC server if necessary.

    This function loads ArcticDEM elevation data from a local icechunk store. If `offline=False`,
    missing data will be automatically downloaded from the AWS-hosted STAC server and stored
    locally for future use. The loaded data is returned in the ArcticDEM's native CRS (EPSG:3413).

    Args:
        geobox (GeoBox): The geobox for which the tile should be loaded. Must be in a meter-based CRS.
        data_dir (Path | str): Path to the icechunk data directory (must have .icechunk suffix).
            This directory stores downloaded ArcticDEM data for faster consecutive access.
        resolution (Literal[2, 10, 32]): The resolution of the ArcticDEM data in meters.
            Must match the resolution indicated in the data_dir name (e.g., "arcticdem_2m.icechunk").
        buffer (int, optional): Buffer around the geobox in pixels. The buffer is applied in the
            ArcticDEM's native CRS (EPSG:3413) after reprojecting the input geobox. Useful for
            edge effect removal in terrain analysis. Defaults to 0.
        offline (bool, optional): If True, only loads data already present in the local store
            without attempting any downloads. If False, missing data is downloaded from AWS.
            Defaults to False.

    Returns:
        xr.Dataset: The ArcticDEM dataset with the following data variables:
            - dem (float32): Elevation values in meters, clipped to [-100, 3000] range
            - arcticdem_data_mask (uint8): Data validity mask (1=valid, 0=invalid)

            The dataset is in the ArcticDEM's native CRS (EPSG:3413) with the buffer applied.
            It is NOT automatically reprojected to match the input geobox's CRS and resolution.

    Note:
        The `offline` parameter controls data fetching behavior:

        - When `offline=False`: Uses `smart_geocubes` accessor's `load()` method which automatically
          downloads missing tiles from AWS and persists them to the icechunk store.
        - When `offline=True`: Uses the accessor's `open_xarray()` method to open the existing store
          and crops it to the requested region. Raises an error if data is missing.

    Warning:
        - The input geobox must be in a meter-based CRS.
        - The data_dir must have an `.icechunk` suffix and contain the resolution in the name.
        - The returned dataset is in EPSG:3413, not the input geobox's CRS.

    Example:
        Load ArcticDEM with a buffer for terrain analysis:

        ```python
        from math import ceil, sqrt
        from darts_acquisition import load_arcticdem

        # Assume "optical" is a loaded Sentinel-2 dataset
        arcticdem = load_arcticdem(
            geobox=optical.odc.geobox,
            data_dir="/data/arcticdem_2m.icechunk",
            resolution=2,
            buffer=ceil(128 / 2 * sqrt(2)),  # Buffer for TPI with 128m radius
            offline=False
        )

        # Reproject to match optical data's CRS and resolution
        arcticdem = arcticdem.odc.reproject(optical.odc.geobox, resampling="cubic")
        ```

    """
    if not offline:
        odc.stac.configure_rio(cloud_defaults=True, aws={"aws_unsigned": True})

    accessor = _validate_and_get_accessor(data_dir, resolution)

    if not offline:
        arcticdem = accessor.load(geobox, buffer=buffer, persist=True)
    else:
        xrcube = accessor.open_xarray()
        reference_geobox = geobox.to_crs(accessor.extent.crs, resolution=accessor.extent.resolution.x).pad(buffer)
        arcticdem = xrcube.odc.crop(reference_geobox.extent, apply_mask=False)
        arcticdem = arcticdem.load()

    # Change dtype of the datamask to uint8 for later reproject_match
    arcticdem["arcticdem_data_mask"] = arcticdem.datamask.astype("uint8")

    # Clip values to -100, 3000 range (see docs about bands)
    arcticdem["dem"] = arcticdem["dem"].clip(-100, 3000)

    # Change dtype of arcticdem to float32 to save memory (original is float64)
    arcticdem["dem"] = arcticdem["dem"].astype("float32")

    return arcticdem


@stopwatch.f("Downloading ArcticDEM", printer=logger.debug, print_kwargs=["data_dir", "resolution"])
def download_arcticdem(
    aoi: gpd.GeoDataFrame,
    data_dir: Path | str,
    resolution: RESOLUTIONS,
) -> None:
    """Download ArcticDEM data for the specified area of interest.

    This function downloads ArcticDEM elevation tiles from AWS S3 for the given area
    of interest and stores them in a local icechunk data store for efficient access.

    Args:
        aoi (gpd.GeoDataFrame): Area of interest for which to download ArcticDEM data.
            Can be in any CRS; will be reprojected to EPSG:3413 (ArcticDEM's native CRS).
        data_dir (Path | str): Path to the icechunk data directory (must have .icechunk suffix).
            Must contain the resolution in the name (e.g., "arcticdem_2m.icechunk").
        resolution (Literal[2, 10, 32]): The resolution of the ArcticDEM data in meters.
            Must match the resolution indicated in the data_dir name.

    Note:
        This function automatically configures AWS access with unsigned requests to the
        public ArcticDEM S3 bucket. No AWS credentials are required.

    Example:
        Download ArcticDEM for a study area:

        ```python
        import geopandas as gpd
        from shapely.geometry import box
        from darts_acquisition import download_arcticdem

        # Define area of interest
        aoi = gpd.GeoDataFrame(
            geometry=[box(-50, 70, -49, 71)],
            crs="EPSG:4326"
        )

        # Download 2m resolution ArcticDEM
        download_arcticdem(
            aoi=aoi,
            data_dir="/data/arcticdem_2m.icechunk",
            resolution=2
        )
        ```

    """
    odc.stac.configure_rio(cloud_defaults=True, aws={"aws_unsigned": True})
    accessor = _validate_and_get_accessor(data_dir, resolution)
    accessor.download(aoi)
