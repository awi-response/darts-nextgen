"""Landsat Trends related Data Loading. Should be used temporary and maybe moved to the acquisition package."""

import logging
from pathlib import Path

import geopandas as gpd
import smart_geocubes
import xarray as xr
from odc.geo.geobox import GeoBox
from stopuhr import stopwatch

logger = logging.getLogger(__name__.replace("darts_", "darts."))


@stopwatch.f("Loading TCVIS", printer=logger.debug, print_kwargs=["data_dir", "buffer", "offline"])
def load_tcvis(
    geobox: GeoBox,
    data_dir: Path | str,
    buffer: int = 0,
    offline: bool = False,
) -> xr.Dataset:
    """Load TCVIS (Tasseled Cap trends) for the given geobox, fetch new data from GEE if necessary.

    This function loads Tasseled Cap trend data from a local icechunk store. If `offline=False`,
    missing data will be automatically downloaded from Google Earth Engine and stored locally.
    The data contains temporal trends in brightness, greenness, and wetness derived from
    Landsat imagery.

    Args:
        geobox (GeoBox): The geobox for which to load the data. Can be in any CRS.
        data_dir (Path | str): Path to the icechunk data directory (must have .icechunk suffix).
            This directory stores downloaded TCVIS data for faster consecutive access.
        buffer (int, optional): Buffer around the geobox in pixels. The buffer is applied in the
            TCVIS dataset's native CRS after reprojecting the input geobox. Defaults to 0.
        offline (bool, optional): If True, only loads data already present in the local store
            without attempting any downloads. If False, missing data is downloaded from GEE.
            Defaults to False.

    Returns:
        xr.Dataset: The TCVIS dataset with the following data variables:
            - tc_brightness (float): Temporal trend in Tasseled Cap brightness component
            - tc_greenness (float): Temporal trend in Tasseled Cap greenness component
            - tc_wetness (float): Temporal trend in Tasseled Cap wetness component

            The dataset is in the TCVIS native CRS with the buffer applied.
            It is NOT automatically reprojected to match the input geobox's CRS.

    Note:
        The `offline` parameter controls data fetching behavior:

        - When `offline=False`: Uses `smart_geocubes` accessor's `load()` method which automatically
          downloads missing tiles from GEE and persists them to the icechunk store.
        - When `offline=True`: Uses the accessor's `open_xarray()` method to open the existing store
          and crops it to the requested region. Raises an error if data is missing.

        Variable naming: The original TCB_slope, TCG_slope, and TCW_slope variables are renamed
        to follow DARTS conventions (tc_brightness, tc_greenness, tc_wetness).

    Example:
        Load TCVIS data aligned with optical imagery:

        ```python
        from darts_acquisition import load_tcvis

        # Assume "optical" is a loaded Sentinel-2 dataset
        tcvis = load_tcvis(
            geobox=optical.odc.geobox,
            data_dir="/data/tcvis.icechunk",
            buffer=0,
            offline=False
        )

        # Reproject to match optical data's CRS and resolution
        tcvis = tcvis.odc.reproject(optical.odc.geobox, resampling="cubic")
        ```

    """
    assert ".icechunk" == data_dir.suffix, f"Data directory {data_dir} must have an .icechunk suffix!"
    accessor = smart_geocubes.TCTrend(data_dir, create_icechunk_storage=False)

    # We want to assume that the datacube is already created to be save in a multi-process environment
    accessor.assert_created()

    if not offline:
        tcvis = accessor.load(geobox, buffer=buffer, persist=True)
    else:
        xrcube = accessor.open_xarray()
        reference_geobox = geobox.to_crs(accessor.extent.crs, resolution=accessor.extent.resolution.x).pad(buffer)
        tcvis = xrcube.odc.crop(reference_geobox.extent, apply_mask=False)
        tcvis = tcvis.load()

    # Rename to follow our conventions
    tcvis = tcvis.rename_vars(
        {
            "TCB_slope": "tc_brightness",
            "TCG_slope": "tc_greenness",
            "TCW_slope": "tc_wetness",
        }
    )

    return tcvis


@stopwatch.f("Downloading TCVIS", printer=logger.debug, print_kwargs=["data_dir"])
def download_tcvis(
    aoi: gpd.GeoDataFrame,
    data_dir: Path | str,
) -> None:
    """Download TCVIS (Tasseled Cap trends) data for the specified area of interest.

    This function downloads Tasseled Cap trend data from Google Earth Engine for the given
    area of interest and stores it in a local icechunk data store for efficient access.

    Args:
        aoi (gpd.GeoDataFrame): Area of interest for which to download TCVIS data.
            Can be in any CRS; will be reprojected to the TCVIS dataset's native CRS.
        data_dir (Path | str): Path to the icechunk data directory (must have .icechunk suffix).

    Note:
        Requires Google Earth Engine authentication to be set up before calling this function.
        Use `ee.Initialize()` or `ee.Authenticate()` as needed.

    Example:
        Download TCVIS for a study area:

        ```python
        import geopandas as gpd
        from shapely.geometry import box
        from darts_acquisition import download_tcvis

        # Define area of interest
        aoi = gpd.GeoDataFrame(
            geometry=[box(-50, 70, -49, 71)],
            crs="EPSG:4326"
        )

        # Download TCVIS
        download_tcvis(
            aoi=aoi,
            data_dir="/data/tcvis.icechunk"
        )
        ```

    """
    assert ".icechunk" == data_dir.suffix, f"Data directory {data_dir} must have an .icechunk suffix!"
    accessor = smart_geocubes.TCTrend(data_dir, create_icechunk_storage=False)
    accessor.assert_created()
    accessor.download(aoi)
