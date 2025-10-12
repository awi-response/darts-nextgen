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
    offline: bool = True,
) -> xr.Dataset:
    """Load the TCVIS for the given geobox, fetch new data from GEE if necessary.

    Args:
        geobox (GeoBox): The geobox to load the data for.
        data_dir (Path | str): The directory to store the downloaded data for faster access for consecutive calls.
        buffer (int, optional): The buffer around the geobox in pixels. Defaults to 0.
        offline (bool, optional): If True, will not attempt to download any missing data. Defaults to False.

    Returns:
        xr.Dataset: The TCVIS dataset.

    Usage:
        Since the API of the `load_tcvis` is based on GeoBox, one can load a specific ROI based on an existing Xarray DataArray:

        ```python
        import xarray as xr
        import odc.geo.xr

        from darts_aquisition import load_tcvis

        # Assume "optical" is an already loaded s2 based dataarray

        tcvis = load_tcvis(
            optical.odc.geobox,
            "/path/to/tcvis-parent-directory",
        )

        # Now we can for example match the resolution and extent of the optical data:
        tcvis = tcvis.odc.reproject(optical.odc.geobox, resampling="cubic")
        ```

    """  # noqa: E501
    assert ".icechunk" == data_dir.suffix, f"Data directory {data_dir} must have an .icechunk suffix!"
    accessor = smart_geocubes.TCTrend(data_dir, create_icechunk_storage=False)

    # We want to assume that the datacube is already created to be save in a multi-process environment
    accessor.assert_created()

    if not offline:
        tcvis = accessor.load(geobox, buffer=buffer, persist=True)
    else:
        xrcube = accessor.open_xarray()
        reference_geobox = geobox.to_crs(accessor.extent.crs, resolution=accessor.extent.resolution.x).pad(buffer)
        xrcube_aoi = xrcube.odc.crop(reference_geobox.extent, apply_mask=False)
        xrcube_aoi = xrcube_aoi.load()

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
    """Download the TCVIS for the given area of interest.

    Args:
        aoi (gpd.GeoDataFrame): The area of interest to download the TCVIS for.
        data_dir (Path | str): The directory to store the downloaded data for faster access for consecutive calls.

    """
    assert ".icechunk" == data_dir.suffix, f"Data directory {data_dir} must have an .icechunk suffix!"
    accessor = smart_geocubes.TCTrend(data_dir, create_icechunk_storage=False)
    accessor.assert_created()
    accessor.download(aoi)
