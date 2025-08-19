"""Downloading and loading related functions for the Zarr-Datacube approach."""

import logging
from pathlib import Path
from typing import Literal

import odc.stac
import smart_geocubes
import xarray as xr
from odc.geo.geobox import GeoBox
from stopuhr import stopwatch

logger = logging.getLogger(__name__.replace("darts_", "darts."))


RESOLUTIONS = Literal[2, 10, 32]


@stopwatch.f("Loading ArcticDEM", printer=logger.debug, print_kwargs=["data_dir", "resolution", "buffer", "persist"])
def load_arcticdem(
    geobox: GeoBox, data_dir: Path | str, resolution: RESOLUTIONS, buffer: int = 0, persist: bool = True
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

    Raises:
        ValueError: If the resolution is not supported.

    """  # noqa: E501
    odc.stac.configure_rio(cloud_defaults=True, aws={"aws_unsigned": True})

    match resolution:
        case 2:
            accessor = smart_geocubes.ArcticDEM2m(data_dir)
        case 10:
            accessor = smart_geocubes.ArcticDEM10m(data_dir)
        case 32:
            accessor = smart_geocubes.ArcticDEM32m(data_dir)
        case _:
            raise ValueError(f"Resolution {resolution} not supported, only 2m, 10m and 32m are supported")

    accessor.assert_created()

    arcticdem = accessor.load(geobox, buffer=buffer, persist=persist)

    # Change dtype of the datamask to uint8 for later reproject_match
    arcticdem["datamask"] = arcticdem.datamask.astype("uint8")

    # Clip values to -100, 3000 range (see docs about bands)
    arcticdem["dem"] = arcticdem["dem"].clip(-100, 3000)

    return arcticdem
