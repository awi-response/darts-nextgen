"""Computation of ArcticDEM derived products."""

import logging
import time

import xarray as xr
from xrspatial import convolution, slope

logger = logging.getLogger(__name__.replace("darts_", "darts."))


def calculate_topographic_position_index(
    arcticdem_ds: xr.Dataset, outer_radius: int = 30, inner_radius: int = 25
) -> xr.Dataset:
    """Calculate the Topographic Position Index (TPI) from an ArcticDEM Dataset.

    Args:
        arcticdem_ds (xr.Dataset): The ArcticDEM Dataset containing the 'dem' variable.
        outer_radius (int, optional): The outer radius of the annulus kernel in number of cells. Defaults to 30.
        inner_radius (int, optional): The inner radius of the annulus kernel in number of cells. Defaults to 25.

    Returns:
        xr.Dataset: The input Dataset with the calculated TPI added as a new variable 'tpi'.

    """
    tick_fstart = time.perf_counter()
    cellsize_x, cellsize_y = convolution.calc_cellsize(arcticdem_ds.dem)  # Should be equal to the resolution of the DEM
    # Use an annulus kernel with a ring at a distance from 25-30 cells away from focal point
    outer_radius_m = str(cellsize_x * outer_radius) + "m"
    inner_radius_m = str(cellsize_x * inner_radius) + "m"
    kernel = convolution.annulus_kernel(cellsize_x, cellsize_y, outer_radius_m, inner_radius_m)
    logger.debug(
        f"Calculating Topographic Position Index with annulus kernel of "
        f"{inner_radius}-{outer_radius} ({inner_radius_m}-{outer_radius_m}) cells."
    )

    tpi = arcticdem_ds.dem - convolution.convolution_2d(arcticdem_ds.dem, kernel) / kernel.sum()
    tpi.attrs = {
        "long_name": "Topographic Position Index",
        "units": "m",
        "description": "The difference between the elevation of a cell and the mean elevation of the surrounding"
        f"cells within a ring at a distance of {inner_radius}-{outer_radius} cells away from the"
        "focal cell.",
        "source": "ArcticDEM",
        "_FillValue": float("nan"),
    }

    arcticdem_ds["tpi"] = tpi.compute()

    tick_fend = time.perf_counter()
    logger.info(f"Topographic Position Index calculated in {tick_fend - tick_fstart:.2f} seconds.")
    return arcticdem_ds


def calculate_slope(arcticdem_ds: xr.Dataset) -> xr.Dataset:
    """Calculate the slope of the terrain surface from an ArcticDEM Dataset.

    Args:
        arcticdem_ds (xr.Dataset): The ArcticDEM Dataset containing the 'dem' variable.

    Returns:
        xr.Dataset: The input Dataset with the calculated slope added as a new variable 'slope'.

    """
    tick_fstart = time.perf_counter()
    logger.debug("Calculating slope of the terrain surface.")

    slope_deg = slope(arcticdem_ds.dem)
    slope_deg.attrs = {
        "long_name": "Slope",
        "units": "degrees",
        "description": "The slope of the terrain surface in degrees.",
        "source": "ArcticDEM",
        "_FillValue": float("nan"),
    }
    arcticdem_ds["slope"] = slope_deg.compute()

    tick_fend = time.perf_counter()
    logger.info(f"Slope calculated in {tick_fend - tick_fstart:.2f} seconds.")
    return arcticdem_ds
