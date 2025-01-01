"""Computation of ArcticDEM derived products."""

import logging
import time
from math import ceil

import xarray as xr
from xrspatial import convolution, slope
from xrspatial.utils import has_cuda_and_cupy

logger = logging.getLogger(__name__.replace("darts_", "darts."))

if has_cuda_and_cupy():
    import cupy as cp
    import cupy_xarray  # noqa: F401


def calculate_topographic_position_index(arcticdem_ds: xr.Dataset, outer_radius: int, inner_radius: int) -> xr.Dataset:
    """Calculate the Topographic Position Index (TPI) from an ArcticDEM Dataset.

    Args:
        arcticdem_ds (xr.Dataset): The ArcticDEM Dataset containing the 'dem' variable.
        outer_radius (int, optional): The outer radius of the annulus kernel in m.
        inner_radius (int, optional): The inner radius of the annulus kernel in m.

    Returns:
        xr.Dataset: The input Dataset with the calculated TPI added as a new variable 'tpi'.

    """
    tick_fstart = time.perf_counter()
    cellsize_x, cellsize_y = convolution.calc_cellsize(arcticdem_ds.dem)  # Should be equal to the resolution of the DEM
    # Use an annulus kernel if inner_radius is greater than 0
    outer_radius_m = f"{outer_radius}m"
    outer_radius_px = f"{ceil(outer_radius / cellsize_x)}px"
    if inner_radius > 0:
        inner_radius_m = f"{inner_radius}m"
        inner_radius_px = f"{ceil(inner_radius / cellsize_x)}px"
        kernel = convolution.annulus_kernel(cellsize_x, cellsize_y, outer_radius_m, inner_radius_m)
        attr_cell_description = (
            f"within a ring at a distance of {inner_radius_px}-{outer_radius_px} cells "
            f"({inner_radius_m}-{outer_radius_m}) away from the focal cell."
        )
        logger.debug(
            f"Calculating Topographic Position Index with annulus kernel of "
            f"{inner_radius_px}-{outer_radius_px} ({inner_radius_m}-{outer_radius_m}) cells."
        )
    else:
        kernel = convolution.circle_kernel(cellsize_x, cellsize_y, outer_radius_m)
        attr_cell_description = (
            f"within a circle at a distance of {outer_radius_px} cells ({outer_radius_m}) away from the focal cell."
        )
        logger.debug(
            f"Calculating Topographic Position Index with circle kernel of {outer_radius_px} ({outer_radius_m}) cells."
        )

    if has_cuda_and_cupy() and arcticdem_ds.cupy.is_cupy:
        kernel = cp.asarray(kernel)

    tpi = arcticdem_ds.dem - convolution.convolution_2d(arcticdem_ds.dem, kernel) / kernel.sum()
    tpi.attrs = {
        "long_name": "Topographic Position Index",
        "units": "m",
        "description": "The difference between the elevation of a cell and the mean elevation of the surrounding"
        f"cells {attr_cell_description}",
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
