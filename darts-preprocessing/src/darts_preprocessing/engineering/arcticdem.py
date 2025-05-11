"""Computation of ArcticDEM derived products."""

import logging
from math import ceil

import stopuhr
import xarray as xr
from xrspatial import aspect, convolution, curvature, hillshade, slope
from xrspatial.utils import has_cuda_and_cupy

logger = logging.getLogger(__name__.replace("darts_", "darts."))

if has_cuda_and_cupy():
    import cupy as cp  # type: ignore
    import cupy_xarray  # noqa: F401 # type: ignore


@stopuhr.funkuhr("Calculating TPI", printer=logger.debug, print_kwargs=["outer_radius", "inner_radius"])
def calculate_topographic_position_index(arcticdem_ds: xr.Dataset, outer_radius: int, inner_radius: int) -> xr.Dataset:
    """Calculate the Topographic Position Index (TPI) from an ArcticDEM Dataset.

    Args:
        arcticdem_ds (xr.Dataset): The ArcticDEM Dataset containing the 'dem' variable.
        outer_radius (int, optional): The outer radius of the annulus kernel in m.
        inner_radius (int, optional): The inner radius of the annulus kernel in m.

    Returns:
        xr.Dataset: The input Dataset with the calculated TPI added as a new variable 'tpi'.

    """
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

    return arcticdem_ds


@stopuhr.funkuhr("Calculating slope", printer=logger.debug)
def calculate_slope(arcticdem_ds: xr.Dataset) -> xr.Dataset:
    """Calculate the slope of the terrain surface from an ArcticDEM Dataset.

    Args:
        arcticdem_ds (xr.Dataset): The ArcticDEM Dataset containing the 'dem' variable.

    Returns:
        xr.Dataset: The input Dataset with the calculated slope added as a new variable 'slope'.

    """
    slope_deg = slope(arcticdem_ds.dem)
    slope_deg.attrs = {
        "long_name": "Slope",
        "units": "degrees",
        "description": "The slope of the terrain surface in degrees.",
        "source": "ArcticDEM",
        "_FillValue": float("nan"),
    }
    arcticdem_ds["slope"] = slope_deg.compute()
    return arcticdem_ds


@stopuhr.funkuhr("Calculating hillshade", printer=logger.debug)
def calculate_hillshade(arcticdem_ds: xr.Dataset) -> xr.Dataset:
    """Calculate the hillshade of the terrain surface from an ArcticDEM Dataset.

    Args:
        arcticdem_ds (xr.Dataset): The ArcticDEM Dataset containing the 'dem' variable.

    Returns:
        xr.Dataset: The input Dataset with the calculated slhillshadeope added as a new variable 'hillshade'.

    """
    hillshade_da = hillshade(arcticdem_ds.dem)
    hillshade_da.attrs = {
        "long_name": "Hillshade",
        "units": "",
        "description": "The hillshade based on azimuth 255 and angle_altitude 25.",
        "source": "ArcticDEM",
        "_FillValue": float("nan"),
    }
    arcticdem_ds["hillshade"] = hillshade_da.compute()
    return arcticdem_ds


@stopuhr.funkuhr("Calculating aspect", printer=logger.debug)
def calculate_aspect(arcticdem_ds: xr.Dataset) -> xr.Dataset:
    """Calculate the aspect of the terrain surface from an ArcticDEM Dataset.

    Args:
        arcticdem_ds (xr.Dataset): The ArcticDEM Dataset containing the 'dem' variable.

    Returns:
        xr.Dataset: The input Dataset with the calculated aspect added as a new variable 'aspect'.

    """
    aspect_deg = aspect(arcticdem_ds.dem)
    aspect_deg.attrs = {
        "long_name": "Aspect",
        "units": "degrees",
        "description": "The compass direction that the slope faces, in degrees clockwise from north.",
        "source": "ArcticDEM",
        "_FillValue": float("nan"),
    }
    arcticdem_ds["aspect"] = aspect_deg.compute()
    return arcticdem_ds


@stopuhr.funkuhr("Calculating curvature", printer=logger.debug)
def calculate_curvature(arcticdem_ds: xr.Dataset) -> xr.Dataset:
    """Calculate the curvature of the terrain surface from an ArcticDEM Dataset.

    Args:
        arcticdem_ds (xr.Dataset): The ArcticDEM Dataset containing the 'dem' variable.

    Returns:
        xr.Dataset: The input Dataset with the calculated curvature added as a new variable 'curvature'.

    """
    curvature_da = curvature(arcticdem_ds.dem)
    curvature_da.attrs = {
        "long_name": "Curvature",
        "units": "",
        "description": "The curvature of the terrain surface.",
        "source": "ArcticDEM",
        "_FillValue": float("nan"),
    }
    arcticdem_ds["curvature"] = curvature_da.compute()
    return arcticdem_ds
