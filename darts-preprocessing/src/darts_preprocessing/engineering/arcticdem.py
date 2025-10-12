"""Computation of ArcticDEM derived products."""

import logging
from dataclasses import dataclass
from math import ceil

import numpy as np
import xarray as xr
from stopuhr import stopwatch
from xrspatial import aspect, convolution, curvature, hillshade, slope
from xrspatial.utils import has_cuda_and_cupy

from darts_preprocessing.engineering.dissection_index import dissection_index

logger = logging.getLogger(__name__.replace("darts_", "darts."))

if has_cuda_and_cupy():
    import cupy as cp  # type: ignore
    import cupy_xarray  # type: ignore


@dataclass(frozen=True)
class Distance:
    """Convenience class to represent a distance in pixels and meters."""

    pixel: int
    meter: float

    def __repr__(self):  # noqa: D105
        return f"{self.pixel}px ({self.meter}m)"

    @classmethod
    def parse(cls, v: int | float | str, res: float) -> "Distance":
        """Parse a distance from a string or numeric value.

        If the input is a string, it can be in the format of "10px" or "10m".
        If it is a numeric value, it is interpreted as meters and converted to pixels based on the resolution.

        Args:
            v (int | float | str): The input distance value.
            res (float): The resolution in meters per pixel.

        Raises:
            ValueError: If the input distance is negative.
            ValueError: If the input distance is not a valid string format.
            TypeError: If the input distance is not a string, int, or float.

        Returns:
            Distance: The parsed distance in pixels and meters.

        """
        if isinstance(v, str):
            if v.endswith("px"):
                pixel = int(v[:-2])
                meter = pixel * res
            elif v.endswith("m"):
                meter = float(v[:-1])
                pixel = ceil(meter / res)
            else:
                raise ValueError(f"Invalid distance format: {v}")
        elif isinstance(v, (int, float)):
            if v < 0:
                raise ValueError("Distance cannot be negative")
            pixel = ceil(v / res)
            meter = pixel * res
        else:
            raise TypeError(f"Invalid type for distance: {type(v)}")
        return cls(pixel=pixel, meter=meter)


@stopwatch.f("Calculating TPI", printer=logger.debug, print_kwargs=["outer_radius", "inner_radius"])
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
    outer_radius: Distance = Distance.parse(outer_radius, cellsize_x)
    if inner_radius > 0:
        inner_radius: Distance = Distance.parse(inner_radius, cellsize_x)
        kernel = convolution.annulus_kernel(cellsize_x, cellsize_y, outer_radius.meter, inner_radius.meter)
        attr_cell_description = (
            f"within a ring at a distance of {inner_radius}-{outer_radius} cells away from the focal cell."
        )
        logger.debug(
            f"Calculating Topographic Position Index with annulus kernel of {inner_radius}-{outer_radius} cells."
        )
    else:
        kernel = convolution.circle_kernel(cellsize_x, cellsize_y, outer_radius.meter)
        attr_cell_description = f"within a circle at a distance of {outer_radius} cells away from the focal cell."
        logger.debug(f"Calculating Topographic Position Index with circle kernel of {outer_radius} cells.")

    # Change dtype of kernel to float32 since we don't need the precision and f32 is faster
    kernel = kernel.astype("float32")

    if has_cuda_and_cupy() and arcticdem_ds.cupy.is_cupy:
        kernel = cp.asarray(kernel)

    tpi = arcticdem_ds.dem - convolution.convolution_2d(arcticdem_ds.dem, kernel) / kernel.sum()
    tpi.attrs = {
        "long_name": "Topographic Position Index",
        "units": "m",
        "description": "The difference between the elevation of a cell and the mean elevation of the surrounding"
        f"cells {attr_cell_description}",
        "source": "ArcticDEM",
    }

    arcticdem_ds["tpi"] = tpi.compute()

    return arcticdem_ds


@stopwatch("Calculating slope", printer=logger.debug)
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
    }
    arcticdem_ds["slope"] = slope_deg.compute()
    return arcticdem_ds


@stopwatch.f("Calculating hillshade", printer=logger.debug, print_kwargs=["azimuth", "angle_altitude"])
def calculate_hillshade(arcticdem_ds: xr.Dataset, azimuth: int = 225, angle_altitude: int = 25) -> xr.Dataset:
    """Calculate the hillshade of the terrain surface from an ArcticDEM Dataset.

    Args:
        arcticdem_ds (xr.Dataset): The ArcticDEM Dataset containing the 'dem' variable.
        azimuth (int, optional): The azimuth angle of the light source in degrees. Defaults to 225.
        angle_altitude (int, optional): The altitude angle of the light source in degrees. Defaults to 25.

    Returns:
        xr.Dataset: The input Dataset with the calculated hillshade added as a new variable 'hillshade'.

    """
    hillshade_da = hillshade(arcticdem_ds.dem, azimuth=azimuth, angle_altitude=angle_altitude)
    hillshade_da.attrs = {
        "long_name": "Hillshade",
        "units": "",
        "description": f"The hillshade based on azimuth {azimuth} and angle_altitude {angle_altitude}.",
        "source": "ArcticDEM",
    }
    arcticdem_ds["hillshade"] = hillshade_da.compute()
    return arcticdem_ds


@stopwatch("Calculating aspect", printer=logger.debug)
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
    }
    arcticdem_ds["aspect"] = aspect_deg.compute()
    return arcticdem_ds


@stopwatch("Calculating curvature", printer=logger.debug)
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
    }
    arcticdem_ds["curvature"] = curvature_da.compute()
    return arcticdem_ds


@stopwatch("Calculating TRI", printer=logger.debug)
def calculate_terrain_ruggedness_index(arcticdem_ds: xr.Dataset, neighborhood_size: int) -> xr.Dataset:
    """Calculate the Terrain Ruggedness Index (TRI) from an ArcticDEM Dataset.

    Definition from ESRI:
    TRI expresses the amount of elevation difference between adjacent cells of a DEM.
    Using methodology developed by Riley et al (1999) and published in the paper
    “A Terrain ruggedness Index That Quantifies Topographic heterogeneity”,
    the tool measures the difference in elevation values from a center cell and eight cells directly surrounding it.
    Then, the eight elevation differences are squared and averaged.
    The square root of this average results is a TRI measurement for the center cell.
    This calculation is then conducted on every cell of the DEM.

    Args:
        arcticdem_ds (xr.Dataset): The ArcticDEM Dataset containing the 'dem' variable.
        neighborhood_size (int): The neighborhood_size in meters for the TRI calculation.

    Returns:
        xr.Dataset: The input Dataset with the calculated TRI added as a new variable 'tri'.

    """
    cellsize_x, _cellsize_y = convolution.calc_cellsize(arcticdem_ds.dem)

    neighborhood_size: Distance = Distance.parse(neighborhood_size, cellsize_x)

    kernel = np.ones((neighborhood_size.pixel, neighborhood_size.pixel), dtype=float)
    kernel[neighborhood_size.pixel // 2, neighborhood_size.pixel // 2] = 0  # Set the center cell to 0
    kernel = convolution.custom_kernel(kernel)
    logger.debug(f"Calculating Terrain Ruggedness Index with square kernel of radius {neighborhood_size} cells.")

    # Change dtype of kernel to float32 since we don't need the precision and f32 is faster
    kernel = kernel.astype("float32")

    if has_cuda_and_cupy() and arcticdem_ds.cupy.is_cupy:
        kernel = cp.asarray(kernel)

    # Kernel compute of TRI as described here:
    # https://sites.utexas.edu/utarima/files/2024/02/terrain_roughness_index.pdf
    dem_squared = arcticdem_ds.dem**2
    focal_sum = convolution.convolution_2d(arcticdem_ds.dem, kernel)
    focal_sum_squared = convolution.convolution_2d(dem_squared, kernel)
    tri = np.sqrt((kernel.size - 1) * dem_squared - 2 * arcticdem_ds.dem * focal_sum + focal_sum_squared)

    tri.attrs = {
        "long_name": "Terrain Ruggedness Index",
        "units": "m",
        "description": (
            "The difference between the elevation of a cell and the mean elevation of the surrounding"
            f" cells within a square kernel of radius {neighborhood_size} cells."
        ),
        "source": "ArcticDEM",
    }

    arcticdem_ds["tri"] = tri.compute()
    return arcticdem_ds


@stopwatch("Calculating Vector Ruggedness Measure", printer=logger.debug)
def calculate_vector_ruggedness_measure(arcticdem_ds: xr.Dataset, neighborhood_size: int) -> xr.Dataset:
    """Calculate the Vector Ruggedness Measure (VRM) from an ArcticDEM Dataset.

    Implementation of the vector ruggedness measure described in Sappington, J.M.,
    K.M. Longshore, and D.B. Thomson. 2007. Quantifying Landscape Ruggedness for
    Animal Habitat Analysis: A case Study Using Bighorn Sheep in the Mojave Desert.
    Journal of Wildlife Management. 71(5): 1419-1426.

    Args:
        arcticdem_ds (xr.Dataset): The ArcticDEM Dataset containing the 'dem' variable.
        neighborhood_size (int): The size of the neighborhood window (in meters) for the calculation.

    Returns:
        xr.Dataset: The input Dataset with the calculated VRM added as a new variable 'vrm'.

    """
    # Calculate slope and aspect
    slope_rad = arcticdem_ds.slope * (np.pi / 180)  # Convert to radians
    aspect_rad = arcticdem_ds.aspect * (np.pi / 180)  # Convert to radians

    # Calculate x, y, and z components of unit vectors
    xy = np.sin(slope_rad)
    z = np.cos(slope_rad)

    # Handle flat areas (where aspect = -1)
    if has_cuda_and_cupy() and arcticdem_ds.cupy.is_cupy:
        aspect_rad.variable._data = cp.where(aspect_rad.variable._data == -1, 0, aspect_rad.variable._data)
        # aspect_rad = aspect_rad.copy(data=aspect_rad_raw)
    else:
        aspect_rad = xr.where(aspect_rad == -1, 0, aspect_rad)
    x = np.sin(aspect_rad) * xy
    y = np.cos(aspect_rad) * xy

    # Get neighborhood_size in pixels
    neighborhood_size: Distance = Distance.parse(neighborhood_size, arcticdem_ds.odc.geobox.resolution.x)
    # Create convolution kernel for focal sum
    kernel = np.ones((neighborhood_size.pixel, neighborhood_size.pixel), dtype=float) / neighborhood_size.pixel**2
    kernel = convolution.custom_kernel(kernel)

    # Change dtype of kernel to float32 since we don't need the precision and f32 is faster
    kernel = kernel.astype("float32")

    if has_cuda_and_cupy() and arcticdem_ds.cupy.is_cupy:
        kernel = cp.asarray(kernel)

    logger.debug(f"Calculating Vector Ruggedness Measure with square kernel of size {neighborhood_size} cells.")

    # TODO: Write a custom kernel for this for speedup and smaller memory footprint
    # Calculate sums of x, y, and z components in the neighborhood
    x_sum = convolution.convolution_2d(x, kernel)
    y_sum = convolution.convolution_2d(y, kernel)
    z_sum = convolution.convolution_2d(z, kernel)

    # Calculate the resultant vector magnitude
    vrm = 1 - np.sqrt(x_sum**2 + y_sum**2 + z_sum**2)

    vrm.attrs = {
        "long_name": "Vector Ruggedness Measure",
        "units": "",
        "description": (
            f"Vector ruggedness measure calculated using a {neighborhood_size} neighborhood. "
            "Values range from 0 (smooth) to 1 (most rugged)."
        ),
        "source": "ArcticDEM",
    }

    arcticdem_ds["vrm"] = vrm.compute()
    return arcticdem_ds


@stopwatch("Calculating Dissection Index", printer=logger.debug)
def calculate_dissection_index(arcticdem_ds: xr.Dataset, neighborhood_size: int) -> xr.Dataset:
    """Calculate the Dissection Index (DI) from an ArcticDEM Dataset.

    Args:
        arcticdem_ds (xr.Dataset): The ArcticDEM Dataset containing the 'dem' variable.
        neighborhood_size (int): The size of the neighborhood window (in meters) for the calculation.

    Returns:
        xr.Dataset: The input Dataset with the calculated DI added as a new variable 'di'.

    """
    # Get neighborhood_size in pixels
    neighborhood_size: Distance = Distance.parse(neighborhood_size, arcticdem_ds.odc.geobox.resolution.x)

    di = dissection_index(arcticdem_ds.dem, window_size=neighborhood_size.pixel)

    di.attrs = {
        "long_name": "Dissection Index",
        "units": "",
        "description": (
            f"Dissection index calculated using a {neighborhood_size} neighborhood. "
            "Values range from 0 (smooth) to 1 (most rugged)."
        ),
        "source": "ArcticDEM",
    }

    arcticdem_ds["di"] = di.compute()
    return arcticdem_ds
