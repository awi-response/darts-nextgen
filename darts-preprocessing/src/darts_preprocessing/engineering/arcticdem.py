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

    TPI measures the relative topographic position of a point by comparing its elevation to
    the mean elevation of the surrounding neighborhood. Positive values indicate higher
    positions (ridges), negative values indicate lower positions (valleys).

    Args:
        arcticdem_ds (xr.Dataset): The ArcticDEM Dataset containing the 'dem' variable (float32).
        outer_radius (int): The outer radius of the neighborhood in meters.
            Can also be specified as string with units (e.g., "100m" or "10px").
        inner_radius (int): The inner radius of the annulus kernel in meters.
            If > 0, creates an annulus (ring) instead of a circle. Set to 0 for a circular kernel.
            Can also be specified as string with units (e.g., "50m" or "5px").

    Returns:
        xr.Dataset: The input Dataset with a new data variable added:

        - tpi (float32): Topographic Position Index values.

            - long_name: "Topographic Position Index (TPI)"
            - description: Details about the kernel used

    Note:
        Kernel shape combinations:

        - inner_radius=0: Circular kernel comparing each cell to all neighbors within outer_radius
        - inner_radius>0: Annulus kernel comparing each cell to neighbors in a ring between
          inner_radius and outer_radius. Useful for multi-scale terrain analysis.

        The actual radii used are rounded to the nearest pixel based on the DEM resolution.

    Example:
        Calculate TPI with circular and annulus kernels:

        ```python
        from darts_preprocessing import calculate_topographic_position_index

        # Circular kernel (100m radius)
        arcticdem_with_tpi = calculate_topographic_position_index(
            arcticdem_ds=arcticdem,
            outer_radius=100,
            inner_radius=0
        )

        # Annulus kernel (50-100m ring)
        arcticdem_multi_scale = calculate_topographic_position_index(
            arcticdem_ds=arcticdem,
            outer_radius=100,
            inner_radius=50
        )
        ```

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
    """Calculate slope of the terrain surface from an ArcticDEM Dataset.

    Slope represents the rate of change of elevation, indicating terrain steepness.

    Args:
        arcticdem_ds (xr.Dataset): Dataset containing:
            - dem (float32): Digital Elevation Model

    Returns:
        xr.Dataset: Input Dataset with new data variable added:

        - slope (float32): Slope in degrees [0-90].

            - long_name: "Slope"
            - units: "degrees"
            - source: "ArcticDEM"

    Note:
        Slope is calculated using finite difference methods on the DEM.
        Values approaching 90° indicate near-vertical terrain.

    Example:
        ```python
        from darts_preprocessing import calculate_slope

        arcticdem_with_slope = calculate_slope(arcticdem_ds)

        # Mask steep terrain
        steep_areas = arcticdem_with_slope.slope > 30
        ```

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
    """Calculate hillshade of the terrain surface from an ArcticDEM Dataset.

    Hillshade simulates illumination of terrain from a specified sun position, useful
    for visualization and terrain analysis.

    Args:
        arcticdem_ds (xr.Dataset): Dataset containing:
            - dem (float32): Digital Elevation Model
        azimuth (int, optional): Light source azimuth in degrees clockwise from north [0-360].
            Defaults to 225 (southwest).
        angle_altitude (int, optional): Light source altitude angle in degrees above horizon [0-90].
            Defaults to 25.

    Returns:
        xr.Dataset: Input Dataset with new data variable added:

        - hillshade (float32): Illumination values [0-255], where 0 is shadow and 255 is fully lit.

            - long_name: "Hillshade"
            - description: Documents azimuth and angle_altitude used
            - source: "ArcticDEM"

    Note:
        Common azimuth/altitude combinations:

        - 315°/45°: Classic northwest illumination (default for many GIS applications)
        - 225°/25°: Southwest with low sun (better for visualizing subtle features)

        The hillshade calculation accounts for both slope and aspect of the terrain.

    Example:
        ```python
        from darts_preprocessing import calculate_hillshade

        # Default southwest illumination
        arcticdem_with_hs = calculate_hillshade(arcticdem_ds)

        # Custom sun position
        arcticdem_custom = calculate_hillshade(
            arcticdem_ds,
            azimuth=315,
            angle_altitude=45
        )
        ```

    """
    x, y = arcticdem_ds.x.mean().item(), arcticdem_ds.y.mean().item()
    correction_offset = np.arctan2(x, y) * (180 / np.pi) + 90
    azimuth_corrected = (azimuth - correction_offset + 360) % 360

    hillshade_da = hillshade(arcticdem_ds.dem, azimuth=azimuth_corrected, angle_altitude=angle_altitude)
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
    """Calculate aspect (compass direction) of the terrain surface from an ArcticDEM Dataset.

    Aspect indicates the downslope direction of the maximum rate of change in elevation.

    Args:
        arcticdem_ds (xr.Dataset): Dataset containing:
            - dem (float32): Digital Elevation Model

    Returns:
        xr.Dataset: Input Dataset with new data variable added:

        - aspect (float32): Aspect in degrees clockwise from north [0-360], or -1 for flat areas.

            - long_name: "Aspect"
            - units: "degrees"
            - description: Compass direction of slope
            - source: "ArcticDEM"

    Note:
        Aspect values:

        - 0° or 360°: North-facing
        - 90°: East-facing
        - 180°: South-facing
        - 270°: West-facing
        - -1: Flat (no dominant direction)

    Example:
        ```python
        from darts_preprocessing import calculate_aspect

        arcticdem_with_aspect = calculate_aspect(arcticdem_ds)

        # Identify south-facing slopes (135-225 degrees)
        south_facing = (arcticdem_with_aspect.aspect > 135) & (arcticdem_with_aspect.aspect < 225)
        ```

    """
    aspect_deg = aspect(arcticdem_ds.dem)

    # Aspect is always calculated in the projection - thus "north" is rather an "up"
    # To get the true north, we need to correct the aspect based on the coordinates
    x = arcticdem_ds.x.expand_dims({"y": arcticdem_ds.y})
    y = arcticdem_ds.y.expand_dims({"x": arcticdem_ds.x})
    if arcticdem_ds.cupy.is_cupy:
        x = cp.asarray(x)
        y = cp.asarray(y)
        correction_offset = cp.arctan2(x, y) * (180 / np.pi) + 90
    else:
        correction_offset = np.arctan2(x, y) * (180 / np.pi) + 90
    aspect_deg = (aspect_deg + correction_offset) % 360

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
    """Calculate curvature of the terrain surface from an ArcticDEM Dataset.

    Curvature measures the rate of change of slope, indicating terrain convexity or concavity.

    Args:
        arcticdem_ds (xr.Dataset): Dataset containing:
            - dem (float32): Digital Elevation Model

    Returns:
        xr.Dataset: Input Dataset with new data variable added:

        - curvature (float32): Curvature values.

            - long_name: "Curvature"
            - description: Rate of change of slope
            - source: "ArcticDEM"

    Note:
        Curvature interpretation:

        - Positive values: Convex terrain (hills, ridges)
        - Negative values: Concave terrain (valleys, depressions)
        - Near zero: Planar terrain

    Example:
        ```python
        from darts_preprocessing import calculate_curvature

        arcticdem_with_curv = calculate_curvature(arcticdem_ds)

        # Identify ridges (convex areas)
        ridges = arcticdem_with_curv.curvature > 0.1
        ```

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

    TRI quantifies topographic heterogeneity by measuring elevation differences between
    a cell and its surrounding cells. Higher values indicate more rugged terrain.

    Args:
        arcticdem_ds (xr.Dataset): Dataset containing:
            - dem (float32): Digital Elevation Model
        neighborhood_size (int): Neighborhood window size for TRI calculation.
            Can be specified as string with units (e.g., "100m" or "10px").

    Returns:
        xr.Dataset: Input Dataset with new data variable added:

        - tri (float32): Terrain Ruggedness Index in meters.

            - long_name: "Terrain Ruggedness Index"
            - units: "m"
            - description: Documents kernel size used
            - source: "ArcticDEM"

    Note:
        TRI methodology from Riley et al (1999):

        1. Measures elevation difference from center cell to 8 surrounding cells
        2. Squares and averages these differences
        3. Takes square root for final TRI value

        The neighborhood_size parameter controls the kernel size. A square kernel is used,
        with the actual size rounded to the nearest pixel based on DEM resolution.

    References:
        Riley, S.J., DeGloria, S.D., Elliot, R., 1999.
        A Terrain Ruggedness Index That Quantifies Topographic Heterogeneity.
        Intermountain Journal of Sciences, vol. 5, No. 1-4, pp. 23-27.

    Example:
        ```python
        from darts_preprocessing import calculate_terrain_ruggedness_index

        # Calculate TRI with 100m neighborhood
        arcticdem_with_tri = calculate_terrain_ruggedness_index(
            arcticdem_ds=arcticdem,
            neighborhood_size=100
        )

        # Identify highly rugged terrain
        rugged = arcticdem_with_tri.tri > 10
        ```

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

    VRM quantifies terrain ruggedness using vector analysis of slope and aspect, providing
    a measure independent of absolute elevation. Values range from 0 (smooth) to 1 (rugged).

    Args:
        arcticdem_ds (xr.Dataset): Dataset containing:
            - dem (float32): Digital Elevation Model
            - slope (float32): Slope in degrees (will be calculated if not present)
            - aspect (float32): Aspect in degrees (will be calculated if not present)
        neighborhood_size (int): Neighborhood window size for VRM calculation.
            Can be specified as string with units (e.g., "100m" or "10px").

    Returns:
        xr.Dataset: Input Dataset with new data variable added:

        - vrm (float32): Vector Ruggedness Measure [0-1].

            - long_name: "Vector Ruggedness Measure"
            - description: Documents neighborhood size used
            - source: "ArcticDEM"

    Note:
        VRM calculation:

        1. Converts slope and aspect to 3D unit vectors (x, y, z components)
        2. Sums vectors in the neighborhood window
        3. Calculates magnitude of resultant vector
        4. VRM = 1 - resultant magnitude

        Flat areas (aspect = -1) are handled by setting aspect to 0.

        Requires slope and aspect to be already calculated on the dataset.

    References:
        Sappington, J.M., K.M. Longshore, and D.B. Thomson. 2007.
        Quantifying Landscape Ruggedness for Animal Habitat Analysis: A case Study Using Bighorn Sheep
        in the Mojave Desert. Journal of Wildlife Management. 71(5): 1419-1426.

    Example:
        ```python
        from darts_preprocessing import (
            calculate_slope, calculate_aspect,
            calculate_vector_ruggedness_measure
        )

        # VRM requires slope and aspect
        arcticdem = calculate_slope(arcticdem)
        arcticdem = calculate_aspect(arcticdem)
        arcticdem_with_vrm = calculate_vector_ruggedness_measure(
            arcticdem_ds=arcticdem,
            neighborhood_size=100
        )
        ```

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

    DI measures the degree to which a landscape has been cut by valleys and ravines.
    Values range from 0 (smooth, undissected) to 1 (highly dissected).

    Args:
        arcticdem_ds (xr.Dataset): Dataset containing:
            - dem (float32): Digital Elevation Model
        neighborhood_size (int): Neighborhood window size for DI calculation.
            Can be specified as string with units (e.g., "100m" or "10px").

    Returns:
        xr.Dataset: Input Dataset with new data variable added:

        - di (float32): Dissection Index [0-1].

            - long_name: "Dissection Index"
            - description: Documents neighborhood size used
            - source: "ArcticDEM"

    Note:
        The dissection index quantifies landscape dissection by comparing elevation
        ranges within the neighborhood window. Higher values indicate more deeply
        incised terrain with greater vertical relief.

        The neighborhood_size parameter is converted to pixels based on DEM resolution.

    Example:
        ```python
        from darts_preprocessing import calculate_dissection_index

        # Calculate DI with 100m neighborhood
        arcticdem_with_di = calculate_dissection_index(
            arcticdem_ds=arcticdem,
            neighborhood_size=100
        )

        # Identify highly dissected terrain
        dissected = arcticdem_with_di.di > 0.5
        ```

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
