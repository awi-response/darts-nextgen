"""PLANET scene based preprocessing."""

import logging
from math import isnan
from typing import Literal

import odc.geo.xr
import xarray as xr
from darts_utils.cuda import DEFAULT_DEVICE, move_to_device, move_to_host
from stopuhr import stopwatch

from darts_preprocessing.engineering.arcticdem import (
    calculate_aspect,
    calculate_curvature,
    calculate_hillshade,
    calculate_slope,
    calculate_topographic_position_index,
)
from darts_preprocessing.engineering.indices import calculate_ndvi

logger = logging.getLogger(__name__.replace("darts_", "darts."))


# TODO: Find a better abstraction for GPU / CPU processing
# Combine it with persisting Stuff on the GPU for later inference
# This is currently blocked because the arcticdem needs to be cropped after the processing happened
# Otherwise it should be possible to do something like this:
# - let the output of the proprocessing stay on the GPU
# - when the data is converted into a torch tensor use the cupy-torch interop: https://docs.cupy.dev/en/stable/user_guide/interoperability.html#pytorch


def get_azimuth_and_elevation(ds_optical: xr.Dataset) -> tuple[float, float]:
    """Get the azimuth and elevation from the optical dataset attributes.

    Args:
        ds_optical (xr.Dataset): The optical dataset.

    Returns:
        tuple[float, float]: The azimuth and elevation.

    """
    azimuth = ds_optical.attrs.get("azimuth", float("nan"))
    elevation = ds_optical.attrs.get("elevation", float("nan"))
    if isnan(azimuth):
        azimuth = 225
        logger.warning("No azimuth found in optical dataset attributes. Using default value of 225 degrees.")
    if isnan(elevation):
        elevation = 25
        logger.warning("No sun elevation found in optical dataset attributes. Using default value of 25 degrees.")
    if not isinstance(azimuth, (int, float)):
        azimuth = 225
        logger.warning(
            f"Azimuth found in optical dataset is {azimuth}, which is not a number. Using default value of 225 degrees."
        )
    if not isinstance(elevation, (int, float)):
        elevation = 25
        logger.warning(
            f"Sun elevation found in optical dataset is {elevation}, which is not a number."
            " Using default value of 25 degrees."
        )

    azimuth = round(azimuth)
    elevation = round(elevation)
    return azimuth, elevation


@stopwatch.f(
    "Preprocessing arcticdem",
    printer=logger.debug,
    print_kwargs=["tpi_outer_radius", "tpi_inner_radius", "azimuth", "angle_altitude"],
)
def preprocess_arcticdem(
    ds_arcticdem: xr.Dataset,
    tpi_outer_radius: int,
    tpi_inner_radius: int,
    azimuth: int,
    angle_altitude: int,
) -> xr.Dataset:
    """Preprocess the ArcticDEM data with mdoern (DARTS v2) preprocessing steps.

    Args:
        ds_arcticdem (xr.Dataset): The ArcticDEM dataset.
        tpi_outer_radius (int): The outer radius of the annulus kernel for the tpi calculation in number of cells.
        tpi_inner_radius (int): The inner radius of the annulus kernel for the tpi calculation in number of cells.
        azimuth (int): The azimuth angle of the light source in degrees for hillshade calculation.
        angle_altitude (int): The altitude angle of the light source in degrees for hillshade

    Returns:
        xr.Dataset: The preprocessed ArcticDEM dataset.

    """
    ds_arcticdem = calculate_topographic_position_index(ds_arcticdem, tpi_outer_radius, tpi_inner_radius)
    ds_arcticdem = calculate_slope(ds_arcticdem)
    ds_arcticdem = calculate_hillshade(ds_arcticdem, azimuth=azimuth, angle_altitude=angle_altitude)
    ds_arcticdem = calculate_aspect(ds_arcticdem)
    ds_arcticdem = calculate_curvature(ds_arcticdem)

    return ds_arcticdem


# TODO: Add possibility to filter out which stuff should be pre-processed
# E.g. in case AlphaEarth Embeddings are added - it does not make sense to merge them here if they are not used
# They are just to big
# However, I think it is still a good approach to have all potential bands available in the training data
# Hence, it would be good to have a possibility to add everything, e.g. filter-preprocessing=False
# The pipelines would then request the bands based on their models
# TODO: Write CUDA reprojection script with cupy
# Needed projections:
# - TCVIS -> Optical: 4326 to UTM
# - ArcticDEM -> Optical: 3413 to UTM
@stopwatch("Preprocessing", printer=logger.debug)
def preprocess_v2(
    ds_optical: xr.Dataset,
    ds_arcticdem: xr.Dataset | None,
    ds_tcvis: xr.Dataset | None,
    tpi_outer_radius: int = 100,
    tpi_inner_radius: int = 0,
    device: Literal["cuda", "cpu"] | int = DEFAULT_DEVICE,
) -> xr.Dataset:
    """Preprocess optical data with modern (DARTS v2) preprocessing steps.

    This function combines optical imagery with terrain (ArcticDEM) and temporal vegetation
    indices (TCVIS) to create a multi-source feature dataset for segmentation. All auxiliary
    data sources are reprojected and cropped to match the optical data's extent and resolution.

    Processing steps:
        1. Calculate NDVI from optical bands
        2. If TCVIS provided: Reproject and merge Tasseled Cap trends
        3. If ArcticDEM provided: Calculate terrain features (TPI, slope, hillshade, aspect, curvature)
           using solar geometry from optical data attributes

    Args:
        ds_optical (xr.Dataset): Optical imagery dataset (PlanetScope or Sentinel-2) containing:
            - Required variables: blue, green, red, nir (float32, reflectance values)
            - Required variables: quality_data_mask, valid_data_mask (uint8)
            - Required attributes: azimuth (float), elevation (float) for hillshade calculation
        ds_arcticdem (xr.Dataset | None): ArcticDEM dataset containing 'dem' (float32) and
            'arcticdem_data_mask' (uint8). If None, terrain features are skipped.
        ds_tcvis (xr.Dataset | None): TCVIS dataset containing tc_brightness, tc_greenness,
            tc_wetness (float). If None, TCVIS features are skipped.
        tpi_outer_radius (int, optional): Outer radius for TPI calculation in meters.
            Defaults to 100m.
        tpi_inner_radius (int, optional): Inner radius for TPI annulus kernel in meters.
            Set to 0 for circular kernel. Defaults to 0.
        device (Literal["cuda", "cpu"] | int, optional): Device for GPU-accelerated computations
            (NDVI, TPI, slope). Use "cuda" for first GPU, int for specific GPU, or "cpu".
            Defaults to "cuda" if available, else "cpu".

    Returns:
        xr.Dataset: Preprocessed dataset with all input optical variables plus:

        Added from optical processing:
            - ndvi (float32): Normalized Difference Vegetation Index
              Attributes: long_name="NDVI"

        Added from TCVIS (if ds_tcvis provided):
            - tc_brightness (float): Tasseled Cap brightness trend
            - tc_greenness (float): Tasseled Cap greenness trend
            - tc_wetness (float): Tasseled Cap wetness trend

        Added from ArcticDEM (if ds_arcticdem provided):
            - dem (float32): Elevation in meters
            - relative_elevation (float32): Topographic Position Index (TPI)
              Attributes: long_name="Topographic Position Index (TPI)"
            - slope (float32): Slope in degrees [0-90]
              Attributes: long_name="Slope"
            - hillshade (uint8): Hillshade values [0-255]
              Attributes: long_name="Hillshade"
            - aspect (float32): Aspect in degrees [0-360]
              Attributes: long_name="Aspect"
            - curvature (float32): Surface curvature
              Attributes: long_name="Curvature"
            - arcticdem_data_mask (uint8): DEM validity mask

    Note:
        Attribute usage:
        - `azimuth` attribute from ds_optical: Used for hillshade calculation (solar azimuth angle).
          Falls back to 225° if missing or invalid.
        - `elevation` attribute from ds_optical: Used for hillshade calculation (solar elevation angle).
          Falls back to 25° if missing or invalid.

        Processing behavior:
        - If both ds_tcvis and ds_arcticdem are None, only NDVI is calculated.
        - ArcticDEM is buffered by tpi_outer_radius before reprojection to avoid edge effects,
          then cropped back to optical extent after terrain feature calculation.
        - Reprojection uses cubic resampling for smooth terrain features.
        - GPU acceleration (if device="cuda") significantly speeds up TPI and slope calculations.

    Example:
        Complete preprocessing with all data sources:

        ```python
        from darts_preprocessing import preprocess_v2
        from darts_acquisition import load_cdse_s2_sr_scene, load_arcticdem, load_tcvis

        # Load optical data
        optical = load_cdse_s2_sr_scene(s2_scene_id, ...)

        # Load auxiliary data
        arcticdem = load_arcticdem(optical.odc.geobox, ...)
        tcvis = load_tcvis(optical.odc.geobox, ...)

        # Preprocess
        preprocessed = preprocess_v2(
            ds_optical=optical,
            ds_arcticdem=arcticdem,
            ds_tcvis=tcvis,
            tpi_outer_radius=100,
            tpi_inner_radius=0,
            device="cuda"
        )

        # Result contains: blue, green, red, nir, ndvi, tc_brightness, tc_greenness,
        # tc_wetness, dem, relative_elevation, slope, hillshade, aspect, curvature
        ```

    """
    # Move to GPU for faster calculations
    ds_optical = move_to_device(ds_optical, device)
    # Calculate NDVI
    ds_optical["ndvi"] = calculate_ndvi(ds_optical)
    ds_optical = move_to_host(ds_optical)

    if ds_tcvis is None and ds_arcticdem is None:
        logger.debug("No auxiliary data provided. Only NDVI was calculated.")
        return ds_optical

    if ds_tcvis is not None:
        # Reproject TCVIS to optical data
        with stopwatch("Reprojecting TCVIS", printer=logger.debug):
            # *: Reprojecting this way will not alter the datatype of the data!
            # Should be uint8 before and after reprojection.
            ds_tcvis = ds_tcvis.odc.reproject(ds_optical.odc.geobox, resampling="cubic")

        # !: Reprojecting with f64 coordinates and values behind the decimal point can result in a coordinate missmatch:
        # E.g. ds_optical has x coordinates [2.123, ...] then is can happen that the
        # reprojected ds_tcvis coordinates are [2.12300001, ...]
        # This results is all-nan assigments later when adding the variables of the reprojected dataset to the original
        assert (ds_optical.x == ds_tcvis.x).all(), "x coordinates do not match! See code comment above"
        assert (ds_optical.y == ds_tcvis.y).all(), "y coordinates do not match! See code comment above"

        # ?: Do ds_tcvis and ds_optical now share the same memory on the GPU?
        #  or do I need to delete ds_tcvis to free memory?
        # Same question for ArcticDEM
        ds_optical["tc_brightness"] = ds_tcvis.tc_brightness
        ds_optical["tc_greenness"] = ds_tcvis.tc_greenness
        ds_optical["tc_wetness"] = ds_tcvis.tc_wetness

    if ds_arcticdem is not None:
        # Calculate TPI and slope from ArcticDEM
        with stopwatch("Reprojecting ArcticDEM", printer=logger.debug):
            ds_arcticdem = ds_arcticdem.odc.reproject(
                ds_optical.odc.geobox.buffered(tpi_outer_radius), resampling="cubic"
            )
        # Move to same device as optical
        ds_arcticdem = move_to_device(ds_arcticdem, device)

        assert (ds_optical.x == ds_arcticdem.x).all(), "x coordinates do not match! See code comment above"
        assert (ds_optical.y == ds_arcticdem.y).all(), "y coordinates do not match! See code comment above"

        azimuth, angle_altitude = get_azimuth_and_elevation(ds_optical)
        ds_arcticdem = preprocess_arcticdem(
            ds_arcticdem,
            tpi_outer_radius,
            tpi_inner_radius,
            azimuth,
            angle_altitude,
        )
        ds_arcticdem = move_to_host(ds_arcticdem)

        # TODO: Check if crop can be done with apply_mask = False
        # -> Then the type conversion of the arcticdem data mask would not be necessary anymore
        # -> And this would also allow to keep the data on the GPU
        ds_arcticdem = ds_arcticdem.odc.crop(ds_optical.odc.geobox.extent)
        # For some reason, we need to reindex, because the reproject + crop of the arcticdem sometimes results
        # in floating point errors. These error are at the order of 1e-10, hence, way below millimeter precision.
        ds_arcticdem["x"] = ds_optical.x
        ds_arcticdem["y"] = ds_optical.y

        ds_optical["dem"] = ds_arcticdem.dem
        ds_optical["relative_elevation"] = ds_arcticdem.tpi
        ds_optical["slope"] = ds_arcticdem.slope
        ds_optical["hillshade"] = ds_arcticdem.hillshade
        ds_optical["aspect"] = ds_arcticdem.aspect
        ds_optical["curvature"] = ds_arcticdem.curvature
        ds_optical["arcticdem_data_mask"] = ds_arcticdem.arcticdem_data_mask.astype("uint8")

    return ds_optical
