"""PLANET scene based preprocessing."""

import logging
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


@stopwatch.f("Preprocessing arcticdem", printer=logger.debug, print_kwargs=["tpi_outer_radius", "tpi_inner_radius"])
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
    ds_arcticdem: xr.Dataset,
    ds_tcvis: xr.Dataset,
    tpi_outer_radius: int = 100,
    tpi_inner_radius: int = 0,
    device: Literal["cuda", "cpu"] | int = DEFAULT_DEVICE,
) -> xr.Dataset:
    """Preprocess optical data with modern (DARTS v2) preprocessing steps.

    The processing steps are:
    - Calculate NDVI
    - Calculate slope, hillshade, aspect, curvature and relative elevation from ArcticDEM
    - Merge everything into a single ds.

    Args:
        ds_optical (xr.Dataset): The Planet scene optical data or Sentinel 2 scene optical dataset including data_masks.
        ds_arcticdem (xr.Dataset): The ArcticDEM dataset.
        ds_tcvis (xr.Dataset): The TCVIS dataset.
        tpi_outer_radius (int, optional): The outer radius of the annulus kernel for the tpi calculation
            in m. Defaults to 100m.
        tpi_inner_radius (int, optional): The inner radius of the annulus kernel for the tpi calculation
            in m. Defaults to 0.
        device (Literal["cuda", "cpu"] | int, optional): The device to run the tpi and slope calculations on.
            If "cuda" take the first device (0), if int take the specified device.
            Defaults to "cuda" if cuda is available, else "cpu".

    Returns:
        xr.Dataset: The preprocessed dataset.

    """
    # Move to GPU for faster calculations
    ds_optical = move_to_device(ds_optical, device)
    # Calculate NDVI
    ds_optical["ndvi"] = calculate_ndvi(ds_optical)
    ds_optical = move_to_host(ds_optical)

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

    # ?: Do ds_tcvis and ds_optical now share the same memory on the GPU or do I need to delte ds_tcvis to free memory?
    # Same question for ArcticDEM
    ds_optical["tc_brightness"] = ds_tcvis.tc_brightness
    ds_optical["tc_greenness"] = ds_tcvis.tc_greenness
    ds_optical["tc_wetness"] = ds_tcvis.tc_wetness

    # Calculate TPI and slope from ArcticDEM
    with stopwatch("Reprojecting ArcticDEM", printer=logger.debug):
        ds_arcticdem = ds_arcticdem.odc.reproject(ds_optical.odc.geobox.buffered(tpi_outer_radius), resampling="cubic")
    # Move to same device as optical
    ds_arcticdem = move_to_device(ds_arcticdem, device)

    assert (ds_optical.x == ds_arcticdem.x).all(), "x coordinates do not match! See code comment above"
    assert (ds_optical.y == ds_arcticdem.y).all(), "y coordinates do not match! See code comment above"

    azimuth = 225  # Default azimuth for hillshade calculation
    angle_altitude = 25  # Default angle altitude for hillshade calculation
    if "view:azimuth" in ds_arcticdem.attrs:
        azimuth = round(ds_arcticdem.attrs["view:azimuth"])
    if "view:sun_elevation" in ds_arcticdem.attrs:
        angle_altitude = round(ds_arcticdem.attrs["view:sun_elevation"])
    ds_arcticdem = preprocess_arcticdem(
        ds_arcticdem,
        tpi_outer_radius,
        tpi_inner_radius,
        azimuth,
        angle_altitude,
    )
    ds_arcticdem = move_to_host(ds_arcticdem)

    ds_arcticdem = ds_arcticdem.odc.crop(ds_optical.odc.geobox.extent)
    # For some reason, we need to reindex, because the reproject + crop of the arcticdem sometimes results
    # in floating point errors. These error are at the order of 1e-10, hence, way below millimeter precision.
    ds_arcticdem = ds_arcticdem.reindex_like(ds_optical)

    ds_optical["dem"] = ds_arcticdem.dem
    ds_optical["relative_elevation"] = ds_arcticdem.tpi
    ds_optical["slope"] = ds_arcticdem.slope
    ds_optical["hillshade"] = ds_arcticdem.hillshade
    ds_optical["aspect"] = ds_arcticdem.aspect
    ds_optical["curvature"] = ds_arcticdem.curvature
    # TODO: Rename datamask to arcticdem_data_mask in the acquisition and change its dtype so it can be cached properly
    ds_optical["arcticdem_data_mask"] = ds_arcticdem.arcticdem_data_mask

    return ds_optical
