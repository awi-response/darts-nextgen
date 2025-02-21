"""Legacy code for creating and handling VRT files from precomputed ArcticDEM data."""

import logging
import os
import time
from pathlib import Path

import rasterio
import rasterio.mask
import rioxarray  # noqa: F401
import xarray as xr

logger = logging.getLogger(__name__.replace("darts_", "darts."))


def create_arcticdem_vrt(dem_data_dir: Path, vrt_target_dir: Path):  # noqa: C901
    """Create a VRT file from ArcticDEM data.

    This command expects the tiles for slope and relative elevation in the subfolders `relative_elevation` and `slope`
    of `dem_data_dir`. The tool requires the python gdal bindings to be installed.

    Args:
        dem_data_dir (Path): The directory containing subfolders `relative_elevation` and `slope` with
            ArcticDEM data (.tif).
        vrt_target_dir (Path): The output directory.

    Raises:
        OSError: If the target directory is not writable.
        ValueError: If the command parameters are invalid

    """
    start_time = time.time()

    if not dem_data_dir.exists():
        raise ValueError(f"The DEM data dir does not exist: {dem_data_dir.resolve().absolute()}")

    logger.debug(f"Creating ArcticDEM VRT file at {vrt_target_dir.resolve()} based on {dem_data_dir.resolve()}")

    try:
        from osgeo import gdal

        logger.debug(f"Found gdal bindings {gdal.__version__}.")
    except ModuleNotFoundError as e:
        logger.exception(
            "The python GDAL bindings where not found. Please install those which are appropriate for your platform."
        )
        raise e

    # decide on the exception behavior of GDAL to supress a warning if we dont
    # don't know if this is necessary in all GDAL versions
    try:
        gdal.UseExceptions()
        logger.debug("Enabled gdal exceptions")
    except AttributeError():
        pass

    # subdirs = {"elevation": "tiles_rel_el", "slope": "tiles_slope"}
    subdirs = {"elevation": "relative_elevation", "slope": "slope"}

    # check first if BOTH files are writable
    non_writable_files = []
    for name in subdirs.keys():
        output_file_path = vrt_target_dir / f"{name}.vrt"
        if not os.access(output_file_path, os.W_OK) and output_file_path.exists():
            non_writable_files.append(output_file_path)
    if len(non_writable_files) > 0:
        raise OSError(f"cannot write to {', '.join([f.name for f in non_writable_files])}")

    # memorize if any files weren't written
    file_written = dict.fromkeys(subdirs.keys(), False)

    for name, subdir in subdirs.items():
        output_file_path = vrt_target_dir / f"{name}.vrt"
        # check the file first if we can write to it

        ds_path = dem_data_dir / subdir
        if not ds_path.exists():
            logger.warning(f"{ds_path.absolute()} does NOT exist!")
            continue

        filelist = [str(f.resolve()) for f in ds_path.glob("*.tif")]
        if len(filelist) < 1:
            logger.warning(f"NO files found in {ds_path.absolute()}")
            continue

        logger.debug(f"Found {len(filelist)} files for {name} at {ds_path}.")
        logger.debug(f"Writing VRT to '{output_file_path.resolve()}'")
        src_nodata = "nan" if name == "slope" else 0
        opt = gdal.BuildVRTOptions(srcNodata=src_nodata, VRTNodata=0)
        gdal.BuildVRT(str(output_file_path.resolve()), filelist, options=opt)
        file_written[name] = True

    for name, is_written in file_written.items():
        if not is_written:
            logger.warning(f"VRT file for {name} was NOT created.")

    logger.debug(f"Creation of VRT took {time.time() - start_time:.2f}s")


def load_vrt(vrt_path: Path, reference_dataset: xr.Dataset) -> xr.DataArray:
    """Load a VRT file and reproject it to match the reference dataset.

    Args:
        vrt_path (Path): Path to the vrt file.
        reference_dataset (xr.Dataset): The reference dataset.

    Raises:
        FileNotFoundError: If the VRT file is not found.

    Returns:
        xr.DataArray: The VRT data reprojected to match the reference dataarray.

    """
    if not vrt_path.exists():
        raise FileNotFoundError(f"Could not find the VRT file at {vrt_path.resolve()}")

    start_time = time.time()

    with rasterio.open(vrt_path) as src:
        with rasterio.vrt.WarpedVRT(
            src, crs=reference_dataset.rio.crs, resampling=rasterio.enums.Resampling.cubic
        ) as vrt:
            bounds = reference_dataset.rio.bounds()
            windows = vrt.window(*bounds)
            shape = (1, len(reference_dataset.y), len(reference_dataset.x))
            data = vrt.read(window=windows, out_shape=shape)[0]  # This is the most time consuming part of the function
            da = xr.DataArray(data, dims=["y", "x"], coords={"y": reference_dataset.y, "x": reference_dataset.x})
            da.rio.write_crs(reference_dataset.rio.crs, inplace=True)
            da.rio.write_transform(reference_dataset.rio.transform(), inplace=True)

    logger.debug(f"Loaded VRT data from {vrt_path.resolve()} in {time.time() - start_time} seconds.")
    return da


def load_arcticdem_from_vrt(slope_vrt: Path, elevation_vrt: Path, reference_dataset: xr.Dataset) -> xr.Dataset:
    """Load ArcticDEM data and reproject it to match the reference dataset.

    Args:
        slope_vrt (Path): Path to the ArcticDEM slope VRT file.
        elevation_vrt (Path): Path to the ArcticDEM elevation VRT file.
        reference_dataset (xr.Dataset): The reference dataset to reproject, resampled and cropped the ArcticDEM data to.

    Returns:
        xr.Dataset: The ArcticDEM data reprojected, resampled and cropped to match the reference dataset.


    """
    start_time = time.time()
    logger.debug(f"Loading ArcticDEM slope from {slope_vrt.resolve()} and elevation from {elevation_vrt.resolve()}")

    slope = load_vrt(slope_vrt, reference_dataset)
    slope: xr.Dataset = (
        slope.assign_attrs({"data_source": "arcticdem", "long_name": "Slope"})
        .rio.write_nodata(float("nan"))
        .astype("float32")
        .to_dataset(name="slope")
    )

    relative_elevation = load_vrt(elevation_vrt, reference_dataset)
    relative_elevation: xr.Dataset = (
        relative_elevation.assign_attrs({"data_source": "arcticdem", "long_name": "Relative Elevation", "units": "m"})
        .fillna(0)
        .rio.write_nodata(0)
        .astype("int16")
        .to_dataset(name="relative_elevation")
    )

    articdem_ds = xr.merge([relative_elevation, slope])
    logger.debug(f"Loaded ArcticDEM data in {time.time() - start_time} seconds.")
    return articdem_ds
