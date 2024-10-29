"""ArcticDEM related data loading."""

import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__.replace("darts_", "darts."))


def create_arcticdem_vrt(dem_data_dir: Path, vrt_target_dir: Path):
    """Create a VRT file from ArcticDEM data.

    Args:
        dem_data_dir (Path): The directory containing the ArcticDEM data (.tif).
        vrt_target_dir (Path): The output directory.

    Raises:
        OSError: If the target directory is not writable.

    """
    start_time = time.time()
    logger.debug(f"Creating ArcticDEM VRT file at {vrt_target_dir} based on {dem_data_dir}.")

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

    for name, subdir in subdirs.items():
        output_file_path = vrt_target_dir / f"{name}.vrt"
        # check the file first if we can write to it

        ds_path = dem_data_dir / subdir
        filelist = [str(f.absolute().resolve()) for f in ds_path.glob("*.tif")]
        logger.debug(f"Found {len(filelist)} files for {name} at {ds_path}.")
        logger.debug(f"Writing VRT to '{output_file_path}'")
        src_nodata = "nan" if name == "slope" else 0
        opt = gdal.BuildVRTOptions(srcNodata=src_nodata, VRTNodata=0)
        gdal.BuildVRT(str(output_file_path.absolute()), filelist, options=opt)

    logger.debug(f"Creation of VRT took {time.time() - start_time:.2f}s")
