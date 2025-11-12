"""PLANET preprocessing functions for training with the v2 data preprocessing."""

import logging
import time
from math import ceil, sqrt
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from darts_utils.paths import DefaultPaths, paths

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import geopandas as gpd


def _path_gen(data_dir: Path):
    return {fpath.parent.name: fpath.parent for fpath in data_dir.rglob("*_SR.tif")}
    # return next(data_dir.rglob(f"{image_id}*_SR.tif")).parent


def _get_region_name(footprint: "gpd.GeoSeries", admin2: "gpd.GeoDataFrame") -> str:
    # Check if any label is intersecting with the test regions
    admin2_of_footprint = admin2[admin2.intersects(footprint.geometry)]

    if admin2_of_footprint.empty:
        raise ValueError("No intersection found between labels and admin2 regions")

    region_name = admin2_of_footprint.iloc[0]["shapeName"]

    if len(admin2_of_footprint) > 1:
        logger.warning(
            f"Found multiple regions for footprint {footprint.image_id}: {admin2_of_footprint.shapeName.to_list()}."
            f" Using the first one ({region_name})"
        )
    return region_name


def preprocess_planet_train_data_pingo(
    *,
    data_dir: Path,
    labels_dir: Path,
    default_dirs: DefaultPaths = DefaultPaths(),
    train_data_dir: Path | None = None,
    arcticdem_dir: Path | None = None,
    tcvis_dir: Path | None = None,
    admin_dir: Path | None = None,
    preprocess_cache: Path | None = None,
    force_preprocess: bool = False,
    device: Literal["cuda", "cpu", "auto"] | int | None = None,
    ee_project: str | None = None,
    ee_use_highvolume: bool = True,
    tpi_outer_radius: int = 100,
    tpi_inner_radius: int = 0,
    patch_size: int = 1024,
    overlap: int = 16,
    exclude_nopositive: bool = False,
    exclude_nan: bool = True,
):
    """Preprocess Planet data for training (Pingo version).

    This function preprocesses Planet scenes into a training-ready format by creating fixed-size patches
    and storing them in a zarr array for efficient random access during training. All data is stored in
    a single zarr group with associated metadata.

    The preprocessing creates patches of the specified size from each Planet scene and stores them as:
    - A zarr group containing 'x' (input data) and 'y' (labels) arrays
    - A geopandas dataframe with metadata including region, position, and label statistics
    - A configuration file with preprocessing parameters

    The x dataarray contains the input data with shape (n_patches, n_bands, patch_size, patch_size).
    The y dataarray contains the labels with shape (n_patches, patch_size, patch_size).
    Both dataarrays are chunked along the n_patches dimension with chunk size 1, resulting in
    each patch being stored in a separate file for super fast random access.

    The metadata dataframe contains information about each patch including:
    - sample_id: Identifier for the source Planet scene
    - region: Administrative region name
    - geometry: Spatial extent of the patch
    - empty: Whether the patch contains positive labeled pixels
    - Additional metadata as specified

    Through `exclude_nopositive` and `exclude_nan`, respective patches can be excluded from the final data.

    A `config.toml` file is saved in the `train_data_dir` containing the configuration used for the
    preprocessing. Additionally, a timestamp-based CLI configuration file is saved for reproducibility.

    The final directory structure of `train_data_dir` will look like this:

    ```sh
    train_data_dir/
    ├── config.toml
    ├── data.zarr/
    │   ├── x/          # Input patches [n_patches, n_bands, patch_size, patch_size]
    │   └── y/          # Label patches [n_patches, patch_size, patch_size]
    ├── metadata.parquet
    └── {timestamp}.cli.json
    ```

    Args:
        data_dir (Path): The directory containing the Planet scenes and orthotiles.
        labels_dir (Path): The directory containing the labels and footprints / extents.
        default_dirs (DefaultPaths, optional): The default directories for DARTS. Defaults to a config filled with None.
        train_data_dir (Path | None, optional): The "output" directory where the tensors are written to.
            If None, will use the default training data directory based on the DARTS paths.
            Defaults to None.
        arcticdem_dir (Path | None, optional): The directory containing the ArcticDEM data
            (the datacube and the extent files).
            Will be created and downloaded if it does not exist.
            If None, will use the default auxiliary directory based on the DARTS paths.
            Defaults to None.
        tcvis_dir (Path | None, optional): The directory containing the TCVis data.
            If None, will use the default TCVis directory based on the DARTS paths.
            Defaults to None.
        admin_dir (Path | None, optional): The directory containing the admin files.
            If None, will use the default auxiliary directory based on the DARTS paths.
            Defaults to None.
        preprocess_cache (Path | None, optional): The directory to store the preprocessed data.
            If None, will neither use nor store preprocessed data.
            Defaults to None.
        force_preprocess (bool, optional): Whether to force the preprocessing of the data. Defaults to False.
        device (Literal["cuda", "cpu"] | int, optional): The device to run the model on.
            If "cuda" take the first device (0), if int take the specified device.
            If "auto" try to automatically select a free GPU (<50% memory usage).
            Defaults to "cuda" if available, else "cpu".
        ee_project (str, optional): The Earth Engine project ID or number to use. May be omitted if
            project is defined within persistent API credentials obtained via `earthengine authenticate`.
        ee_use_highvolume (bool, optional): Whether to use the high volume server (https://earthengine-highvolume.googleapis.com).
        tpi_outer_radius (int, optional): The outer radius of the annulus kernel for the tpi calculation
            in m. Defaults to 100m.
        tpi_inner_radius (int, optional): The inner radius of the annulus kernel for the tpi calculation
            in m. Defaults to 0.
        patch_size (int, optional): The patch size to use for inference. Defaults to 1024.
        overlap (int, optional): The overlap to use for inference. Defaults to 16.
        exclude_nopositive (bool, optional): Whether to exclude patches where the labels do not contain positives.
            Defaults to False.
        exclude_nan (bool, optional): Whether to exclude patches where the input data has nan values.
            Defaults to True.

    """
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    logger.info(f"Starting preprocessing at {current_time}.")

    paths.set_defaults(default_dirs)
    train_data_dir = train_data_dir or paths.train_data_dir("planet_v2_pingo", patch_size)
    arcticdem_dir = arcticdem_dir or paths.arcticdem(2)
    tcvis_dir = tcvis_dir or paths.tcvis()
    admin_dir = admin_dir or paths.admin_boundaries()

    # Storing the configuration as JSON file
    train_data_dir.mkdir(parents=True, exist_ok=True)
    from darts_utils.functools import write_function_args_to_config_file

    write_function_args_to_config_file(
        fpath=train_data_dir / f"{current_time}.cli.toml",
        function=preprocess_planet_train_data_pingo,
        locals_=locals(),
    )

    from stopuhr import Chronometer

    timer = Chronometer(printer=logger.debug)

    from darts.utils.cuda import debug_info

    debug_info()

    # Import here to avoid long loading times when running other commands
    import geopandas as gpd
    import pandas as pd
    import rich
    import smart_geocubes
    import xarray as xr
    from darts_acquisition import load_arcticdem, load_planet_masks, load_planet_scene, load_tcvis
    from darts_acquisition.admin import download_admin_files
    from darts_preprocessing import preprocess_v2
    from darts_segmentation.training.prepare_training import TrainDatasetBuilder
    from darts_utils.tilecache import XarrayCacheManager
    from odc.stac import configure_rio
    from rich.progress import track

    from darts.utils.cuda import decide_device
    from darts.utils.earthengine import init_ee
    from darts.utils.logging import LoggingManager

    device = decide_device(device)
    init_ee(ee_project, ee_use_highvolume)
    configure_rio(cloud_defaults=True, aws={"aws_unsigned": True})
    logger.info("Configured Rasterio")

    # Create the datacubes if they do not exist
    LoggingManager.apply_logging_handlers("smart_geocubes")
    accessor = smart_geocubes.ArcticDEM2m(arcticdem_dir)
    if not accessor.created:
        accessor.create(overwrite=False)
    accessor = smart_geocubes.TCTrend(tcvis_dir)
    if not accessor.created:
        accessor.create(overwrite=False)

    labels = (gpd.read_file(labels_file) for labels_file in labels_dir.glob("*/TrainingLabel*.gpkg"))
    labels = gpd.GeoDataFrame(pd.concat(labels, ignore_index=True))

    footprints = (gpd.read_file(footprints_file) for footprints_file in labels_dir.glob("*/ImageFootprints*.gpkg"))
    footprints = gpd.GeoDataFrame(pd.concat(footprints, ignore_index=True))
    footprints["fpath"] = footprints.image_id.map(_path_gen(data_dir))

    # Download admin files if they do not exist
    admin2_fpath = admin_dir / "geoBoundariesCGAZ_ADM2.shp"
    if not admin2_fpath.exists():
        download_admin_files(admin_dir)
    admin2 = gpd.read_file(admin2_fpath)

    # We hardcode these since they depend on the preprocessing we use
    bands = [
        "red",
        "green",
        "blue",
        "nir",
        "ndvi",
        "relative_elevation",
        "slope",
        "aspect",
        "hillshade",
        "curvature",
        "tc_brightness",
        "tc_greenness",
        "tc_wetness",
    ]

    builder = TrainDatasetBuilder(
        train_data_dir=train_data_dir,
        patch_size=patch_size,
        overlap=overlap,
        bands=bands,
        exclude_nopositive=exclude_nopositive,
        exclude_nan=exclude_nan,
        device=device,
    )
    cache_manager = XarrayCacheManager(preprocess_cache)

    for i, footprint in track(
        footprints.iterrows(), description="Processing samples", total=len(footprints), console=rich.get_console()
    ):
        planet_id = footprint.image_id
        info_id = f"{planet_id=} ({i + 1} of {len(footprint)})"
        try:
            logger.debug(f"Processing sample {info_id}")

            if not footprint.fpath or (not footprint.fpath.exists() and not cache_manager.exists(planet_id)):
                logger.warning(
                    f"Footprint image '{planet_id}' at {footprint.fpath} does not exist. Skipping {info_id}..."
                )
                continue

            def _get_tile():
                tile = load_planet_scene(footprint.fpath)
                arctidem_res = 2
                arcticdem_buffer = ceil(tpi_outer_radius / arctidem_res * sqrt(2))
                arcticdem = load_arcticdem(
                    tile.odc.geobox, arcticdem_dir, resolution=arctidem_res, buffer=arcticdem_buffer
                )
                tcvis = load_tcvis(tile.odc.geobox, tcvis_dir)
                data_masks = load_planet_masks(footprint.fpath)
                tile = xr.merge([tile, data_masks])

                tile: xr.Dataset = preprocess_v2(
                    tile,
                    arcticdem,
                    tcvis,
                    tpi_outer_radius,
                    tpi_inner_radius,
                    device,
                )
                return tile

            with timer("Loading tile"):
                tile = cache_manager.get_or_create(
                    identifier=planet_id,
                    creation_func=_get_tile,
                    force=force_preprocess,
                )

            logger.debug(f"Found tile with size {tile.sizes}")

            footprint_labels = labels[labels.image_id == planet_id]
            region = _get_region_name(footprint, admin2)

            with timer("Save as patches"):
                builder.add_tile(
                    tile=tile,
                    labels=footprint_labels,
                    region=region,
                    sample_id=planet_id,
                    metadata={
                        "planet_id": planet_id,
                        "fpath": footprint.fpath,
                    },
                )

            logger.info(f"Processed sample {info_id}")

        except (KeyboardInterrupt, SystemExit, SystemError):
            logger.info("Interrupted by user.")
            break

        except Exception as e:
            logger.warning(f"Could not process sample {info_id} . Skipping...")
            logger.exception(e)

    timer.summary()

    if len(builder) == 0:
        logger.warning("No samples were processed. Exiting...")
        return

    builder.finalize(
        {
            "data_dir": data_dir,
            "labels_dir": labels_dir,
            "arcticdem_dir": arcticdem_dir,
            "tcvis_dir": tcvis_dir,
            "ee_project": ee_project,
            "ee_use_highvolume": ee_use_highvolume,
            "tpi_outer_radius": tpi_outer_radius,
            "tpi_inner_radius": tpi_inner_radius,
        }
    )
