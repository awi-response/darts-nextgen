"""Preprocessing functions for legacy training."""

import logging
import multiprocessing as mp
from itertools import chain, repeat
from math import ceil, sqrt
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)


def split_dataset_paths(
    s2_paths: list[Path], train_data_dir: Path, test_val_split: float, test_regions: list[str] | None, admin_dir: Path
):
    """Split the dataset into a cross-val, a val-test and a test dataset.

    Returns a generator with: input-path, output-path and split/mode.
    The test set is splitted first by the given regions and is meant to be used to evaluate the regional value shift.
    Then the val-test set is splitted then by random at given size to evaluate the variance value shift.

    Args:
        s2_paths (list[Path]): All paths found with tiffs.
        train_data_dir (Path): Output path.
        test_val_split (float): val-test ratio.
        test_regions (list[str] | None): test regions.
        admin_dir (Path): The directory containing the admin level shape-files.

    Returns:
        [zip[tuple[Path, Path, str]]]: A generator with input-path, output-path and split/mode.

    """
    # Import here to avoid long loading times when running other commands
    import geopandas as gpd
    from darts_acquisition.admin import download_admin_files
    from darts_acquisition.s2 import parse_s2_tile_id
    from sklearn.model_selection import train_test_split

    train_data_dir.mkdir(exist_ok=True, parents=True)

    # 1. Split regions
    test_paths: list[Path] = []
    training_paths: list[Path] = []
    if test_regions:
        # Download admin files if they do not exist
        admin1_fpath = admin_dir / "geoBoundariesCGAZ_ADM1.shp"
        admin2_fpath = admin_dir / "geoBoundariesCGAZ_ADM2.shp"

        if not admin1_fpath.exists() or not admin2_fpath.exists():
            download_admin_files(admin_dir)

        # Load the admin files
        admin1 = gpd.read_file(admin1_fpath)
        admin2 = gpd.read_file(admin2_fpath)

        # Get the regions from the admin files
        test_region_geometries_adm1 = admin1[admin1["shapeName"].isin(test_regions)]
        test_region_geometries_adm2 = admin2[admin2["shapeName"].isin(test_regions)]

        logger.debug(f"Found {len(test_region_geometries_adm1)} admin1-regions in {admin1_fpath}")
        logger.debug(f"Found {len(test_region_geometries_adm2)} admin2-regions in {admin2_fpath}")

        for fpath in s2_paths:
            _, s2_tile_id, _ = parse_s2_tile_id(fpath)
            labels = gpd.read_file(fpath / f"{s2_tile_id}.shp").to_crs("EPSG:4326")
            # Check if any label is intersecting with the test regions
            adm1_intersects = labels.overlay(test_region_geometries_adm1, how="intersection")
            adm2_intersects = labels.overlay(test_region_geometries_adm2, how="intersection")

            if (len(adm1_intersects.index) > 0) or (len(adm2_intersects.index) > 0):
                test_paths.append(fpath)
            else:
                training_paths.append(fpath)
    else:
        training_paths = s2_paths

    # 2. Split by random sampling
    cross_val_paths: list[Path]
    val_test_paths: list[Path]
    if len(training_paths) > 0:
        cross_val_paths, val_test_paths = train_test_split(training_paths, test_size=test_val_split, random_state=42)
    else:
        cross_val_paths, val_test_paths = [], []
        logger.warning("No left over training samples found. Skipping train-val split.")

    logger.info(
        f"Split the data into {len(cross_val_paths)} cross-val (train + val), "
        f"{len(val_test_paths)} val-test (variance) and {len(test_paths)} test (region) samples."
    )

    fpathgen = chain(cross_val_paths, val_test_paths, test_paths)
    modegen = chain(
        repeat("cross-val", len(cross_val_paths)),
        repeat("val-test", len(val_test_paths)),
        repeat("test", len(test_paths)),
    )

    return zip(fpathgen, modegen)


def preprocess_s2_train_data(
    *,
    bands: list[str],
    sentinel2_dir: Path,
    train_data_dir: Path,
    arcticdem_dir: Path,
    tcvis_dir: Path,
    admin_dir: Path,
    preprocess_cache: Path | None = None,
    device: Literal["cuda", "cpu", "auto"] | int | None = None,
    dask_worker: int = min(16, mp.cpu_count() - 1),
    ee_project: str | None = None,
    ee_use_highvolume: bool = True,
    tpi_outer_radius: int = 100,
    tpi_inner_radius: int = 0,
    patch_size: int = 1024,
    overlap: int = 16,
    exclude_nopositive: bool = False,
    exclude_nan: bool = True,
    mask_erosion_size: int = 10,
    test_val_split: float = 0.05,
    test_regions: list[str] | None = None,
):
    """Preprocess Sentinel 2 data for training.

    The data is split into a cross-validation, a validation-test and a test set:

        - `cross-val` is meant to be used for train and validation
        - `val-test` 5% random leave-out for testing the randomness distribution shift of the data
        - `test` leave-out region for testing the spatial distribution shift of the data

    Each split is stored as a zarr group, containing a x and a y dataarray.
    The x dataarray contains the input data with the shape (n_patches, n_bands, patch_size, patch_size).
    The y dataarray contains the labels with the shape (n_patches, patch_size, patch_size).
    Both dataarrays are chunked along the n_patches dimension.
    This results in super fast random access to the data, because each sample / patch is stored in a separate chunk and
    therefore in a separate file.

    Through exclude_nopositve and exclude_nan, respective patches can be excluded from the final data.



    Args:
        bands (list[str]): The bands to be used for training. Must be present in the preprocessing.
        sentinel2_dir (Path): The directory containing the Sentinel 2 scenes.
        train_data_dir (Path): The "output" directory where the tensors are written to.
        arcticdem_dir (Path): The directory containing the ArcticDEM data (the datacube and the extent files).
            Will be created and downloaded if it does not exist.
        tcvis_dir (Path): The directory containing the TCVis data.
        admin_dir (Path): The directory containing the admin files.
        preprocess_cache (Path, optional): The directory to store the preprocessed data. Defaults to None.
        device (Literal["cuda", "cpu"] | int, optional): The device to run the model on.
            If "cuda" take the first device (0), if int take the specified device.
            If "auto" try to automatically select a free GPU (<50% memory usage).
            Defaults to "cuda" if available, else "cpu".
        dask_worker (int, optional): The number of Dask workers to use. Defaults to min(16, mp.cpu_count() - 1).
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
        mask_erosion_size (int, optional): The size of the disk to use for mask erosion and the edge-cropping.
            Defaults to 10.
        test_val_split (float, optional): The split ratio for the test and validation set. Defaults to 0.05.
        test_regions (list[str] | str, optional): The region to use for the test set. Defaults to None.

    """
    # Import here to avoid long loading times when running other commands
    import geopandas as gpd
    import pandas as pd
    import toml
    import xarray as xr
    import zarr
    from darts_acquisition import load_arcticdem, load_s2_masks, load_s2_scene, load_tcvis
    from darts_acquisition.s2 import parse_s2_tile_id
    from darts_preprocessing import preprocess_legacy_fast
    from darts_segmentation.training.prepare_training import create_training_patches
    from dask.distributed import Client, LocalCluster
    from lovely_tensors import monkey_patch
    from numcodecs import Blosc
    from odc.stac import configure_rio
    from rich.progress import track
    from zarr.storage import DirectoryStore

    from darts.utils.cuda import debug_info, decide_device
    from darts.utils.earthengine import init_ee
    from darts.utils.logging import console

    monkey_patch()
    debug_info()
    device = decide_device(device)
    init_ee(ee_project, ee_use_highvolume)

    with LocalCluster(n_workers=dask_worker) as cluster, Client(cluster) as client:
        logger.info(f"Using Dask client: {client} on cluster {cluster}")
        logger.info(f"Dashboard available at: {client.dashboard_link}")
        configure_rio(cloud_defaults=True, aws={"aws_unsigned": True}, client=client)
        logger.info("Configured Rasterio with Dask")

        # We hardcode these because they depend on the preprocessing used
        norm_factors = {
            "red": 1 / 3000,
            "green": 1 / 3000,
            "blue": 1 / 3000,
            "nir": 1 / 3000,
            "ndvi": 1 / 20000,
            "relative_elevation": 1 / 30000,
            "slope": 1 / 90,
            "tc_brightness": 1 / 255,
            "tc_greenness": 1 / 255,
            "tc_wetness": 1 / 255,
        }
        # Filter out bands that are not in the specified bands
        norm_factors = {k: v for k, v in norm_factors.items() if k in bands}

        train_data_dir.mkdir(exist_ok=True, parents=True)

        zgroups = {
            "cross-val": zarr.group(store=DirectoryStore(train_data_dir / "cross-val.zarr"), overwrite=True),
            "val-test": zarr.group(store=DirectoryStore(train_data_dir / "val-test.zarr"), overwrite=True),
            "test": zarr.group(store=DirectoryStore(train_data_dir / "test.zarr"), overwrite=True),
        }
        # We need do declare the number of patches to 0, because we can't know the final number of patches
        for root in zgroups.values():
            root.create(
                name="x",
                shape=(0, len(bands), patch_size, patch_size),
                # shards=(100, len(bands), patch_size, patch_size),
                chunks=(1, len(bands), patch_size, patch_size),
                dtype="float32",
                compressor=Blosc(cname="lz4", clevel=9),
            )
            root.create(
                name="y",
                shape=(0, patch_size, patch_size),
                # shards=(100, patch_size, patch_size),
                chunks=(1, patch_size, patch_size),
                dtype="uint8",
                compressor=Blosc(cname="lz4", clevel=9),
            )

        # Find all Sentinel 2 scenes and split into train+val (cross-val), val-test (variance) and test (region)
        n_patches = 0
        n_patches_by_mode = {"cross-val": 0, "val-test": 0, "test": 0}
        joint_lables = []
        s2_paths = sorted(sentinel2_dir.glob("*/"))
        logger.info(f"Found {len(s2_paths)} Sentinel 2 scenes in {sentinel2_dir}")
        path_gen = split_dataset_paths(s2_paths, train_data_dir, test_val_split, test_regions, admin_dir)
        for i, (fpath, mode) in track(
            enumerate(path_gen), description="Processing samples", total=len(s2_paths), console=console
        ):
            try:
                _, s2_tile_id, tile_id = parse_s2_tile_id(fpath)

                logger.debug(
                    f"Processing sample {i + 1} of {len(s2_paths)} '{fpath.resolve()}' ({tile_id=}) to split '{mode}'"
                )

                # Check for a cached preprocessed file
                if preprocess_cache and (preprocess_cache / f"{tile_id}.nc").exists():
                    cache_file = preprocess_cache / f"{tile_id}.nc"
                    logger.info(f"Loading preprocessed data from {cache_file.resolve()}")
                    tile = xr.open_dataset(preprocess_cache / f"{tile_id}.nc", engine="h5netcdf").set_coords(
                        "spatial_ref"
                    )
                else:
                    optical = load_s2_scene(fpath)
                    logger.info(f"Found optical tile with size {optical.sizes}")
                    arctidem_res = 10
                    arcticdem_buffer = ceil(tpi_outer_radius / arctidem_res * sqrt(2))
                    arcticdem = load_arcticdem(
                        optical.odc.geobox, arcticdem_dir, resolution=arctidem_res, buffer=arcticdem_buffer
                    )
                    tcvis = load_tcvis(optical.odc.geobox, tcvis_dir)
                    data_masks = load_s2_masks(fpath, optical.odc.geobox)

                    tile: xr.Dataset = preprocess_legacy_fast(
                        optical,
                        arcticdem,
                        tcvis,
                        data_masks,
                        tpi_outer_radius,
                        tpi_inner_radius,
                        device,
                    )
                    # Only cache if we have a cache directory
                    if preprocess_cache:
                        preprocess_cache.mkdir(exist_ok=True, parents=True)
                        cache_file = preprocess_cache / f"{tile_id}.nc"
                        logger.info(f"Caching preprocessed data to {cache_file.resolve()}")
                        tile.to_netcdf(cache_file, engine="h5netcdf")

                labels = gpd.read_file(fpath / f"{s2_tile_id}.shp")

                # Save the patches
                gen = create_training_patches(
                    tile,
                    labels,
                    bands,
                    norm_factors,
                    patch_size,
                    overlap,
                    exclude_nopositive,
                    exclude_nan,
                    device,
                    mask_erosion_size,
                )

                zx = zgroups[mode]["x"]
                zy = zgroups[mode]["y"]
                patch_id = None
                for patch_id, (x, y) in enumerate(gen):
                    zx.append(x.unsqueeze(0).numpy().astype("float32"))
                    zy.append(y.unsqueeze(0).numpy().astype("uint8"))
                    n_patches += 1
                    n_patches_by_mode[mode] += 1
                if n_patches > 0 and len(labels) > 0:
                    labels["mode"] = mode
                    joint_lables.append(labels.to_crs("EPSG:3413"))

                logger.info(
                    f"Processed sample {i + 1} of {len(s2_paths)} '{fpath.resolve()}'"
                    f"({tile_id=}) with {patch_id} patches."
                )
            except KeyboardInterrupt:
                logger.info("Interrupted by user.")
                break

            except Exception as e:
                logger.warning(f"Could not process folder sample {i} '{fpath.resolve()}'.\nSkipping...")
                logger.exception(e)

    # Save the used labels
    joint_lables = pd.concat(joint_lables)
    joint_lables.to_file(train_data_dir / "labels.geojson", driver="GeoJSON")

    # Save a config file as toml
    config = {
        "darts": {
            "sentinel2_dir": sentinel2_dir,
            "train_data_dir": train_data_dir,
            "arcticdem_dir": arcticdem_dir,
            "tcvis_dir": tcvis_dir,
            "bands": bands,
            "norm_factors": norm_factors,
            "device": device,
            "ee_project": ee_project,
            "ee_use_highvolume": ee_use_highvolume,
            "tpi_outer_radius": tpi_outer_radius,
            "tpi_inner_radius": tpi_inner_radius,
            "patch_size": patch_size,
            "overlap": overlap,
            "exclude_nopositive": exclude_nopositive,
            "exclude_nan": exclude_nan,
            "n_patches": n_patches,
        }
    }
    with open(train_data_dir / "config.toml", "w") as f:
        toml.dump(config, f)

    logger.info(f"Saved {n_patches} ({n_patches_by_mode}) patches to {train_data_dir}")
