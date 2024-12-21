"""Preprocessing functions for legacy training."""

import logging
import multiprocessing as mp
from itertools import chain, repeat
from math import ceil, sqrt
from pathlib import Path
from typing import Literal

import toml

logger = logging.getLogger(__name__)


def preprocess_s2_train_data(
    *,
    bands: list[str],
    sentinel2_dir: Path,
    train_data_dir: Path,
    arcticdem_dir: Path,
    tcvis_dir: Path,
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
):
    """Preprocess Sentinel 2 data for training.

    Args:
        bands (list[str]): The bands to be used for training. Must be present in the preprocessing.
        sentinel2_dir (Path): The directory containing the Sentinel 2 scenes.
        train_data_dir (Path): The "output" directory where the tensors are written to.
        arcticdem_dir (Path): The directory containing the ArcticDEM data (the datacube and the extent files).
            Will be created and downloaded if it does not exist.
        tcvis_dir (Path): The directory containing the TCVis data.
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

    """
    # Import here to avoid long loading times when running other commands
    import geopandas as gpd
    import torch
    import xarray as xr
    from darts_acquisition.arcticdem import load_arcticdem_tile
    from darts_acquisition.s2 import load_s2_masks, load_s2_scene
    from darts_acquisition.tcvis import load_tcvis
    from darts_preprocessing import preprocess_legacy_fast
    from darts_segmentation.training.prepare_training import create_training_patches
    from dask.distributed import Client, LocalCluster
    from lovely_tensors import monkey_patch
    from odc.stac import configure_rio
    from sklearn.model_selection import train_test_split

    from darts.utils.cuda import debug_info, decide_device
    from darts.utils.earthengine import init_ee

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
        output_dir_train = train_data_dir / "train"
        output_dir_val = train_data_dir / "val"

        # Find all Sentinel 2 scenes
        n_patches = 0
        s2_paths = sorted(sentinel2_dir.glob("*/"))
        logger.info(f"Found {len(s2_paths)} Sentinel 2 scenes in {sentinel2_dir}")
        train_paths: list[Path]
        val_paths: list[Path]
        train_paths, val_paths = train_test_split(s2_paths, test_size=test_val_split, random_state=42)
        logger.info(f"Split the data into {len(train_paths)} training and {len(val_paths)} validation samples.")

        fpathgen = chain(train_paths, val_paths)
        modegen = chain(repeat(output_dir_train, len(train_paths)), repeat(output_dir_val, len(val_paths)))
        for i, (fpath, output_dir) in enumerate(zip(fpathgen, modegen)):
            try:
                optical = load_s2_scene(fpath)
                logger.info(f"Found optical tile with size {optical.sizes}")
                tile_id = optical.attrs["tile_id"]

                # Check for a cached preprocessed file
                if preprocess_cache and (preprocess_cache / f"{tile_id}.nc").exists():
                    cache_file = preprocess_cache / f"{tile_id}.nc"
                    logger.info(f"Loading preprocessed data from {cache_file.resolve()}")
                    tile = xr.open_dataset(preprocess_cache / f"{tile_id}.nc", engine="h5netcdf").set_coords(
                        "spatial_ref"
                    )
                else:
                    arctidem_res = 10
                    arcticdem_buffer = ceil(tpi_outer_radius / arctidem_res * sqrt(2))
                    arcticdem = load_arcticdem_tile(
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

                labels = gpd.read_file(fpath / f"{optical.attrs['s2_tile_id']}.shp")

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

                outdir_x = output_dir / "x"
                outdir_y = output_dir / "y"
                outdir_x.mkdir(exist_ok=True, parents=True)
                outdir_y.mkdir(exist_ok=True, parents=True)
                n_patches = 0
                for patch_id, (x, y) in enumerate(gen):
                    torch.save(x, outdir_x / f"{tile_id}_pid{patch_id}.pt")
                    torch.save(y, outdir_y / f"{tile_id}_pid{patch_id}.pt")
                    n_patches += 1

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

    logger.info(f"Saved {n_patches} patches to {train_data_dir}")
