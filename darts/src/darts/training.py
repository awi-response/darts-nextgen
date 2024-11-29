"""Training module for DARTS."""

import logging
import multiprocessing as mp
from math import ceil, sqrt
from pathlib import Path
from typing import Literal

import toml

logger = logging.getLogger(__name__)


def preprocess_s2_train_data(
    *,
    sentinel2_dir: Path,
    train_data_dir: Path,
    arcticdem_dir: Path,
    tcvis_dir: Path,
    bands: list[str],
    device: Literal["cuda", "cpu", "auto"] | int | None = None,
    ee_project: str | None = None,
    ee_use_highvolume: bool = True,
    tpi_outer_radius: int = 100,
    tpi_inner_radius: int = 0,
    patch_size: int = 1024,
    overlap: int = 16,
    include_allzero: bool = False,
    include_nan_edges: bool = True,
):
    """Preprocess Sentinel 2 data for training.

    Args:
        sentinel2_dir (Path): The directory containing the Sentinel 2 scenes.
        train_data_dir (Path): The "output" directory where the tensors are written to.
        arcticdem_dir (Path): The directory containing the ArcticDEM data (the datacube and the extent files).
            Will be created and downloaded if it does not exist.
        tcvis_dir (Path): The directory containing the TCVis data.
        bands (list[str]): The bands to be used for training. Must be present in the preprocessing.
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
        include_allzero (bool, optional): Whether to include patches where the labels are all zero. Defaults to False.
        include_nan_edges (bool, optional): Whether to include patches where the input data has nan values at the edges.
            Defaults to True.

    """
    # Import here to avoid long loading times when running other commands
    import geopandas as gpd
    import torch
    from darts_acquisition.arcticdem import load_arcticdem_tile
    from darts_acquisition.s2 import load_s2_masks, load_s2_scene
    from darts_acquisition.tcvis import load_tcvis
    from darts_preprocessing import preprocess_legacy_fast
    from darts_segmentation.prepare_training import create_training_patches
    from dask.distributed import Client, LocalCluster
    from odc.stac import configure_rio

    from darts.utils.cuda import debug_info, decide_device
    from darts.utils.earthengine import init_ee

    debug_info()
    device = decide_device(device)
    init_ee(ee_project, ee_use_highvolume)

    cluster = LocalCluster(n_workers=mp.cpu_count() - 1)
    logger.info(f"Created Dask cluster: {cluster}")
    client = Client(cluster)
    logger.info(f"Using Dask client: {client}")
    configure_rio(cloud_defaults=True, aws={"aws_unsigned": True}, client=client)
    logger.info("Configured Rasterio with Dask")

    outpath_x = train_data_dir / "x"
    outpath_y = train_data_dir / "y"

    outpath_x.mkdir(exist_ok=True, parents=True)
    outpath_y.mkdir(exist_ok=True, parents=True)

    # Find all Sentinel 2 scenes
    n_patches = 0
    for fpath in sentinel2_dir.glob("*/"):
        try:
            optical = load_s2_scene(fpath)
            arcticdem = load_arcticdem_tile(
                optical.odc.geobox, arcticdem_dir, resolution=10, buffer=ceil(tpi_outer_radius / 10 * sqrt(2))
            )
            tcvis = load_tcvis(optical.odc.geobox, tcvis_dir)
            data_masks = load_s2_masks(fpath, optical.odc.geobox)

            tile = preprocess_legacy_fast(
                optical,
                arcticdem,
                tcvis,
                data_masks,
                tpi_outer_radius,
                tpi_inner_radius,
                device,
            )

            labels = gpd.read_file(fpath / f"{optical.attrs['tile_id']}.shp")
            tile_id = optical.attrs["tile_id"]

            # Save the patches
            gen = create_training_patches(tile, labels, bands, patch_size, overlap, include_allzero, include_nan_edges)
            for patch_id, (x, y) in enumerate(gen):
                torch.save(x, outpath_x / f"{tile_id}_pid{patch_id}.pt")
                torch.save(y, outpath_y / f"{tile_id}_pid{patch_id}.pt")
                n_patches += 1
            logger.info(f"Processed {tile_id} with {patch_id} patches.")

        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
            break

        except Exception as e:
            logger.warning(f"Could not process folder '{fpath.resolve()}'.\nSkipping...")
            logger.exception(e)

    # Save a config file as toml
    config = {
        "darts": {
            "sentinel2_dir": sentinel2_dir,
            "train_data_dir": train_data_dir,
            "arcticdem_dir": arcticdem_dir,
            "tcvis_dir": tcvis_dir,
            "bands": bands,
            "device": device,
            "ee_project": ee_project,
            "ee_use_highvolume": ee_use_highvolume,
            "tpi_outer_radius": tpi_outer_radius,
            "tpi_inner_radius": tpi_inner_radius,
            "patch_size": patch_size,
            "overlap": overlap,
            "include_allzero": include_allzero,
            "include_nan_edges": include_nan_edges,
            "n_patches": n_patches,
        }
    }
    with open(train_data_dir / "config.toml", "w") as f:
        toml.dump(config, f)

    logger.info(f"Saved {n_patches} patches to {train_data_dir}")

    client.close()
    cluster.close()
