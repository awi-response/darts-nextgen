"""Automated legacy pipeline for Sentinel-2 data."""

import json
import logging
import multiprocessing as mp
import time
from math import ceil, sqrt
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)


def run_native_sentinel2_pipeline_from_aoi(
    aoi_shapefile: Path,
    model_file: Path,
    start_date: str,
    end_date: str,
    max_cloud_cover: int = 10,
    input_cache: Path = Path("data/cache/input"),
    output_data_dir: Path = Path("data/output"),
    tcvis_dir: Path = Path("data/download/tcvis"),
    arcticdem_dir: Path = Path("data/download/arcticdem"),
    device: Literal["cuda", "cpu", "auto"] | int | None = None,
    dask_worker: int = min(16, mp.cpu_count() - 1),
    ee_project: str | None = None,
    ee_use_highvolume: bool = True,
    tpi_outer_radius: int = 100,
    tpi_inner_radius: int = 0,
    patch_size: int = 1024,
    overlap: int = 256,
    batch_size: int = 8,
    reflection: int = 0,
    binarization_threshold: float = 0.5,
    mask_erosion_size: int = 10,
    min_object_size: int = 32,
    quality_level: int | Literal["high_quality", "low_quality", "none"] = 1,
):
    """Pipeline for Sentinel 2 data with optimized preprocessing.

    Args:
        aoi_shapefile (Path): The path to the shapefile containing the AOI. Can be anything readable by geopandas.
        model_file (Path): The path to the model to use for segmentation.
        start_date (str): The start date for the Sentinel-2 data.
        end_date (str): The end date for the Sentinel-2 data.
        max_cloud_cover (int, optional): The maximum cloud cover to use for the Sentinel-2 data. Defaults to 10.
        input_cache(Path): The directory to use for the cache. Stores the downloaded optical data here.
            Defaults to Path("data/cache/input").
        output_data_dir (Path): The "output" directory. Defaults to Path("data/output").
        arcticdem_dir (Path): The directory containing the ArcticDEM data (the datacube and the extent files).
            Will be created and downloaded if it does not exist.
            Defaults to Path("data/download/arcticdem").
        tcvis_dir (Path): The directory containing the TCVis data. Defaults to Path("data/download/tcvis").
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
        batch_size (int, optional): The batch size to use for inference. Defaults to 8.
        reflection (int, optional): The reflection padding to use for inference. Defaults to 0.
        binarization_threshold (float, optional): The threshold to binarize the probabilities. Defaults to 0.5.
        mask_erosion_size (int, optional): The size of the disk to use for mask erosion and the edge-cropping.
            Defaults to 10.
        min_object_size (int, optional): The minimum object size to keep in pixel. Defaults to 32.
        quality_level (int | str, optional): The quality level to use for the mask. If a string maps to int.
            high_quality -> 2, low_quality=1, none=0 (apply no masking). Defaults to 0.

    Raises:
        KeyboardInterrupt: If the user interrupts the process.

    """
    run_id = f"{aoi_shapefile.stem}-{start_date}_{end_date}"
    logger.info(f"Starting processing run {run_id}.")

    # Storing the configuration as JSON file
    config = {k: v if isinstance(v, int | float | str | bool) else str(v) for k, v in locals().items()}
    with open(output_data_dir / f"{run_id}.config.json", "w") as f:
        json.dump(config, f)

    from darts_utils.stopuhr import StopUhr

    stopuhr = StopUhr(logger=logger)

    with stopuhr("Print debug info"):
        from darts.utils.cuda import debug_info

        debug_info()

    with stopuhr("Importing libraries"):
        from darts.utils.earthengine import init_ee

        init_ee(ee_project, ee_use_highvolume)

        import odc.geo.xr
        import pandas as pd
        import torch
        from darts_acquisition import load_arcticdem, load_tcvis
        from darts_acquisition.s2 import get_s2ids_from_shape_ee, load_s2_from_gee
        from darts_export import (
            export_binarized,
            export_extent,
            export_polygonized,
            export_probabilities,
            export_thumbnail,
        )
        from darts_postprocessing import prepare_export
        from darts_preprocessing import preprocess_legacy_fast
        from darts_segmentation.segment import SMPSegmenter
        from dask.distributed import Client, LocalCluster
        from odc.stac import configure_rio

        from darts.utils.cuda import decide_device

    with stopuhr("Decide device"):
        device = decide_device(device)

    with stopuhr("Load model"):
        segmenter = SMPSegmenter(model_file, device=torch.device(device))

    # Init Dask stuff with a context manager
    with LocalCluster(n_workers=dask_worker) as cluster, Client(cluster) as client:
        logger.info(f"Using Dask client: {client} on cluster {cluster}")
        logger.info(f"Dashboard available at: {client.dashboard_link}")
        configure_rio(cloud_defaults=True, aws={"aws_unsigned": True}, client=client)
        logger.info("Configured Rasterio with Dask")

        # Iterate over all the data (_path_generator)
        n_tiles = 0
        # paths = sorted(self._path_generator())
        s2ids = get_s2ids_from_shape_ee(aoi_shapefile, start_date, end_date, max_cloud_cover)
        logger.info(f"Found {len(s2ids)} tiles to process.")

        tile_infos = []
        for i, s2id in enumerate(s2ids):
            try:
                tick_start = time.perf_counter()
                with stopuhr("Loading optical data"):
                    outpath = output_data_dir / s2id
                    optical = load_s2_from_gee(s2id, cache=input_cache)

                with stopuhr("Loading ArcticDEM"):
                    arcticdem = load_arcticdem(
                        optical.odc.geobox, arcticdem_dir, resolution=10, buffer=ceil(tpi_outer_radius / 10 * sqrt(2))
                    )
                with stopuhr("Loading TCVis"):
                    tcvis = load_tcvis(optical.odc.geobox, tcvis_dir)

                with stopuhr("Preprocessing"):
                    tile = preprocess_legacy_fast(
                        optical,
                        arcticdem,
                        tcvis,
                        tpi_outer_radius,
                        tpi_inner_radius,
                        device,
                    )
                with stopuhr("Segmenting"):
                    tile = segmenter.segment_tile(
                        tile,
                        patch_size=patch_size,
                        overlap=overlap,
                        batch_size=batch_size,
                        reflection=reflection,
                    )
                with stopuhr("Postprocessing"):
                    tile = prepare_export(
                        tile,
                        bin_threshold=binarization_threshold,
                        mask_erosion_size=mask_erosion_size,
                        min_object_size=min_object_size,
                        quality_level=quality_level,
                        device=device,
                    )

                with stopuhr("Exporting"):
                    outpath.mkdir(parents=True, exist_ok=True)
                    export_probabilities(tile, outpath)
                    export_binarized(tile, outpath)
                    export_polygonized(tile, outpath)
                    export_extent(tile, outpath)
                    export_thumbnail(tile, outpath)
                    # export_datamask(tile, outpath)
                    # export_arcticdem_datamask(tile, outpath)
                    # export_optical(tile, outpath)
                    # export_tcvis(tile, outpath)
                    # export_dem(tile, outpath)

                n_tiles += 1
                tick_end = time.perf_counter()
                duration = tick_end - tick_start
                tile_infos.append({"s2id": s2id, "path": outpath, "successfull": True, "duration": duration})
                logger.info(f"Processed sample {i + 1} of {len(s2ids)} {s2id=} in {duration:.2f}s.")
            except KeyboardInterrupt:
                logger.warning("Keyboard interrupt detected.\nExiting...")
                raise KeyboardInterrupt
            except Exception as e:
                logger.warning(f"Could not process sample {s2id=}.\nSkipping...")
                logger.exception(e)
        else:
            logger.info(f"Processed {n_tiles} tiles to {output_data_dir.resolve()}.")

        tile_infos = pd.DataFrame(tile_infos)
        tile_infos.to_parquet(output_data_dir / f"{run_id}.tile_infos.parquet")
        stopuhr.summary()
        stopuhr.export().to_parquet(output_data_dir / f"{run_id}.stopuhr.parquet")
