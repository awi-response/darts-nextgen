"""Automated legacy pipeline for Sentinel-2 data."""

import logging
import multiprocessing as mp
from math import ceil, sqrt
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)


def run_native_sentinel2_pipeline_from_aoi(
    aoi_shapefile: Path,
    output_data_dir: Path = Path("data/output"),
    tcvis_dir: Path = Path("data/download/tcvis"),
    arcticdem_dir: Path = Path("data/download/arcticdem"),
    model_dir: Path = Path("models"),
    tcvis_model_name: str = "RTS_v6_tcvis_s2native.pt",
    notcvis_model_name: str = "RTS_v6_notcvis_s2native.pt",
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
    use_quality_mask: bool = False,
    write_model_outputs: bool = False,
):
    """Pipeline for Sentinel 2 data with optimized preprocessing.

    Args:
        aoi_shapefile (Path): The path to the shapefile containing the AOI. Can be anything readable by geopandas.
        output_data_dir (Path): The "output" directory. Defaults to Path("data/output").
        arcticdem_dir (Path): The directory containing the ArcticDEM data (the datacube and the extent files).
            Will be created and downloaded if it does not exist.
            Defaults to Path("data/download/arcticdem").
        tcvis_dir (Path): The directory containing the TCVis data. Defaults to Path("data/download/tcvis").
        model_dir (Path): The path to the models to use for segmentation. Defaults to Path("models").
        tcvis_model_name (str, optional): The name of the model to use for TCVis. Defaults to "RTS_v6_tcvis.pt".
        notcvis_model_name (str, optional): The name of the model to use for not TCVis. Defaults to "RTS_v6_notcvis.pt".
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
        use_quality_mask (bool, optional): Whether to use the "quality" mask instead of the "valid" mask
            to mask the output.
        write_model_outputs (bool, optional): Also save the model outputs, not only the ensemble result.
            Defaults to False.

    Raises:
        KeyboardInterrupt: If the user interrupts the process.

    """
    from darts.utils.cuda import debug_info

    debug_info()

    from darts.utils.earthengine import init_ee

    init_ee(ee_project, ee_use_highvolume)

    import torch
    from darts_acquisition import load_arcticdem, load_tcvis
    from darts_acquisition.s2 import get_s2ids_from_shape_ee, load_s2_from_gee
    from darts_ensemble.ensemble_v1 import EnsembleV1
    from darts_export.inference import InferenceResultWriter
    from darts_postprocessing import prepare_export
    from darts_preprocessing import preprocess_legacy_fast
    from dask.distributed import Client, LocalCluster
    from odc.stac import configure_rio
    from rich.progress import track

    from darts.utils.cuda import decide_device

    device = decide_device(device)

    ensemble = EnsembleV1(
        model_dir / tcvis_model_name,
        model_dir / notcvis_model_name,
        device=torch.device(device),
    )

    # Init Dask stuff with a context manager
    with LocalCluster(n_workers=dask_worker) as cluster, Client(cluster) as client:
        logger.info(f"Using Dask client: {client} on cluster {cluster}")
        logger.info(f"Dashboard available at: {client.dashboard_link}")
        configure_rio(cloud_defaults=True, aws={"aws_unsigned": True}, client=client)
        logger.info("Configured Rasterio with Dask")

        # Iterate over all the data (_path_generator)
        n_tiles = 0
        # paths = sorted(self._path_generator())
        s2ids = get_s2ids_from_shape_ee(aoi_shapefile)
        logger.info(f"Found {len(s2ids)} tiles to process.")
        for i, s2id in track(enumerate(s2ids)):
            try:
                outpath = output_data_dir / s2id
                optical = load_s2_from_gee(s2id)
                arcticdem = load_arcticdem(
                    optical.odc.geobox, arcticdem_dir, resolution=10, buffer=ceil(tpi_outer_radius / 10 * sqrt(2))
                )
                tcvis = load_tcvis(optical.odc.geobox, tcvis_dir)

                tile = preprocess_legacy_fast(
                    optical,
                    arcticdem,
                    tcvis,
                    tpi_outer_radius,
                    tpi_inner_radius,
                    device,
                )
                tile = ensemble.segment_tile(
                    tile,
                    patch_size=patch_size,
                    overlap=overlap,
                    batch_size=batch_size,
                    reflection=reflection,
                    keep_inputs=write_model_outputs,
                )
                tile = prepare_export(
                    tile,
                    binarization_threshold,
                    mask_erosion_size,
                    min_object_size,
                    use_quality_mask,
                    device,
                )

                outpath.mkdir(parents=True, exist_ok=True)
                writer = InferenceResultWriter(tile)
                writer.export_probabilities(outpath)
                writer.export_binarized(outpath)
                writer.export_polygonized(outpath)
                n_tiles += 1
                logger.info(f"Processed sample {i + 1} of {len(s2ids)} '{s2id=}'.")
            except KeyboardInterrupt:
                logger.warning("Keyboard interrupt detected.\nExiting...")
                raise KeyboardInterrupt
            except Exception as e:
                logger.warning(f"Could not process sample '{s2id=}'.\nSkipping...")
                logger.exception(e)
        else:
            logger.info(f"Processed {n_tiles} tiles to {output_data_dir.resolve()}.")
