"""Legacy Pipeline without any other framework, but a faster and improved version."""

import logging
from collections.abc import Generator
from pathlib import Path
from typing import Literal

from darts.legacy_pipeline.shared import AquisitionData, _load_planet, _load_s2, _segment_and_export

logger = logging.getLogger(__name__)


def _process_fast(
    data_generator: Generator[tuple[Path, Path, AquisitionData], None, None],
    model_dir: Path,
    tcvis_model_name: str,
    notcvis_model_name: str,
    device: Literal["cuda", "cpu", "auto"] | int | None,
    ee_project: str | None,
    ee_use_highvolume: bool,
    tpi_outer_radius: int,
    tpi_inner_radius: int,
    patch_size: int,
    overlap: int,
    batch_size: int,
    reflection: int,
    binarization_threshold: float,
    mask_erosion_size: int,
    min_object_size: int,
    use_quality_mask: bool,
    write_model_outputs: bool,
):
    # Import here to avoid long loading times when running other commands
    import torch
    from darts_ensemble.ensemble_v1 import EnsembleV1
    from darts_preprocessing import preprocess_legacy_fast
    from dask.distributed import Client
    from odc.stac import configure_rio

    from darts.utils.cuda import debug_info, decide_device
    from darts.utils.earthengine import init_ee

    debug_info()
    device = decide_device(device)
    init_ee(ee_project, ee_use_highvolume)

    client = Client()
    logger.info(f"Using Dask client: {client}")
    configure_rio(cloud_defaults=True, aws={"aws_unsigned": True}, client=client)
    logger.info("Configured Rasterio with Dask")

    ensemble = EnsembleV1(
        model_dir / tcvis_model_name,
        model_dir / notcvis_model_name,
        device=torch.device(device),
    )

    for fpath, outpath, aqdata in data_generator:
        try:
            tile = preprocess_legacy_fast(
                aqdata.optical,
                aqdata.arcticdem,
                aqdata.tcvis,
                aqdata.data_masks,
                tpi_outer_radius,
                tpi_inner_radius,
                device,
            )

            _segment_and_export(
                tile,
                ensemble,
                outpath,
                device,
                patch_size,
                overlap,
                batch_size,
                reflection,
                binarization_threshold,
                mask_erosion_size,
                min_object_size,
                use_quality_mask,
                write_model_outputs,
            )
        except Exception as e:
            logger.warning(f"Could not process folder '{fpath.resolve()}'.\nSkipping...")
            logger.exception(e)


def run_native_planet_pipeline_fast(
    *,
    orthotiles_dir: Path,
    scenes_dir: Path,
    output_data_dir: Path,
    arcticdem_dir: Path,
    tcvis_dir: Path,
    model_dir: Path,
    tcvis_model_name: str = "RTS_v6_tcvis.pt",
    notcvis_model_name: str = "RTS_v6_notcvis.pt",
    device: Literal["cuda", "cpu", "auto"] | int | None = None,
    ee_project: str | None = None,
    ee_use_highvolume: bool = True,
    tpi_outer_radius: int = 100,
    tpi_inner_radius: int = 0,
    patch_size: int = 1024,
    overlap: int = 16,
    batch_size: int = 8,
    reflection: int = 0,
    binarization_threshold: float = 0.5,
    mask_erosion_size: int = 10,
    min_object_size: int = 32,
    use_quality_mask: bool = False,
    write_model_outputs: bool = False,
):
    """Search for all PlanetScope scenes in the given directory and runs the segmentation pipeline on them.

    Loads the ArcticDEM from a datacube instead of VRT which is a lot faster and does not need manual preprocessing.

    Args:
        orthotiles_dir (Path): The directory containing the PlanetScope orthotiles.
        scenes_dir (Path): The directory containing the PlanetScope scenes.
        output_data_dir (Path): The "output" directory.
        arcticdem_dir (Path): The directory containing the ArcticDEM data (the datacube and the extent files).
            Will be created and downloaded if it does not exist.
        tcvis_dir (Path): The directory containing the TCVis data.
        model_dir (Path): The path to the models to use for segmentation.
        tcvis_model_name (str, optional): The name of the model to use for TCVis. Defaults to "RTS_v6_tcvis.pt".
        notcvis_model_name (str, optional): The name of the model to use for not TCVis. Defaults to "RTS_v6_notcvis.pt".
        device (Literal["cuda", "cpu"] | int, optional): The device to run the model on.
            If "cuda" take the first device (0), if int take the specified device.
            If "auto" try to automatically select a free GPU (<50% memory usage).
            Defaults to "cuda" if available, else "cpu".
        ee_project (str, optional): The Earth Engine project ID or number to use. May be omitted if
            project is defined within persistent API credentials obtained via `earthengine authenticate`.
        ee_use_highvolume (bool, optional): Whether to use the high volume server (https://earthengine-highvolume.googleapis.com).
        tpi_outer_radius (int, optional): The outer radius of the annulus kernel for the tpi calculation
            in m. Defaults 100m.
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

    """
    data_generator = _load_planet(
        orthotiles_dir,
        scenes_dir,
        output_data_dir,
        arcticdem_dir,
        tcvis_dir,
        tpi_outer_radius,
    )
    _process_fast(
        data_generator,
        model_dir,
        tcvis_model_name,
        notcvis_model_name,
        device,
        ee_project,
        ee_use_highvolume,
        tpi_outer_radius,
        tpi_inner_radius,
        patch_size,
        overlap,
        batch_size,
        reflection,
        binarization_threshold,
        mask_erosion_size,
        min_object_size,
        use_quality_mask,
        write_model_outputs,
    )


def run_native_sentinel2_pipeline_fast(
    *,
    sentinel2_dir: Path,
    output_data_dir: Path,
    arcticdem_dir: Path,
    tcvis_dir: Path,
    model_dir: Path,
    tcvis_model_name: str = "RTS_v6_tcvis_s2native.pt",
    notcvis_model_name: str = "RTS_v6_notcvis_s2native.pt",
    device: Literal["cuda", "cpu", "auto"] | int | None = None,
    ee_project: str | None = None,
    ee_use_highvolume: bool = True,
    tpi_outer_radius: int = 100,
    tpi_inner_radius: int = 0,
    patch_size: int = 1024,
    overlap: int = 16,
    batch_size: int = 8,
    reflection: int = 0,
    binarization_threshold: float = 0.5,
    mask_erosion_size: int = 10,
    min_object_size: int = 32,
    use_quality_mask: bool = False,
    write_model_outputs: bool = False,
):
    """Search for all Sentinel 2 scenes in the given directory and runs the segmentation pipeline on them.

    Loads the ArcticDEM from a datacube instead of VRT which is a lot faster and does not need manual preprocessing.

    Args:
        sentinel2_dir (Path): The directory containing the Sentinel 2 scenes.
        scenes_dir (Path): The directory containing the PlanetScope scenes.
        output_data_dir (Path): The "output" directory.
        arcticdem_dir (Path): The directory containing the ArcticDEM data (the datacube and the extent files).
            Will be created and downloaded if it does not exist.
        tcvis_dir (Path): The directory containing the TCVis data.
        model_dir (Path): The path to the models to use for segmentation.
        tcvis_model_name (str, optional): The name of the model to use for TCVis. Defaults to "RTS_v6_tcvis.pt".
        notcvis_model_name (str, optional): The name of the model to use for not TCVis. Defaults to "RTS_v6_notcvis.pt".
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

    """
    data_generator = _load_s2(
        sentinel2_dir,
        output_data_dir,
        arcticdem_dir,
        tcvis_dir,
        tpi_outer_radius,
    )
    _process_fast(
        data_generator,
        model_dir,
        tcvis_model_name,
        notcvis_model_name,
        device,
        ee_project,
        ee_use_highvolume,
        tpi_outer_radius,
        tpi_inner_radius,
        patch_size,
        overlap,
        batch_size,
        reflection,
        binarization_threshold,
        mask_erosion_size,
        min_object_size,
        use_quality_mask,
        write_model_outputs,
    )
