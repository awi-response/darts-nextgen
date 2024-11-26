"""Legacy Pipeline without any other framework."""

import logging
from collections.abc import Generator
from pathlib import Path
from typing import Literal

from darts.legacy_pipeline.shared import AquisitionData, _load_planet, _load_s2, _segment_and_export

logger = logging.getLogger(__name__)


def _process(
    data_generator: Generator[tuple[Path, Path, AquisitionData], None, None],
    model_dir: Path,
    tcvis_model_name: str,
    notcvis_model_name: str,
    device: Literal["cuda", "cpu", "auto"] | int | None,
    ee_project: str | None,
    ee_use_highvolume: bool,
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
    from darts_preprocessing import preprocess_legacy

    from darts.utils.cuda import debug_info, decide_device
    from darts.utils.earthengine import init_ee

    debug_info()
    device = decide_device(device)
    init_ee(ee_project, ee_use_highvolume)

    ensemble = EnsembleV1(
        model_dir / tcvis_model_name,
        model_dir / notcvis_model_name,
        device=torch.device(device),
    )

    for fpath, outpath, aqdata in data_generator:
        try:
            tile = preprocess_legacy(aqdata.optical, aqdata.arcticdem, aqdata.tcvis, aqdata.data_masks)

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


def run_native_planet_pipeline(
    *,
    orthotiles_dir: Path,
    scenes_dir: Path,
    output_data_dir: Path,
    arcticdem_slope_vrt: Path,
    arcticdem_elevation_vrt: Path,
    tcvis_dir: Path,
    model_dir: Path,
    tcvis_model_name: str = "RTS_v6_tcvis.pt",
    notcvis_model_name: str = "RTS_v6_notcvis.pt",
    device: Literal["cuda", "cpu", "auto"] | int | None = None,
    ee_project: str | None = None,
    ee_use_highvolume: bool = True,
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

    Args:
        orthotiles_dir (Path): The directory containing the PlanetScope orthotiles.
        scenes_dir (Path): The directory containing the PlanetScope scenes.
        output_data_dir (Path): The "output" directory.
        arcticdem_slope_vrt (Path): The path to the ArcticDEM slope VRT file.
        arcticdem_elevation_vrt (Path): The path to the ArcticDEM elevation VRT file.
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

    Examples:
        ### PS Orthotile

        Data directory structure:

        ```sh
            data/input
            ├── ArcticDEM
            │   ├── elevation.vrt
            │   ├── slope.vrt
            │   ├── relative_elevation
            │   │   └── 4372514_relative_elevation_100.tif
            │   └── slope
            │       └── 4372514_slope.tif
            └── planet
                └── PSOrthoTile
                    └── 4372514/5790392_4372514_2022-07-16_2459
                        ├── 5790392_4372514_2022-07-16_2459_BGRN_Analytic_metadata.xml
                        ├── 5790392_4372514_2022-07-16_2459_BGRN_DN_udm.tif
                        ├── 5790392_4372514_2022-07-16_2459_BGRN_SR.tif
                        ├── 5790392_4372514_2022-07-16_2459_metadata.json
                        └── 5790392_4372514_2022-07-16_2459_udm2.tif
        ```

        then the config should be

        ```
        ...
        orthotiles_dir: data/input/planet/PSOrthoTile
        arcticdem_slope_vrt: data/input/ArcticDEM/slope.vrt
        arcticdem_elevation_vrt: data/input/ArcticDEM/elevation.vrt
        ```

        ### PS Scene

        Data directory structure:

        ```sh
            data/input
            ├── ArcticDEM
            │   ├── elevation.vrt
            │   ├── slope.vrt
            │   ├── relative_elevation
            │   │   └── 4372514_relative_elevation_100.tif
            │   └── slope
            │       └── 4372514_slope.tif
            └── planet
                └── PSScene
                    └── 20230703_194241_43_2427
                        ├── 20230703_194241_43_2427_3B_AnalyticMS_metadata.xml
                        ├── 20230703_194241_43_2427_3B_AnalyticMS_SR.tif
                        ├── 20230703_194241_43_2427_3B_udm2.tif
                        ├── 20230703_194241_43_2427_metadata.json
                        └── 20230703_194241_43_2427.json
        ```

        then the config should be

        ```
        ...
        scenes_dir: data/input/planet/PSScene
        arcticdem_slope_vrt: data/input/ArcticDEM/slope.vrt
        arcticdem_elevation_vrt: data/input/ArcticDEM/elevation.vrt
        ```


    """
    data_generator = _load_planet(
        orthotiles_dir,
        scenes_dir,
        output_data_dir,
        arcticdem_slope_vrt,
        arcticdem_elevation_vrt,
        tcvis_dir,
        tpi_outer_radius=10,
    )
    _process(
        data_generator,
        model_dir,
        tcvis_model_name,
        notcvis_model_name,
        device,
        ee_project,
        ee_use_highvolume,
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


def run_native_sentinel2_pipeline(
    *,
    sentinel2_dir: Path,
    output_data_dir: Path,
    arcticdem_slope_vrt: Path,
    arcticdem_elevation_vrt: Path,
    tcvis_dir: Path,
    model_dir: Path,
    tcvis_model_name: str = "RTS_v6_tcvis_s2native.pt",
    notcvis_model_name: str = "RTS_v6_notcvis_s2native.pt",
    device: Literal["cuda", "cpu", "auto"] | int | None = None,
    ee_project: str | None = None,
    ee_use_highvolume: bool = True,
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
    """Search for all Sentinel scenes in the given directory and runs the segmentation pipeline on them.

    Args:
        sentinel2_dir (Path): The directory containing the Sentinel 2 scenes.
        output_data_dir (Path): The "output" directory.
        arcticdem_slope_vrt (Path): The path to the ArcticDEM slope VRT file.
        arcticdem_elevation_vrt (Path): The path to the ArcticDEM elevation VRT file.
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

    Examples:
        Data directory structure:

        ```sh
            data/input
            ├── ArcticDEM
            │   ├── elevation.vrt
            │   ├── slope.vrt
            │   ├── relative_elevation
            │   │   └── 4372514_relative_elevation_100.tif
            │   └── slope
            │       └── 4372514_slope.tif
            └── sentinel2
                └── 20220826T200911_20220826T200905_T17XMJ/
                    ├── 20220826T200911_20220826T200905_T17XMJ_SCL_clip.tif
                    └── 20220826T200911_20220826T200905_T17XMJ_SR_clip.tif
        ```

        then the config should be

        ```
        ...
        sentinel2_dir: data/input/sentinel2
        arcticdem_slope_vrt: data/input/ArcticDEM/slope.vrt
        arcticdem_elevation_vrt: data/input/ArcticDEM/elevation.vrt
        ```


    """
    data_generator = _load_s2(
        sentinel2_dir,
        output_data_dir,
        arcticdem_slope_vrt,
        arcticdem_elevation_vrt,
        tcvis_dir,
        tpi_outer_radius=10,
    )
    _process(
        data_generator,
        model_dir,
        tcvis_model_name,
        notcvis_model_name,
        device,
        ee_project,
        ee_use_highvolume,
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
