"""Pipeline without any other framework."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def planet_file_generator(orthotiles_dir: Path, scenes_dir: Path, output_data_dir: Path):
    """Generate a list of files and output paths from planet scenes and orthotiles.

    Args:
        orthotiles_dir (Path): The directory containing the PlanetScope orthotiles.
        scenes_dir (Path): The directory containing the PlanetScope scenes.
        output_data_dir (Path): The "output" directory.

    Yields:
        Tuple[Path, Path]: A tuple containing the input file path and the output directory path.

    """
    # Find all PlanetScope orthotiles
    for fpath in orthotiles_dir.glob("*/*/"):
        tile_id = fpath.parent.name
        scene_id = fpath.name
        outpath = output_data_dir / tile_id / scene_id
        yield fpath, outpath

    # Find all PlanetScope scenes
    for fpath in scenes_dir.glob("*/"):
        scene_id = fpath.name
        outpath = output_data_dir / scene_id
        yield fpath, outpath


def run_native_planet_pipeline(
    orthotiles_dir: Path,
    scenes_dir: Path,
    output_data_dir: Path,
    arcticdem_slope_vrt: Path,
    arcticdem_elevation_vrt: Path,
    model_dir: Path,
    tcvis_model_name: str = "RTS_v6_tcvis.pt",
    notcvis_model_name: str = "RTS_v6_notcvis.pt",
    cache_dir: Path | None = None,
    ee_project: str | None = None,
    patch_size: int = 1024,
    overlap: int = 16,
    batch_size: int = 8,
    reflection: int = 0,
    write_model_outputs: bool = False,
):
    """Search for all PlanetScope scenes in the given directory and runs the segmentation pipeline on them.

    Args:
        orthotiles_dir (Path): The directory containing the PlanetScope orthotiles.
        scenes_dir (Path): The directory containing the PlanetScope scenes.
        output_data_dir (Path): The "output" directory.
        arcticdem_slope_vrt (Path): The path to the ArcticDEM slope VRT file.
        arcticdem_elevation_vrt (Path): The path to the ArcticDEM elevation VRT file.
        model_dir (Path): The path to the models to use for segmentation.
        tcvis_model_name (str, optional): The name of the model to use for TCVis. Defaults to "RTS_v6_tcvis.pt".
        notcvis_model_name (str, optional): The name of the model to use for not TCVis. Defaults to "RTS_v6_notcvis.pt".
        cache_dir (Path | None, optional): The cache directory. If None, no caching will be used. Defaults to None.
        ee_project (str, optional): The Earth Engine project ID or number to use. May be omitted if
            project is defined within persistent API credentials obtained via `earthengine authenticate`.
        patch_size (int, optional): The patch size to use for inference. Defaults to 1024.
        overlap (int, optional): The overlap to use for inference. Defaults to 16.
        batch_size (int, optional): The batch size to use for inference. Defaults to 8.
        reflection (int, optional): The reflection padding to use for inference. Defaults to 0.
        write_model_outputs (bool, optional): Also save the model outputs, not only the ensemble result.
            Defaults to False.

    Todo:
        Document the structure of the input data dir.

    """
    # Import here to avoid long loading times when running other commands
    from darts_ensemble.ensemble_v1 import EnsembleV1
    from darts_export.inference import InferenceResultWriter
    from darts_postprocessing import prepare_export
    from darts_preprocessing import load_and_preprocess_planet_scene

    from darts.utils.earthengine import init_ee

    init_ee(ee_project)

    # Find all PlanetScope orthotiles
    for fpath, outpath in planet_file_generator(orthotiles_dir, scenes_dir, output_data_dir):
        try:
            tile = load_and_preprocess_planet_scene(fpath, arcticdem_slope_vrt, arcticdem_elevation_vrt, cache_dir)

            ensemble = EnsembleV1(model_dir / tcvis_model_name, model_dir / notcvis_model_name)
            tile = ensemble.segment_tile(
                tile,
                patch_size=patch_size,
                overlap=overlap,
                batch_size=batch_size,
                reflection=reflection,
                keep_inputs=write_model_outputs,
            )
            tile = prepare_export(tile)

            outpath.mkdir(parents=True, exist_ok=True)
            writer = InferenceResultWriter(tile)
            writer.export_probabilities(outpath)
            writer.export_binarized(outpath)
            writer.export_polygonized(outpath)
        except Exception as e:
            logger.warning(f"could not process folder '{fpath.absolute()}' (Error: {e})")


def run_native_sentinel2_pipeline(
    sentinel2_dir: Path,
    output_data_dir: Path,
    arcticdem_slope_vrt: Path,
    arcticdem_elevation_vrt: Path,
    model_dir: Path,
    tcvis_model_name: str = "RTS_v6_tcvis.pt",
    notcvis_model_name: str = "RTS_v6_notcvis.pt",
    cache_dir: Path | None = None,
    ee_project: str | None = None,
    patch_size: int = 1024,
    overlap: int = 16,
    batch_size: int = 8,
    reflection: int = 0,
    write_model_outputs: bool = False,
):
    """Search for all PlanetScope scenes in the given directory and runs the segmentation pipeline on them.

    Args:
        sentinel2_dir (Path): The directory containing the Sentinel 2 scenes.
        output_data_dir (Path): The "output" directory.
        arcticdem_slope_vrt (Path): The path to the ArcticDEM slope VRT file.
        arcticdem_elevation_vrt (Path): The path to the ArcticDEM elevation VRT file.
        model_dir (Path): The path to the models to use for segmentation.
        tcvis_model_name (str, optional): The name of the model to use for TCVis. Defaults to "RTS_v6_tcvis.pt".
        notcvis_model_name (str, optional): The name of the model to use for not TCVis. Defaults to "RTS_v6_notcvis.pt".
        cache_dir (Path | None, optional): The cache directory. If None, no caching will be used. Defaults to None.
        ee_project (str, optional): The Earth Engine project ID or number to use. May be omitted if
            project is defined within persistent API credentials obtained via `earthengine authenticate`.
        patch_size (int, optional): The patch size to use for inference. Defaults to 1024.
        overlap (int, optional): The overlap to use for inference. Defaults to 16.
        batch_size (int, optional): The batch size to use for inference. Defaults to 8.
        reflection (int, optional): The reflection padding to use for inference. Defaults to 0.
        write_model_outputs (bool, optional): Also save the model outputs, not only the ensemble result.
            Defaults to False.

    Todo:
        Document the structure of the input data dir.

    """
    # Import here to avoid long loading times when running other commands
    from darts_ensemble.ensemble_v1 import EnsembleV1
    from darts_export.inference import InferenceResultWriter
    from darts_postprocessing import prepare_export
    from darts_preprocessing import load_and_preprocess_sentinel2_scene

    from darts.utils.earthengine import init_ee

    init_ee(ee_project)

    # Find all Sentinel 2 scenes
    for fpath in sentinel2_dir.glob("*/"):
        try:
            scene_id = fpath.name
            outpath = output_data_dir / scene_id

            tile = load_and_preprocess_sentinel2_scene(fpath, arcticdem_slope_vrt, arcticdem_elevation_vrt, cache_dir)

            ensemble = EnsembleV1(model_dir / tcvis_model_name, model_dir / notcvis_model_name)
            tile = ensemble.segment_tile(
                tile,
                patch_size=patch_size,
                overlap=overlap,
                batch_size=batch_size,
                reflection=reflection,
                keep_inputs=write_model_outputs,
            )
            tile = prepare_export(tile)

            outpath.mkdir(parents=True, exist_ok=True)
            writer = InferenceResultWriter(tile)
            writer.export_probabilities(outpath)
            writer.export_binarized(outpath)
            writer.export_polygonized(outpath)

        except Exception as e:
            logger.warning(f"could not process folder '{fpath.relative_to(sentinel2_dir)}' (Error: {e})")
