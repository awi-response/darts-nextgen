"""Pipeline without any other framework."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

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
    # Import here to avoid long loading times when running other commands
    from darts_acquisition.arcticdem import load_arcticdem_from_vrt
    from darts_acquisition.planet import load_planet_masks, load_planet_scene
    from darts_acquisition.tcvis import load_tcvis
    from darts_ensemble.ensemble_v1 import EnsembleV1
    from darts_export.inference import InferenceResultWriter
    from darts_postprocessing import prepare_export
    from darts_preprocessing import preprocess_legacy

    from darts.utils.earthengine import init_ee

    init_ee(ee_project)

    # Find all PlanetScope orthotiles
    for fpath, outpath in planet_file_generator(orthotiles_dir, scenes_dir, output_data_dir):
        try:
            optical = load_planet_scene(fpath)
            arcticdem = load_arcticdem_from_vrt(arcticdem_slope_vrt, arcticdem_elevation_vrt, optical)
            tcvis = load_tcvis(optical, cache_dir)
            data_masks = load_planet_masks(fpath)

            tile = preprocess_legacy(optical, arcticdem, tcvis, data_masks)

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
            logger.warning(f"Could not process folder '{fpath.resolve()}'.\nSkipping...")
            logger.exception(e)


def run_native_planet_pipeline_fast(
    orthotiles_dir: Path,
    scenes_dir: Path,
    output_data_dir: Path,
    arcticdem_dir: Path,
    model_dir: Path,
    tcvis_model_name: str = "RTS_v6_tcvis.pt",
    notcvis_model_name: str = "RTS_v6_notcvis.pt",
    cache_dir: Path | None = None,
    ee_project: str | None = None,
    tpi_outer_radius: int = 30,
    tpi_inner_radius: int = 25,
    patch_size: int = 1024,
    overlap: int = 16,
    batch_size: int = 8,
    reflection: int = 0,
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
        model_dir (Path): The path to the models to use for segmentation.
        tcvis_model_name (str, optional): The name of the model to use for TCVis. Defaults to "RTS_v6_tcvis.pt".
        notcvis_model_name (str, optional): The name of the model to use for not TCVis. Defaults to "RTS_v6_notcvis.pt".
        cache_dir (Path | None, optional): The cache directory. If None, no caching will be used. Defaults to None.
        ee_project (str, optional): The Earth Engine project ID or number to use. May be omitted if
            project is defined within persistent API credentials obtained via `earthengine authenticate`.
        tpi_outer_radius (int, optional): The outer radius of the annulus kernel for the tpi calculation
            in number of cells. Defaults to 30.
        tpi_inner_radius (int, optional): The inner radius of the annulus kernel for the tpi calculation
            in number of cells. Defaults to 25.
        patch_size (int, optional): The patch size to use for inference. Defaults to 1024.
        overlap (int, optional): The overlap to use for inference. Defaults to 16.
        batch_size (int, optional): The batch size to use for inference. Defaults to 8.
        reflection (int, optional): The reflection padding to use for inference. Defaults to 0.
        write_model_outputs (bool, optional): Also save the model outputs, not only the ensemble result.
            Defaults to False.

    """
    # Import here to avoid long loading times when running other commands
    from darts_acquisition.arcticdem import load_arcticdem_tile
    from darts_acquisition.planet import load_planet_masks, load_planet_scene
    from darts_acquisition.tcvis import load_tcvis
    from darts_ensemble.ensemble_v1 import EnsembleV1
    from darts_export.inference import InferenceResultWriter
    from darts_postprocessing import prepare_export
    from darts_preprocessing import preprocess_legacy_fast
    from dask.distributed import Client
    from odc.stac import configure_rio

    from darts.utils.earthengine import init_ee

    init_ee(ee_project)

    client = Client()
    logger.info(f"Using Dask client: {client}")
    configure_rio(cloud_defaults=True, aws={"aws_unsigned": True}, client=client)
    logger.info("Configured Rasterio with Dask")

    # Find all PlanetScope orthotiles
    for fpath, outpath in planet_file_generator(orthotiles_dir, scenes_dir, output_data_dir):
        try:
            optical = load_planet_scene(fpath)
            arcticdem = load_arcticdem_tile(optical.odc.geobox, arcticdem_dir, resolution=2, buffer=tpi_outer_radius)
            tcvis = load_tcvis(optical, cache_dir)
            data_masks = load_planet_masks(fpath)

            tile = preprocess_legacy_fast(optical, arcticdem, tcvis, data_masks, tpi_outer_radius, tpi_inner_radius)

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
            logger.warning(f"Could not process folder '{fpath.resolve()}'.\nSkipping...")
            logger.exception(e)


def run_native_sentinel2_pipeline(
    sentinel2_dir: Path,
    output_data_dir: Path,
    arcticdem_slope_vrt: Path,
    arcticdem_elevation_vrt: Path,
    model_dir: Path,
    tcvis_model_name: str = "RTS_v6_tcvis_s2native.pt",
    notcvis_model_name: str = "RTS_v6_notcvis_s2native.pt",
    cache_dir: Path | None = None,
    ee_project: str | None = None,
    patch_size: int = 1024,
    overlap: int = 16,
    batch_size: int = 8,
    reflection: int = 0,
    write_model_outputs: bool = False,
):
    """Search for all Sentinel scenes in the given directory and runs the segmentation pipeline on them.

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
    # Import here to avoid long loading times when running other commands
    from darts_acquisition.arcticdem import load_arcticdem_from_vrt
    from darts_acquisition.s2 import load_s2_masks, load_s2_scene
    from darts_acquisition.tcvis import load_tcvis
    from darts_ensemble.ensemble_v1 import EnsembleV1
    from darts_export.inference import InferenceResultWriter
    from darts_postprocessing import prepare_export
    from darts_preprocessing import preprocess_legacy

    from darts.utils.earthengine import init_ee

    init_ee(ee_project)

    # Find all Sentinel 2 scenes
    for fpath in sentinel2_dir.glob("*/"):
        try:
            scene_id = fpath.name
            outpath = output_data_dir / scene_id

            optical = load_s2_scene(fpath)
            arcticdem = load_arcticdem_from_vrt(arcticdem_slope_vrt, arcticdem_elevation_vrt, optical)
            tcvis = load_tcvis(optical, cache_dir)
            data_masks = load_s2_masks(fpath, optical.odc.geobox)

            tile = preprocess_legacy(optical, arcticdem, tcvis, data_masks)

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
            logger.warning(f"Could not process folder '{fpath.resolve()}'.\nSkipping...")
            logger.exception(e)


def run_native_sentinel2_pipeline_fast(
    sentinel2_dir: Path,
    output_data_dir: Path,
    arcticdem_dir: Path,
    model_dir: Path,
    tcvis_model_name: str = "RTS_v6_tcvis_s2native.pt",
    notcvis_model_name: str = "RTS_v6_notcvis_s2native.pt",
    cache_dir: Path | None = None,
    ee_project: str | None = None,
    tpi_outer_radius: int = 30,
    tpi_inner_radius: int = 25,
    patch_size: int = 1024,
    overlap: int = 16,
    batch_size: int = 8,
    reflection: int = 0,
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
        model_dir (Path): The path to the models to use for segmentation.
        tcvis_model_name (str, optional): The name of the model to use for TCVis. Defaults to "RTS_v6_tcvis.pt".
        notcvis_model_name (str, optional): The name of the model to use for not TCVis. Defaults to "RTS_v6_notcvis.pt".
        cache_dir (Path | None, optional): The cache directory. If None, no caching will be used. Defaults to None.
        ee_project (str, optional): The Earth Engine project ID or number to use. May be omitted if
            project is defined within persistent API credentials obtained via `earthengine authenticate`.
        tpi_outer_radius (int, optional): The outer radius of the annulus kernel for the tpi calculation
            in number of cells. Defaults to 30.
        tpi_inner_radius (int, optional): The inner radius of the annulus kernel for the tpi calculation
            in number of cells. Defaults to 25.
        patch_size (int, optional): The patch size to use for inference. Defaults to 1024.
        overlap (int, optional): The overlap to use for inference. Defaults to 16.
        batch_size (int, optional): The batch size to use for inference. Defaults to 8.
        reflection (int, optional): The reflection padding to use for inference. Defaults to 0.
        write_model_outputs (bool, optional): Also save the model outputs, not only the ensemble result.
            Defaults to False.

    """
    # Import here to avoid long loading times when running other commands
    from darts_acquisition.arcticdem import load_arcticdem_tile
    from darts_acquisition.s2 import load_s2_masks, load_s2_scene
    from darts_acquisition.tcvis import load_tcvis
    from darts_ensemble.ensemble_v1 import EnsembleV1
    from darts_export.inference import InferenceResultWriter
    from darts_postprocessing import prepare_export
    from darts_preprocessing import preprocess_legacy_fast
    from dask.distributed import Client
    from odc.stac import configure_rio

    from darts.utils.earthengine import init_ee

    init_ee(ee_project)

    client = Client()
    logger.info(f"Using Dask client: {client}")
    configure_rio(cloud_defaults=True, aws={"aws_unsigned": True}, client=client)
    logger.info("Configured Rasterio with Dask")

    # Find all Sentinel 2 scenes
    for fpath in sentinel2_dir.glob("*/"):
        try:
            scene_id = fpath.name
            outpath = output_data_dir / scene_id

            optical = load_s2_scene(fpath)
            arcticdem = load_arcticdem_tile(optical.odc.geobox, arcticdem_dir, resolution=2, buffer=tpi_outer_radius)
            tcvis = load_tcvis(optical, cache_dir)
            data_masks = load_s2_masks(fpath, optical.odc.geobox)

            tile = preprocess_legacy_fast(optical, arcticdem, tcvis, data_masks, tpi_outer_radius, tpi_inner_radius)

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
            logger.warning(f"Could not process folder '{fpath.resolve()}'.\nSkipping...")
            logger.exception(e)
