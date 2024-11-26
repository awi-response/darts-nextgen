"""Data loading for legacy Pipeline."""

import logging
from collections import namedtuple
from math import ceil, sqrt
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

AquisitionData = namedtuple("AquisitionData", ["optical", "arcticdem", "tcvis", "data_masks"])


def _planet_file_generator(orthotiles_dir: Path, scenes_dir: Path, output_data_dir: Path):
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


def _load_s2(sentinel2_dir: Path, output_data_dir: Path, arcticdem_dir: Path, tcvis_dir: Path, tpi_outer_radius: int):
    from darts_acquisition.arcticdem import load_arcticdem_tile
    from darts_acquisition.s2 import load_s2_masks, load_s2_scene
    from darts_acquisition.tcvis import load_tcvis

    for fpath in sentinel2_dir.glob("*/"):
        scene_id = fpath.name
        outpath = output_data_dir / scene_id
        try:
            optical = load_s2_scene(fpath)
            arcticdem = load_arcticdem_tile(
                optical.odc.geobox, arcticdem_dir, resolution=10, buffer=ceil(tpi_outer_radius / 10 * sqrt(2))
            )
            tcvis = load_tcvis(optical.odc.geobox, tcvis_dir)
            data_masks = load_s2_masks(fpath, optical.odc.geobox)
            aqdata = AquisitionData(optical, arcticdem, tcvis, data_masks)
            yield fpath, outpath, aqdata
        except Exception as e:
            logger.warning(f"Could not process folder '{fpath.resolve()}'.\nSkipping...")
            logger.exception(e)
            continue


def _load_planet(
    orthotiles_dir: Path,
    scenes_dir: Path,
    output_data_dir: Path,
    arcticdem_dir: Path,
    tcvis_dir: Path,
    tpi_outer_radius: int,
):
    from darts_acquisition.arcticdem import load_arcticdem_tile
    from darts_acquisition.planet import load_planet_masks, load_planet_scene
    from darts_acquisition.tcvis import load_tcvis

    # Find all PlanetScope orthotiles
    for fpath, outpath in _planet_file_generator(orthotiles_dir, scenes_dir, output_data_dir):
        try:
            optical = load_planet_scene(fpath)
            arcticdem = load_arcticdem_tile(
                optical.odc.geobox, arcticdem_dir, resolution=2, buffer=ceil(tpi_outer_radius / 2 * sqrt(2))
            )
            tcvis = load_tcvis(optical.odc.geobox, tcvis_dir)
            data_masks = load_planet_masks(fpath)
            aqdata = AquisitionData(optical, arcticdem, tcvis, data_masks)
            yield fpath, outpath, aqdata
        except Exception as e:
            logger.warning(f"Could not process folder '{fpath.resolve()}'.\nSkipping...")
            logger.exception(e)
            continue


def _segment_and_export(
    tile,
    ensemble,
    outpath: Path,
    device: Literal["cuda", "cpu", "auto"] | int | None,
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
    from darts_export.inference import InferenceResultWriter
    from darts_postprocessing import prepare_export

    tile = ensemble.segment_tile(
        tile,
        patch_size=patch_size,
        overlap=overlap,
        batch_size=batch_size,
        reflection=reflection,
        keep_inputs=write_model_outputs,
    )
    tile = prepare_export(tile, binarization_threshold, mask_erosion_size, min_object_size, use_quality_mask, device)

    outpath.mkdir(parents=True, exist_ok=True)
    writer = InferenceResultWriter(tile)
    writer.export_probabilities(outpath)
    writer.export_binarized(outpath)
    writer.export_polygonized(outpath)
