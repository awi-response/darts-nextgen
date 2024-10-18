"""Pipeline without any other framework."""

import logging
from pathlib import Path

import xarray as xr
from darts_export.inference import InferenceResultWriter
from darts_postprocessing.prepare_export import prepare_export
from darts_preprocessing.preprocess_tobi import load_and_preprocess_planet_scene
from darts_segmentation.segment import SMPSegmenter
from lovely_tensors import monkey_patch
from rich import traceback
from rich.logging import RichHandler

xr.set_options(display_expand_data=False)

# Set up logging
logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
logging.getLogger("darts_preprocessing").setLevel(logging.DEBUG)
logging.getLogger("darts_segmentation").setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)

monkey_patch()
traceback.install(show_locals=True)


def run_native_orthotile_pipeline(input_data_dir: Path, output_data_dir: Path):
    """Search for all PlanetScope scenes in the given directory and runs the.

    Args:
        input_data_dir (Path): The "input" directory.
        output_data_dir (Path): The "output" directory.

    Todo:
        Document the structure of the input data dir.

    """
    # Find all PlanetScope scenes
    for fpath in (input_data_dir / "planet" / "PSOrthoTile").glob("*/*/"):
        scene_id = fpath.parent.name
        elevation_path = input_data_dir / "ArcticDEM" / "relative_elevation" / f"{scene_id}_relative_elevation_100.tif"
        slope_path = input_data_dir / "ArcticDEM" / "slope" / f"{scene_id}_slope.tif"
        outpath = output_data_dir / scene_id

        tile = load_and_preprocess_planet_scene(fpath, elevation_path, slope_path)

        model = SMPSegmenter("../models/RTS_v6_notcvis.pt")
        tile = model.segment_tile(tile, patch_size=256, batch_size=2)
        tile = prepare_export(tile)

        outpath.mkdir(parents=True, exist_ok=True)
        writer = InferenceResultWriter(tile)
        writer.export_probabilities(outpath)
        writer.export_binarized(outpath)
