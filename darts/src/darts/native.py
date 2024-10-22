"""Pipeline without any other framework."""

from pathlib import Path


def run_native_orthotile_pipeline(
    input_data_dir: Path,
    output_data_dir: Path,
    model_dir: Path,
    patch_size: int = 1024,
    overlap: int = 16,
    batch_size: int = 8,
):
    """Search for all PlanetScope scenes in the given directory and runs the segmentation pipeline on them.

    Args:
        input_data_dir (Path): The "input" directory.
        output_data_dir (Path): The "output" directory.
        model_dir (Path): The path to the models to use for segmentation.
        patch_size (int, optional): The patch size to use for inference. Defaults to 1024.
        overlap (int, optional): The overlap to use for inference. Defaults to 16.
        batch_size (int, optional): The batch size to use for inference. Defaults to 8.

    Todo:
        Document the structure of the input data dir.

    """
    # Import here to avoid long loading times when running other commands
    from darts_export.inference import InferenceResultWriter
    from darts_postprocessing import prepare_export
    from darts_preprocessing import load_and_preprocess_planet_scene
    from darts_segmentation import SMPSegmenter

    # Find all PlanetScope scenes
    for fpath in (input_data_dir / "planet" / "PSOrthoTile").glob("*/*/"):
        scene_id = fpath.parent.name
        elevation_path = input_data_dir / "ArcticDEM" / "relative_elevation" / f"{scene_id}_relative_elevation_100.tif"
        slope_path = input_data_dir / "ArcticDEM" / "slope" / f"{scene_id}_slope.tif"
        outpath = output_data_dir / scene_id

        tile = load_and_preprocess_planet_scene(fpath, elevation_path, slope_path)

        model = SMPSegmenter(model_dir / "RTS_v6_notcvis.pt")
        tile = model.segment_tile(tile, patch_size=patch_size, overlap=overlap, batch_size=batch_size)
        tile = prepare_export(tile)

        outpath.mkdir(parents=True, exist_ok=True)
        writer = InferenceResultWriter(tile)
        writer.export_probabilities(outpath)
        writer.export_binarized(outpath)
        writer.export_polygonized(outpath)
