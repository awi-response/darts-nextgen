"""Check if outputpath already contains files."""

import logging
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__.replace("darts_", "darts."))


def _missing_files(output_dir: Path, file_names: list[str]) -> list[str]:
    """Check if the given files exist in the output directory.

    Args:
        output_dir (Path): The directory to check for files.
        file_names (list[str]): The list of file names to check.

    Returns:
        list[str]: A list of missing file names.

    """
    missing_files = []
    for file_name in file_names:
        file_path = output_dir / file_name
        if not file_path.exists():
            missing_files.append(file_name)
    return missing_files


def missing_outputs(  # noqa: C901
    out_dir: Path,
    bands: list[str] = ["probabilities", "binarized", "polygonized", "extent", "thumbnail"],
    ensemble_subsets: list[str] = [],
) -> Literal["all", "some", "none"]:
    """Check for missing output files in the given directory.

    Args:
        out_dir (Path): The directory to check for missing files.
        bands (list[str], optional): The bands to export. Defaults to ["probabilities"].
        ensemble_subsets (list[str], optional): The ensemble subsets to export. Defaults to [].

    Returns:
        Literal["all", "some", "none"]: A string indicating the status of missing files:
            - "none": No files are missing.
            - "some": Some files are missing, which one will be logged to debug.
            - "all": All files are missing.

    Raises:
        ValueError: If the output path is not a directory.

    """
    if not out_dir.exists():
        return []
    if not out_dir.is_dir():
        raise ValueError(f"Output path {out_dir} is not a directory.")
    expected_files = []
    for band in bands:
        match band:
            case "polygonized":
                expected_files += ["prediction_segments.gpkg"] + [
                    f"prediction_segments-{es}.gpkg" for es in ensemble_subsets
                ]
                expected_files += ["prediction_segments.parquet"] + [
                    f"prediction_segments-{es}.parquet" for es in ensemble_subsets
                ]
            case "binarized":
                expected_files += ["binarized.tif"] + [f"binarized-{es}.tif" for es in ensemble_subsets]
            case "probabilities":
                expected_files += ["probabilities.tif"] + [f"probabilities-{es}.tif" for es in ensemble_subsets]
            case "extent":
                expected_files += ["extent.gpkg", "extent.parquet"]
            case "thumbnail":
                expected_files += ["thumbnail.jpg"]
            case _:
                expected_files += [f"{band}.tif"]

    missing_files = _missing_files(out_dir, expected_files)
    if len(missing_files) == 0:
        return "none"
    elif len(missing_files) == len(expected_files):
        return "all"
    else:
        logger.debug(
            f"Missing files in {out_dir}: {', '.join(missing_files)}. Expected files: {', '.join(expected_files)}."
        )
        return "some"
