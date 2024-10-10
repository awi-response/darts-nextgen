"""Prepare the export, e.g. binarizes the data and convert the float probabilities to uint8."""

import xarray as xr


def prepare_export(tile: xr.Dataset) -> xr.Dataset:
    """Prepare the export, e.g. binarizes the data and convert the float probabilities to uint8.

    Args:
        tile (xr.Dataset): Input tile from inference and / or an ensemble.

    Returns:
        xr.Dataset: Output tile.

    """
    # Binarize the segmentation
    tile["binarized_segmentation"] = (tile["probabilities"] > 0.5).astype("uint8")

    # Convert the probabilities to uint8
    intprobs = (tile["probabilities"] * 100).astype("uint8")
    tile["probabilities"] = xr.where(~tile["probabilities"].isnull(), intprobs, 255)  # noqa: PD003

    return tile
