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
    # Where the output from the ensemble / segmentation is nan turn it into 0, else threshold it
    # Also, where there was no valid input data, turn it into 0
    binarized = xr.where(~tile["probabilities"].isnull(), (tile["probabilities"] > 0.5), 0).astype("uint8")  # noqa: PD003
    tile["binarized_segmentation"] = xr.where(tile["valid_data_mask"], binarized, 0)

    # Convert the probabilities to uint8
    # Same but this time with 255 as no-data
    intprobs = (tile["probabilities"] * 100).astype("uint8")
    intprobs = xr.where(~tile["probabilities"].isnull(), intprobs, 255)  # noqa: PD003
    tile["probabilities"] = xr.where(tile["valid_data_mask"], intprobs, 255)

    return tile
