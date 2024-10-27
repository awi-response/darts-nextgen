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
    binarized = (tile["probabilities"].fillna(0) > 0.5).astype("uint8")
    tile["binarized_segmentation"] = xr.where(tile["valid_data_mask"], binarized, 0)
    tile["binarized_segmentation"].attrs = {
        "long_name": "Binarized Segmentation",
    }

    # Convert the probabilities to uint8
    # Same but this time with 255 as no-data
    intprobs = (tile["probabilities"] * 100).fillna(255).astype("uint8")
    tile["probabilities"] = xr.where(tile["valid_data_mask"], intprobs, 255)
    tile["probabilities"].attrs = {
        "long_name": "Probabilities",
        "units": "%",
    }
    tile["probabilities"] = tile["probabilities"].rio.write_nodata(255)

    return tile
