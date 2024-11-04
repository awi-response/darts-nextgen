"""Prepare the export, e.g. binarizes the data and convert the float probabilities to uint8."""

import xarray as xr


def prepare_export(tile: xr.Dataset) -> xr.Dataset:
    """Prepare the export, e.g. binarizes the data and convert the float probabilities to uint8.

    Args:
        tile (xr.Dataset): Input tile from inference and / or an ensemble.

    Returns:
        xr.Dataset: Output tile.

    """

    def _prep_layer(tile, layername, binarized_layer_name):
        # Binarize the segmentation
        # Where the output from the ensemble / segmentation is nan turn it into 0, else threshold it
        # Also, where there was no valid input data, turn it into 0
        binarized = (tile[layername].fillna(0) > 0.5).astype("uint8")
        tile[binarized_layer_name] = xr.where(tile["valid_data_mask"], binarized, 0)
        tile[binarized_layer_name].attrs = {
            "long_name": "Binarized Segmentation",
        }

        # Convert the probabilities to uint8
        # Same but this time with 255 as no-data
        intprobs = (tile[layername] * 100).fillna(255).astype("uint8")
        tile[layername] = xr.where(tile["valid_data_mask"], intprobs, 255)
        tile[layername].attrs = {
            "long_name": "Probabilities",
            "units": "%",
        }
        tile[layername] = tile[layername].rio.write_nodata(255)
        return tile

    tile = _prep_layer(tile, "probabilities", "binarized_segmentation")
    if "probabilities-tcvis" in tile:
        tile = _prep_layer(tile, "probabilities-tcvis", "binarized_segmentation-tcvis")
    if "probabilities-notcvis" in tile:
        tile = _prep_layer(tile, "probabilities-notcvis", "binarized_segmentation-notcvis")

    return tile
