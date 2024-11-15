"""Utility functions related to storage of e.g. zarr arrays."""

from typing import TypedDict

import numpy as np
import zarr
import zarr.codecs
from numcodecs.abc import Codec


class CoordEncoding(TypedDict):
    """TypedDict for the encoding of regularly spaced coordinates."""

    compressor: zarr.Blosc
    filters: tuple[Codec, Codec]


def optimize_coord_encoding(values: np.ndarray, dx: int) -> CoordEncoding:
    """Optimize zarr encoding of regularly spaced coordinates.

    Taken from https://github.com/earth-mover/serverless-datacube-demo/blob/a15423b9734898f52468bebc441e29ccf3789410/src/lib.py#L280

    Args:
        values (np.ndarray): The coordinates to encode
        dx (int): The spacing between the coordinates

    Returns:
        CoordEncoding: A dictionary containing the zarr compressor and filters to use

    """
    dx_all = np.diff(values)
    # dx = dx_all[0]
    np.testing.assert_allclose(dx_all, dx), "must be regularly spaced"

    offset_codec = zarr.FixedScaleOffset(offset=values[0], scale=1 / dx, dtype=values.dtype, astype="i8")
    delta_codec = zarr.Delta("i8", "i2")
    compressor = zarr.Blosc(cname="zstd")

    enc0 = offset_codec.encode(values)
    # everything should be offset by 1 at this point
    np.testing.assert_equal(np.unique(np.diff(enc0)), [1])
    enc1 = delta_codec.encode(enc0)
    # now we should be able to compress the shit out of this
    enc2 = compressor.encode(enc1)
    decoded = offset_codec.decode(delta_codec.decode(compressor.decode(enc2)))

    # will produce numerical precision differences
    # np.testing.assert_equal(values, decoded)
    np.testing.assert_allclose(values, decoded)

    return {"compressor": compressor, "filters": (offset_codec, delta_codec)}
