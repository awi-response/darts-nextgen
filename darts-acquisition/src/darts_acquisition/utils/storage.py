import numpy as np
import zarr


def optimize_coord_encoding(values, dx):
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
