from functools import partial  # noqa: D100

import dask.array as da
import numpy as np
import xarray as xr
from numba import cuda
from xrspatial.utils import ArrayTypeFunctionMapping, cuda_args, ngjit, not_implemented_func

# 3rd-party
try:
    import cupy
except ImportError:

    class cupy:  # noqa: D101, N801
        ndarray = False


RADIAN = 180 / np.pi


@ngjit
def _run_numpy(data: np.ndarray, d: int):
    data = data.astype(np.float32)
    out = np.zeros_like(data, dtype=np.float32)
    out[:] = np.nan
    rows, cols = data.shape
    for y in range(d, rows - d):
        for x in range(d, cols - d):
            aoi = data[y - d : y + d + 1, x - d : x + d + 1]
            aoi_min = aoi.min()
            aoi_max = aoi.max()
            if aoi_max == aoi_min:
                out[y, x] = 0
            elif aoi_max == 0:
                out[y, x] = float("nan")
            else:
                out[y, x] = (aoi_max - aoi_min) / (aoi_max)
    return out


@cuda.jit
def _run_gpu(arr, d, out):
    i, j = cuda.grid(2)
    di = d
    dj = d
    if i - di >= 0 and i + di < out.shape[0] and j - dj >= 0 and j + dj < out.shape[1]:
        # aoi = arr[i - di : i + di + 1, j - dj : j + dj + 1]
        aoi_min = np.inf
        aoi_max = -np.inf
        for y in range(i - di, i + di + 1):
            for x in range(j - dj, j + dj + 1):
                v = arr[y, x]
                if v < aoi_min:
                    aoi_min = v
                if v > aoi_max:
                    aoi_max = v
        if aoi_max != 0:
            out[i, j] = (aoi_max - aoi_min) / aoi_max


def _run_cupy(data: cupy.ndarray, d: int) -> cupy.ndarray:
    data = data.astype(cupy.float32)
    griddim, blockdim = cuda_args(data.shape)
    out = cupy.empty(data.shape, dtype="float32")
    out[:] = cupy.nan
    _run_gpu[griddim, blockdim](data, d, out)
    return out


def _run_dask_numpy(data: da.Array, d: int) -> da.Array:
    data = data.astype(np.float32)
    _func = partial(_run_numpy, d=d)
    out = data.map_overlap(_func, depth=(d, d), boundary=np.nan, meta=np.array(()))
    return out


def dissection_index(agg: xr.DataArray, window_size: int = 3, name: str | None = "dissection_index") -> xr.DataArray:
    mapper = ArrayTypeFunctionMapping(
        numpy_func=_run_numpy,
        dask_func=_run_dask_numpy,
        cupy_func=_run_cupy,
        dask_cupy_func=lambda *args: not_implemented_func(
            *args, messages="dissection_index() does not support dask with cupy backed DataArray"
        ),
    )

    out = mapper(agg)(agg.data, (window_size - 1) // 2)

    return xr.DataArray(out, name=name, coords=agg.coords, dims=agg.dims, attrs=agg.attrs)
