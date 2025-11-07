# Devices

!!! info "Supported devices"

    As of right now, only CUDA and CPU devices are supported.
    How to install a working Python environment for either case please refer to the [installation guide](installation.md).

Some functions can be accelerated by a GPU if a CUDA device is available and the python environment is properly installed with CUDA enabled.
These functions will automatically detect if a CUDA device is available and will use it if so and not specified otherwise.

!!! Fix "Debuggin CUDA"

    There is a complete [page about CUDA related errors](../dev/cuda_fixes.md).

## Device selection with the CLI

The CLIs pipeline commands, e.g. `darts inference sentinel2-sequential` allow for specifying a device with the `--device` flag:

```sh
uv run darts inference sentinel2-sequential --device cuda
```

One can pass either

- `auto`: This will automatically select a free GPU based on memory usage (<50%). If no GPU is available will use the CPU instead.
- `cuda`: This will use the first (0) device, fails if no GPU is available.
- `cpu`: This will use the CPU.
- an integer: This will use the specified GPU, e.g. `2` will take the third availble GPU.

Of course the device can also be set by setting the `CUDA_VISIBLE_DEVICES` environment variable.

## Device selection within specific functions

It is possible to force the use of a specific device through the `device` parameter of the respective function.
For most GPU-capable functions it is possible to pass either `cpu` or `cuda` as a string to the `device` parameter.
In a multi-GPU setup, the device can be specified by passing the device index as an integer (e.g. `0` for the first GPU, `1` for the second GPU, etc.).
However, functions which use PyTorch expect the device to be a PyTorch device object, so you need to pass `torch.device("cuda:0")` instead of just `0`.
Which type of device is expected is documented in the respective function documentation.

As of now, the following functions can be accelerated by the GPU:

- [darts_acquisition.load_cdse_s2_sr_scene][] - `device`: `"cpu" | "cuda" | int`
- [darts_acquisition.load_gee_s2_sr_scene][] - `device`: `"cpu" | "cuda" | int`
- [darts_preprocessing.preprocess_v2][] - `device`: `"cpu" | "cuda" | int`
- [darts_segmentation.segment.SMPSegmenter][] - `device`: `torch.device`
- [darts_ensemble.EnsembleV1][] - `device`: `torch.device`
- [darts_postprocessing.prepare_export][] - `device`: `"cpu" | "cuda" | int`

!!! tip "Compute backends"

    All preprocessing-engineering functions are written without the need of a specific compute backend, thanks to xarrays widely compatibility.
    This allows for passing xarray Datasets with e.g. a `dask` or `cupy` backend and the computations will happens with this backend automatically.
