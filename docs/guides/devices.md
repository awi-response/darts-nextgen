# Devices

!!! info "Supported devices"

    As of right now, only CUDA and CPU devices are supported.
    How to install a working Python environment for either case please refer to the [installation guide](installation.md).

Some functions can be run on the GPU if a CUDA device is available and the python environment is properly installed with CUDA enabled.
These functions will automatically detect if a CUDA device is available and will use it if so.
It is possible to also force the use of a specific device through the `device` parameter of the respective function.
For most GPU-capable functions it is possible to pass either `cpu` or `cuda` as a string to the `device` parameter.
In a multi-GPU setup, the device can be specified by passing the device index as an integer (e.g. `0` for the first GPU, `1` for the second GPU, etc.).
However, functions which use PyTorch expect the device to be a PyTorch device object, so you need to pass `torch.device("cuda:0")` instead of just `0`.
Which type of device is expected is documented in the respective function documentation.

As of now, the following functions can be run on the GPU:

- [darts_preprocessing.preprocess_legacy_fast](../reference/darts_preprocessing/preprocess_legacy_fast.md) - `device`: `"cpu" | "cuda" | int`
- [darts_postprocessing.prepare_export](../reference/darts_postprocessing/prepare_export.md) - `device`: `"cpu" | "cuda" | int`
- [darts_segmentation.SMPSegmenter](../reference/darts_segmentation/SMPSegmenter.md#darts_segmentation.SMPSegmenter) - `device`: `torch.device`
- [darts_ensemble.EnsembleV1](../reference/darts_ensemble/EnsembleV1.md#darts_ensemble.EnsembleV1) - `device`: `torch.device`
