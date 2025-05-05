"""Utility functions for working with CUDA devices."""

import logging
from typing import Literal

logger = logging.getLogger(__name__)


def debug_info():
    """Print debug information about the CUDA devices and library installations."""
    import os

    import torch
    from xrspatial.utils import has_cuda_and_cupy

    logger.debug("=== CUDA DEBUG INFO ===")
    logger.debug(f"PyTorch version: {torch.__version__}")
    logger.debug(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    logger.debug(f"Cupy+Numba CUDA available: {has_cuda_and_cupy()}")
    logger.debug(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH')}")

    try:
        from pynvml import (  # type: ignore
            nvmlDeviceGetCount,
            nvmlDeviceGetHandleByIndex,
            nvmlDeviceGetMemoryInfo,
            nvmlDeviceGetName,
            nvmlInit,
            nvmlShutdown,
            nvmlSystemGetCudaDriverVersion_v2,
            nvmlSystemGetDriverVersion,
        )

        nvmlInit()
        driver_version = nvmlSystemGetDriverVersion().decode()
        logger.debug(f"CUDA driver version: {driver_version}")
        cuda_driver_version = nvmlSystemGetCudaDriverVersion_v2()
        logger.debug(f"CUDA runtime version: {cuda_driver_version}")
        ndevices = nvmlDeviceGetCount()
        logger.debug(f"Number of CUDA devices: {ndevices}")

        for i in range(ndevices):
            handle = nvmlDeviceGetHandleByIndex(i)
            device_name = nvmlDeviceGetName(handle).decode()
            meminfo = nvmlDeviceGetMemoryInfo(handle)
            logger.debug(f"Device {i} ({device_name}): {meminfo.used / meminfo.total:.2%} memory usage.")
        nvmlShutdown()

    except ImportError:
        logger.debug("Module 'pynvml' not found, darts is probably installed without CUDA support.")

    try:
        import cupy  # type: ignore

        logger.debug(f"Cupy version: {cupy.__version__}")
        # This is the version which is installed (dynamically linked via PATH or LD_LIBRARY_PATH) in the environment
        env_runtime_version = cupy.cuda.get_local_runtime_version()
        # This is the version which is used by cupy (statically linked)
        cupy_runtime_version = cupy.cuda.runtime.runtimeGetVersion()
        if env_runtime_version != cupy_runtime_version:
            logger.warning(
                "Cupy CUDA runtime versions don't match!\n"
                f"Got {env_runtime_version} as local (dynamically linked) runtime version.\n"
                f"Got {cupy_runtime_version} as by cupy statically linked runtime version.\n"
                "Cupy will use the statically linked runtime version!"
            )
        else:
            logger.debug(f"Cupy CUDA runtime version: {cupy_runtime_version}")
        logger.debug(f"Cupy CUDA driver version: {cupy.cuda.runtime.driverGetVersion()}")
    except ImportError:
        logger.debug("Module 'cupy' not found, darts is probably installed without CUDA support.")

    try:
        import numba.cuda

        cuda_available = numba.cuda.is_available()
        logger.debug(f"Numba CUDA is available: {cuda_available}")
        if cuda_available:
            logger.debug(f"Numba CUDA runtime: {numba.cuda.runtime.get_version()}")
            # logger.debug(f"Numba CUDA has supported devices: {numba.cuda.detect()}")
    except ImportError:
        logger.debug("Module 'numba.cuda' not found, darts is probably installed without CUDA support.")

    try:
        import cucim  # type: ignore

        logger.debug(f"Cucim version: {cucim.__version__}")
    except ImportError:
        logger.debug("Module 'cucim' not found, darts is probably installed without CUDA support.")


def decide_device(device: Literal["cuda", "cpu", "auto"] | int | None) -> Literal["cuda", "cpu"] | int:
    """Decide the device based on the input.

    Args:
        device (Literal["cuda", "cpu", "auto"] | int): The device to run the model on.

    Returns:
        Literal["cuda", "cpu"] | int: The device to run the model on.

    """
    import torch
    from xrspatial.utils import has_cuda_and_cupy

    # We can't provide a default value for device in the parameter list because then we would need to import torch at
    # top-level, which would make the CLI slow.
    if device is None:
        device = "cuda" if torch.cuda.is_available() and has_cuda_and_cupy() else "cpu"
        logger.info(f"Device not provided. Using {device}.")
        return device

    # Automatically select a free GPU (<50% memory usage)
    if device == "auto":
        logger.info(f"{device=}. Trying to automatically select a free GPU. (<50% memory usage)")

        # Check if torch and cupy are available
        if not has_cuda_and_cupy() or not torch.cuda.is_available():
            logger.info("CUDA not available. Using CPU.")
            return "cpu"

        try:
            from pynvml import (  # type: ignore
                nvmlDeviceGetCount,
                nvmlDeviceGetHandleByIndex,
                nvmlDeviceGetMemoryInfo,
                nvmlInit,
                nvmlShutdown,
            )
        except ImportError:
            logger.warning("Module 'pynvml' not found. Using CPU.")
            return "cpu"

        nvmlInit()
        ndevices = nvmlDeviceGetCount()

        # If there are multiple devices, we need to check which one is free
        for i in range(ndevices):
            handle = nvmlDeviceGetHandleByIndex(i)
            meminfo = nvmlDeviceGetMemoryInfo(handle)
            perc_used = meminfo.used / meminfo.total
            logger.debug(f"Device {i}: {perc_used:.2%} memory usage.")
            # If the device is less than 50% used, we skip it
            if perc_used > 0.5:
                continue
            else:
                nvmlShutdown()
                logger.info(f"Using free GPU {i} ({perc_used:.2%} memory usage).")
                return i
        else:
            nvmlShutdown()
            logger.warning(
                "No free GPU found (<50% memory usage). Using CPU. "
                "If you want to use a GPU, please select a device manually with the 'device' parameter."
            )
            return "cpu"

    # If device is int or "cuda" or "cpu", we just return it
    logger.info(f"Using {device=}.")
    return device
