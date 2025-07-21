"""Utility functions for working with CUDA devices."""

import logging
import os
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)


def debug_info():  # noqa: C901
    """Print debug information about the CUDA devices and library installations."""  # noqa: DOC501
    logger.debug("===vvv CUDA DEBUG INFO vvv===")
    important_env_vars = [
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDA_VISIBLE_DEVICES",
        "LD_LIBRARY_PATH",
        "NUMBA_CUDA_DRIVER",
        "NUMBA_CUDA_INCLUDE_PATH",
    ]
    for v in important_env_vars:
        value = os.environ.get(v, "UNSET")
        logger.debug(f"{v}: {value}")

    logger.debug("Quicknote: CUDA driver is something different than CUDA runtime, hence versions can mismatch")
    try:
        from pynvml import (  # type: ignore
            NVMLError,
            nvmlDeviceGetCount,
            nvmlDeviceGetHandleByIndex,
            nvmlDeviceGetMemoryInfo,
            nvmlDeviceGetName,
            nvmlInit,
            nvmlShutdown,
            nvmlSystemGetCudaDriverVersion_v2,
            nvmlSystemGetDriverVersion,
        )

        try:
            nvmlInit()
            cuda_driver_version_legacy = nvmlSystemGetDriverVersion().decode()
            cuda_driver_version = nvmlSystemGetCudaDriverVersion_v2()
            logger.debug(f"CUDA driver version: {cuda_driver_version} ({cuda_driver_version_legacy})")
            ndevices = nvmlDeviceGetCount()
            logger.debug(f"Number of CUDA devices: {ndevices}")

            for i in range(ndevices):
                handle = nvmlDeviceGetHandleByIndex(i)
                device_name = nvmlDeviceGetName(handle).decode()
                meminfo = nvmlDeviceGetMemoryInfo(handle)
                logger.debug(f"Device {i} ({device_name}): {meminfo.used / meminfo.total:.2%} memory usage.")
            nvmlShutdown()
        except NVMLError:
            raise ImportError

    except ImportError:
        logger.debug("Module 'pynvml' could not be imported. darts is probably installed without CUDA support.")
    except Exception as e:
        logger.error(
            "Error while trying to get CUDA driver version or device info."
            " Is is possible that this device has no CUDA device?"
        )
        logger.exception(e, exc_info=True)

    try:
        import torch

        logger.debug(f"PyTorch version: {torch.__version__}")
        logger.debug(f"PyTorch CUDA available: {torch.cuda.is_available()}")
        logger.debug(f"PyTorch CUDA runtime version: {torch.version.cuda}")
    except ImportError as e:
        logger.error("Module 'torch' could not be imported:")
        logger.exception(e, exc_info=True)

    try:
        import cupy  # type: ignore

        logger.debug(f"Cupy version: {cupy.__version__}")
        cupy_driver_version = cupy.cuda.runtime.driverGetVersion()
        logger.debug(f"Cupy CUDA driver version: {cupy_driver_version}")
        # This is the version which is installed (dynamically linked via PATH or LD_LIBRARY_PATH) in the environment
        env_runtime_version = cupy.cuda.get_local_runtime_version()
        logger.debug(f"Cupy CUDA runtime version: {env_runtime_version}")
        if cupy_driver_version < env_runtime_version:
            logger.warning(
                "CUDA runtime version is newer than CUDA driver version!"
                " The CUDA environment is probably not setup correctly!"
                " Consider linking CUDA to an older version with CUDA_HOME and LD_LIBRARY_PATH environment variables,"
                " or in case of a setup done by pixi choose a different environment with the -e flag."
            )
        # This is the version which is was used when cupy was compiled (statically linked)
        cupy_runtime_version = cupy.cuda.runtime.runtimeGetVersion()
        if env_runtime_version != cupy_runtime_version:
            logger.debug(
                "Cupy CUDA runtime versions don't match!\n"
                f"Got {env_runtime_version} as local (dynamically linked) runtime version.\n"
                f"Got {cupy_runtime_version} as by cupy statically linked runtime version.\n"
                "This can happen if cupy was compiled using a different CUDA runtime version. "
                "Things should still work, note that Cupy will use the dynamically linked version."
            )
    except ImportError:
        logger.debug("Module 'cupy' not found, darts is probably installed without CUDA support.")

    try:
        import numba.cuda

        cuda_available = numba.cuda.is_available()
        logger.debug(f"Numba CUDA is available: {cuda_available}")
        if cuda_available:
            logger.debug(f"Numba CUDA runtime version: {numba.cuda.runtime.get_version()}")
            # logger.debug(f"Numba CUDA has supported devices: {numba.cuda.detect()}")
    except ImportError:
        logger.debug("Module 'numba.cuda' not found, darts is probably installed without CUDA support.")

    from xrspatial.utils import has_cuda_and_cupy

    logger.debug(f"Cupy+Numba CUDA available: {has_cuda_and_cupy()}")

    try:
        import cucim  # type: ignore

        logger.debug(f"Cucim version: {cucim.__version__}")
    except ImportError:
        logger.debug("Module 'cucim' not found, darts is probably installed without CUDA support.")

    logger.debug("===^^^ CUDA DEBUG INFO ^^^===")


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


def set_pixi_cuda_env(version: Literal["cuda121", "cuda124", "cuda126", "cuda128"], prefix: str = "../"):
    """Set the CUDA environment variables.

    This is useful when working with a notebook wich does not load the Pixi environment.

    !!! warnign

        This does not work currently as expected!

    Args:
        version (str): The version string of the pixi CUDA envirnonent.
        prefix (str): The prefix to the PIXI installation. E.g. when in the `./notebooks` directory, the prefix is `../`

    """
    prefix = str(Path(prefix).resolve())
    os.environ["PATH"] = f"{prefix}/.pixi/envs/{version}/bin:{os.environ.get('PATH', '')}"
    os.environ["CUDA_HOME"] = f"{prefix}/.pixi/envs/{version}"
    os.environ["CUDA_PATH"] = f"{prefix}/.pixi/envs/{version}"
    os.environ["LD_LIBRARY_PATH"] = f"{prefix}/.pixi/envs/{version}/lib:{prefix}/.pixi/envs/{version}/lib/stubs"
    os.environ["NUMBA_CUDA_DRIVER"] = f"{prefix}/.pixi/envs/{version}/lib/stubs/libcuda.so"
    os.environ["NUMBA_CUDA_INCLUDE_PATH"] = f"{prefix}/.pixi/envs/{version}/include"
