import torch
from xrspatial.utils import has_cuda_and_cupy
import os
print(os.environ.get('CUDA_PATH'))
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
print(f"Cupy+Numba CUDA available: {has_cuda_and_cupy()}")
try:
    from pynvml import (
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
    print(f"CUDA driver version: {driver_version}")
    cuda_driver_version = nvmlSystemGetCudaDriverVersion_v2()
    print(f"CUDA runtime version: {cuda_driver_version}")
    ndevices = nvmlDeviceGetCount()
    print(f"Number of CUDA devices: {ndevices}")
except Exception as e:
    print(e)