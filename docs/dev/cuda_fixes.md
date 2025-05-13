# Problems with CUDA

This is a collection of known issues and potential fixes for CUDA-related problems in the codebase.

## CUCIM import

```sh
python3: cufile_worker_thread.h:57: virtual void CUFileThreadPoolWorker::run(): Assertion `0' failed.
```

`cufile.log`:

```log
 07-05-2025 20:29:31:49 [pid=747532 tid=748005] ERROR  cufio_core:55 Threadpool Thread ID:  139915782243904 cuDevicePrimaryCtxRetain failed with error 2
```

**What happend?**
Probably, another user on the cluster is using a GPU in exclusive mode, which prevents other users from using it.
CUCIM is trying to access the GPU, but fails because the GPU is not available.

Please read this [related community issue](https://forums.developer.nvidia.com/t/cufilethreadpoolworker-run-assertion-0-failed/295318):

> From the snippet pasted here, it looks like the cuda device is not available at this time and as a result the cuDevicePrimaryCtxRetain cuda call fails.
> CUDA_ERROR_DEVICE_UNAVAILABLE = 46
> This indicates that requested CUDA device is unavailable at the current time. Devices are often unavailable due to use of CU_COMPUTEMODE_EXCLUSIVE_PROCESS or CU_COMPUTEMODE_PROHIBITED.
> My suspicion is that one of the compute mode settings are effected in this environment preventing the thread to retain the context on the same device.

**How to fix?**
Just set the `CUDA_VISIBLE_DEVICES` environment variable to the device you want to use.
Don't forget to set the `--device` argument of the CLI to `cuda` is such case, since our auto detection will not work in this case.

```sh
export CUDA_VISIBLE_DEVICES=0
```

## NVRTC

```txt
Unable to replicate
```

**What happend?**
The LD_LIBRARY_PATH is either not set correctly or the `cuda-nvrtc` package is not installed in the conda environment.

**How to fix?**
Create a conda environment with the `cuda-nvrtc` package installed. You can do this by running the following command:

```sh
conda create -n cuda_nvrtc -c nvidia cuda-nvrtc
```

This environment must NOT be activated, it is just to install the library files somewhere. Check:

```sh
$ conda list -n cuda120
# packages in environment at /path/to/.conda/envs/cuda_nvrtc:
#
# Name                    Version                   Build  Channel
cuda-nvrtc                12.1.105                      0    nvidia
```

Now you can set the `LD_LIBRARY_PATH` to include the path to the `cuda-nvrtc` library:

```sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/.conda/envs/cuda120/lib/
```

This will allow numba to find the `libnvrtc.so` library when it is needed:

```sh
$ ls -l /path/to/.conda/envs/cuda120/lib/
total 62234
lrwxrwxrwx 1 jokuep001 hpc_user       29 Mar 24 16:25 libnvrtc-builtins.so.12.1 -> libnvrtc-builtins.so.12.1.105
-rwxrwxr-x 3 jokuep001 hpc_user  6846016 Apr  4  2023 libnvrtc-builtins.so.12.1.105
lrwxrwxrwx 1 jokuep001 hpc_user       20 Mar 24 16:25 libnvrtc.so.12 -> libnvrtc.so.12.1.105
-rwxrwxr-x 3 jokuep001 hpc_user 56875328 Apr  4  2023 libnvrtc.so.12.1.105
```

## PTXAS

```sh
LinkerError: [222] Call to cuLinkAddData results in CUDA_ERROR_UNSUPPORTED_PTX_VERSION                                                                                                                                                                                                 
ptxas application ptx input, line 9; fatal   : Unsupported .version 8.7; current version is '8.2'
```

**What happend?**
There is a mismatch between the CUDA version used by the projects code and the CUDA version used by the system.
This error is caused by a mismatch between the OS and the CUDA version of the system.
In our case, the default CUDA version of the system was 12.6 and Ubuntu 22.04.
CUDA 12.6 expects for our use case a PTX version of 8.7, but Ubutnu 22.04 only supports PTX version 8.2.

**How to fix?**

Write a script to set all the environment variables (these may differ on your system):

```sh
export CUDA_PATH="/usr/local/cuda-12.2"
export CUDA_HOME="/usr/local/cuda-12.2"
export PATH="/usr/local/cuda-12.2/bin:$PATH"
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/compat:$LD_LIBRARY_PATH

export NUMBA_CUDA_DRIVER="/usr/local/cuda-12.2/compat/libcuda.so"
export NUMBA_CUDA_INCLUDE_PATH="/usr/local/cuda-12.2/include"
```

and then **source** it:

```sh
source /path/to/your/script.sh
```

Just running `sh script.sh` will not work, since it will create a new shell and the environment variables will not be set in the current shell.
You can also add these lines to your `.bashrc` file, so they are set automatically when you open a new terminal.
However, this may cause problems for other CUDA applications and projects, so be careful with this approach.
Of course, you can also set these variables manually in the terminal before running the code.

## Template

"Error message"

**What happend?**
...

**How to fix?**
...
