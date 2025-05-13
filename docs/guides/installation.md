# Advanced Installation

Prereq:

- [uv](https://docs.astral.sh/uv/): `curl -LsSf https://astral.sh/uv/install.sh | sh`
- [gcloud CLI](https://cloud.google.com/sdk/docs/install)
- [cuda](https://developer.nvidia.com/cuda-downloads) (optional for GPU support)

This project uses `uv` to manage the python environment.
If you are not familiar with `uv` yet, please read their documentation first.
Please don't use `pip` or `conda` to install the dependencies, as this often leads to problems.
We have spend a lot of time making sure that the install process is easy and quick, but this is only possible with `uv`.
So please use it.

In general the environment can be installed with `uv sync`.
However, this project depends on some libraries (torch and torchvision) which don't get installed per default.
Therefore you need to specify an extra flag to install the correct dependencies, e.g.

```sh
uv sync --extra cuda126
```

This will install the environment with the correct dependencies for CUDA 12.6.
The following sections will explain the different extra flags and groups which can be used to install the environment for different purposes and systems.

## CUDA and CPU-only installations

Several CUDA versions can be used, but it may happen that some problems occur on different systems.
Currently CUDA 11.8, 12.1, and 12.6 are supported, but sometimes other versions work as well.
We use python extra dependencies, so it is possible to specify the CUDA version via an `--extra` flag in the `uv sync` command.

You can check the currently installed CUDA version via:

```sh
nvidia-smi
# Look at the top right corner for the CUDA version
```

!!! warning
    If the `nvidia-smi` command is not found, you might need to install the nvidia drivers.
    Be very cautious with the installation of the driver, rather read the documentation with care.

To install the python environment for a specific CUDA version use one of the following commands respectively:

```sh
uv sync --extra cuda118
uv sync --extra cuda121
uv sync --extra cuda126
```

!!! info "CUDA version missmatch"
    Sometimes it is possible to use a different CUDA version for the python packages than the one installed.
    E.g. we tested our code on a system with CUDA 12.2 installed, but used the python packages for CUDA 12.1.
    This is not recommended, but sometimes it works.

Install the python environment for CPU-only use:

```sh
uv sync --extra cpu
```

!!! danger
    Either `--extra cpu` or `--extra cudaXXX` must be specified.
    Without important libraries like PyTorch will not be installed and the environment will not work.

### Workaround for CUDA related errors

If CUDA is not installed correctly, some CUDA optional packages are missing or the wrong version of CUDA is installed, conda / mamba can be used as a workaround.

First create a new conda environment and activate it:

```sh
mamba create -n darts-nextgen-cuda-env
mamba activate darts-nextgen-cuda-env
```

Then install CUDA toolkit and required system packages via conda / mamba:

```sh
mamba install cuda-toolkit nvidia::cuda-nvrtc
...
```

Now you can (while the conda / mamba environment is active) sync your uv environment.

```sh
uv sync --...
```

## Training specific dependencies

Training specific dependencies are optional and therefore not installed by default.
To install them, add `--extra training` to the `uv sync` command, e.g.:

```sh
uv sync --extra cuda126 --extra training
```

!!! info "psycopg2"
    The training dependencies depend on psycopg2, which requires postgresql installed on your system.

## Packages for the documentation

Packages which are used to create this documentation are not installed by default and are not available via as an extra.
Instead they are installed as part of an optional `dependency-group`, or `group` for short.
To install the documentation dependencies, add `--group docs` to the `uv sync` command, e.g.:

```sh
uv sync --extra cuda126 --extra training --group docs
```
