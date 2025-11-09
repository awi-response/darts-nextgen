# Advanced Installation

!!! danger "A lot of caveats and side effects"

    Please read this complete guide before doing anything on your machine!
    Especially handling CUDA stuff can be tricky and lead to problems.
    This project tries to make the installation as easy as possible while still being flexible about the OS and GPU setup.
    Furthermore, we try to **not infer with existing setups** done on your machine - that is why we use `uv` and `pixi` in the first place.
    But this is only possible if you follow the instructions carefully.

Prereq:

- [uv](https://docs.astral.sh/uv/): `curl -LsSf https://astral.sh/uv/install.sh | sh`
- [gcloud CLI](https://cloud.google.com/sdk/docs/install)
- [cuda](https://developer.nvidia.com/cuda-downloads) (optional for GPU support)

This project uses `uv` to manage the python environment.
If you are not familiar with `uv` yet, please read their documentation first.
Please don't use `pip` or `conda` to install the dependencies, as this often leads to problems.
We have spend a lot of time making sure that the install process is easy and quick, but this is only possible with `uv`.
So please use it.

!!! info "Using conda"

    Sometimes, it is not possible to only use `uv`, e.g. if CUDA is not installed correctly and one does not have access to fix this.
    In these scenarios, one would use `conda` or `mamba` to create a new environment and install the required packages.
    However, we strongly discourage this, since it requires a lot of manual work and knowledge about what to install.
    E.g. did you know, that installing the `cuda-toolkit` from conda-forge will always install version 11.8?
    To install the correct version, one would need to use the `nvidia` channel and install the `cuda-toolkit` from there.
    Or even better, the `cuda` package, which installs all CUDA-related stuff.
    **Instead use `pixi`***, we provide a ready to use environment setup in our project.
    Read more about this in the respective section below.

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
Currently CUDA 12.1, and 12.6 are supported, but sometimes other versions work as well.
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
uv sync --extra cuda118 # Not well tested
uv sync --extra cuda121
uv sync --extra cuda124 # Not well tested
uv sync --extra cuda126 
uv sync --extra cuda128 # Not well tested
```

!!! info "CUDA version missmatch"
    Sometimes it is possible to use a different CUDA version for the python packages than the one installed.
    E.g. we tested our code on a system with CUDA 12.2 installed, but used the python packages for CUDA 12.1.
    This is not recommended, but sometimes it works.
    In general, one should NOT use a higher CUDA version for the python packages than the one installed on the system.

Install the python environment for CPU-only use:

```sh
uv sync --extra cpu
```

!!! danger
    Either `--extra cpu` or `--extra cudaXXX` must be specified.
    Without important libraries like PyTorch will not be installed and the environment will not work.

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

## Installation with pixi

!!! warning "Experimental"

    The following section is experimental and not yet fully tested.

!!! warning

    Using `pixi` instead of `uv` is not as ergonomic, because one needs to further wrap the commands or work in an shell.

Prereq:

- [pixi](https://pixi.sh/dev/): `curl -fsSL https://pixi.sh/install.sh | sh`

Pixi is a tool to manage system-environment using conda packages.
Hence, a lot of the functionality is similar to normal `conda`, however, `pixi` follows a project related approach.

In case you need to install `cuda` in an isolated environment, this is the way to go.
The idea is to use `pixi` to create a conda environment and within this environment use `uv` to manage the python packages in an virtual environment.
The conda environment will contain the necessary system packages, like `cuda-nvcc` or `cuda-toolkit`.
To ensure that the python packages all refereing (linking) to these packages, all the installation done with `uv` will be done inside the `pixi` environment.
The following graphic should illustrate this:

![Environment Setup](../assets/environments.drawio.png){ loading=lazy }

We provide different conda environments over pixi: `cuda121`, `cuda124`, `cuda126`, `cuda128`, and `default` (no cuda).
To run the installation in one of these environments use the `pixi run` command:

```sh
pixi run -e cuda126 uv sync --extra cuda126 --extra training
pixi run -e cuda126 uv run darts --help
```

However, this results in a lot of chaining of commands, which beccome very verbose when using multiple commands.
Instead, it is possible to create a pixi shell, in which the uv commands can be run directly, as described in the other sections:

```sh
pixi shell -e cuda126
uv sync --extra cuda126 --extra training
uv run darts --help
```

This is similar to `conda activate ...`, but in a better and more encapsulated way.
It is also possible to exit this environment any time:

```sh
exit
```

Another way is to run the `darts` command directly via pixi, since we enable a passthrough, which will run `uv run darts` for you:

```sh
pixi run darts --help
```

However, with this method, it is still required that the venv is installed inside the conda environment of pixi like described above.
