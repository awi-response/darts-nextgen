# DARTS nextgen

[![Lint](https://github.com/awi-response/darts-nextgen/actions/workflows/ruff.yml/badge.svg)](https://github.com/awi-response/darts-nextgen/actions/workflows/ruff.yml)
[![CI](https://github.com/awi-response/darts-nextgen/actions/workflows/update_version.yml/badge.svg)](https://github.com/awi-response/darts-nextgen/actions/workflows/update_version.yml)

> Early Alpha!

Panarctic Database of Active Layer Detachment Slides and Retrogressive Thaw Slumps from Deep Learning on High Resolution Satellite Imagery.
This is te successor of the thaw-slump-segmentation (pipeline), with which the first version of the DARTS dataset was created.

## Documentation

The documentation is available at [https://awi-response.github.io/darts-nextgen/](https://awi-response.github.io/darts-nextgen/).
It is recommended to read the [getting started guide](https://awi-response.github.io/darts-nextgen/getting_started) before working with the project.

## Editor setup

There is only setup files provided for VSCode and no other editor (yet).
A list of extensions and some settings can be found in the `.vscode`.
At the first start, VSCode should ask you if you want to install the recommended extension.
The settings should be automaticly used by VSCode.
Both should provide the developers with a better experience and enforce code-style.

## Environment setup

Prereq:

- [uv](https://docs.astral.sh/uv/): `curl -LsSf https://astral.sh/uv/install.sh | sh`
- [cuda](https://developer.nvidia.com/cuda-downloads) (optional)

Install the python environment for CPU-only use:

```sh
uv sync --extra cpu
```

For CUDA support, first look up the version of CUDA you have installed:

```sh
nvidia-smi
# Look at the top right corner for the CUDA version
```

> If the `nvidia-smi` command is not found, you might need to install the nvidia drivers.
> Be very cautious with the installation of the driver, rather read the documentation with care.

We currently support CUDA 11.8, 12.1, 12.4, and 12.6. Use one of the following commands respectively:

```sh
uv sync --extra cuda118
uv sync --extra cuda121
uv sync --extra cuda124
uv sync --extra cuda126
```

> Sometimes the CUDA version must not match exactly, but it is recommended to use the exact version.

Training specific dependencies are optional and therefore not installed by default.
To install them, add `--extra training` to the `uv sync` command, e.g.:

```sh
uv sync --extra cuda126 --extra training
```

!!! info "psycopg2"
    The training dependencies depend on psycopg2, which requires postgresql installed on your system.

To see if the installation was successful, you can run the following command:

```sh
uv run darts --version
```

## Writing docs

The documentation is managed with [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/).
The documentation related dependencies are separated from the main dependencies and can be installed with:

```sh
uv sync --group docs
```

!!! note
    You should combine the `--group docs` with the extras you previously used, e.g. `uv sync --extra training --extra cuda126 --group docs`.

To start the documentation server for live-update, run:

```sh
uv run mkdocs serve
```

In general all mkdocs commands can be run with `uv run mkdocs ...`.

## Recommended Notebook header

The following code snipped can be put in the very first cell of a notebook to already to add logging and initialize earth engine.

```python
import logging

from rich.logging import RichHandler
from rich import traceback

from darts.utils.earthengine import init_ee
from darts.utils.logging import LoggingManager

LoggingManager.setup_logging()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
traceback.install(show_locals=True)  # Change to False if you encounter too large tracebacks
init_ee("ee-project")  # Replace with your project
```
