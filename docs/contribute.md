# Contribute

This page is also meant for internal documentation.

## Editor setup

There is only setup files provided for VSCode and no other editor (yet).
A list of extensions and some settings can be found in the `.vscode`.
At the first start, VSCode should ask you if you want to install the recommended extension.
The settings should be automaticly used by VSCode.
Both should provide the developers with a better experience and enforce code-style.

## Environment setup

Please read and follow the [installation guide](guides/installation.md) to setup the environment.

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
