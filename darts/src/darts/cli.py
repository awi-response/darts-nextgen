"""Entrypoint for the darts-pipeline CLI."""

import logging
from pathlib import Path
from typing import Annotated

import cyclopts
from darts_acquisition.arcticdem import create_arcticdem_vrt
from rich.console import Console

from darts import __version__
from darts.native import run_native_orthotile_pipeline
from darts.utils.config import config_parser
from darts.utils.logging import add_logging_handlers, setup_logging

logger = logging.getLogger(__name__)
console = Console()


app = cyclopts.App(
    version=__version__,
    console=console,
    config=config_parser,  # config=cyclopts.config.Toml("config.toml", root_keys=["darts"], search_parents=True)
)

pipeline_group = cyclopts.Group.create_ordered("Pipeline Commands")
data_group = cyclopts.Group.create_ordered("Data Commands")


# @app.command
def hello(name: str, n: int = 1):
    """Say hello to someone.

    Args:
        name (str): The name of the person to say hello to
        n (int, optional): The number of times to say hello. Defaults to 1.

    Raises:
        ValueError: If n is 3.

    """
    for i in range(n):
        logger.debug(f"Currently at {i=}")
        if n == 3:
            raise ValueError("I don't like 3")
        logger.info(f"Hello {name}")


app.command(group=pipeline_group)(run_native_orthotile_pipeline)
app.command(group=data_group)(create_arcticdem_vrt)


# Intercept the logging behavior to add a file handler
@app.meta.default
def launcher(  # noqa: D103
    *tokens: Annotated[str, cyclopts.Parameter(show=False, allow_leading_hyphen=True)], log_dir: Path = Path("logs")
):
    command, bound = app.parse_args(tokens)
    add_logging_handlers(command.__name__, console, log_dir)
    return command(*bound.args, **bound.kwargs)


def start_app():
    """Wrapp to start the app."""
    try:
        setup_logging()
        app.meta()
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Closing...")
    except SystemExit:
        logger.info("Closing...")
    except Exception as e:
        logger.exception(e)
