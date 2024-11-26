"""Entrypoint for the darts-pipeline CLI."""

import logging
import sys
from pathlib import Path
from typing import Annotated

import cyclopts
from rich.console import Console

from darts import __version__
from darts.legacy_pipeline import (
    run_native_planet_pipeline,
    run_native_planet_pipeline_fast,
    run_native_sentinel2_pipeline,
    run_native_sentinel2_pipeline_fast,
)
from darts.utils.config import ConfigParser
from darts.utils.logging import add_logging_handlers, setup_logging

root_file = Path(__file__).resolve()
logger = logging.getLogger(__name__)
console = Console()

config_parser = ConfigParser()
app = cyclopts.App(
    version=__version__,
    console=console,
    config=config_parser,
    help_format="plaintext",
    version_format="plaintext",
)

pipeline_group = cyclopts.Group.create_ordered("Pipeline Commands")
data_group = cyclopts.Group.create_ordered("Data Commands")


@app.command
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


@app.command
def env_info():
    """Print debug information about the environment."""
    from darts.utils.cuda import debug_info

    debug_info()


app.command(group=pipeline_group)(run_native_planet_pipeline)
app.command(group=pipeline_group)(run_native_planet_pipeline_fast)
app.command(group=pipeline_group)(run_native_sentinel2_pipeline)
app.command(group=pipeline_group)(run_native_sentinel2_pipeline_fast)


# Custom wrapper for the create_arcticdem_vrt function, which dodges the loading of all the heavy modules
@app.command(group=data_group)
def create_arcticdem_vrt(dem_data_dir: Path, vrt_target_dir: Path):
    """Create a VRT file from ArcticDEM data.

    Args:
        dem_data_dir (Path): The directory containing the ArcticDEM data (.tif).
        vrt_target_dir (Path): The output directory.

    """
    from darts_acquisition.arcticdem.vrt import create_arcticdem_vrt as _create_arcticdem_vrt

    _create_arcticdem_vrt(dem_data_dir, vrt_target_dir)


# Intercept the logging behavior to add a file handler
@app.meta.default
def launcher(  # noqa: D103
    *tokens: Annotated[str, cyclopts.Parameter(show=False, allow_leading_hyphen=True)],
    log_dir: Path = Path("logs"),
    config_file: Path = Path("config.toml"),
):
    command, bound, _ = app.parse_args(tokens)
    add_logging_handlers(command.__name__, console, log_dir)
    logger.debug(f"Running on Python version {sys.version} from {__name__} ({root_file})")
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
