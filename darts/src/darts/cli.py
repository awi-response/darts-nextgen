"""Entrypoint for the darts-pipeline CLI."""

import logging
import os
import sys
from pathlib import Path
from typing import Annotated

import cyclopts
import rich
from darts_segmentation.training import (
    convert_lightning_checkpoint,
    cross_validation_smp,
    test_smp,
    train_smp,
    tune_smp,
)

from darts import __version__
from darts.pipelines import (
    AOISentinel2BlockPipeline,
    AOISentinel2Pipeline,
    AOISentinel2RayPipeline,
    PlanetBlockPipeline,
    PlanetPipeline,
    PlanetRayPipeline,
    Sentinel2BlockPipeline,
    Sentinel2Pipeline,
    Sentinel2RayPipeline,
)
from darts.training import (
    preprocess_planet_train_data,
    preprocess_planet_train_data_for_nina,
    preprocess_planet_train_data_pingo,
    preprocess_s2_train_data,
)
from darts.utils.bench import benchviz
from darts.utils.config import ConfigParser
from darts.utils.logging import LoggingManager

root_file = Path(__file__).resolve()
logger = logging.getLogger(__name__)

config_parser = ConfigParser()
app = cyclopts.App(
    version=__version__,
    console=rich.get_console(),
    config=config_parser,
    help_format="plaintext",
    version_format="plaintext",
)

pipeline_group = cyclopts.Group.create_ordered("Pipeline Commands")
data_group = cyclopts.Group.create_ordered("Data Commands")
train_group = cyclopts.Group.create_ordered("Training Commands")


@app.command
def hello(name: str, *, n: int = 1):
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
def shell():
    """Open an interactive shell."""
    app.interactive_shell()


@app.command
def help():
    """Display the help screen."""
    app.help_print()


@app.command
def env_info():
    """Print debug information about the environment."""
    from darts.utils.cuda import debug_info

    logger.debug(f"PATH: {os.environ.get('PATH', 'UNSET')}")
    debug_info()


app.command(name="run-sequential-aoi-sentinel2-pipeline", group=pipeline_group)(AOISentinel2Pipeline.cli)
app.command(name="run-sequential-sentinel2-pipeline", group=pipeline_group)(Sentinel2Pipeline.cli)
app.command(name="run-sequential-planet-pipeline", group=pipeline_group)(PlanetPipeline.cli)
app.command(name="run-ray-aoi-sentinel2-pipeline", group=pipeline_group)(AOISentinel2RayPipeline.cli)
app.command(name="run-ray-sentinel2-pipeline", group=pipeline_group)(Sentinel2RayPipeline.cli)
app.command(name="run-ray-planet-pipeline", group=pipeline_group)(PlanetRayPipeline.cli)
app.command(name="run-block-sentinel2-pipeline", group=pipeline_group)(Sentinel2BlockPipeline.cli)
app.command(name="run-block-aoi-sentinel2-pipeline", group=pipeline_group)(AOISentinel2BlockPipeline.cli)
app.command(name="run-block-planet-pipeline", group=pipeline_group)(PlanetBlockPipeline.cli)
app.command(group=pipeline_group)(benchviz)

app.command(group=train_group)(preprocess_planet_train_data)
app.command(group=train_group)(preprocess_planet_train_data_pingo)
app.command(group=train_group)(preprocess_planet_train_data_for_nina)
app.command(group=train_group)(preprocess_s2_train_data)
app.command(group=train_group)(train_smp)
app.command(group=train_group)(test_smp)
app.command(group=train_group)(convert_lightning_checkpoint)
app.command(group=train_group)(cross_validation_smp)
app.command(group=train_group)(tune_smp)


# Intercept the logging behavior to add a file handler
@app.meta.default
def launcher(  # noqa: D103
    *tokens: Annotated[str, cyclopts.Parameter(show=False, allow_leading_hyphen=True)],
    log_dir: Path = Path("logs"),
    config_file: Path = Path("config.toml"),
    verbose: bool = False,
    tracebacks_show_locals: bool = False,
    log_plain: bool = False,
):
    command, bound, ignored = app.parse_args(tokens, verbose=verbose)
    # Set verbose to true for debug stuff like env_info
    if command.__name__ == "env_info":
        verbose = True
    LoggingManager.add_logging_handlers(command.__name__, log_dir, verbose, tracebacks_show_locals, log_plain=log_plain)
    logger.debug(f"Running on Python version {sys.version} from {__name__} ({root_file})")
    additional_args = {}
    if "config_file" in ignored:
        additional_args["config_file"] = config_file
    if "log_dir" in ignored:
        additional_args["log_dir"] = log_dir
    if "verbose" in ignored:
        additional_args["verbose"] = verbose
    return command(*bound.args, **bound.kwargs, **additional_args)


def start_app():
    """Wrapp to start the app."""
    try:
        # First time initialization of the logging manager
        LoggingManager.setup_logging()
        app.meta()
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Closing...")
    except SystemExit:
        logger.info("Closing...")
    except Exception as e:
        logger.exception(e)
