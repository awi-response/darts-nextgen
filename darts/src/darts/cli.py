"""Entrypoint for the darts-pipeline CLI."""

import logging
import os
import sys
from pathlib import Path
from typing import Annotated

import cyclopts
import rich
from darts_segmentation.training import (
    cross_validation_smp,
    test_smp,
    train_smp,
    tune_smp,
    validate_dataset,
)
from darts_utils.paths import DefaultPaths, paths

from darts import __version__
from darts.pipelines import (
    PlanetPipeline,
    PlanetRayPipeline,
    Sentinel2Pipeline,
    Sentinel2RayPipeline,
)
from darts.training import (
    preprocess_planet_train_data,
    preprocess_planet_train_data_pingo,
    preprocess_s2_train_data,
)
from darts.utils.bench import benchviz
from darts.utils.config import ConfigParser
from darts.utils.logging import LoggingManager, VerbosityLevel

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

subcommands_group = cyclopts.Group.create_ordered("Pipelines & Scripts")


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


@app.command
def debug_paths(default_paths: DefaultPaths = DefaultPaths()):
    """Debug and print the current DARTS paths.

    Args:
        default_paths (DefaultPaths, optional): Default paths to set before logging.
            Defaults to DefaultPaths().

    """
    paths_instance = paths
    paths_instance.set_defaults(default_paths)
    paths_instance.log_all_paths(level=logging.INFO)


inference_app = cyclopts.App(name="inference", group=subcommands_group, help="Predefined inference pipelines")
app.command(inference_app)
sequential_group = cyclopts.Group.create_ordered("Sequential Pipelines")
inference_app.command(name="sentinel2-sequential", group=sequential_group)(Sentinel2Pipeline.cli)
inference_app.command(name="planet-sequential", group=sequential_group)(PlanetPipeline.cli)
ray_group = cyclopts.Group.create_ordered("Ray Pipelines")
inference_app.command(name="sentinel2-ray", group=ray_group)(Sentinel2RayPipeline.cli)
inference_app.command(name="planet-ray", group=ray_group)(PlanetRayPipeline.cli)
utilities_group = cyclopts.Group.create_ordered("Utilities")
inference_app.command(group=utilities_group)(benchviz)

inference_data_app = cyclopts.App(name="prep-data", group=utilities_group, help="Data preparation for offline use")
inference_app.command(inference_data_app)
inference_data_app.command(name="sentinel2")(Sentinel2Pipeline.cli_prepare_data)
inference_data_app.command(name="planet")(PlanetPipeline.cli_prepare_data)

training_app = cyclopts.App(name="training", group=subcommands_group, help="Predefined training pipelines")
app.command(training_app)
training_app.command()(validate_dataset)
training_app.command()(train_smp)
training_app.command()(test_smp)
training_app.command(name="crossval-smp")(cross_validation_smp)
training_app.command()(tune_smp)

training_data_app = cyclopts.App(name="create-dataset", help="Dataset creation")
training_app.command(training_data_app)
training_data_app.command(name="planet")(preprocess_planet_train_data)
training_data_app.command(name="planet-pingo")(preprocess_planet_train_data_pingo)
training_data_app.command(name="sentinel2")(preprocess_s2_train_data)


# Intercept the logging behavior to add a file handler
@app.meta.default
def launcher(  # noqa: D103
    *tokens: Annotated[str, cyclopts.Parameter(show=False, allow_leading_hyphen=True)],
    log_dir: Path = Path("logs"),
    config_file: Path = Path("config.toml"),
    verbose: Annotated[bool, cyclopts.Parameter(alias="-v")] = False,
    very_verbose: Annotated[bool, cyclopts.Parameter(alias="-vv")] = False,
    debug: Annotated[bool, cyclopts.Parameter(alias="-vvv")] = False,
    log_plain: bool = False,
):
    verbosity = VerbosityLevel.from_cli(verbose, very_verbose, debug)
    command, bound, ignored = app.parse_args(tokens, verbose=verbosity == VerbosityLevel.VERBOSE)
    # Set verbosity to 1 for debug stuff like env_info
    if command.__name__ == "env_info" and verbosity == VerbosityLevel.NORMAL:
        verbosity = VerbosityLevel.VERBOSE
    LoggingManager.add_logging_handlers(command.__name__, log_dir, verbosity, log_plain=log_plain)
    logger.debug(f"Running on Python version {sys.version} from {__name__} ({root_file})")
    additional_args = {}
    if "config_file" in ignored:
        additional_args["config_file"] = config_file
    if "log_dir" in ignored:
        additional_args["log_dir"] = log_dir
    if "verbosity" in ignored:
        additional_args["verbosity"] = verbosity
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
