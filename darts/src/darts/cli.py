"""Entrypoint for the darts-pipeline CLI."""

import logging
import sys
from pathlib import Path
from typing import Annotated

import cyclopts
from darts_utils.rich import RichManager

from darts import __version__
from darts.automated_pipeline.s2 import run_native_sentinel2_pipeline_from_aoi
from darts.legacy_pipeline import (
    run_native_planet_pipeline_fast,
    run_native_sentinel2_pipeline_fast,
)
from darts.legacy_training import (
    convert_lightning_checkpoint,
    optuna_sweep_smp,
    preprocess_planet_train_data,
    preprocess_s2_train_data,
    test_smp,
    train_smp,
    wandb_sweep_smp,
)
from darts.utils.config import ConfigParser
from darts.utils.logging import LoggingManager

root_file = Path(__file__).resolve()
logger = logging.getLogger(__name__)

config_parser = ConfigParser()
app = cyclopts.App(
    version=__version__,
    console=RichManager.console,
    config=config_parser,
    help_format="plaintext",
    version_format="plaintext",
)

pipeline_group = cyclopts.Group.create_ordered("Pipeline Commands")
data_group = cyclopts.Group.create_ordered("Data Commands")
train_group = cyclopts.Group.create_ordered("Training Commands")


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


app.command(group=pipeline_group)(run_native_planet_pipeline_fast)
app.command(group=pipeline_group)(run_native_sentinel2_pipeline_fast)
app.command(group=pipeline_group)(run_native_sentinel2_pipeline_from_aoi)

app.command(group=train_group)(preprocess_planet_train_data)
app.command(group=train_group)(preprocess_s2_train_data)
app.command(group=train_group)(train_smp)
app.command(group=train_group)(test_smp)
app.command(group=train_group)(convert_lightning_checkpoint)
app.command(group=train_group)(wandb_sweep_smp)
app.command(group=train_group)(optuna_sweep_smp)


# Intercept the logging behavior to add a file handler
@app.meta.default
def launcher(  # noqa: D103
    *tokens: Annotated[str, cyclopts.Parameter(show=False, allow_leading_hyphen=True)],
    log_dir: Path = Path("logs"),
    config_file: Path = Path("config.toml"),
    tracebacks_show_locals: bool = False,
):
    command, bound, _ = app.parse_args(tokens)
    LoggingManager.add_logging_handlers(command.__name__, log_dir, tracebacks_show_locals)
    logger.debug(f"Running on Python version {sys.version} from {__name__} ({root_file})")
    return command(*bound.args, **bound.kwargs)


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
