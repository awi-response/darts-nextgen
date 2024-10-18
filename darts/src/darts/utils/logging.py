"""Utility functions for logging."""

import logging
import time
from pathlib import Path

import cyclopts
import xarray as xr
from lovely_tensors import monkey_patch
from rich.console import Console
from rich.logging import RichHandler

# A global level to easy change the log level for interal darts modules
# -> is different from the log level of other libraries (always INFO)
DARTS_LEVEL = logging.DEBUG


def setup_logging():
    """Set up logging for the application."""
    # Disable data prints in xarray for better tracebacks
    xr.set_options(display_expand_data=False)
    # Disable data prints in pytorch, instead show a summary of these
    monkey_patch()

    # Set up logging for our own modules
    logging.getLogger("darts_acquisition").setLevel(DARTS_LEVEL)
    logging.getLogger("darts_ensemble").setLevel(DARTS_LEVEL)
    logging.getLogger("darts_export").setLevel(DARTS_LEVEL)
    logging.getLogger("darts_postprocessing").setLevel(DARTS_LEVEL)
    logging.getLogger("darts_preprocessing").setLevel(DARTS_LEVEL)
    logging.getLogger("darts_segmentation").setLevel(DARTS_LEVEL)
    logging.getLogger("darts_superresolution").setLevel(DARTS_LEVEL)
    logging.getLogger("darts").setLevel(DARTS_LEVEL)


def add_logging_handlers(command: str, console: Console, log_dir: Path):
    """Add logging handlers (rich-console and file) to the application.

    Args:
        command (str): The command that is run.
        console (Console): The rich console to log everything to.
        log_dir (Path): The directory to save the logs to.

    """
    log_dir.mkdir(parents=True, exist_ok=True)
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S")

    # Configure the rich console handler
    rich_handler = RichHandler(console=console, rich_tracebacks=True, tracebacks_suppress=[cyclopts])
    rich_handler.setFormatter(
        logging.Formatter(
            "%(message)s",
            datefmt="[%Y-%m-%d %H:%M:%S]",
        )
    )

    # Configure the file handler (no fancy)
    file_handler = logging.FileHandler(log_dir / f"darts_{command}_{current_time}.log")
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(name)s:%(levelname)s - %(message)s",
            datefmt="[%Y-%m-%d %H:%M:%S]",
        )
    )

    logging.basicConfig(
        level=logging.INFO,
        handlers=[rich_handler, file_handler],
    )
