"""Utility functions for logging."""

import logging
import time
from pathlib import Path

import cyclopts
from rich.console import Console
from rich.logging import RichHandler

# A global level to easy change the log level for interal darts modules
# -> is different from the log level of other libraries (always INFO)
DARTS_LEVEL = logging.DEBUG


def setup_logging(monkey_patch: bool = False):
    """Set up logging for the application.

    Args:
        monkey_patch (bool): Whether to monkey patch the logging of pytorch and disable xarray data expansion.
            This is useful to avoid printing large tensors and arrays to the console.

    """
    if monkey_patch:
        # Import here to avoid heavy imports in other modules if not needed
        import lovely_tensors
        import xarray as xr

        # Disable data prints in xarray for better tracebacks
        xr.set_options(display_expand_data=False)
        # Disable data prints in pytorch, instead show a summary of these
        lovely_tensors.monkey_patch()

    # Set up logging for our own modules
    logging.getLogger("darts").setLevel(DARTS_LEVEL)

    logging.captureWarnings(True)


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
