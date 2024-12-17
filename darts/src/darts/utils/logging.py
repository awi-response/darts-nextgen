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

logger = logging.getLogger(__name__)


class LoggingManagerSingleton:
    """A singleton class to manage logging handlers for the application."""

    _instance = None

    def __new__(cls):
        """Create a new instance of the LoggingManager if it does not exist yet."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)

        return cls._instance

    def __init__(self):
        """Initialize the LoggingManager."""
        self._rich_handler = None
        self._file_handler = None
        self._managed_loggers = []

    def setup_logging(self):
        """Set up logging for the application."""
        # Set up logging for our own modules
        logging.getLogger("darts").setLevel(DARTS_LEVEL)
        logging.captureWarnings(True)

    def add_logging_handlers(self, command: str, console: Console, log_dir: Path):
        """Add logging handlers (rich-console and file) to the application.

        Args:
            command (str): The command that is run.
            console (Console): The rich console to log everything to.
            log_dir (Path): The directory to save the logs to.

        """
        import distributed
        import lightning as L  # noqa: N812
        import torch
        import torch.utils.data
        import xarray as xr

        if self._rich_handler is not None or self._file_handler is not None:
            logger.warning("Logging handlers already added.")
            return

        log_dir.mkdir(parents=True, exist_ok=True)
        current_time = time.strftime("%Y-%m-%d_%H-%M-%S")

        # Configure the rich console handler
        rich_handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            tracebacks_suppress=[cyclopts, L, torch, torch.utils.data, xr, distributed],
            tracebacks_show_locals=True,
        )
        rich_handler.setFormatter(
            logging.Formatter(
                "%(message)s",
                datefmt="[%Y-%m-%d %H:%M:%S]",
            )
        )
        self._rich_handler = rich_handler

        # Configure the file handler (no fancy)
        file_handler = logging.FileHandler(log_dir / f"darts_{command}_{current_time}.log")
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(name)s@%(processName)s(%(process)d)-%(threadName)s(%(thread)d):%(levelname)s - %(message)s",  # noqa: E501
                datefmt="[%Y-%m-%d %H:%M:%S]",
            )
        )
        self._file_handler = file_handler

        darts_logger = logging.getLogger("darts")
        darts_logger.addHandler(rich_handler)
        darts_logger.addHandler(file_handler)
        darts_logger.setLevel(DARTS_LEVEL)

    def apply_logging_handlers(self, *names: str, level: int = logging.INFO):
        """Apply the logging handlers to a (third-party) logger.

        Args:
            names (str): The names of the loggers to apply the handlers to.
            level (int): The log level to set for the logger.

        """
        for name in names:
            if name in self._managed_loggers:
                continue
            third_party_logger = logging.getLogger(name)
            # Check if logger has a StreamHandler already and remove it if so
            for handler in third_party_logger.handlers:
                if isinstance(handler, logging.StreamHandler):
                    third_party_logger.removeHandler(handler)
            third_party_logger.addHandler(self._rich_handler)
            third_party_logger.addHandler(self._file_handler)
            # Set level for all handlers
            third_party_logger.setLevel(level)

            self._managed_loggers.append(name)


LoggingManager = LoggingManagerSingleton()
