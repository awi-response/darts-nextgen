"""Utility functions for logging."""

import importlib
import logging
import sys
import time
from pathlib import Path

import cyclopts
import rich
from rich.logging import RichHandler

# A global level to easy change the log level for interal darts modules
# -> is different from the log level of other libraries (always INFO)
DARTS_LEVEL = logging.INFO

logger = logging.getLogger(__name__)

# Console singleton to access the console from anywhere
# console = Console()


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
        self._console_handler = None
        self._file_handler = None
        self._managed_loggers = []
        self._log_level = DARTS_LEVEL

    @property
    def logger(self):
        """Get the logger for the application."""
        return logging.getLogger("darts")

    def setup_logging(self, verbose: bool = False):
        """Set up logging for the application.

        Args:
            verbose (bool): Whether to set the log level to DEBUG.

        """
        # Set up logging for our own modules
        self._log_level = logging.DEBUG if verbose else DARTS_LEVEL
        logging.getLogger("darts").setLevel(DARTS_LEVEL)
        logging.captureWarnings(True)

    def add_logging_handlers(
        self,
        command: str,
        log_dir: Path,
        verbose: bool = False,
        tracebacks_show_locals: bool = False,
        log_plain: bool = False,
    ):
        """Add logging handlers (rich-console and file) to the application.

        Args:
            command (str): The command that is run.
            log_dir (Path): The directory to save the logs to.
            verbose (bool): Whether to set the log level to DEBUG.
            tracebacks_show_locals (bool): Whether to show local variables in tracebacks.
            log_plain (bool, optional): uses the RichHandler as output by default,
                enable this to use a common print handler

        """
        if self._console_handler is not None or self._file_handler is not None:
            logger.warning("Logging handlers already added.")
            return

        log_dir.mkdir(parents=True, exist_ok=True)
        current_time = time.strftime("%Y-%m-%d_%H-%M-%S")

        # Configure the rich console handler
        supress_module_names = [
            "torch",
            "torch.utils.data",
            "xarray",
            "distributed",
            "pandas",
            # "lightning",
        ]
        traceback_suppress = [cyclopts]
        for module_name in supress_module_names:
            try:
                module = importlib.import_module(module_name)
                traceback_suppress.append(module)
            except ImportError:
                logger.warning(f"Module {module_name} not found, skipping traceback suppression for it.")
                continue

        if not log_plain:
            console_fmt = (
                "%(message)s"
                if not verbose
                else "%(processName)s(%(process)d)-%(threadName)s(%(thread)d)@%(name)s - %(message)s"
            )
            console_handler = RichHandler(
                console=rich.get_console(),
                rich_tracebacks=True,
                tracebacks_suppress=traceback_suppress,
                tracebacks_show_locals=tracebacks_show_locals,
            )
        else:
            console_fmt = "** %(levelname)s %(asctime)s **\n   [%(pathname)s:%(lineno)d]\n"
            console_fmt += (
                "%(message)s"
                if not verbose
                else "   [%(name)s@%(processName)s(%(process)d)-%(threadName)s(%(thread)d)]\n%(message)s\n"
            )
            console_handler = logging.StreamHandler(sys.stdout)

        console_formatter = logging.Formatter(console_fmt, datefmt="[%Y-%m-%d %H:%M:%S]")
        console_handler.setFormatter(console_formatter)
        self._console_handler = console_handler

        # Configure the file handler (no fancy)
        file_handler = logging.FileHandler(log_dir / f"darts_{command}_{current_time}.log")
        file_fmt = "%(processName)s(%(process)d)-%(threadName)s(%(thread)d)@%(name)s:%(levelname)s - %(message)s (in %(filename)s:%(lineno)d)"  # noqa: E501
        file_handler.setFormatter(
            logging.Formatter(
                file_fmt,
                datefmt="[%Y-%m-%d %H:%M:%S]",
            )
        )
        self._file_handler = file_handler

        self._log_level = logging.DEBUG if verbose else DARTS_LEVEL

        darts_logger = logging.getLogger("darts")
        darts_logger.addHandler(console_handler)
        darts_logger.addHandler(file_handler)
        darts_logger.setLevel(self._log_level)

    def apply_logging_handlers(self, *names: str, level: int | None = None):
        """Apply the logging handlers to a (third-party) logger.

        Args:
            names (str): The names of the loggers to apply the handlers to.
            level (int | None, optional): The log level to set for the loggers. If None, use the manager level.
                Defaults to None.

        """
        if level is None:
            level = self._log_level

        for name in names:
            if name in self._managed_loggers:
                continue
            third_party_logger = logging.getLogger(name)
            # Check if logger has a StreamHandler already and remove it if so
            for handler in third_party_logger.handlers:
                if isinstance(handler, logging.StreamHandler):
                    third_party_logger.removeHandler(handler)
            third_party_logger.addHandler(self._console_handler)
            third_party_logger.addHandler(self._file_handler)
            # Set level for all handlers
            third_party_logger.setLevel(level)

            self._managed_loggers.append(name)


LoggingManager = LoggingManagerSingleton()
