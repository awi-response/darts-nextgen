"""Utility functions for logging."""

import importlib
import logging
import re
import sys
import time
from enum import IntEnum
from pathlib import Path

import cyclopts
import rich
from rich.logging import RichHandler

logger = logging.getLogger(__name__)

# Console singleton to access the console from anywhere
# console = Console()


class VerbosityLevel(IntEnum):
    """Enum for verbosity levels."""

    NORMAL = 0
    VERBOSE = 1
    VERY_VERBOSE = 2
    DEBUG = 3

    @classmethod
    def from_cli(cls, verbose: bool, very_verbose: bool, debug: bool) -> "VerbosityLevel":
        """Get the verbosity level from CLI flags.

        Args:
            verbose (bool): Whether the verbose flag is set.
            very_verbose (bool): Whether the very verbose flag is set.
            debug (bool): Whether the debug flag is set.

        Returns:
            VerbosityLevel: The corresponding verbosity level.

        """
        if debug:
            return cls.DEBUG
        if very_verbose:
            return cls.VERY_VERBOSE
        if verbose:
            return cls.VERBOSE
        return cls.NORMAL


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
        self._verbosity: VerbosityLevel = VerbosityLevel.NORMAL

    @property
    def logger(self):
        """Get the logger for the application."""
        return logging.getLogger("darts")

    def _overwrite_wandb_logger(self):
        # Based on https://github.com/wandb/wandb/issues/9840#issuecomment-2888302624
        import wandb

        wandb_logger = logging.getLogger("darts.wandb")

        def _format_string(s: str) -> str:
            # Remove the wandb color codes
            return re.sub(r"\x1b\[[0-9;]*m", "'", s)

        def custom_termlog(string="", newline=True, repeat=True):
            # Log to your custom logger
            if string:
                wandb_logger.info(_format_string(string))

        def custom_termwarn(string="", newline=True, repeat=True):
            if string:
                wandb_logger.warning(_format_string(string))

        def custom_termerror(string="", newline=True, repeat=True):
            if string:
                wandb_logger.error(_format_string(string))

        # Replace wandb's terminal output functions with our custom versions
        wandb.termlog = custom_termlog
        wandb.termwarn = custom_termwarn
        wandb.termerror = custom_termerror

    def setup_logging(self):
        """Set up logging for the application."""
        # Set up logging for our own modules
        logging.getLogger("darts").setLevel(logging.INFO)
        logging.captureWarnings(True)

    def add_logging_handlers(
        self,
        command: str,
        log_dir: Path,
        verbosity: VerbosityLevel,
        log_plain: bool = False,
    ):
        """Add logging handlers (rich-console and file) to the application.

        Args:
            command (str): The command that is run.
            log_dir (Path): The directory to save the logs to.
            verbosity (VerbosityLevel): The verbosity level.
            log_plain (bool, optional): uses the RichHandler as output by default,
                enable this to use a common print handler

        """
        self._verbosity = verbosity

        if self._console_handler is not None or self._file_handler is not None:
            logger.warning("Logging handlers already added.")
            return

        log_dir.mkdir(parents=True, exist_ok=True)
        current_time = time.strftime("%Y-%m-%d_%H-%M-%S")

        # Configure the rich console handler
        if verbosity <= VerbosityLevel.VERY_VERBOSE:
            supress_module_names = [
                "torch",
                "torch.utils.data",
                "xarray",
                "distributed",
                "pandas",
                # "lightning",
                "stopuhr",
                "contextlib",
            ]
            traceback_suppress = [cyclopts]
            for module_name in supress_module_names:
                try:
                    module = importlib.import_module(module_name)
                    traceback_suppress.append(module)
                except ImportError:
                    logger.warning(f"Module {module_name} not found, skipping traceback suppression for it.")
                    continue
        else:
            traceback_suppress = []

        if not log_plain:
            console_fmt = (
                "%(message)s"
                if not verbosity >= VerbosityLevel.DEBUG
                else "%(processName)s(%(process)d)-%(threadName)s(%(thread)d)@%(name)s - %(message)s"
            )
            console_handler = RichHandler(
                console=rich.get_console(),
                rich_tracebacks=True,
                tracebacks_suppress=traceback_suppress,
                tracebacks_show_locals=verbosity >= VerbosityLevel.VERY_VERBOSE,
            )
        else:
            console_fmt = "** %(levelname)s %(asctime)s **\n   [%(pathname)s:%(lineno)d]\n"
            console_fmt += (
                "%(message)s"
                if not verbosity >= VerbosityLevel.DEBUG
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

        darts_logger = logging.getLogger("darts")
        darts_logger.addHandler(console_handler)
        darts_logger.addHandler(file_handler)
        darts_logger.setLevel(logging.DEBUG if verbosity >= VerbosityLevel.VERBOSE else logging.INFO)

        if verbosity >= VerbosityLevel.VERY_VERBOSE:
            very_verbose_modules = [
                "smart_geocubes",
                "dask",
                "lightning",
                "pytorch_lightning",
                "torch",
                "torch.utils.data",
                "xarray",
                "distributed",
                "pandas",
            ]
            module_level = logging.DEBUG if verbosity >= VerbosityLevel.DEBUG else logging.INFO
            self.apply_logging_handlers(*very_verbose_modules, level=module_level)

    def apply_logging_handlers(self, *names: str, level: int = logging.INFO):
        """Apply the logging handlers to a (third-party) logger.

        Args:
            names (str): The names of the loggers to apply the handlers to.
            level (int, optional): The log level to set for the loggers.
                Defaults to logging.INFO.

        """
        for name in names:
            third_party_logger = logging.getLogger(name)
            if name in self._managed_loggers:
                # Set level for existing managed logger (will overwrite pot. verbosity settings)
                third_party_logger.setLevel(level)
                continue
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
