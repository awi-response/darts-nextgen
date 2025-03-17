"""Very high level benchmarking tool."""

import logging
import time
from collections import defaultdict
from contextlib import contextmanager
from statistics import mean, stdev


class StopUhr:
    """Very high level benchmarking tool."""

    def __init__(self, logger: logging.Logger | None = None):
        """StopUhr: a very high level benchmarking tool.

        Args:
            logger (logging.Logger | None, optional): A logger to print the output to instead of stdout.
                Defaults to None.

        """
        self.printer = logger.debug if logger else print
        self.reset()

    def reset(self):
        """Reset the durations."""
        self.durations = defaultdict(list)

    def export(self):
        """Export the durations as a pandas DataFrame.

        Returns:
            pd.DataFrame: A pandas DataFrame with the durations.

        """
        import pandas as pd

        # bring durations to same length
        durations = {}
        max_len = max(len(v) for v in self.durations.values())
        for key, values in self.durations.items():
            durations[key] = values + [pd.NA] * (max_len - len(values))

        return pd.DataFrame(durations)

    def summary(self, res: int = 2):
        """Print a summary of the durations.

        Args:
            res (int, optional): The number of decimal places to round to. Defaults to 2.

        """
        for key, values in self.durations.items():
            if not values:
                self.printer(f"'{key}' No durations recorded")
                continue

            if len(values) == 1:
                self.printer(f"'{key}' took {values[0]:.{res}f} s")
                continue

            mean_val = mean(values)
            stdev_val = stdev(values)
            self.printer(f"{key} took {mean_val:.{res}f} ± {stdev_val:.{res}f} s")

    @contextmanager
    def __call__(self, key: str, res: int = 2, log: bool = True):
        """Context manager to measure the time taken in a block.

        Args:
            key (str): The key to store the duration under.
            res (int, optional): The number of decimal places to round to. Defaults to 2.
            log (bool, optional): Whether to log the duration. Defaults to True.

        """
        start = time.perf_counter()
        yield
        duration = time.perf_counter() - start
        self.durations[key].append(duration)
        if log:
            self.printer(f"{key} took {duration:.{res}f} s")


@contextmanager
def stopuhr(msg: str, printer: callable = print, res: int = 2):
    """Context manager to measure the time taken in a block.

    Args:
        msg (str): The message to print.
        printer (callable, optional): The function to print with. Defaults to print.
        res (int, optional): The number of decimal places to round to. Defaults to 2.

    """
    start = time.perf_counter()
    yield
    duration = time.perf_counter() - start
    printer(f"{msg} took {duration:.{res}f} s")
