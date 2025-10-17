"""Default Path management for all DARTS modules.

Places where this default path management should be used:

- The root darts CLI
- The root darts pipelines
- The segmentation training functions

This module allows for setting and getting default paths for DARTS data storage.

Intended usage is to use the provided root_dir, fast_dir, and vast_dir functions to build paths.

Example:
    ```python
    from darts_utils.paths import paths

    my_data_pool = paths.vast / "my_data_pool"
    my_fast_cache = paths.fast / "my_fast_cache"
    ```

The default paths can be set using the set_default_paths function, this should be done at the start of the CLI:

    ```python
    from darts_utils.paths import paths

    def cli(data_dir: str = "data"):
        paths.set_defaults(darts_dir=data_dir)
        ...
    ```

## Structure:

Fast Storage:
    - training
    - models

Vast Storage:
    - aux
    - artifacts
    - cache
    - logs
    - out
    - input

Other directories must be provided by the user.

"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import cyclopts

logger = logging.getLogger(__name__.replace("darts_", "darts."))


def _parse_path(p: Path | str | None) -> Path | None:
    if isinstance(p, str):
        p = Path(p)
    if p is not None:
        p = p.resolve()
    return p


@cyclopts.Parameter(name="*")
@dataclass(frozen=True)
class DefaultPaths:
    """Dataclass for holding default DARTS paths.

    Attributes:
        darts_dir (Path | str | None): The default DARTS data directory.
            If None, defaults to the current working directory.
            Defaults to None.
        fast_dir (Path | str | None): The default DARTS fast data directory.
            If None, defaults to the DARTS data directory.
            Defaults to None.
        vast_dir (Path | str | None): The default DARTS vast data directory.
            If None, defaults to the DARTS data directory.
            Defaults to None.

    """

    darts_dir: Path | str | None = None
    fast_dir: Path | str | None = None
    vast_dir: Path | str | None = None


class PathManagerSingleton:
    """Singleton class for managing DARTS paths."""

    _instance = None

    def __new__(cls):
        """Create a new instance of PathsSingleton or return the existing instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_paths()
        return cls._instance

    def _initialize_paths(self):
        """Initialize the default paths for DARTS."""
        darts_dir = _parse_path(os.environ.get("DARTS_DATA_DIR")) or Path.cwd()
        self.fast_dir = _parse_path(os.environ.get("DARTS_FAST_DATA_DIR")) or darts_dir
        self.vast_dir = _parse_path(os.environ.get("DARTS_VAST_DATA_DIR")) or darts_dir

    def set_defaults(self, defaults: DefaultPaths) -> None:
        """Set the default directories for DARTS.

        The priority for setting the directories is as follows:
        1. Directly set paths via this function.
        2. Environment variables.
        3. Current working directory.

        Where the fast_dir and vast_dir default to darts_dir if not set.

        Args:
            defaults (DefaultPaths): The default paths to set.

        """
        darts_dir = _parse_path(defaults.darts_dir)
        self.fast_dir = _parse_path(defaults.fast_dir) or darts_dir or self.fast_dir
        self.vast_dir = _parse_path(defaults.vast_dir) or darts_dir or self.vast_dir

    @property
    def fast(self) -> Path:  # noqa: D102
        return self.fast_dir

    @property
    def vast(self) -> Path:  # noqa: D102
        return self.vast_dir

    @property
    def aux(self) -> Path:
        """Get the default aux data directory.

        Returns:
            Path: The default aux data directory.

        """
        return self.vast_dir / "aux"

    @property
    def artifacts(self) -> Path:  # noqa: D102
        return self.vast_dir / "artifacts"

    @property
    def training(self) -> Path:  # noqa: D102
        return self.fast_dir / "training"

    @property
    def cache(self) -> Path:  # noqa: D102
        return self.vast_dir / "cache"

    @property
    def logs(self) -> Path:  # noqa: D102
        return self.vast_dir / "logs"

    @property
    def out(self) -> Path:  # noqa: D102
        return self.vast_dir / "out"

    @property
    def models(self) -> Path:  # noqa: D102
        return self.fast_dir / "models"

    @property
    def input(self) -> Path:  # noqa: D102
        return self.vast_dir / "input"


paths = PathManagerSingleton()
