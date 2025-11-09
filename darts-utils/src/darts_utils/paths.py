"""Default Path management for all DARTS modules.

Places where this default path management should be used:

- The root darts CLI
- The root darts pipelines
- The segmentation training functions

This module allows for setting and getting default paths for DARTS data storage.

Intended usage is to use the provided root_dir, fast_dir, and large_dir functions to build paths.

Example:
    ```python
    from darts_utils.paths import paths

    my_data_pool = paths.large / "my_data_pool"
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

Large Storage:
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
from typing import Literal

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
        large_dir (Path | str | None): The default DARTS large data directory.
            If None, defaults to the DARTS data directory.
            Defaults to None.

    """

    darts_dir: Path | str | None = None
    fast_dir: Path | str | None = None
    large_dir: Path | str | None = None


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
        self.large_dir = _parse_path(os.environ.get("DARTS_LARGE_DATA_DIR")) or darts_dir

    def set_defaults(self, defaults: DefaultPaths) -> None:
        """Set the default directories for DARTS.

        The priority for setting the directories is as follows:
        1. Directly set paths via this function.
        2. Environment variables.
        3. Current working directory.

        Where the fast_dir and large_dir default to darts_dir if not set.

        Args:
            defaults (DefaultPaths): The default paths to set.

        """
        darts_dir = _parse_path(defaults.darts_dir)
        self.fast_dir = _parse_path(defaults.fast_dir) or darts_dir or self.fast_dir
        self.large_dir = _parse_path(defaults.large_dir) or darts_dir or self.large_dir
        logger.debug(f"Set DARTS default paths: fast_dir={self.fast_dir}, large_dir={self.large_dir}")

    def log_all_paths(self, level: int = logging.DEBUG):
        """Log all paths managed."""
        logger.log(level, f"DARTS Fast Directory: {self.fast_dir}")
        logger.log(level, f"DARTS Large Directory: {self.large_dir}")
        logger.log(level, f"DARTS Aux Directory: {self.aux}")
        logger.log(level, f"DARTS Artifacts Directory: {self.artifacts}")
        logger.log(level, f"DARTS Training Directory: {self.training}")
        logger.log(level, f"DARTS Cache Directory: {self.cache}")
        logger.log(level, f"DARTS Logs Directory: {self.logs}")
        logger.log(level, f"DARTS Out Directory: {self.out}")
        logger.log(level, f"DARTS Models Directory: {self.models}")
        logger.log(level, f"DARTS Input Directory: {self.input}")
        logger.log(level, f"DARTS Archive Directory: {self.archive}")
        logger.log(level, f"DARTS Admin Directory: {self.admin()}")
        logger.log(level, f"DARTS ArcticDEM Directory (2m): {self.arcticdem(2)}")
        logger.log(level, f"DARTS ArcticDEM Directory (10m): {self.arcticdem(10)}")
        logger.log(level, f"DARTS ArcticDEM Directory (32m): {self.arcticdem(32)}")
        logger.log(level, f"DARTS TCVIS Directory: {self.tcvis()}")
        logger.log(level, f"DARTS Planet Orthotiles Directory: {self.planet_orthotiles()}")
        logger.log(level, f"DARTS Planet Scenes Directory: {self.planet_scenes()}")
        logger.log(level, f"DARTS Sentinel-2 Grid Directory: {self.sentinel2_grid()}")
        logger.log(level, f"DARTS Sentinel-2 Raw Data Directory (CDSE): {self.sentinel2_raw_data('cdse')}")
        logger.log(level, f"DARTS Sentinel-2 Raw Data Directory (GEE): {self.sentinel2_raw_data('gee')}")
        logger.log(
            level, f"DARTS Training Data Directory ('pipeline', 256x256): {self.train_data_dir('pipeline', 256)}"
        )

    @property
    def fast(self) -> Path:  # noqa: D102
        return self.fast_dir

    @property
    def large(self) -> Path:  # noqa: D102
        return self.large_dir

    @property
    def aux(self) -> Path:  # noqa: D102
        return self.large_dir / "aux"

    @property
    def artifacts(self) -> Path:  # noqa: D102
        return self.large_dir / "artifacts"

    @property
    def training(self) -> Path:  # noqa: D102
        return self.fast_dir / "training"

    @property
    def cache(self) -> Path:  # noqa: D102
        return self.large_dir / "cache"

    @property
    def logs(self) -> Path:  # noqa: D102
        return self.large_dir / "logs"

    @property
    def out(self) -> Path:  # noqa: D102
        return self.large_dir / "output"

    @property
    def models(self) -> Path:  # noqa: D102
        return self.fast_dir / "models"

    @property
    def input(self) -> Path:  # noqa: D102
        return self.large_dir / "input"

    @property
    def archive(self) -> Path:  # noqa: D102
        return self.large_dir / "archive"

    def admin_boundaries(self) -> Path:  # noqa: D102
        d = (self.aux / "admin_boundaries").resolve()
        logger.debug(f"Administrative boundaries path: {d}")
        return d

    def arcticdem(self, res: Literal[2, 10, 32]) -> Path:  # noqa: D102
        d = (self.aux / f"arcticdem_{res}m.icechunk").resolve()
        logger.debug(f"ArcticDEM path for resolution {res}m: {d}")
        return d

    def tcvis(self) -> Path:  # noqa: D102
        d = (self.aux / "tcvis.icechunk").resolve()
        logger.debug(f"TCVIS path: {d}")
        return d

    def planet_orthotiles(self) -> Path:  # noqa: D102
        d = (self.input / "planet" / "tiles").resolve()
        logger.debug(f"Planet orthotiles path: {d}")
        return d

    def planet_scenes(self) -> Path:  # noqa: D102
        d = (self.input / "planet" / "scenes").resolve()
        logger.debug(f"Planet scenes path: {d}")
        return d

    def sentinel2_grid(self) -> Path:  # noqa: D102
        d = (self.input / "sentinel2" / "grid").resolve()
        logger.debug(f"Sentinel-2 grid path: {d}")
        return d

    def sentinel2_raw_data(self, source: Literal["cdse", "gee"]) -> Path:  # noqa: D102
        d = (self.input / "sentinel2" / f"{source}-scenes").resolve()
        logger.debug(f"Sentinel-2 raw data path for source '{source}': {d}")
        return d

    def train_data_dir(self, pipeline: str, patch_size: int) -> Path:  # noqa: D102
        d = (self.training / f"{pipeline}_{patch_size}").resolve()
        logger.debug(f"Training data directory for pipeline '{pipeline}' and patch size {patch_size}: {d}")
        return d


paths = PathManagerSingleton()
