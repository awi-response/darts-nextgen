"""Default Path management for all DARTS modules.

Unused prototype.
"""

import os
from pathlib import Path


def _parse_path(p: Path | str | None) -> Path | None:
    if isinstance(p, str):
        p = Path(p)
    return p


DEFAULT_DARTS_DIR = _parse_path(os.environ.get("DARTS_DIR")) or Path(".")
DEFAULT_FAST_DIR = _parse_path(os.environ.get("DARTS_FAST_DIR")) or DEFAULT_DARTS_DIR
DEFAULT_VAST_DIR = _parse_path(os.environ.get("DARTS_VAST_DIR")) or DEFAULT_DARTS_DIR


def set_default_paths(
    *,
    fast_dir: Path | str | None,
    vast_dir: Path | str | None,
    darts_dir: Path | str = DEFAULT_DARTS_DIR,
) -> None:
    """Set the default directories for DARTS.

    Args:
        fast_dir (Path | str | None): The directory for fast data.
        vast_dir (Path | str | None): The directory for vast data.
        darts_dir (Path | str, optional): The directory for DARTS data. Defaults to ".".

    """
    global DEFAULT_DARTS_DIR, DEFAULT_FAST_DIR, DEFAULT_VAST_DIR
    DEFAULT_DARTS_DIR = Path(darts_dir) if isinstance(darts_dir, str) else darts_dir

    DEFAULT_FAST_DIR = _parse_path(fast_dir) or DEFAULT_FAST_DIR or DEFAULT_DARTS_DIR
    DEFAULT_VAST_DIR = _parse_path(vast_dir) or DEFAULT_VAST_DIR or DEFAULT_DARTS_DIR
    DEFAULT_FAST_DIR = DEFAULT_FAST_DIR.resolve()
    DEFAULT_VAST_DIR = DEFAULT_VAST_DIR.resolve()
