"""Image superresolution of Sentinel 2 imagery for the DARTS dataset."""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("darts-nextgen")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"
