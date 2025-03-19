"""Postprocessing steps for the DARTS dataset."""

import importlib.metadata

from darts_postprocessing.prepare_export import prepare_export as prepare_export

try:
    __version__ = importlib.metadata.version("darts-nextgen")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"
