"""Dataset export for the DARTS dataset."""

import importlib.metadata

from darts_export.check import missing_outputs as missing_outputs
from darts_export.export import export_tile as export_tile

try:
    __version__ = importlib.metadata.version("darts-nextgen")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"
