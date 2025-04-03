"""Postprocessing steps for the DARTS dataset."""

import importlib.metadata

from darts_postprocessing.postprocess import binarize as binarize
from darts_postprocessing.postprocess import erode_mask as erode_mask
from darts_postprocessing.postprocess import prepare_export as prepare_export

try:
    __version__ = importlib.metadata.version("darts-nextgen")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"
