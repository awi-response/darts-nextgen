"""Image segmentation of thaw-slumps for the DARTS dataset."""

import importlib.metadata

from darts_segmentation.segment import SMPSegmenter as SMPSegmenter
from darts_segmentation.segment import SMPSegmenterConfig as SMPSegmenterConfig
from darts_segmentation.utils import create_patches as create_patches
from darts_segmentation.utils import patch_coords as patch_coords
from darts_segmentation.utils import predict_in_patches as predict_in_patches

try:
    __version__ = importlib.metadata.version("darts-nextgen")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"
