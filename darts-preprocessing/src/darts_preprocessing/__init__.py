"""Data preprocessing and feature engineering for the DARTS dataset."""

import importlib.metadata

from darts_preprocessing.engineering.arcticdem import calculate_aspect as calculate_aspect
from darts_preprocessing.engineering.arcticdem import calculate_curvature as calculate_curvature
from darts_preprocessing.engineering.arcticdem import calculate_hillshade as calculate_hillshade
from darts_preprocessing.engineering.arcticdem import calculate_slope as calculate_slope
from darts_preprocessing.engineering.arcticdem import (
    calculate_topographic_position_index as calculate_topographic_position_index,
)
from darts_preprocessing.engineering.indices import calculate_ndvi as calculate_ndvi
from darts_preprocessing.legacy import preprocess_legacy_fast as preprocess_legacy_fast
from darts_preprocessing.v2 import preprocess_v2 as preprocess_v2

try:
    __version__ = importlib.metadata.version("darts-nextgen")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"
