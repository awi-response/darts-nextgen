"""Data preprocessing and feature engineering for the DARTS dataset."""

import importlib.metadata

from darts_preprocessing.engineering.arcticdem import calculate_slope as calculate_slope
from darts_preprocessing.engineering.arcticdem import (
    calculate_topographic_position_index as calculate_topographic_position_index,
)
from darts_preprocessing.engineering.indices import calculate_ndvi as calculate_ndvi
from darts_preprocessing.preprocess import preprocess_legacy_fast as preprocess_legacy_fast

try:
    __version__ = importlib.metadata.version("darts-nextgen")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"
