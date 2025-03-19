"""Data preprocessing and feature engineering for the DARTS dataset."""

import importlib.metadata

from darts_preprocessing.preprocess import preprocess_legacy as preprocess_legacy
from darts_preprocessing.preprocess import preprocess_legacy_fast as preprocess_legacy_fast

try:
    __version__ = importlib.metadata.version("darts-nextgen")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"
