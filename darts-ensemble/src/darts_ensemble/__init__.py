"""Inference and model ensembling for the DARTS dataset."""

import importlib.metadata

from darts_ensemble.ensemble_v1 import EnsembleV1 as EnsembleV1

try:
    __version__ = importlib.metadata.version("darts-nextgen")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"
