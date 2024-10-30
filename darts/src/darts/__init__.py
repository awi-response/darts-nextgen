"""DARTS processing pipeline."""

from importlib.metadata import version

from darts.native import run_native_planet_pipeline as run_native_planet_pipeline
from darts.native import run_native_sentinel2_pipeline as run_native_sentinel2_pipeline

__version__ = version("darts-nextgen")
