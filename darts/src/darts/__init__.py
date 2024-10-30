"""DARTS processing pipeline."""

from importlib.metadata import version

from darts.native import run_native_planet_pipeline as run_native_planet_pipeline

__version__ = version("darts-nextgen")
