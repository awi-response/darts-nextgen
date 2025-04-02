"""Acquisition of data from various sources for the DARTS dataset."""

import importlib.metadata

from darts_acquisition.admin import download_admin_files as download_admin_files
from darts_acquisition.arcticdem import load_arcticdem as load_arcticdem
from darts_acquisition.planet import load_planet_masks as load_planet_masks
from darts_acquisition.planet import load_planet_scene as load_planet_scene
from darts_acquisition.planet import parse_planet_type as parse_planet_type
from darts_acquisition.s2 import load_s2_from_gee as load_s2_from_gee
from darts_acquisition.s2 import load_s2_from_stac as load_s2_from_stac
from darts_acquisition.s2 import load_s2_masks as load_s2_masks
from darts_acquisition.s2 import load_s2_scene as load_s2_scene
from darts_acquisition.s2 import parse_s2_tile_id as parse_s2_tile_id
from darts_acquisition.tcvis import load_tcvis as load_tcvis

try:
    __version__ = importlib.metadata.version("darts-nextgen")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"
