"""Acquisition of data from various sources for the DARTS dataset."""

import importlib.metadata

from darts_acquisition.admin import download_admin_files as download_admin_files
from darts_acquisition.arcticdem import download_arcticdem as download_arcticdem
from darts_acquisition.arcticdem import load_arcticdem as load_arcticdem
from darts_acquisition.planet import get_planet_geometry as get_planet_geometry
from darts_acquisition.planet import load_planet_masks as load_planet_masks
from darts_acquisition.planet import load_planet_scene as load_planet_scene
from darts_acquisition.planet import parse_planet_type as parse_planet_type
from darts_acquisition.s2.s2cdse import (
    get_cdse_s2_sr_scene_ids_from_geodataframe as get_cdse_s2_sr_scene_ids_from_geodataframe,
)
from darts_acquisition.s2.s2cdse import load_cdse_s2_sr_scene as load_cdse_s2_sr_scene
from darts_acquisition.s2.s2cdse import (
    match_cdse_s2_sr_scene_ids_from_geodataframe as match_cdse_s2_sr_scene_ids_from_geodataframe,
)
from darts_acquisition.s2.s2cdse import search_cdse_s2_sr as search_cdse_s2_sr
from darts_acquisition.s2.s2gee import (
    get_gee_s2_sr_scene_ids_from_geodataframe as get_gee_s2_sr_scene_ids_from_geodataframe,
)
from darts_acquisition.s2.s2gee import load_gee_s2_sr_scene as load_gee_s2_sr_scene
from darts_acquisition.tcvis import download_tcvis as download_tcvis
from darts_acquisition.tcvis import load_tcvis as load_tcvis

try:
    __version__ = importlib.metadata.version("darts-nextgen")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"
