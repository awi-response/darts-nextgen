"""Acquisition of data from various sources for the DARTS dataset."""

from darts_acquisition.admin import download_admin_files as download_admin_files
from darts_acquisition.arcticdem.datacube import load_arcticdem as load_arcticdem
from darts_acquisition.arcticdem.vrt import load_arcticdem_from_vrt as load_arcticdem_from_vrt
from darts_acquisition.planet import load_planet_masks as load_planet_masks
from darts_acquisition.planet import load_planet_scene as load_planet_scene
from darts_acquisition.s2 import load_s2_masks as load_s2_masks
from darts_acquisition.s2 import load_s2_scene as load_s2_scene
from darts_acquisition.tcvis import load_tcvis as load_tcvis
