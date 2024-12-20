"""ArcticDEM related data loading."""

from darts_acquisition.arcticdem.datacube import create_empty_datacube as create_empty_datacube
from darts_acquisition.arcticdem.datacube import load_arcticdem_tile as load_arcticdem_tile
from darts_acquisition.arcticdem.datacube import procedural_download_datacube as procedural_download_datacube
from darts_acquisition.arcticdem.vrt import create_arcticdem_vrt as create_arcticdem_vrt
from darts_acquisition.arcticdem.vrt import load_arcticdem_from_vrt as load_arcticdem_from_vrt
