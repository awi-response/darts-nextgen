"""ArcticDEM related data loading."""

# Importing these function would result in long import period when using the CLI,
# since pytorch, xarray etc. are loaded here.
# TODO: Find a way to avoid this.

# from darts_acquisition.arcticdem.datacube import create_empty_datacube as create_empty_datacube
# from darts_acquisition.arcticdem.datacube import download_arcticdem_extend as download_arcticdem_extend
# from darts_acquisition.arcticdem.datacube import download_arcticdem_stac as download_arcticdem_stac
# from darts_acquisition.arcticdem.datacube import get_arcticdem_tile as get_arcticdem_tile
# from darts_acquisition.arcticdem.datacube import procedural_download_datacube as procedural_download_datacube
# from darts_acquisition.arcticdem.vrt import create_arcticdem_vrt as create_arcticdem_vrt
