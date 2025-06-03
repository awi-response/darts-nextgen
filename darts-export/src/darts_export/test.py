import os

from export import export_tile
import xarray as xr
import logging
import time
from pathlib import Path
logger = logging.getLogger(__name__)

tile = xr.load_dataset('sample_tile.zarr')
outpath = Path(os.path.join(os.getcwd(), '20240711T210031_20240711T210026_T08WNB'))
print('we loaded tile')

from stopuhr import Chronometer

timer = Chronometer(printer=logger.debug)

with timer("Exporting", log=False):
    try:
        export_tile(
            tile,
            outpath,
            bands=["probabilities", "binarized", "polygonized", "extent", "thumbnail"],
            ensemble_subsets= [],
        )
        print("done exporting")
    except Exception as e:
        print("Error in exporting")
        print(e)
        time.sleep(10)