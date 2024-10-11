from pathlib import Path

import xarray as xr
from utils.data_pre_processing import calculate_ndvi, load_auxiliary, load_data_masks, load_planet_scene

"""
darts-nextgen preprocessing
"""

# load planet data

# DATA Path
planet_scene_path = Path(
    "/isipd/projects-noreplica/p_aicore_dev/initze/rts_pipeline_data/v2/input/planet/PSOrthoTile/4372514/5790392_4372514_2022-07-16_2459"
)

# TODO: change to vrt
elevation_path = Path(
    "/isipd/projects-noreplica/p_aicore_dev/initze/rts_pipeline_data/v2/input/ArcticDEM/relative_elevation/4372514_relative_elevation_100.tif"
)

# TODO: change to vrt
slope_path = Path(
    "/isipd/projects-noreplica/p_aicore_dev/initze/rts_pipeline_data/v2/input/ArcticDEM/slope/4372514_slope.tif"
)

# load planet scene
ds_planet = load_planet_scene(planet_scene_path)

# calculate xr.dataset ndvi
ds_ndvi = calculate_ndvi(ds_planet)

# get xr.dataset for elevation
ds_elevation = load_auxiliary(planet_scene_path, elevation_path, xr_dataset_name="relative_elevation")

# get xr.dataset for slope
ds_slope = load_auxiliary(planet_scene_path, slope_path, xr_dataset_name="slope")

# # get xr.dataset for tcvis
# ds_tcvis = load_auxiliary(planet_scene_path, tcvis_path)

# load udm2
ds_data_masks = load_data_masks(planet_scene_path)

# merge to final dataset
ds_merged = xr.merge([ds_planet, ds_ndvi, ds_elevation, ds_slope, ds_data_masks])
