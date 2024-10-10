from pathlib import Path

from utils.data_pre_processing import calculate_ndvi, load_auxiliary, load_planet_scene

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

# calculate ndvi
ds_ndvi = calculate_ndvi(ds_planet)

# get dataarray for elevation
ds_elevation = load_auxiliary(planet_scene_path, elevation_path, xr_dataset_name="relative_elevation")

# # get dataarray for slope
da_slope = load_auxiliary(planet_scene_path, slope_path, xr_dataset_name="slope")

# # get dataarray for tcvis
# da_elevation = load_auxiliary(planet_scene_path, elevation_path)

# merge all datasets
