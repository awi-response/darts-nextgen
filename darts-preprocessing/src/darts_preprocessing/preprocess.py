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

elevation_path = Path(
    "/isipd/projects-noreplica/p_aicore_dev/initze/rts_pipeline_data/v2/input/ArcticDEM/relative_elevation/4372514_relative_elevation_100.tif"
)

slope_path = Path(
    "/isipd/projects-noreplica/p_aicore_dev/initze/rts_pipeline_data/v2/input/ArcticDEM/slope/4372514_slope_100.tif"
)

# load planet scene
da_planet = load_planet_scene(planet_scene_path)
# calculate ndvi
da_ndvi = calculate_ndvi(da_planet)

# get dataarray for elevation
da_elevation = load_auxiliary(planet_scene_path, elevation_path)

# # get dataarray for slope
da_slope = load_auxiliary(planet_scene_path, slope_path)

# # get dataarray for tcvis
# da_elevation = load_auxiliary(planet_scene_path, elevation_path)
