from darts.native import run_native_planet_pipeline
from pathlib import Path
orthotiles_dir = "data/input/planet/PSOrthoTile"
scenes_dir = "data/input/planet/PSScene"
arcticdem_slope_vrt = "data/input/ArcticDEM/slope.vrt"
arcticdem_elevation_vrt = "data/input/ArcticDEM/elevation.vrt"
output_data_dir = "data/output"
model_dir = "models"

if __name__ == "__main__":
    print("Running the pipeline")
    run_native_planet_pipeline(orthotiles_dir=Path(orthotiles_dir),
                               scenes_dir=Path(scenes_dir),
                               output_data_dir=Path(output_data_dir),
                               arcticdem_slope_vrt=Path(arcticdem_slope_vrt),
                               arcticdem_elevation_vrt=Path(arcticdem_elevation_vrt),
                               model_dir=Path(model_dir))