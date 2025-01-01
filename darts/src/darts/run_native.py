from darts.native import run_native_planet_pipeline, run_native_planet_pipeline_fast
from pathlib import Path
orthotiles_dir = "/home/toddn/pdg/darts-nextgen/rts_dtaset01/input/planet/PSOrthoTile"
scenes_dir = "/home/toddn/pdg/darts-nextgen/rts_dtaset01/input/planet/PSScene"
arcticdem_slope_vrt = "/home/toddn/pdg/darts-nextgen/rts_dtaset01/input/ArcticDEM/slope.vrt"
arcticdem_elevation_vrt = "/home/toddn/pdg/darts-nextgen/rts_dtaset01/input/ArcticDEM/elevation.vrt"
output_data_dir = "/home/toddn/pdg/darts-nextgen/data/output"
model_dir = "/home/toddn/pdg/darts-nextgen/rts_dtaset01/models"
arctic_dem_dir = "/home/toddn/pdg/darts-nextgen/rts_dtaset01/input/ArcticDEM"

tcvis_dir = "/home/toddn/pdg/darts-nextgen/tcvis_dir"

if __name__ == "__main__":
    print("Running the pipeline")
    run_native_planet_pipeline_fast(orthotiles_dir=Path(orthotiles_dir),
                                scenes_dir=Path(scenes_dir),
                                output_data_dir=Path(output_data_dir),
                                arcticdem_dir=Path(arctic_dem_dir),
                                model_dir=Path(model_dir),
                                tcvis_dir=Path(tcvis_dir),
                                ee_project='pdg-project-406720')

    run_native_planet_pipeline(orthotiles_dir=Path(orthotiles_dir),
                               scenes_dir=Path(scenes_dir),
                               output_data_dir=Path(output_data_dir),
                               arcticdem_slope_vrt=Path(arcticdem_slope_vrt),
                               arcticdem_elevation_vrt=Path(arcticdem_elevation_vrt),
                               model_dir=Path(model_dir),
                               tcvis_dir=Path(tcvis_dir),
                               ee_project='pdg-project-406720')