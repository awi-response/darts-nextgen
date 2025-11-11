# Pixi shell instructions

If you are unable to install the nvidia related dependencies directly, you will be
using a pixi shell. This requires a few extra steps before you can run the darts pipeline

First, run this command

`pixi shell -e cuda128`

`conda install -c nvidia cuda-toolkit=12 -y
`

then 

`uv sync --extra cuda128 --extra torchdeps --extra cuda12deps`

This will install the dependencies.

Once you run those, you will activate the environment using this command: 

`source .venv/bin/activate`  

commands are then run like this. Note that there is no `uv run` at the start.

`darts inference sentinel2-ray --aoi-shapefile 
/taiga/toddn/rts-files/tiles_nwt_2010_2016_small.geojson
--start-date 2024-07 --max-cloud-cover 100 --max-snow-cover 100 
--end-date 2024-09 --verbose`
