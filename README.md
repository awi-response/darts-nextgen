# DARTS nextgen

[![Lint](https://github.com/awi-response/darts-nextgen/actions/workflows/ruff.yml/badge.svg)](https://github.com/awi-response/darts-nextgen/actions/workflows/ruff.yml)
[![CI](https://github.com/awi-response/darts-nextgen/actions/workflows/update_version.yml/badge.svg)](https://github.com/awi-response/darts-nextgen/actions/workflows/update_version.yml)

> Early Alpha!

Panarctic Database of Active Layer Detachment Slides and Retrogressive Thaw Slumps from Deep Learning on High Resolution Satellite Imagery.
This is te successor of the thaw-slump-segmentation (pipeline), with which the first version of the DARTS dataset was created.

## Documentation

The documentation is available at [https://awi-response.github.io/darts-nextgen/](https://awi-response.github.io/darts-nextgen/).
It is recommended to read the [overview](https://awi-response.github.io/darts-nextgen/latest/overview) before working with the project.

## Quick Start

1. Download source code from the [GitHub repository](https://https://github.com/awi-response/darts-nextgen):

    ```sh
    git clone git@github.com:awi-response/darts-nextgen.git
    cd darts-nextgen
    ```

2. Install the required dependencies using uv:

    ```sh
    uv sync --extra cuda126 --extra training
    ```

    > For other installation options, e.g. using conda, see the [installation guide](https://awi-response.github.io/darts-nextgen/latest/guides/installation/).

3.  Install the required dependencies using pixi

start by running this command:

`pixi shell -e cuda128`

then 

`uv sync --extra cuda128 --extra torchdeps --extra cuda12deps
`

and finally 

`source .venv/bin/activate`  
        
4. Run the Sentinel 2 based pipeline on an area of interest:

    If using the uv environment

    ```sh
    uv run darts run-sequential-aoi-sentinel2-pipeline \
      --aoi-shapefile path/to/your/aoi.geojson \
      --model-files path/to/your/model/checkpoint \
      --start-date 2024-07 \
      --end-date 2024-09
    ```
   
    If using the pixi shell, leave off the 'uv run'
    ```sh
    darts run-sequential-aoi-sentinel2-pipeline \
      --aoi-shapefile path/to/your/aoi.geojson \
      --model-files path/to/your/model/checkpoint \
      --start-date 2024-07 \
      --end-date 2024-09
    ```

## Contribute

Before contributing please contact one of the authors and make sure to read the [Contribution Guidelines](https://awi-response.github.io/darts-nextgen/latest/contribute).
