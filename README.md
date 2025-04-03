# DARTS nextgen

[![Lint](https://github.com/awi-response/darts-nextgen/actions/workflows/ruff.yml/badge.svg)](https://github.com/awi-response/darts-nextgen/actions/workflows/ruff.yml)
[![CI](https://github.com/awi-response/darts-nextgen/actions/workflows/update_version.yml/badge.svg)](https://github.com/awi-response/darts-nextgen/actions/workflows/update_version.yml)

> Early Alpha!

Panarctic Database of Active Layer Detachment Slides and Retrogressive Thaw Slumps from Deep Learning on High Resolution Satellite Imagery.
This is te successor of the thaw-slump-segmentation (pipeline), with which the first version of the DARTS dataset was created.

## Documentation

The documentation is available at [https://awi-response.github.io/darts-nextgen/](https://awi-response.github.io/darts-nextgen/).
It is recommended to read the [getting started guide](https://awi-response.github.io/darts-nextgen/getting_started) before working with the project.

## Quick Start

1. Download source code from the [GitHub repository](https://https://github.com/awi-response/darts-nextgen):

    ```sh
    git clone git@github.com:awi-response/darts-nextgen.git
    cd darts-nextgen
    ```

2. Install the required dependencies:

    ```sh
    uv sync --extra cuda126 --extra training
    ```

3. Run the Sentinel 2 based pipeline on an area of interest:

    ```sh
    uv run darts run-sequential-aoi-sentinel2-pipeline \
      --aoi-shapefile path/to/your/aoi.geojson \
      --model-files path/to/your/model/checkpoint \
      --start-date 2024-07 \
      --end-date 2024-09
    ```

## Contribute

Before contributing please contact one of the authors and make sure to read the [Contribution Guidelines](docs/contribute.md).
