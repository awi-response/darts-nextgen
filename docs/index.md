---
hide:
  - navigation
---

# DARTS nextgen

Panarctic Database of Active Layer Detachment Slides and Retrogressive Thaw Slumps from Deep Learning on High Resolution Satellite Imagery.
This is te successor of the thaw-slump-segmentation (pipeline), with which the first version of the DARTS dataset was created.

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
    uv run darts run-native-sentinel2-pipeline-from-aoi \
      --aoi-shapefile path/to/your/aoi.geojson \
      --model-file path/to/your/model/checkpoint \
      --start-date 2024-07 \
      --end-date 2024-09
    ```

Continue reading with an [Overview](overview.md) for more detailed information or the [Install Guide](guides/installation.md) for detailed information about the installation.

## Contribute

Before contributing please contact one of the authors and make sure to read the [Contribution Guidelines](contribute.md).
