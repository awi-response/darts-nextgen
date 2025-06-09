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
    uv run darts run-sequential-aoi-sentinel2-pipeline \
      --aoi-shapefile path/to/your/aoi.geojson \
      --model-files path/to/your/model/checkpoint \
      --start-date 2024-07 \
      --end-date 2024-09
    ```

<div class="grid cards" markdown>

-   :material-lightbulb-on:{ .lg .middle } __Overview__

    ---

    Get an overview on how this project works and how to run different pipelines.

    [:octicons-arrow-right-24: Get Started](overview.md)

-   :material-folder-download:{ .lg .middle } __Install__

    ---

    View detailed instructions on how to install the project for different environments and setup, e.g. with CUDA or conda.

    [:octicons-arrow-right-24: Install](guides/installation.md)

-   :material-book-cog:{ .lg .middle } __Pipeline Components__

    ---

    Learn about the different components of the pipeline and how they work together.

    [:octicons-arrow-right-24: Components](guides/components.md)

-   :material-scale-balance:{ .lg .middle } __API Reference__

    ---

    View the API reference of the components.

    [:octicons-arrow-right-24: Reference](reference/darts/index.md)

</div>

## Contribute

Before contributing please contact one of the authors and make sure to read the [Contribution Guidelines](contribute.md).
