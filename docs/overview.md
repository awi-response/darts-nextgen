---
hide:
  - navigation
---

# Overview

This is a guide to help you, as a user / data engineer, get started with the project.

[TOC]

## Installation

To setup the environment for the project, you need to install [uv](https://docs.astral.sh/uv/) and run the following command, assuming CUDA 12.6 is installed:

```sh
uv sync --extra cuda126
```

For other CUDA versions, see the [installation guide](guides/installation.md).

Training specific dependencies are optional and therefore not installed by default.
To install them, add `--extra training` to the `uv sync` command, e.g.:

```sh
uv sync --extra cuda126 --extra training
```

To see if the installation was successful, you can run the following command:

```sh
uv run darts env-info
```

## Running stuff via the CLI

The project provides a CLI to run different pipelines, training and other utility functions.
Because the environment is setup with `uv`, you can run the CLI commands with `uv run darts ...`.
If you manually active the environment with `source .venv/bin/activate`, you can run the CLI commands just via `darts ...` without `uv run`.

To see a list of all available commands, run:

```sh
uv run darts --help
```

To get help for a specific command, run:

```sh
uv run darts the-specific-command --help
```

### Device selection

By default, the CLI will automatic select the device for functions that support it.
To force a specific device, you can use the `--device` parameter.
Read more about the device selection in the [device guide](guides/devices.md).

### Config files

The CLI supports config files in TOML format to reduce the amount of parameters you need to pass or to safe different configurations.
By default the CLI tries to load a `config.toml` file from the current directory.
However, you can specify a different file with the `--config-file` parameter.

As of right now, the CLI tries to match all parameters under the `darts` key of the config file, skipping not needed ones.
For more information about the config file,  see the [config guide](guides/config.md)..

### Log files

By default the CLI sets up a logging handler at `INFO` level for the `darts` specific packages found in this workspace.
The log-level can be changed via the `--verbose` flag of the CLI to set it to `DEBUG`.
Running any command will output a logging file at the logging directory, which can be specified via the `--log-dir` parameter.
The logging file will be named after the command and the current timestamp.
If you want to change the logging behavior in python code, you can check out the [logging guide](guides/logging.md).

## Running a pipeline based on Sentinel 2 data

The `run-sequential-aoi-sentinel2-pipeline` automatically downloads and processes Sentinel 2 data based on an Area of Interest (AOI) in GeoJSON format.
Before running you need access to a trained model.
Note, that only special checkpoints can be used, as described in the [architecture guide](dev/arch.md#pytorch-model-checkpoints).
In future versions, downloading of the model via huggingface will be supported, but for now you need to ask the developers for a valid model checkpoint.

To run the pipeline run:

```sh
uv run darts run-sequential-aoi-sentinel2-pipeline --aoi-shapefile path/to/your/aoi.geojson --model-files path/to/your/model/checkpoint --start-date 2024-07 --end-date 2024-09
```

Run `uv run darts run-sequential-aoi-sentinel2-pipeline --help` for more configuration options.

!!! tip "Pipeline v2"

    The [Pipeline v2 Guide](./guides/pipeline-v2.md) provides a more in-depth explanation of the pipeline and its components.

## Running a pipeline based on PLANET data

PLANET data cannot be downloaded automatically.
Hence, you need to download the data manually and place it a directory of you choice.

Example directory structure of a PLANET Orthotile:

```sh
    data/input/planet/PSOrthoTile/
    ├── 4372514/
    │  └── 5790392_4372514_2022-07-16_2459/
    │      ├── 5790392_4372514_2022-07-16_2459_BGRN_Analytic_metadata.xml
    │      ├── 5790392_4372514_2022-07-16_2459_BGRN_DN_udm.tif
    │      ├── 5790392_4372514_2022-07-16_2459_BGRN_SR.tif
    │      ├── 5790392_4372514_2022-07-16_2459_metadata.json
    │      ├── 5790392_4372514_2022-07-16_2459_udm2.tif
    │      └── Thumbs.db
    └── 4974017/
        └── 5854937_4974017_2022-08-14_2475/
            ├── 5854937_4974017_2022-08-14_2475_BGRN_Analytic_metadata.xml
            ├── 5854937_4974017_2022-08-14_2475_BGRN_DN_udm.tif
            ├── 5854937_4974017_2022-08-14_2475_BGRN_SR.tif
            ├── 5854937_4974017_2022-08-14_2475_metadata.json
            ├── 5854937_4974017_2022-08-14_2475_udm2.tif
            └── Thumbs.db
```

Example directory structure of a PLANET Scene:

```sh
    data/input/planet/PSScene/
    ├── 20230703_194241_43_2427/
    │  ├── 20230703_194241_43_2427.json
    │  ├── 20230703_194241_43_2427_3B_AnalyticMS_metadata.xml
    │  ├── 20230703_194241_43_2427_3B_AnalyticMS_SR.tif
    │  ├── 20230703_194241_43_2427_3B_udm2.tif
    │  └── 20230703_194241_43_2427_metadata.json
    └── 20230703_194243_54_2427/
       ├── 20230703_194243_54_2427.json
       ├── 20230703_194243_54_2427_3B_AnalyticMS_metadata.xml
       ├── 20230703_194243_54_2427_3B_AnalyticMS_SR.tif
       ├── 20230703_194243_54_2427_3B_udm2.tif
       └── 20230703_194243_54_2427_metadata.json
```

!!! info "Backcompatability of Sentinel 2 data"
    For historical reasons, it is possible to run similar pipelines with Sentinel 2 data.
    For this, the Sentinel 2 data is expected to be in the same directory structure as the PLANET data.
    Hence, data from Google EarthEngine or from the Copernicus Cloud needs to be adjusted and scaled by the factor of `0.0001`.

    ```sh
    data/input/sentinel2/
    ├── 20210818T223529_20210818T223531_T03WXP/
    │  ├── 20210818T223529_20210818T223531_T03WXP_SCL_clip.tif
    │  └── 20210818T223529_20210818T223531_T03WXP_SR_clip.tif
    └── 20220826T200911_20220826T200905_T17XMJ/
    ├── 20220826T200911_20220826T200905_T17XMJ_SCL_clip.tif
    └── 20220826T200911_20220826T200905_T17XMJ_SR_clip.tif
    ```

### Create a config file

Because the minimal amount of parameters to pass for the PLANET pipeline, it is recommended to use a config file.

An example config file can be found in the root of this repository called `config.toml.example`.
You can copy this file to either `configs/` or copy and rename it to `config.toml`, so that you personal config will be ignored by git.

Please change  `orthotiles-dir` and `scenes-dir` according to your PLANET download directory.

You also need to specify the paths the model checkpoints (`model-dir`, `tcvis-model-name` and `notcvis-model-name`) you want to use.
Note, that only special checkpoints can be used, as described in the [architecture guide](dev/arch.md)
By setting `notcvis-model-name` to `None`, the pipeline will only use the TCVIS model.

Auxiliary data (TCVIS and ArcticDEM) will be downloaded on demand into a datacube, which paths needs to be specified as well (`arcticdem-dir` and `tcvis-dir`).

Finally, specify an output directory (`output-dir`), where you want to save the results of the pipeline.

Of course you can tweak all other options aswell, also via the CLI.
A list of all options can be found in the [config guide](guides/config.md) or by running a command with the `--help` parameter.

???+ tip "Example config file"
    This is how an example config file could look like for the automated Sentinel 2 pipeline:

    ```toml title="config.toml"
    [darts]
    ee-project = "ee-tobias-hoelzer"

    [darts.aoi]
    aoi-shapefile = "./data/banks_island.shp"
    start-date = "2024-07"
    end-date = "2024-10"
    max-cloud-cover = 1 # %

    [darts.paths]
    input-cache = "./data/cache/s2gee"
    output-data-dir = "./data/out"
    arcticdem-dir = "./data/download/arcticdem"
    tcvis-dir = "./data/download/tcvis"
    model-file = "./models/s2-tcvis-final-large_2025-02-12.ckpt"

    [darts.tiling]
    batch-size = 8  # Reduce incase of memory issues
    patch-size = 512  # Reduce incase of memory issues
    overlap = 128  # Recommended to be 1/4 of patch-size
    ```

### Run a the pipeline

Finally run the pipeline with the following command. Additional parameters can be passed via the CLI, which will overwrite the config file.

```sh
rye run darts run-sequential-planet-pipeline-fast --config-file path/to/your/config.toml
```

## Creating your own pipeline

The project was build with the idea in mind, that it is easy to create a new pipeline, with e.g. different parallelisation techniques.
The [architecture guide](dev/arch.md) provides an overview of the project structure and the key components.
A good starting point to understand the components is the [intro to components](guides/components.md).
The build-in pipelines are a good example how the components can be used and put together to create a new pipeline.
