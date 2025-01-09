---
hide:
  - navigation
---

# Getting Started

This is a guide to help you, as a user / data engineer, get started with the project.

## Installation

To setup the environment for the project, you need to install [Rye](https://rye.astral.sh/) and run the following command:

```sh
UV_INDEX_STRATEGY="unsafe-best-match" rye sync --features cuda12
```

For other CUDA versions or optional GDAL functionality, see the [contribution guide](contribute.md).

## Data Preparation

As of now, none of the implemented pipelines dowloads PLANET or Sentinel 2 data automatically.
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

Example directory structure of a Sentinel 2 Scene:

!!! warning
    For backcompatability reasons the Sentinel 2 data is expected to be in the same directory structure as the PLANET data.
    Hence, data from Google EarthEngine or from the Copernicus Cloud needs to be adjusted.
    This will probably change in the future.

```sh
data/input/sentinel2/
├── 20210818T223529_20210818T223531_T03WXP/
│  ├── 20210818T223529_20210818T223531_T03WXP_SCL_clip.tif
│  └── 20210818T223529_20210818T223531_T03WXP_SR_clip.tif
└── 20220826T200911_20220826T200905_T17XMJ/
   ├── 20220826T200911_20220826T200905_T17XMJ_SCL_clip.tif
   └── 20220826T200911_20220826T200905_T17XMJ_SR_clip.tif
```

## Create a config file

An example config file can be found in the root of this repository called `config.toml.example`.
You can copy this file to either `configs/` or copy and rename it to `config.toml`, so that you personal config will be ignored by git.

Please change `sentinel2-dir`, `orthotiles-dir` and `scenes-dir` according to your PLANET or Sentinel 2 download directory.

You also need to specify the paths the model checkpoints (`model-dir`, `tcvis-model-name` and `notcvis-model-name`) you want to use.
Note, that only special checkpoints can be used, as described in the [architecture guide](dev/arch.md)

Auxiliary data (TCVIS and ArcticDEM) will be downloaded on demand into a datacube, which paths needs to be specified as well (`arcticdem-dir` and `tcvis-dir`).

Finally, specify an output directory (`output-dir`), where you want to save the results of the pipeline.

Of course you can tweak all other options aswell, also via the CLI.
A list of all options can be found in the [config guide](dev/config.md) or by running a command with the `--help` parameter.

## Run a pipeline

Example for PLANET

```sh
rye run darts run-native-planet-pipeline-fast --config-file path/to/your/config.toml
```

Example for Sentinel 2

```sh
rye run darts run-native-sentinel2-pipeline-fast --config-file path/to/your/config.toml
```

## Getting help

The CLI provides a help view with short explanations on the input settings with the `--help` parameter.

Of course, you are also welcome to contact the developers.
