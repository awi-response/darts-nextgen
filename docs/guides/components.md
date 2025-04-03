# Introduction to the darts components

!!! tip "Components"

    The idea behind the [Architecture](../dev/arch.md#api-paradigms) of `darts-nextgen` is to provide `Components` to the user.
    Users should pick their components and put them together in their custom pipeline, utilizing their own parallelization framework.
    [Pipeline v2](./pipeline-v2.md) shows how this could look like for a simple sequential pipeline - hence without parallelization framework.

Currently, the implemented components are:

| Component                                                                                                                                    | Statefull? and why?      | Bound                        |
| -------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ | ---------------------------- |
| [darts_acquisition.load_arcticdem](../reference/darts_acquisition/load_arcticdem.md)                                                         | Stateless                | Network-IO + Disk-IO         |
| [darts_acquisition.load_tcvis](../reference/darts_acquisition/load_tcvis.md)                                                                 | Stateless                | Network-IO + Disk-IO         |
| [darts_acquisition.load_planet_scene](../reference/darts_acquisition/load_planet_scene.md)                                                   | Stateless                | Disk-IO                      |
| [darts_acquisition.load_planet_masks](../reference/darts_acquisition/load_planet_masks.md)                                                   | Stateless                | Disk-IO                      |
| [darts_acquisition.load_s2_scene](../reference/darts_acquisition/load_s2_scene.md)                                                           | Stateless                | Disk-IO                      |
| [darts_acquisition.load_s2_from_gee](../reference/darts_acquisition/load_s2_from_gee.md)                                                     | Stateless                | Network-IO (+ Disk-IO) + CPU |
| [darts_acquisition.load_s2_from_stac](../reference/darts_acquisition/load_s2_from_stac.md)                                                   | Stateless                | Network-IO (+ Disk-IO)       |
| [darts_acquisition.load_s2_masks](../reference/darts_acquisition/load_s2_masks.md)                                                           | Stateless                | Disk-IO + CPU                |
| [darts_preprocessing.preprocess_legacy_fast](../reference/darts_preprocessing/preprocess_legacy_fast.md)                                     | Stateless                | CPU + GPU                    |
| [darts_segmentation.SMPSegmenter.segment_tile](../reference/darts_segmentation/SMPSegmenter.md#darts_segmentation.SMPSegmenter.segment_tile) | Statefull: Model-Weights | GPU                          |
| [darts_ensemble.EnsembleV1.segment_tile](../reference/darts_ensemble/EnsembleV1.md#darts_segmentation.SMPSegmenter.segment_tile)             | Statefull: Model-Weights | GPU                          |
| [darts_postprocessing.prepare_export](../reference/darts_postprocessing/prepare_export.md)                                                   | Stateless                | CPU + GPU                    |
| [darts_export.export_tile](../reference/darts_export/export_tile.md)                                                                         | Stateless                | Disk-IO                      |

## Component Outputs

!!! danger "Incomplete"

    This section is incomplete and will be updated in the future.

All component-tiles are `xr.Datasets` which have geospatial coordinates `x`, `y` and a spatial reference  `spatial_ref` (from rioxarray / odc-geo) as coordinates..
The following documents the input and output of each component.

### Acquisition: Load ArcticDEM

#### Input

(Acquisition components do not have an input in form of a `xr.Dataset`)

#### Output

| DataVariable   | shape  | dtype   | no-data | attrs                         | note |
| -------------- | ------ | ------- | ------- | ----------------------------- | ---- |
| `dem`          | (x, y) | float32 | nan     | data_source, long_name, units |      |
| `dem_datamask` | (x, y) | bool    | False   | data_source, long_name, units |      |

### Preprocessing: Legacy Fast

#### Input

| DataVariable    | shape  | dtype   | no-data | attrs                         | note |
| --------------- | ------ | ------- | ------- | ----------------------------- | ---- |
| `blue`          | (x, y) | uint16  | 0       | data_source, long_name, units |      |
| `green`         | (x, y) | uint16  | 0       | data_source, long_name, units |      |
| `red`           | (x, y) | uint16  | 0       | data_source, long_name, units |      |
| `nir`           | (x, y) | uint16  | 0       | data_source, long_name, units |      |
| `dem`           | (x, y) | float32 | nan     | data_source, long_name, units |      |
| `dem_datamask`  | (x, y) | bool    | False   | data_source, long_name, units |      |
| `tc_brightness` | (x, y) | uint8   | -       | data_source, long_name        |      |
| `tc_greenness`  | (x, y) | uint8   | -       | data_source, long_name        |      |
| `tc_wetness`    | (x, y) | uint8   | -       | data_source, long_name        |      |

#### Output

| DataVariable         | shape  | dtype   | no-data | attrs                         | note                               |
| -------------------- | ------ | ------- | ------- | ----------------------------- | ---------------------------------- |
| [Input]              |        |         |         |                               |                                    |
| `ndvi`               | (x, y) | uint16  | 0       | data_source, long_name        | Values between 0-20.000 (+1, *1e4) |
| `relative_elevation` | (x, y) | int16   | 0       | data_source, long_name, units |                                    |
| `slope`              | (x, y) | float32 | nan     | data_source, long_name        |                                    |

### Segmentation / Ensemble Output

Coordinates: `x`, `y` and `spatial_ref` (from rioxarray)

| DataVariable                | shape  | dtype   | no-data | attrs     |
| --------------------------- | ------ | ------- | ------- | --------- |
| [Output from Preprocessing] |        |         |         |           |
| `probabilities`             | (x, y) | float32 | nan     | long_name |
| `probabilities-model-X*`    | (x, y) | float32 | nan     | long_name |

\*: optional intermedia probabilities in an ensemble

### Postprocessing Output

Coordinates: `x`, `y` and `spatial_ref` (from rioxarray)

| DataVariable                | shape  | dtype | no-data | attrs            | note                 |
| --------------------------- | ------ | ----- | ------- | ---------------- | -------------------- |
| [Output from Preprocessing] |        |       |         |                  |                      |
| `probabilities_percent`     | (x, y) | uint8 | 255     | long_name, units | Values between 0-100 |
| `binarized_segmentation`    | (x, y) | uint8 | -       | long_name        |                      |
