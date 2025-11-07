---
hide:
  - toc
---
# DARTS components

!!! tip "Components"

    The idea behind the [Architecture](../dev/arch.md#api-paradigms) of `darts-nextgen` is to provide `Components` to the user.
    Users should pick their components and put them together in their custom pipeline, utilizing their own parallelization framework.
    [Pipeline v2](./pipeline-v2.md) shows how this could look like for a simple sequential pipeline - hence without parallelization framework.

There are many different components implemented, all within the different `darts packages`.
Here is an overview over the currently implemented components and their hardware requirements / bounds.

Note: the following table was generated from the public re-exports in each package's `__init__.py`.
I inferred resource requirements conservatively from each component's role (data acquisition -> network/disk; preprocessing/postprocessing/metrics -> CPU/disk; training/ensembling -> heavy compute and often GPU). Where unclear I made reasonable assumptions and marked them in the table. If you want tighter/verified bounds I can inspect individual implementations and tests to refine the entries.

| Component                                                    | Stateful? and why?                               | Network [^1]             | Disk [^2]                | Compute [^3]       | GPU [^4]           |
| ------------------------------------------------------------ | ------------------------------------------------ | ------------------------ | ------------------------ | ------------------ | ------------------ |
| **Acquisition**                                              |                                                  |                          |                          |                    |                    |
| [darts_acquisition.download_arcticdem][]                     | Stateless                                        | :white_check_mark:       | :white_check_mark:       |                    |                    |
| [darts_acquisition.load_arcticdem][]                         | Stateless                                        | :material-toggle-switch: | :material-toggle-switch: |                    |                    |
| [darts_acquisition.load_planet_masks][]                      | Stateless                                        |                          | :white_check_mark:       |                    |                    |
| [darts_acquisition.load_planet_scene][]                      | Stateless                                        |                          | :white_check_mark:       | :white_check_mark: |                    |
| [darts_acquisition.download_cdse_s2_sr_scene][]              | Stateless                                        | :white_check_mark:       | :white_check_mark:       |                    |                    |
| [darts_acquisition.load_cdse_s2_sr_scene][]                  | Stateless                                        | :material-toggle-switch: | :material-toggle-switch: | :white_check_mark: | :white_check_mark: |
| [darts_acquisition.download_gee_s2_sr_scene][]               | Stateless                                        | :white_check_mark:       | :white_check_mark:       |                    |                    |
| [darts_acquisition.load_gee_s2_sr_scene][]                   | Stateless                                        | :material-toggle-switch: | :material-toggle-switch: | :white_check_mark: | :white_check_mark: |
| [darts_acquisition.download_tcvis][]                         | Stateless                                        | :white_check_mark:       | :white_check_mark:       |                    |                    |
| [darts_acquisition.load_tcvis][]                             | Stateless                                        | :material-toggle-switch: | :material-toggle-switch: |                    |                    |
| **Preprocessing**                                            |                                                  |                          |                          |                    |                    |
| [darts_preprocessing.calculate_aspect][]                     | Stateless                                        |                          |                          | :white_check_mark: | :white_check_mark: |
| [darts_preprocessing.calculate_curvature][]                  | Stateless                                        |                          |                          | :white_check_mark: | :white_check_mark: |
| [darts_preprocessing.calculate_dissection_index][]           | Stateless                                        |                          |                          | :white_check_mark: | :white_check_mark: |
| [darts_preprocessing.calculate_hillshade][]                  | Stateless                                        |                          |                          | :white_check_mark: | :white_check_mark: |
| [darts_preprocessing.calculate_slope][]                      | Stateless                                        |                          |                          | :white_check_mark: | :white_check_mark: |
| [darts_preprocessing.calculate_terrain_ruggedness_index][]   | Stateless                                        |                          |                          | :white_check_mark: | :white_check_mark: |
| [darts_preprocessing.calculate_topographic_position_index][] | Stateless                                        |                          |                          | :white_check_mark: | :white_check_mark: |
| [darts_preprocessing.calculate_vector_ruggedness_measure][]  | Stateless                                        |                          |                          | :white_check_mark: | :white_check_mark: |
| [darts_preprocessing.calculate_ctvi][]                       | Stateless                                        |                          |                          | :white_check_mark: | :white_check_mark: |
| [darts_preprocessing.calculate_evi][]                        | Stateless                                        |                          |                          | :white_check_mark: | :white_check_mark: |
| [darts_preprocessing.calculate_exg][]                        | Stateless                                        |                          |                          | :white_check_mark: | :white_check_mark: |
| [darts_preprocessing.calculate_gli][]                        | Stateless                                        |                          |                          | :white_check_mark: | :white_check_mark: |
| [darts_preprocessing.calculate_gndvi][]                      | Stateless                                        |                          |                          | :white_check_mark: | :white_check_mark: |
| [darts_preprocessing.calculate_grvi][]                       | Stateless                                        |                          |                          | :white_check_mark: | :white_check_mark: |
| [darts_preprocessing.calculate_ndvi][]                       | Stateless                                        |                          |                          | :white_check_mark: | :white_check_mark: |
| [darts_preprocessing.calculate_nrvi][]                       | Stateless                                        |                          |                          | :white_check_mark: | :white_check_mark: |
| [darts_preprocessing.calculate_rvi][]                        | Stateless                                        |                          |                          | :white_check_mark: | :white_check_mark: |
| [darts_preprocessing.calculate_savi][]                       | Stateless                                        |                          |                          | :white_check_mark: | :white_check_mark: |
| [darts_preprocessing.calculate_tgi][]                        | Stateless                                        |                          |                          | :white_check_mark: | :white_check_mark: |
| [darts_preprocessing.calculate_ttvi][]                       | Stateless                                        |                          |                          | :white_check_mark: | :white_check_mark: |
| [darts_preprocessing.calculate_tvi][]                        | Stateless                                        |                          |                          | :white_check_mark: | :white_check_mark: |
| [darts_preprocessing.calculate_vari][]                       | Stateless                                        |                          |                          | :white_check_mark: | :white_check_mark: |
| [darts_preprocessing.calculate_vdvi][]                       | Stateless                                        |                          |                          | :white_check_mark: | :white_check_mark: |
| [darts_preprocessing.calculate_vigreen][]                    | Stateless                                        |                          |                          | :white_check_mark: | :white_check_mark: |
| [darts_preprocessing.calculate_spyndex][]                    | Stateless                                        |                          |                          | :white_check_mark: | :white_check_mark: |
| [darts_preprocessing.preprocess_legacy_fast][] [^5]          | Stateless                                        |                          |                          | :white_check_mark: | :white_check_mark: |
| [darts_preprocessing.preprocess_v2][] [^5]                   | Stateless                                        |                          |                          | :white_check_mark: | :white_check_mark: |
| **Segmentation**                                             |                                                  |                          |                          |                    |                    |
| [darts_segmentation.segment.SMPSegmenter][]                  | Stateful — holds model state (weights/config)    |                          |                          | :white_check_mark: | :white_check_mark: |
| **Ensemble**                                                 |                                                  |                          |                          |                    |                    |
| [darts_ensemble.EnsembleV1][]                                | Stateful — holds ensemble state (weights/config) |                          |                          | :white_check_mark: | :white_check_mark: |
| **Postprocessing**                                           |                                                  |                          |                          |                    |                    |
| [darts_postprocessing.binarize][]                            | Stateless                                        |                          |                          | :white_check_mark: | :white_check_mark: |
| [darts_postprocessing.erode_mask][]                          | Stateless                                        |                          |                          | :white_check_mark: | :white_check_mark: |
| [darts_postprocessing.prepare_export][] [^6]                 | Stateless                                        |                          |                          | :white_check_mark: | :white_check_mark: |
| **Export**                                                   |                                                  |                          |                          |                    |                    |
| [darts_export.export_tile][]                                 | Stateless                                        |                          | :white_check_mark:       |                    |                    |

[^1]: Network: Requires network access if
[^2]: Disk: Reads from or writes to disk
[^3]: Compute: Does some heavy compute, utilizing the CPU
[^4]: GPU: Supports offloading compute to the GPU
[^5]: Wrapper for multiple preprocessing steps, handling compatibility with each other
[^6]: Wrapper for multiple postprocessing steps

Next to the components, there exist several helper functions for e.g. searching Sentinel-2 scenes or metrics for the training.
Have a look at the [Reference](/darts-nextgen/reference/darts/) for a list of all functions and a describtion of what they do.
