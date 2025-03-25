# Pipeline v1

The following document describes the pipeline which lead to the DARTS v1 dataset.
The orginial dataset was not created with this repository, however, a newer, faster version of this pipeline is implemented here, which still uses the exact same pipeline-steps.
Hence, it should be possible to re-create the DARTS v1 dataset with this repository.
The implemented pipeline in this repository could potentially be used for future iterations and releases of the DARTS dataset.

In addition to the PLANET version of the DARTS dataset, the pipeline also supports Sentinel 2 imagery as optical input, resulting in a lower spatial resolution (10m instead of 3m).

!!! note

    The v1 pipeline is also aliased by `legacy` pipeline.

As of right now, three basic realisation of the v1 pipeline are implemented:

- `run-native-planet-pipeline-fast`
- `run-native-sentinel2-pipeline-fast`
- `run-native-sentinel2-pipeline-from-aoi`

The naming convention has grown historically and is not very consistent or up-to-date.
`native` indicates that the pipeline runs without any parallelization framework.
`fast` indicates that the pipeline is using an faster and more efficient implementation of the data acquisition and preprocessing steps.
However, this is now the default, since older implementations are now deprecated and will potentially be removed in the future.
The `run-native-sentinel2-pipeline-from-aoi` also uses the fast implementation, despite not having it in the same.

The pipeline currently consists of the following steps:

1. Load the optical data
    This step depends on the realisation of the pipeline.
    Either [darts_acquisition.load_planet_scene], [darts_acquisition.load_s2_scene] [darts_acquisition.load_s2_from_gee] or [darts_acquisition.load_s2_from_stac].
    Also loads the masks if not loaded from gee or stac: [darts_acquisition.load_planet_masks] or [darts_acquisition.load_s2_masks].
2. Preprocess the optical data: [darts_preprocessing.preprocess_legacy_fast]
3. Segment the optical data: [darts_segmentation.SMPSegmenter.segment_tile]
    Note that this implementation doesn't use an ensemble of models like the original DARTS pipeline.
4. Postprocess the segmentation and make it ready for export: [darts_postprocessing.prepare_export]
5. Export the data via various methods from the [darts_export] module.
