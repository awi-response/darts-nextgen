# Documentation organization

Guides should help users to get familar on how to use the pipelines or how they could implement certain parts of our project / components.

The dev section should help us developers to understand the codebase and decision making.

## ToDos

- [x] Write ToDo List
- [ ] Update Training Guide -> Add Sweep
- [ ] Update Architecture Docs
- [ ] Add components guide
- [x] Move the guides pages to the guides folder
- [ ] Guide on how to measure times
- [ ] Deprecate old pipeliens
- [ ] Add components and public commands API ref and further document them and specify import behaviour through inits:
  - [ ] darts
    - [ ] automated_pipeline.run_native_sentinel2_pipeline_from_aoi
    - [ ] legacy_pipeline.run_native_planet_pipeline_fast
    - [ ] legacy_pipeline.run_native_sentinel2_pipeline_fast
    - [ ] legacy_training.preprocess_planet_train_data
    - [ ] legacy_training.preprocess_s2_train_data
    - [ ] legacy_training.train_smp
    - [ ] legacy_training.test_smp
    - [ ] legacy_training.convert_lightning_checkpoint
    - [ ] legacy_training.optuna_sweep_smp
    - [ ] legacy_training.wandb_sweep_smp
  - [ ] darts_acquisition
    - [ ] load_s2_scene
    - [ ] load_s2_masks
    - [ ] load_s2_from_gee
    - [ ] load_s2_from_stac
    - [ ] load_planet_scene
    - [ ] load_planet_masks
    - [ ] load_tcvis
    - [ ] load_arcticdem
  - [ ] darts_ensemble
    - [ ] EnsembleV1
  - [ ] darts_export
    - [ ] export_probabilities
    - [ ] export_binarized
    - [ ] export_polygonized
    - [ ] export_datamask
    - [ ] export_arcticdem_datamask
    - [ ] export_extent
    - [ ] export_optical
    - [ ] export_dem
    - [ ] export_tcvis
    - [ ] export_thumbnail
  - [ ] darts_postprocessing
    - [ ] prepare_export
  - [ ] darts_preprocessing
    - [ ] preprocess_legacy_fast
    - [ ] calculate_ndvi
    - [ ] calculate_topographic_position_index
    - [ ] calculate_slope
  - [ ] darts_segmentation
    - [ ] SMPSegmenter
    - [ ] predict_in_patches
    - [ ] create_patches
    - [ ] patch_coords
    - [ ] training.BinarySegmentationMetrics
    - [ ] training.DartsDataset
    - [ ] training.DartsDatasetZarr
    - [ ] training.DartsDatasetInMemory
    - [ ] training.DartsDataModule
    - [ ] training.SMPSegmenter
    - [ ] training.create_training_patches
  - [ ] darts_superresolution
  - [ ] darts_utils
    - [ ] free_cupy
    - [ ] free_torch
    - [ ] RichManagerSingleton
    - [ ] RichManager
    - [ ] StopUhr
    - [ ] stopuhr
