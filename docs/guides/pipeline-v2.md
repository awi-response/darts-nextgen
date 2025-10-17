# Pipeline v2

The following document describes the pipeline which lead to the DARTS v1 dataset and will potentially lead to the DARTS v2 dataset.
The orginial v1 dataset was not created with this repository, however, a newer, faster version of this pipeline is implemented here, which still uses the exact same pipeline-steps.
Hence, it should be possible to re-create the DARTS v1 dataset with this repository.
The implemented pipeline in this repository could potentially be used for future iterations and releases of the DARTS dataset.

In addition to the PLANET version of the DARTS dataset, the pipeline also supports Sentinel 2 imagery as optical input, resulting in a lower spatial resolution (10m instead of 3m).

!!! note

    The v1 / v2 pipeline is also aliased by `legacy` pipeline somewhere deep in the code.

As of right now, three basic realisation of the v1 pipeline are implemented:

- `run-sequential-planet-pipeline-fast`
- `run-sequential-sentinel2-pipeline-fast`
- `run-sequential-aoi-sentinel2-pipeline`

The naming convention has changed a lot and probably will further change with more pipeline realisations becoming implemented.
`sequential` indicates that the pipeline runs without any parallelization framework.

The pipeline currently consists of the following steps:

1. Load the optical and auxiliary data
    This step depends on the realisation of the pipeline.
    Either [darts_acquisition.load_planet_scene][], [darts_acquisition.load_s2_scene][], [darts_acquisition.load_gee_s2_sr_scene][] or [darts_acquisition.load_cdse_s2_sr_scene][].
    Also loads the masks if not loaded from gee or stac: [darts_acquisition.load_planet_masks][] or [darts_acquisition.load_s2_masks][], for the gee and stac versions the masks are already included.
    For the auxiliary data: [darts_acquisition.load_arcticdem][] and [darts_acquisition.load_tcvis][]
2. Preprocess the optical data: [darts_preprocessing.preprocess_legacy_fast][] or [darts_preprocessing.preprocess_v2][].
3. Segment the optical data: [darts_segmentation.segment.SMPSegmenter.segment_tile][] or [darts_ensemble.EnsembleV1.segment_tile][].
4. Postprocess the segmentation and make it ready for export: [darts_postprocessing.prepare_export][].
5. Export the data: [darts_export.export_tile][].

![DARTS nextgen pipeline v2](../assets/darts_nextgen_pipeline_v2.png){ loading=lazy }

A very simplified version of this implementation looks like this:

```python
from darts_acquisition import load_arcticdem, load_tcvis
from darts_segmentation import SMPSegmenter
from darts_export import export_tile, missing_outputs
from darts_postprocessing import prepare_export
from darts_preprocessing import preprocess_legacy_fast
from darts_acquisition.s2 import load_gee_s2_sr_scene

s2id = "20230701T194909_20230701T195350_T11XNA"
arcticdem_dir = "/path/to/arcticdem"
tcvis_dir = "/path/to/tcvis"
model_file = "/path/to/model.pt"
outpath = "/path/to/output"

segmenter = SMPSegmenter(model_file)

tile = load_gee_s2_sr_scene(s2id)

arcticdem = load_arcticdem(
    tile.odc.geobox,
    arcticdem_dir,
    resolution=10,
    buffer=ceil(100 / 2 * sqrt(2)),
)

tcvis = load_tcvis(tile.odc.geobox, tcvis_dir)

tile = preprocess_legacy_fast(tile, arcticdem, tcvis)

tile = segmenter.segment_tile(tile)

tile = prepare_export(tile)

export_tile(tile, outpath)
```

!!! abstract "Further reading"

    To learn more about how the pipeline and their components steps work, please refer to the following materials:

    - The Paper about the DARTS Dataset (No link yet)
    - The [Components Guide](components.md)
    - The [API Reference](../reference/darts/index.md)

There are further features implemented, which do not come from the components:

- Time tracking of processing steps
- Skipping of already processed tiles
- Environment debugging info

## Minimal configuration example

```toml
[darts]
ee-project = "ee-tobias-hoelzer"
model-files = "./models/s2-tcvis-final-large_2025-02-12.ckpt"
aoi-shapefile = "./data/myaoi.gpkg"
start-date = "2024-07"
end-date = "2024-09"
```

## Full configuration explaination

!!! danger "TODO"
