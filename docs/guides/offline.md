# Downloads and Offline Mode

DARTS is designed to work with large geospatial datasets from various sources.
To ensure efficient workflows, especially when working on HPC systems or in environments with limited internet connectivity, DARTS implements automatic data downloading with intelligent caching and a dedicated offline mode.

## Overview

DARTS handles two types of data downloads:

1. **Optical data** (Sentinel-2): Downloaded from remote STAC servers or Google Earth Engine
2. **Auxiliary data** (ArcticDEM, TCVIS): Downloaded procedurally into local datacubes as needed

All downloaded data is cached locally by default, so subsequent runs can reuse previously downloaded data without internet access.
For optical data it is possible to deactivate caching, which may be useful for very large runs where the each image is only processed once.

## Sentinel-2 Scene Downloads

When running a Sentinel-2 pipeline, DARTS automatically downloads scenes from either:

- **CDSE (Copernicus Data Space Ecosystem)**: The default source, accessed via STAC API
- **GEE (Google Earth Engine)**: Alternative source, useful when working already on Google Cloud

!!! "danger" Different processing pipelines

    Sentinel 2 data from Google Earth Engine comes in different processing levels, since they loaded data only once.
    Thus, the spectral values are not super comparable across the years and can reduce model performance.

The default download process works as follows:

1. **Scene Discovery**: Based on your input (scene IDs, tile IDs, or AOI), DARTS queries the respective service to find matching scenes
2. **Local Caching**: Scenes are downloaded and stored in a compressed zarr format in a local "raw data store"
3. **Automatic Reuse**: On subsequent runs, if a scene is already cached, it's loaded from the local store instead of being re-downloaded

By default, Sentinel-2 raw data is stored in:

```text
<DARTS_DATA_DIR>/sentinel2/<source>/
```

Where `<source>` is either `cdse` or `gee`. You can customize this location using the `raw_data_store` parameter in your pipeline configuration.

The caching functionality can be disabled by passing `--no-raw-data-store` to the CLI when running `uv run darts inference sentinel2-sequential`.

## Auxiliary Data Downloads

DARTS uses auxiliary datasets to enhance segmentation performance:

- **ArcticDEM**: High-resolution elevation data (2m, 10m, or 32m resolution)
- **TCVIS**: Tasseled Cap vegetation trends from Landsat

These datasets are stored in **Zarr datacubes** powered by Icechunk and downloaded **procedurally** - only the tiles you actually need are downloaded and stored.
This approach is much more efficient than downloading entire continental datasets.

For detailed information about how procedural downloading works, see the [Auxiliary Data documentation](../dev/auxiliary.md).

By default auxiliary data is stored under:

- ArcticDEM: `<DARTS_DATA_DIR>/auxiliary/arcticdem_<resolution>m.icechunk`
- TCVIS: `<DARTS_DATA_DIR>/auxiliary/tcvis.icechunk`

You can customize these using the `arcticdem_dir` and `tcvis_dir` parameters.

## Using DARTS on machines without internet

When you need to run DARTS on a system without internet access (e.g., an HPC cluster), you can pre-download all necessary data using the `prep-data` command.
The `prep-data` command allows you to download optical data and/or auxiliary data before running your pipeline offline. E.g.

```sh
darts inference prep-data sentinel2 \
    --pipeline.scene-ids S2A_MSIL2A_20230615T123456_N0509_R012_T33UUP_20230615T145678 \
    --pipeline.raw-data-source cdse \
    --pipeline.raw-data-store /data/s2_raw_data \
    --pipeline.arcticdem-dir /data/arcticdem_10m.icechunk \
    --pipeline.tcvis-dir /data/tcvis.icechunk \
    --optical # Don't forget this
    --aux # Don't forget this
```

When using `tile-ids` or `aoi-file` for scene discovery, DARTS generates a list of scene IDs. You can save this list for offline use:

```bash
# Online: Discover scenes and save IDs
darts inference prep-data sentinel2 \
    --pipeline.tile-ids 33UUP \
    --pipeline.prep-data-scene-id-file /data/scene_ids.json \
    ...

# This creates a file with the discovered scene IDs that can be used offline
```

To actually run the pipeline in offline-mode, set `offline=True` in your pipeline configuration or use the `--pipeline.offline` flag:

```bash
darts inference sentinel2-sequential \
    --pipeline.scene-ids S2A_MSIL2A_20230615T123456_N0509_R012_T33UUP_20230615T145678 \
    --pipeline.offline \
    # ... other parameters
```

This of course expects that all necessary data is present at the specified paths.

## Storage Considerations

- **Sentinel-2 Raw Data**: ~500-800 MB per scene (compressed)
- **ArcticDEM Datacube**: Grows as needed, ~1-2 GB per 10,000 km² at 10m resolution
- **TCVIS Datacube**: ~50-100 MB per 10,000 km²

Plan your storage accordingly, especially for large-scale processing.

## See Also

- [Auxiliary Data Documentation](../dev/auxiliary.md) - Detailed information about datacube downloads
- [Pipeline Configuration](pipeline-v2.md) - Complete pipeline configuration options
- [DARTS Paths](paths.md) - Understanding DARTS directory structure
