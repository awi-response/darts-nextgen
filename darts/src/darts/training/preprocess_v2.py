"""Preprocessing functions for training with the v2 data preprocessing."""

import logging
from math import ceil, sqrt
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import geopandas as gpd

logger = logging.getLogger(__name__)


def _get_region_name(labels: "gpd.GeoDataFrame", admin_dir: Path) -> str:
    import geopandas as gpd
    from darts_acquisition.admin import download_admin_files

    # Download admin files if they do not exist
    admin1_fpath = admin_dir / "geoBoundariesCGAZ_ADM1.shp"
    admin2_fpath = admin_dir / "geoBoundariesCGAZ_ADM2.shp"

    if not admin1_fpath.exists() or not admin2_fpath.exists():
        download_admin_files(admin_dir)

    # Load the admin files
    # admin1 = gpd.read_file(admin1_fpath)
    admin2 = gpd.read_file(admin2_fpath)

    # Check if any label is intersecting with the test regions
    # adm1_intersects = labels.overlay(admin1, how="intersection")
    adm2_intersects = labels.to_crs("EPSG:4326").overlay(admin2, how="intersection")
    if adm2_intersects.empty:
        raise ValueError("No intersection found between labels and admin2 regions")
    return adm2_intersects.iloc[0]["shapeName"]


def preprocess_s2_train_data(
    *,
    bands: list[str],
    sentinel2_dir: Path,
    train_data_dir: Path,
    arcticdem_dir: Path,
    tcvis_dir: Path,
    admin_dir: Path,
    preprocess_cache: Path | None = None,
    device: Literal["cuda", "cpu", "auto"] | int | None = None,
    ee_project: str | None = None,
    ee_use_highvolume: bool = True,
    tpi_outer_radius: int = 100,
    tpi_inner_radius: int = 0,
    patch_size: int = 1024,
    overlap: int = 16,
    exclude_nopositive: bool = False,
    exclude_nan: bool = True,
    mask_erosion_size: int = 10,
    test_val_split: float = 0.05,
    test_regions: list[str] | None = None,
):
    """Preprocess Sentinel 2 data for training.

    The data is split into a cross-validation, a validation-test and a test set:

        - `cross-val` is meant to be used for train and validation
        - `val-test` (5%) random leave-out for testing the randomness distribution shift of the data
        - `test` leave-out region for testing the spatial distribution shift of the data

    Each split is stored as a zarr group, containing a x and a y dataarray.
    The x dataarray contains the input data with the shape (n_patches, n_bands, patch_size, patch_size).
    The y dataarray contains the labels with the shape (n_patches, patch_size, patch_size).
    Both dataarrays are chunked along the n_patches dimension.
    This results in super fast random access to the data, because each sample / patch is stored in a separate chunk and
    therefore in a separate file.

    Through the parameters `test_val_split` and `test_regions`, the test and validation split can be controlled.
    To `test_regions` can a list of admin 1 or admin 2 region names, based on the region shapefile maintained by
    https://github.com/wmgeolab/geoBoundaries, be supplied to remove intersecting scenes from the dataset and
    put them in the test-split.
    With the `test_val_split` parameter, the ratio between further splitting of a test-validation set can be controlled.

    Through `exclude_nopositve` and `exclude_nan`, respective patches can be excluded from the final data.

    Further, a `config.toml` file is saved in the `train_data_dir` containing the configuration used for the
    preprocessing.
    Addionally, a `labels.geojson` file is saved in the `train_data_dir` containing the joined labels geometries used
    for the creation of the binarized label-masks, containing also information about the split via the `mode` column.

    The final directory structure of `train_data_dir` will look like this:

    ```sh
    train_data_dir/
    ├── config.toml
    ├── cross-val.zarr/
    ├── test.zarr/
    ├── val-test.zarr/
    └── labels.geojson
    ```

    Args:
        bands (list[str]): The bands to be used for training. Must be present in the preprocessing.
        sentinel2_dir (Path): The directory containing the Sentinel 2 scenes.
        train_data_dir (Path): The "output" directory where the tensors are written to.
        arcticdem_dir (Path): The directory containing the ArcticDEM data (the datacube and the extent files).
            Will be created and downloaded if it does not exist.
        tcvis_dir (Path): The directory containing the TCVis data.
        admin_dir (Path): The directory containing the admin files.
        preprocess_cache (Path, optional): The directory to store the preprocessed data. Defaults to None.
        device (Literal["cuda", "cpu"] | int, optional): The device to run the model on.
            If "cuda" take the first device (0), if int take the specified device.
            If "auto" try to automatically select a free GPU (<50% memory usage).
            Defaults to "cuda" if available, else "cpu".
        ee_project (str, optional): The Earth Engine project ID or number to use. May be omitted if
            project is defined within persistent API credentials obtained via `earthengine authenticate`.
        ee_use_highvolume (bool, optional): Whether to use the high volume server (https://earthengine-highvolume.googleapis.com).
        tpi_outer_radius (int, optional): The outer radius of the annulus kernel for the tpi calculation
            in m. Defaults to 100m.
        tpi_inner_radius (int, optional): The inner radius of the annulus kernel for the tpi calculation
            in m. Defaults to 0.
        patch_size (int, optional): The patch size to use for inference. Defaults to 1024.
        overlap (int, optional): The overlap to use for inference. Defaults to 16.
        exclude_nopositive (bool, optional): Whether to exclude patches where the labels do not contain positives.
            Defaults to False.
        exclude_nan (bool, optional): Whether to exclude patches where the input data has nan values.
            Defaults to True.
        mask_erosion_size (int, optional): The size of the disk to use for mask erosion and the edge-cropping.
            Defaults to 10.
        test_val_split (float, optional): The split ratio for the test and validation set. Defaults to 0.05.
        test_regions (list[str] | str, optional): The region to use for the test set. Defaults to None.

    """
    # Import here to avoid long loading times when running other commands
    import geopandas as gpd
    import pandas as pd
    import toml
    import xarray as xr
    import zarr
    from darts_acquisition import load_arcticdem, load_s2_masks, load_s2_scene, load_tcvis
    from darts_acquisition.s2 import parse_s2_tile_id
    from darts_preprocessing import preprocess_legacy_fast
    from darts_segmentation.training.prepare_training import create_training_patches
    from lovely_tensors import monkey_patch
    from odc.stac import configure_rio
    from rich.progress import track
    from zarr.codecs import BloscCodec
    from zarr.storage import LocalStore

    from darts.utils.cuda import debug_info, decide_device
    from darts.utils.earthengine import init_ee
    from darts.utils.logging import console

    monkey_patch()
    debug_info()
    device = decide_device(device)
    init_ee(ee_project, ee_use_highvolume)

    configure_rio(cloud_defaults=True, aws={"aws_unsigned": True})
    logger.info("Configured Rasterio with Dask")

    # We hardcode these because they depend on the preprocessing used
    norm_factors = {
        "red": 1 / 3000,
        "green": 1 / 3000,
        "blue": 1 / 3000,
        "nir": 1 / 3000,
        "ndvi": 1 / 20000,
        "relative_elevation": 1 / 30000,
        "slope": 1 / 90,
        "tc_brightness": 1 / 255,
        "tc_greenness": 1 / 255,
        "tc_wetness": 1 / 255,
    }
    # Filter out bands that are not in the specified bands
    norm_factors = {k: v for k, v in norm_factors.items() if k in bands}

    train_data_dir.mkdir(exist_ok=True, parents=True)

    zroot = zarr.group(store=LocalStore(train_data_dir / "data.zarr"), overwrite=True)
    # We need do declare the number of patches to 0, because we can't know the final number of patches

    zroot.create(
        name="x",
        shape=(0, len(bands), patch_size, patch_size),
        # shards=(100, len(bands), patch_size, patch_size),
        chunks=(1, len(bands), patch_size, patch_size),
        dtype="float32",
        compressors=BloscCodec(cname="lz4", clevel=9),
    )
    zroot.create(
        name="y",
        shape=(0, patch_size, patch_size),
        # shards=(100, patch_size, patch_size),
        chunks=(1, patch_size, patch_size),
        dtype="uint8",
        compressors=BloscCodec(cname="lz4", clevel=9),
    )

    # Find all Sentinel 2 scenes and preprocess them
    n_patches = 0
    joint_lables = []
    metadata = []
    s2_paths = sorted(sentinel2_dir.glob("*/"))
    logger.info(f"Found {len(s2_paths)} Sentinel 2 scenes in {sentinel2_dir}")
    for i, fpath in track(enumerate(s2_paths), description="Processing samples", total=len(s2_paths), console=console):
        try:
            _, s2_tile_id, tile_id = parse_s2_tile_id(fpath)

            logger.debug(f"Processing sample {i + 1} of {len(s2_paths)} '{fpath.resolve()}' ({tile_id=})")

            # Check for a cached preprocessed file
            if preprocess_cache and (preprocess_cache / f"{tile_id}.nc").exists():
                cache_file = preprocess_cache / f"{tile_id}.nc"
                logger.info(f"Loading preprocessed data from {cache_file.resolve()}")
                tile = xr.open_dataset(preprocess_cache / f"{tile_id}.nc", engine="h5netcdf").set_coords("spatial_ref")
            else:
                optical = load_s2_scene(fpath)
                logger.info(f"Found optical tile with size {optical.sizes}")
                arctidem_res = 10
                arcticdem_buffer = ceil(tpi_outer_radius / arctidem_res * sqrt(2))
                arcticdem = load_arcticdem(
                    optical.odc.geobox, arcticdem_dir, resolution=arctidem_res, buffer=arcticdem_buffer
                )
                tcvis = load_tcvis(optical.odc.geobox, tcvis_dir)
                data_masks = load_s2_masks(fpath, optical.odc.geobox)

                tile: xr.Dataset = preprocess_legacy_fast(
                    optical,
                    arcticdem,
                    tcvis,
                    data_masks,
                    tpi_outer_radius,
                    tpi_inner_radius,
                    device,
                )
                # Only cache if we have a cache directory
                if preprocess_cache:
                    preprocess_cache.mkdir(exist_ok=True, parents=True)
                    cache_file = preprocess_cache / f"{tile_id}.nc"
                    logger.info(f"Caching preprocessed data to {cache_file.resolve()}")
                    tile.to_netcdf(cache_file, engine="h5netcdf")

            labels = gpd.read_file(fpath / f"{s2_tile_id}.shp")

            # Get region name
            region = _get_region_name(labels, admin_dir)

            # Save the patches
            gen = create_training_patches(
                tile,
                labels,
                bands,
                norm_factors,
                patch_size,
                overlap,
                exclude_nopositive,
                exclude_nan,
                device,
                mask_erosion_size,
            )

            patch_id = None
            for patch_id, (x, y) in enumerate(gen):
                zroot["x"].append(x.unsqueeze(0).numpy().astype("float32"))
                zroot["y"].append(y.unsqueeze(0).numpy().astype("uint8"))
                metadata.append(
                    {
                        "patch_id": patch_id,
                        "tile_id": tile_id,
                        "region": region,
                        "sample_id": tile_id,
                        "empty": not (y == 1).any(),
                    }
                )
                n_patches += 1
            if n_patches > 0 and len(labels) > 0:
                labels["region"] = region
                labels["sample_id"] = tile_id
                joint_lables.append(labels.to_crs("EPSG:3413"))

            logger.info(
                f"Processed sample {i + 1} of {len(s2_paths)} '{fpath.resolve()}'({tile_id=}) with {patch_id} patches."
            )
        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
            break

        except Exception as e:
            logger.warning(f"Could not process folder sample {i} '{fpath.resolve()}'.\nSkipping...")
            logger.exception(e)

    # Save the used labels
    joint_lables = pd.concat(joint_lables)
    joint_lables.to_file(train_data_dir / "labels.geojson", driver="GeoJSON")

    # Save the metadata
    metadata = pd.DataFrame(metadata)
    metadata.to_parquet(train_data_dir / "metadata.parquet")

    # Save a config file as toml
    config = {
        "darts": {
            "sentinel2_dir": sentinel2_dir,
            "train_data_dir": train_data_dir,
            "arcticdem_dir": arcticdem_dir,
            "tcvis_dir": tcvis_dir,
            "bands": bands,
            "norm_factors": norm_factors,
            "device": device,
            "ee_project": ee_project,
            "ee_use_highvolume": ee_use_highvolume,
            "tpi_outer_radius": tpi_outer_radius,
            "tpi_inner_radius": tpi_inner_radius,
            "patch_size": patch_size,
            "overlap": overlap,
            "exclude_nopositive": exclude_nopositive,
            "exclude_nan": exclude_nan,
            "n_patches": n_patches,
        }
    }
    with open(train_data_dir / "config.toml", "w") as f:
        toml.dump(config, f)

    logger.info(f"Saved {n_patches} patches to {train_data_dir}")
