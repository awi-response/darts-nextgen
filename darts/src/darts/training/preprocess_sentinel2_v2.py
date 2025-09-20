"""PLANET preprocessing functions for training with the v2 data preprocessing."""

import json
import logging
import time
from math import ceil, sqrt
from pathlib import Path
from typing import TYPE_CHECKING, Literal

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import geopandas as gpd
    import xarray as xr


def _planet_legacy_path_gen(data_dir: Path):
    for iterdir in data_dir.iterdir():
        if iterdir.stem == "iteration001":
            for sitedir in (iterdir).iterdir():
                for imgdir in (sitedir).iterdir():
                    if not imgdir.is_dir():
                        continue
                    try:
                        yield next(imgdir.glob("*_SR.tif")).parent
                    except StopIteration:
                        yield next(imgdir.glob("*_SR_clip.tif")).parent
        else:
            for imgdir in (iterdir).iterdir():
                if not imgdir.is_dir():
                    continue
                try:
                    yield next(imgdir.glob("*_SR.tif")).parent
                except StopIteration:
                    yield next(imgdir.glob("*_SR_clip.tif")).parent


def _get_region_name(footprint: "gpd.GeoSeries", admin2: "gpd.GeoDataFrame") -> str:
    # Check if any label is intersecting with the test regions
    admin2_of_footprint = admin2[admin2.intersects(footprint.geometry)]

    if admin2_of_footprint.empty:
        raise ValueError("No intersection found between labels and admin2 regions")

    region_name = admin2_of_footprint.iloc[0]["shapeName"]

    if len(admin2_of_footprint) > 1:
        logger.warning(
            f"Found multiple regions for footprint {footprint.image_id}: {admin2_of_footprint.shapeName.to_list()}."
            f" Using the first one ({region_name})"
        )
    return region_name


def _parse_date(row):
    import pandas as pd

    orthotile = row["datasource"] == "PlanetScope OrthoTile"
    if orthotile:
        return pd.to_datetime(row["image_id"].split("_")[-2], format="%Y-%m-%d", utc=True)
    else:
        return pd.to_datetime(row["image_id"].split("_")[0], format="%Y%m%d", utc=True)


def _align_offsets(
    tile: "xr.Dataset", footprint: "gpd.GeoSeries", labels: "gpd.GeoDataFrame"
) -> tuple["gpd.GeoDataFrame", dict[str, float]]:
    from darts_acquisition import (
        load_planet_masks,
        load_planet_scene,
    )
    from darts_acquisition.utils.arosics import get_offsets

    assert tile.odc.crs == labels.crs, "Tile and labels must have the same CRS"
    # Align S2 data to Planet data if planet_data_dir is provided
    try:
        planetds = load_planet_scene(footprint.fpath)
        planet_mask = load_planet_masks(footprint.fpath)
        offsets_info = get_offsets(
            tile,
            planetds,
            bands=["red", "green", "blue", "nir"],
            window_size=128,
            target_mask=tile.quality_data_mask == 2,
            reference_mask=planet_mask.quality_data_mask == 2,
            resample_to="target",
        )
        logger.debug(f"Aligned S2 dataset to Planet dataset with offsets {offsets_info}.")
        if not offsets_info.is_valid():
            return labels, {"x_offset": 0, "y_offset": 0}
        x_offset = (offsets_info.x_offset or 0) * tile.odc.geobox.resolution.x
        y_offset = (offsets_info.y_offset or 0) * tile.odc.geobox.resolution.y
        labels["geometry"] = labels.geometry.translate(xoff=-x_offset, yoff=-y_offset)
        return labels, {
            "x_offset": x_offset,
            "y_offset": y_offset,
            "reliability": offsets_info.avg_reliability,
            "ssim_improvement": offsets_info.avg_ssim_improvement,
        }

    except Exception:
        logger.error("Error while aligning S2 dataset to Planet dataset, continue without alignment", exc_info=True)
        return labels, {}


def preprocess_s2_train_data(  # noqa: C901
    *,
    labels_dir: Path,
    train_data_dir: Path,
    arcticdem_dir: Path,
    tcvis_dir: Path,
    admin_dir: Path,
    planet_data_dir: Path | None = None,
    s2_download_cache: Path | None = None,
    preprocess_cache: Path | None = None,
    matching_cache: Path | None = None,
    force_preprocess: bool = False,
    append: bool = True,
    device: Literal["cuda", "cpu", "auto"] | int | None = None,
    ee_project: str | None = None,
    ee_use_highvolume: bool = True,
    matching_day_range: int = 7,
    matching_max_cloud_cover: int = 10,
    matching_min_intersects: float = 0.7,
    tpi_outer_radius: int = 100,
    tpi_inner_radius: int = 0,
    patch_size: int = 1024,
    overlap: int = 16,
    exclude_nopositive: bool = False,
    exclude_nan: bool = True,
    save_matching_scores: bool = False,
):
    """Preprocess Planet data for training.

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
        labels_dir (Path): The directory containing the labels and footprints / extents.
        train_data_dir (Path): The "output" directory where the tensors are written to.
        arcticdem_dir (Path): The directory containing the ArcticDEM data (the datacube and the extent files).
            Will be created and downloaded if it does not exist.
        tcvis_dir (Path): The directory containing the TCVis data.
        admin_dir (Path): The directory containing the admin files.
        planet_data_dir (Path, optional): The directory containing the Planet scenes and orthotiles.
            The planet data is used to align the Sentinel-2 data to the Planet data, spatially.
            Can be set to None if no alignment is wished.
            Defaults to None.
        s2_download_cache (Path): The directory to use for caching the raw downloaded sentinel 2 data. Defaults to None.
        preprocess_cache (Path, optional): The directory to store the preprocessed data. Defaults to None.
        matching_cache (Path, optional): The path to a file where the matchings are stored.
            Note: this is different from the matching scores.
        force_preprocess (bool, optional): Whether to force the preprocessing of the data. Defaults to False.
        append (bool, optional): Whether to append the data to the existing data. Defaults to True.
        device (Literal["cuda", "cpu"] | int, optional): The device to run the model on.
            If "cuda" take the first device (0), if int take the specified device.
            If "auto" try to automatically select a free GPU (<50% memory usage).
            Defaults to "cuda" if available, else "cpu".
        ee_project (str, optional): The Earth Engine project ID or number to use. May be omitted if
            project is defined within persistent API credentials obtained via `earthengine authenticate`.
        ee_use_highvolume (bool, optional): Whether to use the high volume server (https://earthengine-highvolume.googleapis.com).
            Defaults to True.
        matching_day_range (int, optional): The day range to use for matching S2 scenes to Planet footprints.
            Defaults to 7.
        matching_max_cloud_cover (int, optional): The maximum cloud cover percentage to use for matching S2 scenes
            to Planet footprints. Defaults to 10.
        matching_min_intersects (float, optional): The minimum intersection percentage to use for matching S2 scenes
            to Planet footprints. Defaults to 0.7.
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
        save_matching_scores (bool, optional): Whether to save the matching scores. Defaults to False.

    """
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    logger.info(f"Starting preprocessing at {current_time}.")

    # Storing the configuration as JSON file
    train_data_dir.mkdir(parents=True, exist_ok=True)
    from darts_utils.functools import write_function_args_to_config_file

    write_function_args_to_config_file(
        fpath=train_data_dir / f"{current_time}.cli.json",
        function=preprocess_s2_train_data,
        locals_=locals(),
    )

    from stopuhr import Chronometer

    timer = Chronometer(printer=logger.debug)

    from darts.utils.cuda import debug_info

    debug_info()

    # Import here to avoid long loading times when running other commands
    import geopandas as gpd
    import pandas as pd
    import rich
    import xarray as xr
    from darts_acquisition import (
        load_arcticdem,
        load_s2_from_stac,
        load_tcvis,
        match_s2ids_from_geodataframe_stac,
    )
    from darts_acquisition.admin import download_admin_files
    from darts_preprocessing import preprocess_v2
    from darts_segmentation.training.prepare_training import TrainDatasetBuilder
    from darts_utils.tilecache import XarrayCacheManager
    from odc.geo.geom import Geometry
    from pystac import Item
    from rich.progress import track

    from darts.utils.cuda import decide_device
    from darts.utils.earthengine import init_ee

    device = decide_device(device)
    init_ee(ee_project, ee_use_highvolume)
    logger.info("Configured Rasterio")

    labels = (gpd.read_file(labels_file) for labels_file in labels_dir.glob("*/TrainingLabel*.gpkg"))
    labels = gpd.GeoDataFrame(pd.concat(labels, ignore_index=True))

    footprints = (gpd.read_file(footprints_file) for footprints_file in labels_dir.glob("*/ImageFootprints*.gpkg"))
    footprints = gpd.GeoDataFrame(pd.concat(footprints, ignore_index=True))
    footprints["geometry"] = footprints["geometry"].simplify(0.001)  # Simplify to reduce compute
    footprints["date"] = footprints.apply(_parse_date, axis=1)
    if planet_data_dir is not None:
        fpaths = {fpath.stem: fpath for fpath in _planet_legacy_path_gen(planet_data_dir)}
        footprints["fpath"] = footprints.image_id.map(fpaths)

    # Find S2 scenes that intersect with the Planet footprints
    if matching_cache is None or not matching_cache.exists():
        matches = match_s2ids_from_geodataframe_stac(
            aoi=footprints,
            day_range=matching_day_range,
            max_cloud_cover=matching_max_cloud_cover,
            min_intersects=matching_min_intersects,
            simplify_geometry=0.001,
            save_scores=train_data_dir / "matching-scores.parquet" if save_matching_scores else None,
        )
        if matching_cache is not None:
            matches_serializable = {k: v.to_dict() if isinstance(v, Item) else "None" for k, v in matches.items()}
            with matching_cache.open("w") as f:
                json.dump(matches_serializable, f)
            logger.info(f"Saved matching scores to {matching_cache}")
            del matches_serializable  # Free memory
    else:
        logger.info(f"Loading matching scores from {matching_cache}")
        with matching_cache.open("r") as f:
            matches_serializable = json.load(f)
        matches = {int(k): Item.from_dict(v) if v != "None" else None for k, v in matches_serializable.items()}
        del matches_serializable  # Free memory
    footprints["s2_item"] = footprints.index.map(matches)

    # Filter out footprints without a matching S2 item
    logger.info(f"Found {len(footprints)} footprints, {footprints.s2_item.notna().sum()} with matching S2 items.")
    footprints = footprints[footprints.s2_item.notna()]

    # Download admin files if they do not exist
    admin2_fpath = admin_dir / "geoBoundariesCGAZ_ADM2.shp"
    if not admin2_fpath.exists():
        download_admin_files(admin_dir)
    admin2 = gpd.read_file(admin2_fpath)

    # We hardcode these since they depend on the preprocessing we use
    bands = [
        "red",
        "green",
        "blue",
        "nir",
        "ndvi",
        "relative_elevation",
        "slope",
        "aspect",
        "hillshade",
        "curvature",
        "tc_brightness",
        "tc_greenness",
        "tc_wetness",
    ]

    builder = TrainDatasetBuilder(
        train_data_dir=train_data_dir,
        patch_size=patch_size,
        overlap=overlap,
        bands=bands,
        exclude_nopositive=exclude_nopositive,
        exclude_nan=exclude_nan,
        device=device,
        append=append,
    )
    cache_manager = XarrayCacheManager(preprocess_cache)

    if append and (train_data_dir / "metadata.parquet").exists():
        metadata = gpd.read_parquet(train_data_dir / "metadata.parquet")
        already_processed_planet_ids = set(metadata["planet_id"].unique())
        logger.info(f"Already processed {len(already_processed_planet_ids)} samples.")
        footprints = footprints[~footprints.image_id.isin(already_processed_planet_ids)]

    for i, footprint in track(
        footprints.iterrows(), description="Processing samples", total=len(footprints), console=rich.get_console()
    ):
        s2_item = footprint.s2_item
        # Convert to stac item if dictionary
        if isinstance(s2_item, dict):
            s2_item = Item.from_dict(s2_item)

        s2_id = s2_item.id
        planet_id = footprint.image_id
        info_id = f"{s2_id=} -> {planet_id=} ({i + 1} of {len(footprints)})"
        try:
            logger.info(f"Processing sample {info_id}")

            if planet_data_dir is not None and (
                not footprint.fpath or (not footprint.fpath.exists() and not cache_manager.exists(planet_id))
            ):
                logger.warning(
                    f"Footprint image {planet_id} at {footprint.fpath} does not exist. Skipping sample {info_id}..."
                )
                continue

            def _get_tile():
                s2ds = load_s2_from_stac(s2_item, cache=s2_download_cache)

                # Crop to footprint geometry
                geom = Geometry(footprint.geometry, crs=footprints.crs)
                s2ds = s2ds.odc.crop(geom, apply_mask=True)
                # Crop above will change all dtypes to float32 -> change them back for s2_scl and qa mask
                s2ds["s2_scl"] = s2ds["s2_scl"].fillna(0.0).astype("uint8")
                s2ds["quality_data_mask"] = s2ds["quality_data_mask"].fillna(0.0).astype("uint8")

                # Preprocess as usual
                arctidem_res = 10
                arcticdem_buffer = ceil(tpi_outer_radius / arctidem_res * sqrt(2))
                arcticdem = load_arcticdem(
                    s2ds.odc.geobox, arcticdem_dir, resolution=arctidem_res, buffer=arcticdem_buffer
                )
                tcvis = load_tcvis(s2ds.odc.geobox, tcvis_dir)

                s2ds: xr.Dataset = preprocess_v2(
                    s2ds,
                    arcticdem,
                    tcvis,
                    tpi_outer_radius,
                    tpi_inner_radius,
                    device,
                )
                return s2ds

            with timer("Loading tile"):
                tile = cache_manager.get_or_create(
                    identifier=f"preprocess-s2train-v2-{s2_id}_{planet_id}",
                    creation_func=_get_tile,
                    force=force_preprocess,
                )
            logger.debug(f"Found tile with size {tile.sizes}")

            # Skip if the size is too small
            if tile.sizes["x"] < patch_size or tile.sizes["y"] < patch_size:
                logger.info(f"Skipping sample {info_id} due to small size {tile.sizes}.")
                continue

            footprint_labels = labels[labels.image_id == planet_id].to_crs(tile.odc.crs)
            region = _get_region_name(footprint, admin2)

            if planet_data_dir is not None:
                with timer("Align to PLANET"):
                    footprint_labels, offsets_info = _align_offsets(tile, footprint, footprint_labels)

            with timer("Save as patches"):
                builder.add_tile(
                    tile=tile,
                    labels=footprint_labels,
                    region=region,
                    sample_id=f"{s2_id}_{planet_id}",
                    metadata={
                        "planet_id": planet_id,
                        "s2_id": s2_id,
                        "fpath": footprint.fpath,
                        **offsets_info,
                    },
                )

            logger.info(f"Processed sample {info_id}")

        except (KeyboardInterrupt, SystemExit, SystemError):
            logger.info("Interrupted by user.")
            break

        except Exception as e:
            logger.warning(f"Could not process sample {info_id}. Skipping...")
            logger.exception(e)

    timer.summary()

    if len(builder) == 0:
        logger.warning("No samples were processed. Exiting...")
        return

    builder.finalize(
        {
            "planet_data_dir": planet_data_dir,
            "labels_dir": labels_dir,
            "arcticdem_dir": arcticdem_dir,
            "tcvis_dir": tcvis_dir,
            "ee_project": ee_project,
            "ee_use_highvolume": ee_use_highvolume,
            "tpi_outer_radius": tpi_outer_radius,
            "tpi_inner_radius": tpi_inner_radius,
        }
    )
