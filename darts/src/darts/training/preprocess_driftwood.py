"""PLANET preprocessing functions for training with the v2 data preprocessing."""

import logging
import time
from collections.abc import Generator
from math import ceil, sqrt
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)


def preprocess_driftwood_train_data(
    *,
    labels_dir: Path,
    footprints_dir: Path,
    train_data_dir: Path,
    arcticdem_dir: Path,
    tcvis_dir: Path,
    admin_dir: Path,
    preprocess_cache: Path | None = None,
    force_preprocess: bool = False,
    append: bool = True,
    device: Literal["cuda", "cpu", "auto"] | int | None = None,
    ee_project: str | None = None,
    ee_use_highvolume: bool = True,
    tpi_outer_radius: int = 100,
    tpi_inner_radius: int = 0,
    patch_size: int = 1024,
    overlap: int = 16,
    exclude_nopositive: bool = False,
    exclude_nan: bool = True,
    mask_erosion_size: int = 3,
):
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    logger.info(f"Starting preprocessing at {current_time}.")

    from stopuhr import Chronometer

    timer = Chronometer(printer=logger.debug)

    from darts.utils.cuda import debug_info

    debug_info()

    # Import here to avoid long loading times when running other commands
    import geopandas as gpd
    import rich
    import xarray as xr
    from darts_acquisition import load_arcticdem, load_planet_masks, load_planet_scene, load_tcvis
    from darts_preprocessing import preprocess_v2
    from darts_segmentation.training.prepare_training import TrainDatasetBuilder
    from darts_segmentation.utils import Bands
    from darts_utils.tilecache import XarrayCacheManager
    from odc.stac import configure_rio
    from rich.progress import track

    from darts.utils.cuda import decide_device
    from darts.utils.earthengine import init_ee

    device = decide_device(device)
    init_ee(ee_project, ee_use_highvolume)
    configure_rio(cloud_defaults=True, aws={"aws_unsigned": True})
    logger.info("Configured Rasterio")

    # We hardcode these because they depend on the preprocessing used
    bands = Bands.from_dict(
        {
            "red": (1 / 3000, 0.0),
            "green": (1 / 3000, 0.0),
            "blue": (1 / 3000, 0.0),
            "nir": (1 / 3000, 0.0),
            "ndvi": (1 / 20000, 0.0),
        }
    )

    builder = TrainDatasetBuilder(
        train_data_dir=train_data_dir,
        patch_size=patch_size,
        overlap=overlap,
        bands=bands,
        exclude_nopositive=exclude_nopositive,
        exclude_nan=exclude_nan,
        mask_erosion_size=mask_erosion_size,
        device=device,
        append=append,
    )
    cache_manager = XarrayCacheManager(preprocess_cache / "planet_v2")

    def _utm_zones_gen() -> Generator[tuple[int, str], None, None]:
        for year_dir in labels_dir.iterdir():
            year = int(year_dir.name)
            for utm_dir in year_dir.iterdir():
                if not utm_dir.name.startswith("utm_"):
                    continue
                utm_zone = utm_dir.name
                yield (year, utm_zone)

    utm_zones = list(_utm_zones_gen())
    if not utm_zones:
        raise ValueError(
            f"No UTM zones found in {labels_dir=}. Please check the structure of your labels directory."
            " It should be like: year/utm_zone/[aoi.gpkg | dw.gpkg]"
        )
    logger.info(f"Found {len(utm_zones)} UTM zones in {labels_dir=}.")

    if append and (train_data_dir / "metadata.parquet").exists():
        metadata = gpd.read_parquet(train_data_dir / "metadata.parquet")
        already_processed_utm_zones = set(metadata["zone_id"].unique())
        logger.info(f"Already processed {len(already_processed_utm_zones)} samples.")
        utm_zones = [(year, zone) for year, zone in utm_zones if f"{year}-{zone}" not in already_processed_utm_zones]

    for i, (year, zone) in track(
        enumerate(utm_zones), description="Processing samples", total=len(utm_zones), console=rich.get_console()
    ):
        zone_id = f"{year}-{zone}"
        logger.info(f"Processing UTM zone {zone} in {year} ({zone_id}: {i + 1} of {len(utm_zones)})")

        try:
            utm_dir = labels_dir / str(year) / zone
            planet_mosaic = gpd.read_file(footprints_dir / f"footprints_plant_{year}.gpkg")
            aoi = gpd.read_file(utm_dir / "aoi.gpkg")
            dw = gpd.read_file(utm_dir / "dw.gpkg")
            aoi = aoi.sjoin(planet_mosaic, how="left", predicate="intersects")

            def _get_tile():
                tile = load_planet_scene(footprint.fpath)
                arctidem_res = 2
                arcticdem_buffer = ceil(tpi_outer_radius / arctidem_res * sqrt(2))
                arcticdem = load_arcticdem(
                    tile.odc.geobox, arcticdem_dir, resolution=arctidem_res, buffer=arcticdem_buffer
                )
                tcvis = load_tcvis(tile.odc.geobox, tcvis_dir)
                data_masks = load_planet_masks(footprint.fpath)
                tile = xr.merge([tile, data_masks])

                tile: xr.Dataset = preprocess_v2(
                    tile,
                    arcticdem,
                    tcvis,
                    tpi_outer_radius,
                    tpi_inner_radius,
                    device,
                )
                return tile

            with timer("Loading tile"):
                tile = cache_manager.get_or_create(
                    identifier=planet_id,
                    creation_func=_get_tile,
                    force=force_preprocess,
                )

            logger.debug(f"Found tile with size {tile.sizes}")

            footprint_labels = labels[labels.image_id == planet_id]

            with timer("Save as patches"):
                builder.add_tile(
                    tile=tile,
                    labels=footprint_labels,
                    region=zone,
                    sample_id=zone_id,
                    metadata={
                        "year": year,
                        "zone": zone,
                        "zone_id": zone_id,
                    },
                )

            logger.info(f"Processed UTM zone {zone} in {year} ({zone_id}: {i + 1} of {len(utm_zones)})")

        except (KeyboardInterrupt, SystemExit, SystemError):
            logger.info("Interrupted by user.")
            break

        except Exception as e:
            logger.warning(
                f"Could not process UTM zone {zone} in {year} ({zone_id}: {i + 1} of {len(utm_zones)}) \nSkipping..."
            )
            logger.exception(e)

    builder.finalize(
        {
            "labels_dir": labels_dir,
            "footprints_dir": footprints_dir,
            "arcticdem_dir": arcticdem_dir,
            "tcvis_dir": tcvis_dir,
            "ee_project": ee_project,
            "ee_use_highvolume": ee_use_highvolume,
            "tpi_outer_radius": tpi_outer_radius,
            "tpi_inner_radius": tpi_inner_radius,
        }
    )
    timer.summary()
