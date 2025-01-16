"""Landsat Trends related Data Loading. Should be used temporary and maybe moved to the acquisition package."""

import logging
import multiprocessing as mp
import time
import warnings
from pathlib import Path

import ee
import numpy as np
import xarray as xr
import xee  # noqa: F401
import zarr
import zarr.storage
from odc.geo.geobox import GeoBox, GeoboxTiles

from darts_acquisition.utils.storage import create_empty_datacube

logger = logging.getLogger(__name__.replace("darts_", "darts."))

EE_WARN_MSG = "Unable to retrieve 'system:time_start' values from an ImageCollection due to: No 'system:time_start' values found in the 'ImageCollection'."  # noqa: E501

CHUNK_SIZE = 3600
DATA_EXTENT = GeoBox.from_bbox((-180, 60, 180, 90), "epsg:4326", resolution=0.00026949)
DATA_VARS = ["tc_brightness", "tc_greenness", "tc_wetness"]
DATA_VARS_META = {
    "tc_brightness": {
        "long_name": "Tasseled Cap Brightness",
        "data_source": "ee:ingmarnitze/TCTrend_SR_2000-2019_TCVIS",
    },
    "tc_greenness": {
        "long_name": "Tasseled Cap Greenness",
        "data_source": "ee:ingmarnitze/TCTrend_SR_2000-2019_TCVIS",
    },
    "tc_wetness": {
        "long_name": "Tasseled Cap Wetness",
        "data_source": "ee:ingmarnitze/TCTrend_SR_2000-2019_TCVIS",
    },
}

DATA_VARS_ENCODING = {
    "tc_brightness": {"dtype": "uint8"},
    "tc_greenness": {"dtype": "uint8"},
    "tc_wetness": {"dtype": "uint8"},
}

# Lock for downloading the data
download_lock = mp.Lock()


def procedural_download_datacube(storage: zarr.storage.Store, geobox: GeoBox):
    """Download the TCVIS data procedurally and add it to the datacube.

    Args:
        storage (zarr.storage.Store): The zarr storage object where the datacube will be saved.
        geobox (GeoBox): The geobox to download the data for.

    References:
        - https://earthmover.io/blog/serverless-datacube-pipeline

    Warning:
        This function is not thread-safe. Thread-safety might be added in the future.

    """
    tick_fstart = time.perf_counter()

    # Check if data already exists
    tcvis_datacube = xr.open_zarr(storage, mask_and_scale=False)
    loaded_tiles: list[str] = tcvis_datacube.attrs.get("loaded_tiles", []).copy()

    # Get chunk size of datacube and create global grid
    chunk_size = tcvis_datacube.chunks["y"][0]
    tiles = GeoboxTiles(DATA_EXTENT, (chunk_size, chunk_size))

    # Get new, intersecting tiles
    new_tiles = [(xidx, yidx) for xidx, yidx in tiles.tiles(geobox.extent) if f"{xidx}_{yidx}" not in loaded_tiles]

    if not len(new_tiles):
        logger.debug("No new tiles to download")
        return
    logger.debug(f"Downloading {len(new_tiles)} new tiles")

    for yidx, xidx in new_tiles:
        tick_stile = time.perf_counter()
        tileid = f"{yidx}_{xidx}"
        geobox_tile = tiles[yidx, xidx]
        # Note: This is a little bit weird: First we create an own grid which overlaps to the chunks
        # of the zarr array. Then we create a mosaic of the data and clip it to a single chunk.
        # We could load the images from the collection directly instead of creating a mosaic.
        # However, this would require more testing and probably results a lot of manual computation
        # of slices etc. like in the arcticdem. So for now, we just use the mosaic.
        logging.getLogger("urllib3.connectionpool").disabled = True
        geom = ee.Geometry.Rectangle(geobox_tile.geographic_extent.boundingbox)
        ee_image_tcvis = ee.ImageCollection("users/ingmarnitze/TCTrend_SR_2000-2019_TCVIS").mosaic().clip(geom)
        ee_image_tcvis = ee.ImageCollection(ee_image_tcvis)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=EE_WARN_MSG)
            ds = xr.open_dataset(
                ee_image_tcvis,
                engine="ee",
                geometry=geom,
                crs="epsg:4326",
                scale=DATA_EXTENT.resolution.x,
            )
        # Update dataset properties to fit our pipeline-api
        ds = ds.isel(time=0).drop_vars("time").rename({"lon": "x", "lat": "y"}).transpose("y", "x")
        ds = ds.rename_vars(
            {
                "TCB_slope": "tc_brightness",
                "TCG_slope": "tc_greenness",
                "TCW_slope": "tc_wetness",
            }
        )
        tick_search = time.perf_counter()
        logger.debug(f"Found a dataset with shape {ds.sizes} for tile {tileid} in {tick_search - tick_stile} seconds.")

        # Download the data
        tick_downloads = time.perf_counter()
        ds = ds.compute()
        tick_downloade = time.perf_counter()
        logger.debug(f"Downloaded the data for tile {tileid} in {tick_downloade - tick_downloads} seconds.")
        logging.getLogger("urllib3.connectionpool").disabled = False

        # Recrop the data to the geobox_tile, since gee does not always return the exact extent
        ds = ds.odc.crop(geobox_tile.extent)

        # Save original min-max values for each band for clipping later
        clip_values = {band: (ds[band].min().values.item(), ds[band].max().values.item()) for band in ds.data_vars}

        # Interpolate missing values (there are very few, so we actually can interpolate them)
        ds.rio.write_crs(ds.attrs["crs"], inplace=True)
        ds.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
        for band in ds.data_vars:
            ds[band] = ds[band].rio.write_nodata(np.nan).rio.interpolate_na()

        # Convert to uint8
        for band in ds.data_vars:
            band_min, band_max = clip_values[band]
            ds[band] = ds[band].clip(band_min, band_max, keep_attrs=True).astype("uint8").rio.write_nodata(None)

        x_start = chunk_size * xidx
        y_start = chunk_size * yidx
        target_slice = {
            "x": slice(x_start, x_start + chunk_size),
            "y": slice(y_start, y_start + chunk_size),
        }
        # Save the data to the zarr storage
        tick_save = time.perf_counter()
        ds.drop_vars(["spatial_ref"]).reindex(y=list(reversed(ds.y))).to_zarr(storage, region=target_slice)
        tick_saved = time.perf_counter()
        logger.debug(f"Saved the data for tile {tileid} in {tick_saved - tick_save} seconds.")

        # Update loaded_tiles (with native zarr, since xarray does not support this yet)
        loaded_tiles.append(tileid)
        zarr.open(storage).attrs["loaded_tiles"] = loaded_tiles
        # Xarray default behaviour is to read the consolidated metadata, hence, we must update it
        zarr.consolidate_metadata(storage)

    tick_fend = time.perf_counter()
    logger.info(f"Procedural download of {len(new_tiles)} tiles completed in {tick_fend - tick_fstart:.2f} seconds")


def load_tcvis(
    geobox: GeoBox,
    data_dir: Path | str,
    buffer: int = 0,
    persist: bool = True,
) -> xr.Dataset:
    """Load the TCVIS for the given geobox, fetch new data from GEE if necessary.

    Args:
        geobox (GeoBox): The geobox to load the data for.
        data_dir (Path | str): The directory to store the downloaded data for faster access for consecutive calls.
        buffer (int, optional): The buffer around the geobox in pixels. Defaults to 0.
        persist (bool, optional): If the data should be persisted in memory.
            If not, this will return a Dask backed Dataset. Defaults to True.

    Returns:
        xr.Dataset: The TCVIS dataset.

    Usage:
        Since the API of the `load_tcvis` is based on GeoBox, one can load a specific ROI based on an existing Xarray DataArray:

        ```python
        import xarray as xr
        import odc.geo.xr

        from darts_aquisition import load_tcvis

        # Assume "optical" is an already loaded s2 based dataarray

        tcvis = load_tcvis(
            optical.odc.geobox,
            "/path/to/tcvis-parent-directory",
        )

        # Now we can for example match the resolution and extent of the optical data:
        tcvis = tcvis.odc.reproject(optical.odc.geobox, resampling="cubic")
        ```

    """  # noqa: E501
    tick_fstart = time.perf_counter()

    data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir

    datacube_fpath = data_dir / "tcvis_2000-2019.zarr"
    storage = zarr.storage.FSStore(datacube_fpath)
    logger.debug(f"Loading TCVis from {datacube_fpath.resolve()}")

    if not datacube_fpath.exists():
        logger.debug(f"Creating a new zarr datacube at {datacube_fpath.resolve()} with {storage=}")
        create_empty_datacube(
            title="Landsat Trends TCVIS 2000-2019",
            storage=storage,
            geobox=DATA_EXTENT,
            chunk_size=CHUNK_SIZE,
            data_vars=DATA_VARS,
            meta=DATA_VARS_META,
            var_encoding=DATA_VARS_ENCODING,
        )

    # Download the adjacent tiles (if necessary)
    reference_geobox = geobox.to_crs("epsg:4326", resolution=DATA_EXTENT.resolution.x).pad(buffer)
    with download_lock:
        procedural_download_datacube(storage, reference_geobox)

    # Load the datacube and set the spatial_ref since it is set as a coordinate within the zarr format
    chunks = None if persist else "auto"
    tcvis_datacube = xr.open_zarr(storage, mask_and_scale=False, chunks=chunks).set_coords("spatial_ref")

    # Get an AOI slice of the datacube
    tcvis_aoi = tcvis_datacube.odc.crop(reference_geobox.extent, apply_mask=False)

    # The following code would load the lazy zarr data from disk into memory
    if persist:
        tick_sload = time.perf_counter()
        tcvis_aoi = tcvis_aoi.load()
        tick_eload = time.perf_counter()
        logger.debug(f"TCVIS AOI loaded from disk in {tick_eload - tick_sload:.2f} seconds")

    logger.info(
        f"TCVIS tile {'loaded' if persist else 'lazy-opened'} in {time.perf_counter() - tick_fstart:.2f} seconds"
    )
    return tcvis_aoi
