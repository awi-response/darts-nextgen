"""Darts export module for inference results."""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import xarray as xr
from darts_utils.bands import manager
from stopuhr import stopwatch

from darts_export import miniviz, vectorization

logger = logging.getLogger(__name__.replace("darts_", "darts."))


def _export_raster(tile: xr.Dataset, name: str, out_dir: Path, fname: str | None = None, tags={}):
    if fname is None:
        fname = name
    fpath = out_dir / f"{fname}.tif"
    with stopwatch(f"Exporting {name} to {fpath.resolve()}", printer=logger.debug):
        if tile[name].dtype == "bool":
            tile[name].astype("uint8").rio.to_raster(fpath, driver="GTiff", compress="LZW", tags=tags)
        else:
            tile[name].rio.to_raster(fpath, driver="GTiff", compress="LZW", tags=tags)


def _export_vector(tile: xr.Dataset, name: str, out_dir: Path, fname: str | None = None):
    if fname is None:
        fname = name
    fpath_gpkg = out_dir / f"{fname}.gpkg"
    fpath_parquet = out_dir / f"{fname}.parquet"
    with stopwatch(f"Exporting {name} to {fpath_gpkg.resolve()} and {fpath_parquet.resolve()}", printer=logger.debug):
        polygon_gdf = vectorization.vectorize(tile, name)
        polygon_gdf.to_file(fpath_gpkg, layer=f"{fname}")
        polygon_gdf.to_parquet(fpath_parquet)


def _export_polygonized(tile: xr.Dataset, out_dir: Path, ensemble_subsets: list[str] = []):
    _export_vector(tile, "binarized_segmentation", out_dir, fname="prediction_segments")
    for ensemble_subset in ensemble_subsets:
        _export_vector(
            tile,
            f"binarized_segmentation-{ensemble_subset}",
            out_dir,
            fname=f"prediction_segments-{ensemble_subset}",
        )


def _export_binarized(tile: xr.Dataset, out_dir: Path, ensemble_subsets: list[str] = [], tags={}):
    _export_raster(tile, "binarized_segmentation", out_dir, fname="binarized")
    for ensemble_subset in ensemble_subsets:
        _export_raster(
            tile,
            f"binarized_segmentation-{ensemble_subset}",
            out_dir,
            fname=f"binarized-{ensemble_subset}",
            tags=tags,
        )


def _export_probabilities(tile: xr.Dataset, out_dir: Path, ensemble_subsets: list[str] = [], tags={}):
    tile["probabilities"] = (tile["probabilities"] * 100).fillna(255).astype("uint8").rio.write_nodata(255)
    _export_raster(tile, "probabilities", out_dir, fname="probabilities", tags=tags)
    for ensemble_subset in ensemble_subsets:
        tile[f"probabilities-{ensemble_subset}"] = (
            (tile[f"probabilities-{ensemble_subset}"] * 100).fillna(255).astype("uint8").rio.write_nodata(255)
        )
        _export_raster(
            tile,
            f"probabilities-{ensemble_subset}",
            out_dir,
            fname=f"probabilities-{ensemble_subset}",
            tags=tags,
        )


def _export_thumbnail(tile: xr.Dataset, out_dir: Path):
    fpath = out_dir / "thumbnail.jpg"
    with stopwatch(f"Exporting thumbnail to {fpath}", printer=logger.debug):
        fig = miniviz.thumbnail(tile)
        fig.savefig(fpath)
        fig.clear()


def _export_metadata(out_dir: Path, metadata: dict):
    with (out_dir / "darts_inference.json").open("w") as fp:
        json.dump(metadata, fp, indent=2)


@stopwatch.f("Exporting tile", printer=logger.debug, print_kwargs=["bands", "ensemble_subsets"])
def export_tile(  # noqa: C901
    tile: xr.Dataset,
    out_dir: Path,
    bands: list[str] = ["probabilities", "binarized", "polygonized", "extent", "thumbnail"],
    ensemble_subsets: list[str] = [],
    metadata: dict = {},
    debug: bool = False,
):
    """Export segmentation results to multiple file formats for analysis and distribution.

    This function exports a processed tile to an output directory, creating multiple file
    formats including GeoTIFFs, GeoPackages, Parquet files, and visualizations. It handles
    both ensemble-averaged results and individual model outputs.

    Args:
        tile (xr.Dataset): Processed tile from prepare_export() containing segmentation results.
            Must include spatial reference information (CRS, coordinates).
        out_dir (Path): Output directory path. Created if it doesn't exist.
        bands (list[str], optional): List of data products to export. Options:
            - "probabilities": Probability maps as GeoTIFF (uint8, 0-100 scale, 255=nodata)
            - "binarized": Binary segmentation masks as GeoTIFF (uint8, 0/1)
            - "polygonized": Vectorized segmentations as GeoPackage and Parquet
            - "extent": Valid data extent as vector (GeoPackage and Parquet)
            - "thumbnail": RGB visualization as JPEG
            - "optical": Optical bands (red, green, blue, nir) as multi-band GeoTIFF
            - "dem": Terrain features (slope, relative_elevation) as multi-band GeoTIFF
            - "tcvis": TCVIS features (tc_brightness, tc_greenness, tc_wetness) as GeoTIFF
            - "metadata": Metadata JSON file
            - Any other variable name: Exported as single-band GeoTIFF
            Defaults to ["probabilities", "binarized", "polygonized", "extent", "thumbnail"].
        ensemble_subsets (list[str], optional): Names of individual ensemble models to export
            separately (e.g., ["with_tcvis", "without_tcvis"]). Creates suffixed files for
            each subset. Defaults to [].
        metadata (dict, optional): Metadata dictionary to embed in raster tags and export as JSON.
            Automatically adds DARTS_exportdate timestamp. Defaults to {}.
        debug (bool, optional): If True, exports complete tile as NetCDF file
            (darts_inference_debug.nc) for debugging. Defaults to False.

    Raises:
        ValueError: If a specified band is not found in the tile's data variables.

    Note:
        Output files created (depending on `bands` parameter):

        Raster outputs (GeoTIFF with LZW compression):
        - probabilities.tif: Uint8 [0-100], 255=nodata
        - binarized.tif: Uint8 binary mask
        - optical.tif: Multi-band optical imagery
        - dem.tif: Multi-band terrain features
        - tcvis.tif: Multi-band TCVIS features
        - {custom_band}.tif: Single-band custom exports

        Vector outputs (GeoPackage + Parquet):
        - prediction_segments.gpkg/.parquet: Polygonized segmentation
        - prediction_extent.gpkg/.parquet: Valid data extent

        Visualization:
        - thumbnail.jpg: RGB composite with overlay

        Metadata:
        - darts_inference.json: Metadata dictionary

        Ensemble subsets:
        All raster and vector outputs get suffixed versions for each subset:
        - probabilities-{subset}.tif
        - binarized-{subset}.tif
        - prediction_segments-{subset}.gpkg/.parquet

    Example:
        Standard export workflow:

        ```python
        from pathlib import Path
        from darts_export import export_tile

        # After prepare_export()
        export_tile(
            tile=processed_tile,
            out_dir=Path("/output/scene_12345"),
            bands=["probabilities", "binarized", "polygonized", "extent", "thumbnail"],
            ensemble_subsets=["with_tcvis", "without_tcvis"],
            metadata={
                "scene_id": "S2A_MSIL2A_20230615...",
                "model_version": "v2.1",
                "processing_date": "2023-06-20"
            },
            debug=False
        )

        # Creates:
        # /output/scene_12345/
        #   ├── probabilities.tif
        #   ├── probabilities-with_tcvis.tif
        #   ├── probabilities-without_tcvis.tif
        #   ├── binarized.tif
        #   ├── binarized-with_tcvis.tif
        #   ├── binarized-without_tcvis.tif
        #   ├── prediction_segments.gpkg
        #   ├── prediction_segments.parquet
        #   ├── prediction_segments-with_tcvis.gpkg
        #   ├── prediction_segments-with_tcvis.parquet
        #   ├── prediction_segments-without_tcvis.gpkg
        #   ├── prediction_segments-without_tcvis.parquet
        #   ├── prediction_extent.gpkg
        #   ├── prediction_extent.parquet
        #   ├── thumbnail.jpg
        #   └── darts_inference.json
        ```

    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if len(metadata) > 0:
        metadata["DARTS_exportdate"] = str(datetime.now(UTC))

    raster_tags = metadata

    if debug:
        manager.to_netcdf(tile, out_dir / "darts_inference_debug.nc", crop=False)

    for band in bands:
        match band:
            case "polygonized":
                _export_polygonized(tile, out_dir, ensemble_subsets)
            case "binarized":
                _export_binarized(tile, out_dir, ensemble_subsets, tags=raster_tags)
            case "probabilities":
                _export_probabilities(tile, out_dir, ensemble_subsets, tags=raster_tags)
            case "extent":
                _export_vector(tile, "extent", out_dir, fname="prediction_extent")
            case "thumbnail":
                _export_thumbnail(tile, out_dir)
            case "optical":
                _export_raster(tile, ["red", "green", "blue", "nir"], out_dir, fname="optical", tags=raster_tags)
            case "dem":
                _export_raster(tile, ["slope", "relative_elevation"], out_dir, fname="dem", tags=raster_tags)
            case "tcvis":
                _export_raster(
                    tile, ["tc_brightness", "tc_greenness", "tc_wetness"], out_dir, fname="tcvis", tags=raster_tags
                )
            case "metadata":
                _export_metadata(out_dir, metadata)
            case _:
                if band not in tile.data_vars:
                    raise ValueError(
                        f"Band {band} not found in tile for export. Available bands are: {list(tile.data_vars.keys())}"
                    )
                # Export the band as a raster
                _export_raster(tile, band, out_dir, tags=raster_tags)
