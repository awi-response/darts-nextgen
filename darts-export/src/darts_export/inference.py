"""Darts export module for inference results."""

import logging
import time
from pathlib import Path

import xarray as xr
from darts_utils.stopuhr import stopuhr

from darts_export import vectorization
from darts_export.miniviz import thumbnail

logger = logging.getLogger(__name__.replace("darts_", "darts."))


def _get_subset_names(ds: xr.Dataset):
    return [subset.removeprefix("probabilities-") for subset in ds.keys() if subset.startswith("probabilities-")]


def export_probabilities(tile: xr.Dataset, out_dir: Path, export_ensemble_inputs: bool = False, tags: dict = {}):
    """Export the probabilities layer to a file.

    If `export_ensemble_inputs` is set to True and the ensemble used at least two models for inference,
    the probabilities of the models will be written as individual files as well.

    Args:
        tile (xr.Dataset): The inference result.
        out_dir (Path): The path where to export to.
        export_ensemble_inputs (bool, optional): Also save the model outputs, not only the ensemble result.
            Only applies if the inference result is an ensemble result and has at least two inputs.
            Defaults to False.
        tags (dict, optional): optional GeoTIFF metadata to be written. Defaults to no additional metadata.

    """
    subset_names = _get_subset_names(tile)
    if export_ensemble_inputs and len(subset_names) > 1:
        for subset in _get_subset_names(tile):
            tick_estart = time.perf_counter()
            layer_name = f"probabilities-{subset}"
            fpath = out_dir / f"{layer_name}.tif"
            tile[layer_name].rio.to_raster(fpath, driver="GTiff", tags=tags, compress="LZW")
            tick_eend = time.perf_counter()
            logger.debug(f"Exported probabilities for {subset} to {fpath} in {tick_eend - tick_estart:.2f}s")

    fpath = out_dir / "probabilities.tif"
    with stopuhr(f"Exporting probabilities to {fpath}", logger.debug):
        tile["probabilities"].rio.to_raster(fpath, driver="GTiff", tags=tags, compress="LZW")


def export_binarized(tile: xr.Dataset, out_dir: Path, export_ensemble_inputs: bool = False, tags: dict = {}):
    """Export the binarized segmentation layer to a file.

    If `export_ensemble_inputs` is set to True and the ensemble used at least two models for inference,
    the binarized segmentation of the models will be written as individual files as well.

    Args:
        tile (xr.Dataset): The inference result.
        out_dir (Path): The path where to export to.
        export_ensemble_inputs (bool, optional): Also save the model outputs, not only the ensemble result.
            Only applies if the inference result is an ensemble result and has at least two inputs.
            Defaults to False.
        tags (dict, optional): optional GeoTIFF metadata to be written. Defaults to no additional metadata.

    """
    subset_names = _get_subset_names(tile)
    if export_ensemble_inputs and len(subset_names) > 1:
        for subset in _get_subset_names(tile):
            tick_estart = time.perf_counter()
            layer_name = f"binarized_segmentation-{subset}"
            fpath = out_dir / f"{layer_name}.tif"
            tile[layer_name].rio.to_raster(fpath, driver="GTiff", tags=tags, compress="LZW")
            tick_eend = time.perf_counter()
            logger.debug(f"Exported binarized segmentation for {subset} to {fpath} in {tick_eend - tick_estart:.2f}s")

    fpath = out_dir / "binarized.tif"
    with stopuhr(f"Exporting binarized segmentation to {fpath}", logger.debug):
        tile["binarized_segmentation"].rio.to_raster(fpath, driver="GTiff", tags=tags, compress="LZW")


def export_polygonized(
    tile: xr.Dataset, out_dir: Path, export_ensemble_inputs: bool = False, minimum_mapping_unit: int = 32
):
    """Export the binarized probabilities as a vector dataset in GeoPackage and GeoParquet format.

    If `export_ensemble_inputs` is set to True and the ensemble used at least two models for inference,
    the vectorized binarized segmentation of the models will be written as individual files as well.

    Args:
        tile (xr.Dataset): The inference result.
        out_dir (Path): The path where to export to.
        export_ensemble_inputs (bool, optional): Also save the model outputs, not only the ensemble result.
            Only applies if the inference result is an ensemble result and has at least two inputs.
            Defaults to False.
        minimum_mapping_unit (int, optional): segments covering less pixel are removed. Defaults to 32.

    """
    subset_names = _get_subset_names(tile)
    if export_ensemble_inputs and len(subset_names) > 1:
        for subset in _get_subset_names(tile):
            tick_estart = time.perf_counter()
            layer_name = f"binarized_segmentation-{subset}"
            fpath_gpkg = out_dir / f"prediction_segments-{subset}.gpkg"
            fpath_parquet = out_dir / f"prediction_segments-{subset}.parquet"
            polygon_gdf = vectorization.vectorize(tile, layer_name, minimum_mapping_unit=minimum_mapping_unit)
            polygon_gdf.to_file(fpath_gpkg, layer=f"prediction_segments-{subset}")
            polygon_gdf.to_parquet(fpath_parquet)
            tick_eend = time.perf_counter()
            logger.debug(
                f"Exported binarized segmentation for {subset} to {fpath_gpkg} and {fpath_parquet}"
                f" in {tick_eend - tick_estart:.2f}s"
            )

    fpath_gpkg = out_dir / "prediction_segments.gpkg"
    fpath_parquet = out_dir / "prediction_segments.parquet"
    with stopuhr(f"Exporting binarized segmentation to {fpath_gpkg} and {fpath_parquet}", logger.debug):
        polygon_gdf = vectorization.vectorize(tile, "binarized_segmentation", minimum_mapping_unit=minimum_mapping_unit)
        polygon_gdf.to_file(fpath_gpkg, layer="prediction_segments")
        polygon_gdf.to_parquet(fpath_parquet)


def export_datamask(tile: xr.Dataset, out_dir: Path):
    """Export the data mask as a GeoTIFF file.

    Args:
        tile (xr.Dataset): The inference result.
        out_dir (Path): The path where to export to.

    """
    fpath = out_dir / "data_mask.tif"
    with stopuhr(f"Exporting data mask to {fpath}", logger.debug):
        tile["quality_data_mask"].rio.to_raster(fpath, driver="GTiff", compress="LZW")


def export_arcticdem_datamask(tile: xr.Dataset, out_dir: Path):
    """Export the arcticdem data mask as a GeoTIFF file.

    Args:
        tile (xr.Dataset): The inference result.
        out_dir (Path): The path where to export to.

    """
    fpath = out_dir / "arcticdem_data_mask.tif"
    with stopuhr(f"Exporting arcticdem data mask to {fpath}", logger.debug):
        tile["arcticdem_data_mask"].rio.to_raster(fpath, driver="GTiff", compress="LZW")


def export_extent(tile: xr.Dataset, out_dir: Path):
    """Export the extent of the prediction as a vector dataset in GeoPackage and GeoParquet format.

    Args:
        tile (xr.Dataset): The inference result.
        out_dir (Path): The path where to export to.

    """
    fpath_gpkg = out_dir / "prediction_extent.gpkg"
    fpath_parquet = out_dir / "prediction_extent.parquet"
    with stopuhr(f"Exporting extent to {fpath_gpkg} and {fpath_parquet}", logger.debug):
        polygon_gdf = vectorization.vectorize(tile, "quality_data_mask", minimum_mapping_unit=0)
        polygon_gdf.to_file(fpath_gpkg, layer="prediction_extent")
        polygon_gdf.to_parquet(fpath_parquet)


def export_optical(tile: xr.Dataset, out_dir: Path):
    """Export the optical data as a GeoTIFF file.

    Args:
        tile (xr.Dataset): The inference result.
        out_dir (Path): The path where to export to.

    """
    fpath = out_dir / "optical.tif"
    with stopuhr(f"Exporting optical data to {fpath}", logger.debug):
        tile[["red", "green", "blue", "nir"]].rio.to_raster(fpath, driver="GTiff", compress="LZW")


def export_dem(tile: xr.Dataset, out_dir: Path):
    """Export the DEM data as a GeoTIFF file.

    Args:
        tile (xr.Dataset): The inference result.
        out_dir (Path): The path where to export to.

    """
    fpath = out_dir / "dem.tif"
    with stopuhr(f"Exporting DEM data to {fpath}", logger.debug):
        tile[["slope", "relative_elevation"]].rio.to_raster(fpath, driver="GTiff", compress="LZW")


def export_tcvis(tile: xr.Dataset, out_dir: Path):
    """Export the TCVIS data as a GeoTIFF file.

    Args:
        tile (xr.Dataset): The inference result.
        out_dir (Path): The path where to export to.

    """
    fpath = out_dir / "tcvis.tif"
    with stopuhr(f"Exporting TCVIS data to {fpath}", logger.debug):
        tile[["tc_brightness", "tc_greenness", "tc_wetness"]].rio.to_raster(fpath, driver="GTiff", compress="LZW")


def export_thumbnail(tile: xr.Dataset, out_dir: Path):
    """Export a thumbnail of the optical data.

    Args:
        tile (xr.Dataset): The inference result.
        out_dir (Path): The path where to export to.

    """
    fpath = out_dir / "thumbnail.jpg"
    with stopuhr(f"Exporting thumbnail to {fpath}", logger.debug):
        fig = thumbnail(tile)
        fig.savefig(fpath)
        fig.clear()
