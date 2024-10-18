"""Darts export module for inference results."""

import logging
from pathlib import Path

import xarray

from darts_export import vectorization

L = logging.getLogger("darts.export")


class InferenceResultWriter:
    """Writer class to export inference result datasets."""

    def __init__(self, ds) -> None:
        """Initialize the dataset."""
        self.ds: xarray.Dataset = ds

    def export_probabilities(self, path: Path, filename="pred_probabilities.tif", tags={}):
        """Export the probabilities layer to a file.

        Args:
            path (Path): The path where to export to.
            filename (str, optional): the filename. Defaults to "pred_probabilities.tif".
            tags (dict, optional): optional GeoTIFF metadate to be written. Defaults to no additional metadata.

        Returns:
            the Path of the written file

        """
        # write the probability layer from the raster to a GeoTiff
        file_path = path / filename
        self.ds.probabilities.rio.to_raster(file_path, driver="GTiff", tags=tags, compress="LZW")
        return file_path

    def export_binarized(self, path: Path, filename="pred_binarized.tif", tags={}):
        """Export the binarized segmentation result of the inference Result.

        Args:
            path (Path): The path where to export to.
            filename (str, optional): the filename. Defaults to "pred_binarized.tif".
            tags (dict, optional): optional GeoTIFF metadate to be written. Defaults to no additional metadata.

        Returns:
            the Path of the written file

        """
        file_path = path / filename
        self.ds.binarized_segmentation.rio.to_raster(file_path, driver="GTiff", tags=tags, compress="LZW")
        return file_path

    def export_polygonized(self, path: Path, filename_prefix="pred_segments", minimum_mapping_unit=32):
        """Export the binarized probabilities as a vector dataset in GeoPackage and GeoParquet format.

        Args:
            path (Path): The path where to export the files
            filename_prefix (str, optional): the file prefix of the exported files. Defaults to "pred_segments".
            minimum_mapping_unit (int, optional): segments covering less pixel are removed. Defaults to 32.

        """
        polygon_gdf = vectorization.vectorize(self.ds, minimum_mapping_unit=minimum_mapping_unit)

        path_gpkg = path / f"{filename_prefix}.gpkg"
        path_parquet = path / f"{filename_prefix}.parquet"

        polygon_gdf.to_file(path_gpkg, layer=filename_prefix)
        polygon_gdf.to_parquet(path_parquet)
