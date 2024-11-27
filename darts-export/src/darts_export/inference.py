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

    def export_geotiff(self, path: Path, filename: str, layername: str, tags={}):
        """Export a GeoTiff file from the inference result, specifying the layer to export.

        Args:
            path (Path): the folder path where to write the result as GeoTIFF
            filename (str): The filename (basename) of the GeoTIFF to write
            layername (str): the name of the layer to write
            tags (dict, optional): optional GeoTIFF metadata to be written. Defaults to no additional metadata.

        Returns:
            _type_: _description_

        """
        # write the probability layer from the raster to a GeoTiff
        file_path = path / filename
        self.ds[layername].rio.to_raster(file_path, driver="GTiff", tags=tags, compress="LZW")
        return file_path

    def export_probabilities(self, path: Path, filename="pred_probabilities.tif", tags={}):
        """Export the probabilities layer to a file.

        If the inference result is an ensemble result and it contains also results of the models,
        also the probabilities of the models will be written as individual files as well.

        Args:
            path (Path): The path where to export to.
            filename (str, optional): the filename. Defaults to "pred_probabilities.tif".
            tags (dict, optional): optional GeoTIFF metadata to be written. Defaults to no additional metadata.

        Returns:
            the Path of the written file

        """
        # check if the ds as also the model outputs in it
        for check_subset in ["tcvis", "notcvis"]:
            check_layer_name = "probabilities-" + check_subset
            if check_layer_name in self.ds:
                fname_p = Path(filename)
                fname = fname_p.stem + "-" + check_subset + ".tif"
                self.export_geotiff(path, fname, check_layer_name, tags)

        return self.export_geotiff(path, filename, "probabilities", tags)

    def export_binarized(self, path: Path, filename="pred_binarized.tif", tags={}):
        """Export the binarized segmentation result of the inference result.

        If the inference result is an ensemble result and it contains also results of the models,
        also the binarized probabilities of the models will be written as individual files as well.

        Args:
            path (Path): The path where to export to.
            filename (str, optional): the filename. Defaults to "pred_binarized.tif".
            tags (dict, optional): optional GeoTIFF metadata to be written. Defaults to no additional metadata.

        Returns:
            the Path of the written file

        """
        # check if the ds as also the model outputs in it
        for check_subset in ["tcvis", "notcvis"]:
            check_layer_name = "binarized_segmentation-" + check_subset
            if check_layer_name in self.ds:
                fname_p = Path(filename)
                fname = fname_p.stem + "-" + check_subset + ".tif"
                self.export_geotiff(path, fname, check_layer_name, tags)

        return self.export_geotiff(path, filename, "binarized_segmentation", tags)

    def export_polygonized(self, path: Path, filename_prefix="pred_segments", minimum_mapping_unit=32):
        """Export the binarized probabilities as a vector dataset in GeoPackage and GeoParquet format.

        If the inference result is an ensemble result and it contains also results of the models,
        these datasets will also be polygonized. In that case a parquet file for each result (ensemble + models) as
        well as a GeoPackage file containing all polygonization results as individual layers will be written.

        Args:
            path (Path): The path where to export the files
            filename_prefix (str, optional): the file prefix of the exported files. Defaults to "pred_segments".
            minimum_mapping_unit (int, optional): segments covering less pixel are removed. Defaults to 32.

        """
        polygon_gdf = vectorization.vectorize(
            self.ds, "binarized_segmentation", minimum_mapping_unit=minimum_mapping_unit
        )

        path_gpkg = path / f"{filename_prefix}.gpkg"
        path_parquet = path / f"{filename_prefix}.parquet"

        polygon_gdf.to_file(path_gpkg, layer=filename_prefix)
        polygon_gdf.to_parquet(path_parquet)

        for subset_name in ["tcvis", "notcvis"]:
            layer_name = "binarized_segmentation-" + subset_name
            if layer_name in self.ds:
                polygon_gdf = vectorization.vectorize(self.ds, layer_name, minimum_mapping_unit=minimum_mapping_unit)
                polygon_gdf.to_file(path_gpkg, layer=f"{filename_prefix} ({subset_name.upper()})")
                polygon_gdf.to_parquet(path / f"{filename_prefix}-{subset_name}.parquet")
