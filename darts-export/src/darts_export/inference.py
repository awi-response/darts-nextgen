"""Darts export module for inference results."""

from pathlib import Path

import xarray


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
            path (Path): _description_
            filename (str, optional): _description_. Defaults to "pred_binarized.tif".
            tags (dict, optional): _description_. Defaults to {}.

        """
        self.ds.binarized_segmentation.rio.to_raster(path / filename, driver="GTiff", tags=tags, compress="LZW")

    # def export_vectors(self, path: Path, filename_prefix="pred_segments"):
    #    pass
