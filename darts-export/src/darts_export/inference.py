from pathlib import Path
import xarray
import rioxarray

class InferenceResultWriter:

    def __init__(self, ds) -> None:
        self.ds:xarray.Dataset = ds

    def export_probabilities(self, path:Path, filename="pred_probabilities.tif", tags=dict()):

        # write the probability layer from the raster to a GeoTiff
        self.ds.probabilities.rio.to_raster(path / filename, driver="GTiff", tags=tags, compress="LZW")

    def export_binarized(self, path:Path, filename="pred_binarized.tif", tags=dict()):

        self.ds.binarized_segmentation.rio.to_raster(path / filename, driver="GTiff", tags=tags, compress="LZW")


    def export_vectors(self, path:Path, filename_prefix="pred_segments"):
        pass

