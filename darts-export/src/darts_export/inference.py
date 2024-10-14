"""Darts export module for inference results."""

import logging
from pathlib import Path

import xarray
from osgeo import gdal, gdal_array, ogr

gdal.UseExceptions()
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

    def export_polygonized(self, path: Path, filename_prefix="pred_segments"):
        """Export the binarized probabilities as a vector dataset in GeoPackage format.

        If the gdal installation supports GeoParquet files, an additional file will be written in this format.

        Args:
            path (Path): The path where to export the files
            filename_prefix (str, optional): the file prefix of the exported files. Defaults to "pred_segments".

        """
        # extract the layer with the binarized probabilities and create an in-memory
        # gdal dataset from it:
        layer = self.ds.binarized_segmentation
        dta = gdal_array.OpenArray(layer.values)

        # copy over the geodata
        # the transform object of rasterio has to be converted into a tuple
        affine_transform = layer.rio.transform()
        geotransform = (
            affine_transform.c,
            affine_transform.a,
            affine_transform.b,
            affine_transform.f,
            affine_transform.d,
            affine_transform.e,
        )
        dta.SetGeoTransform(geotransform)
        dta.SetProjection(layer.rio.crs.to_wkt())

        # create the vector output datasets
        gpkg_drv = ogr.GetDriverByName("GPKG")
        gpkg_ds = gpkg_drv.CreateDataSource(path / f"{filename_prefix}.gpkg")
        output_layer = gpkg_ds.CreateLayer("filename_prefix", geom_type=ogr.wkbPolygon, srs=dta.GetSpatialRef())

        # add the field where to store the polygonization threshold
        field = ogr.FieldDefn("DN", ogr.OFTInteger)
        output_layer.CreateField(field)

        # do the polygonization
        gdal.Polygonize(
            dta.GetRasterBand(1),
            None,  # no masking, polygonize everything
            output_layer,  # where to write the vector data to
            0,  # write the polygonization threshold in the first attribute ("DN")
        )

        # remove all features where DN is not 1
        output_layer.SetAttributeFilter("DN != 1")
        for feature in output_layer:
            feature_id = feature.GetFID()
            output_layer.DeleteFeature(feature_id)

        output_layer.SetAttributeFilter(None)

        parquet_drv = ogr.GetDriverByName("Parquet")
        if parquet_drv is not None:
            parquet_ds = parquet_drv.CreateDataSource(path / f"{filename_prefix}.parquet")
            parquet_ds.CopyLayer(output_layer, filename_prefix)
        else:
            L.warning(
                "export of polygonized inference data in geoparquet failed,"
                " because the GDAL installation does not support it."
            )

        gpkg_ds = None
        output_layer = None
