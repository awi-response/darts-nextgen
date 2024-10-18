"""Collection of conversion functions to translate data for export and processing."""

import geopandas as gpd
import numpy as np
import xarray
from osgeo import gdal, gdal_array, ogr

gdal.UseExceptions()


def ogrlyr_to_geopandas(ogr_layer: ogr.Layer) -> gpd.GeoDataFrame:
    """Convert a GDAL/OGR layer object to a geopandas dataframe.

    Args:
        ogr_layer (ogr.Layer): the ogr layer object to convert

    Returns:
        gpd.GeoDataFrame: the resulting GeoDataFrame

    """
    # Initialize an empty list to store geometries and attributes
    features = []

    # Iterate over OGR features in the layer
    for feature in ogr_layer:
        geom = feature.GetGeometryRef()
        # geom_json = geom.ExportToJson()  # Convert to GeoJSON format
        geom_wkt = geom.ExportToWkt()  # Convert to Well-Known Text (WKT)
        attributes = feature.items()  # Get attribute data as a dictionary

        # Append a tuple (geometry, attributes)
        features.append((geom_wkt, attributes))

    # Create a GeoDataFrame from the geometries and attributes
    gdf = gpd.GeoDataFrame(
        [attr for geom, attr in features], geometry=gpd.GeoSeries.from_wkt([geom for geom, attr in features])
    )

    return gdf


def numpy_to_gdal(nparray: np.ndarray, rio_georef: xarray.DataArray | xarray.Dataset) -> gdal.Dataset:
    """Convert a numpy ndarray into a gdal Dataset.

    Georeference is to be passed in terms
    of an xarray object augmented by the rioxarray module, meaning the '.rio' accessor is
    available.

    Args:
        nparray (np.ndarray): The data to convert
        rio_georef (xarray.DataArray | xarray.Dataset): an xarray with rio accessor as georeference

    Returns:
        gdal.Dataset: _description_

    """
    # convert the xarray to a gdal dataset
    dta = gdal_array.OpenArray(nparray)

    # copy over the geodata
    # the transform object of rasterio has to be converted into a tuple
    affine_transform = rio_georef.rio.transform()
    geotransform = (
        affine_transform.c,
        affine_transform.a,
        affine_transform.b,
        affine_transform.f,
        affine_transform.d,
        affine_transform.e,
    )
    dta.SetGeoTransform(geotransform)
    dta.SetProjection(rio_georef.rio.crs.to_wkt())
    return dta


def rioxarrayds_to_gdal(rix: xarray.DataArray) -> gdal.Dataset:
    """Convert a rioxarray object to a gdal dataset.

    Args:
        rix (xarray.DataArray): data to convert

    Returns:
        gdal.Dataset: the converted data

    """
    return numpy_to_gdal(rix.to_numpy(), rix)
