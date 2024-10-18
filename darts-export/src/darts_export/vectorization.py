"""Module for various tasks during export."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely
import xarray
from osgeo import gdal, ogr
from rasterio.features import shapes, sieve
from skimage import measure

from darts_export import conversion

gdal.UseExceptions()


def gdal_polygonization(
    labels: np.ndarray, rio_georef: xarray.Dataset, as_gdf=True, gpkg_path: Path | None = None
) -> ogr.Layer | gpd.GeoDataFrame:
    """Polygonize a numpy array using GDAL.

    Detects regions with the same value in the numpy array and converts those
    into polygons. Can return the initial gdal result as ogr.Dataset or converts into a geopandas dataframe if
    as_gdf is enabled. If `gpkg_path` is set, the polsgonization result will be written one go
    into a GeoPackage file, otherwise the OGR dataset will reside purely in memory.

    Args:
        labels (np.ndarray): The input dataset as a ndarray with values designating labels
        rio_georef (xarray.Dataset): an xarray with the rioxarray accessor to determine the CRS of the dataset
        as_gdf (bool, optional): returns result as geopandas.GeoDataFrame. Defaults to True.
        gpkg_path (Path | None, optional): Path where a GPKG file is written backing the OGR dataset. Defaults to None.

    Returns:
        ogr.Layer | gpd.GeoDataFrame: the polyginization result

    """
    # convert to a GDAL dataset
    dta = conversion.numpy_to_gdal(labels, rio_georef)

    # prepare the vector output datasets to write to
    if gpkg_path is not None:
        gpkg_path = Path(gpkg_path)
        ds = ogr.GetDriverByName("GPKG").CreateDataSource(gpkg_path)
        ogr_layer = ds.CreateLayer(gpkg_path.stem, geom_type=ogr.wkbPolygon, srs=dta.GetSpatialRef())
    else:
        # work only in memory
        ds = ogr.GetDriverByName("Memory").CreateDataSource("gdal_polygonization")
        ogr_layer = ds.CreateLayer("gdal_polygonization", geom_type=ogr.wkbPolygon, srs=dta.GetSpatialRef())

    # add the field where to store region ID
    field = ogr.FieldDefn("Region_ID", ogr.OFTInteger)
    ogr_layer.CreateField(field)

    # do the polygonization
    gdal.Polygonize(
        dta.GetRasterBand(1),
        None,  # no masking, polygonize everything
        ogr_layer,  # where to write the vector data to
        0,  # write the polygonization threshold in the first attribute ("DN")
    )
    # the region with the ID zero is the region unlabelled by measure label
    # remove features polygonized from that region, that is all features where DN is not 1
    ogr_layer.SetAttributeFilter("Region_ID = 0")
    for feature in ogr_layer:
        feature_id = feature.GetFID()
        ogr_layer.DeleteFeature(feature_id)
    ogr_layer.SetAttributeFilter(None)

    if not as_gdf:
        return ogr_layer

    # convert the gdal vector object zo a geopandas gdf
    gdf_polygons = conversion.ogrlyr_to_geopandas(ogr_layer)
    gdf_polygons.set_crs(rio_georef.rio.crs, inplace=True)
    return gdf_polygons


def rasterio_polygonization(labels: np.ndarray, rio_georef: xarray.Dataset) -> gpd.GeoDataFrame:
    """Polygonize a numpy array with rasterio.

    Detects regions with the same value in the numpy array and converts those
    into polygons. The `rio_georef` agrument determines the final CRS of
    the returned geopandas GeoDataFrame.

    Args:
        labels (np.ndarray): the array of regionalizable labels
        rio_georef (xarray.Dataset): the CRS as an xarray/rioxarray Dataset with rio accessor

    Returns:
        geopandas.GeoDataFrame: the resolut of the polygonization

    """
    # shapes() needs int32 data, while scikit labels puts out int64
    # cast with astype()
    gdf = (
        gpd.GeoDataFrame(
            [
                (shapely.geometry.shape(geom), int(region_Id))
                for geom, region_Id in shapes(labels.astype(np.int32), transform=rio_georef.rio.transform())
            ],
            columns=["geometry", "Region_ID"],
        )
        .set_crs(rio_georef.rio.crs)
        .query("Region_ID > 0")
    )
    return gdf


def vectorize(xdat: xarray.Dataset, polygonization_func: str = "rasterio", minimum_mapping_unit=32) -> gpd.GeoDataFrame:
    """Vectorize an inference result dataset.

    Detects connected regions in the with the same value `binarized_segmentation` layer, polygonizes
    this into a vector dataset.
    Additionally this function writes zonal statistics of the `probabilities` layer to the polygon attributes.

    Args:
        xdat (xarray.Dataset): the input dataset augmented with the rioxarray `rio` accessor
        polygonization_func (str): the method to utilize for polygonization, either 'gdal' or 'rasterio', the default.
        minimum_mapping_unit (int, optional): polygons smaller than this number are removed. Defaults to 32.

    Returns:
        _type_: _description_

    """
    layer = xdat.binarized_segmentation

    # MIN POLYGON for sieving
    if minimum_mapping_unit > 0:
        sieved = sieve(layer.to_numpy(), minimum_mapping_unit)
        bin_labelled = measure.label(sieved)
    else:
        bin_labelled = measure.label(layer)

    if polygonization_func.lower() == "gdal":
        gdf_polygons = gdal_polygonization(bin_labelled, layer)
    else:
        gdf_polygons = rasterio_polygonization(bin_labelled, layer)

    # execute the zonal stats:
    # arguments must be in the specified order, matching regionprops
    def median_intensity(region, intensities):
        # note the ddof arg to get the sample var if you so desire!
        return np.median(intensities[region])

    region_stats = measure.regionprops(
        bin_labelled, intensity_image=xdat.probabilities.values, extra_properties=[median_intensity]
    )

    # collect stats data:
    stats_dict = {}
    for region in region_stats:
        stats_dict[region.label] = {
            "min": int(region.min_intensity),
            "max": int(region.max_intensity),
            "mean": region.mean_intensity,
            "median": region.median_intensity,
            "std": region.intensity_std,
            "npixel": region.num_pixels,
        }

    # add the zonal stats to the GeoPandas DataFrame
    stats_df = gpd.pd.DataFrame.from_dict(stats_dict, orient="index")
    return gdf_polygons.merge(stats_df, left_on="Region_ID", right_index=True)
