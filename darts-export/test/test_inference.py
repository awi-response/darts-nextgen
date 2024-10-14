from pathlib import Path  # noqa: I001

import geopandas as gpd
import rasterio
from darts_export import inference
from osgeo import ogr
from xarray import Dataset


def test_writeProbabilities(probabilities: Dataset, tmp_path: Path):
    ds = inference.InferenceResultWriter(probabilities)

    ds.export_probabilities(tmp_path)

    assert (tmp_path / "pred_probabilities.tif").is_file()

    rio_ds = rasterio.open(tmp_path / "pred_probabilities.tif")
    assert rio_ds.crs.to_epsg() == 32601
    assert rio_ds.width == 16
    assert rio_ds.height == 16
    assert rio_ds.dtypes[0] == "int8"


def test_writeBinarization(probabilities: Dataset, tmp_path: Path):
    ds = inference.InferenceResultWriter(probabilities)
    ds.export_binarized(tmp_path)

    rio_ds = rasterio.open(tmp_path / "pred_binarized.tif")
    assert rio_ds.crs.to_epsg() == 32601
    assert rio_ds.width == 16
    assert rio_ds.height == 16
    assert rio_ds.dtypes[0] == "uint8"


def test_writeVectors(probabilities: Dataset, tmp_path: Path):
    gdal_has_parquet = ogr.GetDriverByName("Parquet") is not None

    ds = inference.InferenceResultWriter(probabilities)
    ds.export_polygonized(tmp_path)

    assert (tmp_path / "pred_segments.gpkg").is_file()
    if gdal_has_parquet:
        assert (tmp_path / "pred_segments.parquet").is_file()

    gpkg_layers = gpd.list_layers(tmp_path / "pred_segments.gpkg")
    assert len(gpkg_layers) == 1
    assert gpkg_layers.iloc[0]["name"] == "pred_segments"

    gdf = gpd.read_file(tmp_path / "pred_segments.gpkg")
    assert gdf.shape == (1, 2)  # one feature, two attributes (fid, DN)
    assert gdf.iloc[0].geometry.wkt == (
        "POLYGON (("
        "500007.5 4500003.5, 500007.5 4500004.5, 500006.5 4500004.5, 500006.5 4500008.5, 500007.5 4500008.5, "
        "500007.5 4500009.5, 500009.5 4500009.5, 500009.5 4500008.5, 500010.5 4500008.5, 500010.5 4500004.5, "
        "500009.5 4500004.5, 500009.5 4500003.5, 500007.5 4500003.5))"
    )
