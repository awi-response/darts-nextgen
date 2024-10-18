from pathlib import Path  # noqa: I001

import geopandas as gpd
import rasterio
from darts_export import inference
from xarray import Dataset

POLYGONOUTPUT_EXPECTED_COLUMNS = {"geometry", "Region_ID", "min", "max", "mean", "median", "std", "npixel"}


def test_writeProbabilities(probabilities_1: Dataset, tmp_path: Path):
    ds = inference.InferenceResultWriter(probabilities_1)

    ds.export_probabilities(tmp_path)

    assert (tmp_path / "pred_probabilities.tif").is_file()

    rio_ds = rasterio.open(tmp_path / "pred_probabilities.tif")
    assert rio_ds.crs.to_epsg() == 32601
    assert rio_ds.width == 16
    assert rio_ds.height == 16
    assert rio_ds.dtypes[0] == "int8"


def test_writeBinarization(probabilities_1: Dataset, tmp_path: Path):
    ds = inference.InferenceResultWriter(probabilities_1)
    ds.export_binarized(tmp_path)

    rio_ds = rasterio.open(tmp_path / "pred_binarized.tif")
    assert rio_ds.crs.to_epsg() == 32601
    assert rio_ds.width == 16
    assert rio_ds.height == 16
    assert rio_ds.dtypes[0] == "uint8"


def test_writeVectorsSimple(probabilities_1: Dataset, tmp_path: Path):
    ds = inference.InferenceResultWriter(probabilities_1)
    ds.export_polygonized(tmp_path, minimum_mapping_unit=0)

    assert (tmp_path / "pred_segments.gpkg").is_file()
    assert (tmp_path / "pred_segments.parquet").is_file()

    gpkg_layers = gpd.list_layers(tmp_path / "pred_segments.gpkg")
    assert len(gpkg_layers) == 1
    assert gpkg_layers.iloc[0]["name"] == "pred_segments"

    expected_polygon = (
        "POLYGON (("
        "500007.5 4500003.5, 500007.5 4500004.5, 500006.5 4500004.5, 500006.5 4500008.5, 500007.5 4500008.5, "
        "500007.5 4500009.5, 500009.5 4500009.5, 500009.5 4500008.5, 500010.5 4500008.5, 500010.5 4500004.5, "
        "500009.5 4500004.5, 500009.5 4500003.5, 500007.5 4500003.5))"
    )

    for suffix in ["gpkg", "parquet"]:
        if suffix == "parquet":
            gdf = gpd.read_parquet(tmp_path / f"pred_segments.{suffix}")
        else:
            gdf = gpd.read_file(tmp_path / f"pred_segments.{suffix}")
        assert gdf.shape == (1, len(POLYGONOUTPUT_EXPECTED_COLUMNS))
        assert gdf.iloc[0].geometry.wkt == expected_polygon
        assert set(gdf.columns) == POLYGONOUTPUT_EXPECTED_COLUMNS


def test_writeVectorsComplex(probabilities_2: Dataset, tmp_path: Path):
    ds = inference.InferenceResultWriter(probabilities_2)
    ds.export_polygonized(tmp_path)

    assert (tmp_path / "pred_segments.gpkg").is_file()
    assert (tmp_path / "pred_segments.parquet").is_file()

    gpkg_layers = gpd.list_layers(tmp_path / "pred_segments.gpkg")
    assert len(gpkg_layers) == 1
    assert gpkg_layers.iloc[0]["name"] == "pred_segments"

    for suffix in ["gpkg", "parquet"]:
        if suffix == "parquet":
            gdf = gpd.read_parquet(tmp_path / f"pred_segments.{suffix}")
        else:
            gdf = gpd.read_file(tmp_path / f"pred_segments.{suffix}")
        assert gdf.shape == (4, len(POLYGONOUTPUT_EXPECTED_COLUMNS))
