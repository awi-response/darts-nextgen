from pathlib import Path  # noqa: I001

import rasterio
from darts_export import inference
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
