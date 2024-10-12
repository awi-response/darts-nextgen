from pathlib import Path

from xarray import Dataset

from darts_export import inference


def test_writeProbabilities(probabilities: Dataset, tmp_path: Path):
    ds = inference.InferenceResultWriter(probabilities)

    ds.export_probabilities(tmp_path)

    assert (tmp_path / "pred_probabilities.tif").is_file()
