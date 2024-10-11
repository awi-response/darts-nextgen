from pathlib import Path
from darts_export import inference

def test_writeProbabilities(probabilities, tmp_path:Path):
    ds = inference.InferenceResultWriter(probabilities)

    ds.export_probabilities(tmp_path)

    assert (tmp_path / "pred_probabilities.tif").is_file()
