import pandas as pd

from darts_segmentation.training.tune import _split_metadata

metadata = pd.DataFrame(
    [
        {"region": "North", "sample_id": "1", "empty": True},
        {"region": "South", "sample_id": "2", "empty": False},
        {"region": "South", "sample_id": "3", "empty": False},
        {"region": "West", "sample_id": "4", "empty": True},
        {"region": "North", "sample_id": "5", "empty": False},
        {"region": "East", "sample_id": "6", "empty": True},
        {"region": "North", "sample_id": "7", "empty": True},
        {"region": "West", "sample_id": "8", "empty": False},
        {"region": "West", "sample_id": "9", "empty": False},
        {"region": "South", "sample_id": "10", "empty": True},
    ]
)


def test_datasplit_none():
    train_metadata, test_metadata = _split_metadata(metadata)
    assert len(train_metadata) == 10
    assert test_metadata is None


def test_datasplit_random():
    train_metadata, test_metadata = _split_metadata(metadata, data_split_method="random")
    assert len(train_metadata) == 8
    assert len(test_metadata) == 2

    train_metadata, test_metadata = _split_metadata(metadata, data_split_method="random", data_split_by=0.6)
    assert len(train_metadata) == 6
    assert len(test_metadata) == 4


def test_datasplit_region():
    train_metadata, test_metadata = _split_metadata(metadata, data_split_method="region", data_split_by="West")
    assert len(train_metadata) == 7
    assert len(test_metadata) == 3

    assert (train_metadata["region"].isin(["North", "South", "East"])).all()
    assert (test_metadata["region"] == "West").all()

    train_metadata, test_metadata = _split_metadata(metadata, data_split_method="region", data_split_by="East")
    assert len(train_metadata) == 9
    assert len(test_metadata) == 1

    assert (train_metadata["region"].isin(["North", "South", "West"])).all()
    assert (test_metadata["region"] == "East").all()

    train_metadata, test_metadata = _split_metadata(
        metadata, data_split_method="region", data_split_by=["West", "East"]
    )
    assert len(train_metadata) == 6
    assert len(test_metadata) == 4

    assert (train_metadata["region"].isin(["North", "South"])).all()
    assert (test_metadata["region"].isin(["East", "West"])).all()


def test_datasplit_sample():
    train_metadata, test_metadata = _split_metadata(
        metadata,
        data_split_method="sample",
        data_split_by=["2", "5", "8", "10"],
    )
    assert len(train_metadata) == 6
    assert len(test_metadata) == 4
