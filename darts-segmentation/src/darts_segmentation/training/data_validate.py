"""CLI-ready function for validating a training dataset config based on a training dataset."""

import logging
from pathlib import Path
from typing import Literal


def validate_dataset(
    train_data_dir: str | Path,
    # data_split is for the test split
    data_split_method: Literal["random", "region", "sample"] | None = None,
    data_split_by: list[str | float] | None = None,
    # fold is for cross-validation split (train/val)
    fold_method: Literal["kfold", "shuffle", "stratified", "region", "region-stratified", "none"] = "kfold",
    total_folds: int = 5,
    subsample: int | None = None,
    bands: list[str] | None = None,
):
    """Validate a training dataset config based on a training dataset.

    Please see the DartsDataModule for more information.

    Args:
        train_data_dir (Path): The path to the data to be used for training.
            Expects a directory containing:
            1. a zarr group called "data.zarr" containing a "x" and "y" array
            2. a geoparquet file called "metadata.parquet" containing the metadata for the data.
                This metadata should contain at least the following columns:
                - "sample_id": The id of the sample
                - "region": The region the sample belongs to
                - "empty": Whether the image is empty
                The index should refer to the index of the sample in the zarr data.
            This directory should be created by a preprocessing script.
        data_split_method (Literal["random", "region", "sample"] | None, optional):
            The method to use for splitting the data into a train and a test set.
            "random" will split the data randomly, the seed is always 42 and the test size can be specified
            by providing a list with a single a float between 0 and 1 to data_split_by
            This will be the fraction of the data to be used for testing.
            E.g. [0.2] will use 20% of the data for testing.
            "region" will split the data by one or multiple regions,
            which can be specified by providing a str or list of str to data_split_by.
            "sample" will split the data by sample ids, which can also be specified similar to "region".
            If None, no split is done and the complete dataset is used for both training and testing.
            The train split will further be split in the cross validation process.
            Defaults to None.
        data_split_by (list[str | float] | None, optional): Select by which regions/samples to split or
            the size of test set. Defaults to None.
        fold_method (Literal["kfold", "shuffle", "stratified", "region", "region-stratified", "none"], optional):
            Method for cross-validation split. Defaults to "kfold".
        total_folds (int, optional): Total number of folds in cross-validation. Defaults to 5.
        subsample (int | None, optional): If set, will subsample the dataset to this number of samples.
            This is useful for debugging and testing. Defaults to None.
        bands (Bands | list[str] | None, optional): List of bands to use.
            Expects the data_dir to contain a config.toml with a "darts.bands" key,
            with which the indices of the bands will be mapped to.
            Defaults to None.

    """
    import geopandas as gpd
    import toml

    from darts_segmentation.training.data import _get_fold, _log_stats, _split_metadata

    data_dir = Path(train_data_dir)

    config_file = data_dir / "config.toml"
    assert config_file.exists(), f"Config file {config_file} not found!"
    config = toml.load(config_file)
    assert "darts" in config and "bands" in config["darts"], f"Config file {config_file} is missing 'darts.bands' key!"

    if bands:
        # Check if bands are in config
        data_bands = config["darts"]["bands"]
        for b in bands:
            assert b in data_bands, f"Band {b} not found in config file {config_file}!"

    metadata_file = data_dir / "metadata.parquet"
    assert metadata_file.exists(), f"Metadata file {metadata_file} not found!"
    metadata = gpd.read_parquet(data_dir / "metadata.parquet")

    if subsample is not None:
        metadata = metadata.sample(n=subsample, random_state=42)
    train_metadata, test_metadata = _split_metadata(metadata, data_split_method, data_split_by)

    _log_stats(train_metadata, "Non-Test (train-split)", level=logging.INFO)
    _log_stats(test_metadata, "Test (test-split)", level=logging.INFO)

    for fold in range(total_folds):
        train_index, val_index = _get_fold(train_metadata, fold_method, total_folds, fold)
        _log_stats(metadata.loc[train_index], f"Training fold {fold} (train-fold)", level=logging.INFO)
        _log_stats(metadata.loc[val_index], f"Validation fold {fold} (val-fold)", level=logging.INFO)
