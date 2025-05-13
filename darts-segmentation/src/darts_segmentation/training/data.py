# ruff: noqa: D101
# ruff: noqa: D102
# ruff: noqa: D105
# ruff: noqa: D107
"""Training script for DARTS segmentation."""

import logging
from pathlib import Path
from typing import Literal

import geopandas as gpd
import lightning as L  # noqa: N812
import toml
import zarr
from sklearn.model_selection import (
    GroupShuffleSplit,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
)
from torch.utils.data import DataLoader, Dataset
from zarr.storage import LocalStore

from darts_segmentation.training.augmentations import Augmentation, get_augmentation
from darts_segmentation.utils import Bands

logger = logging.getLogger(__name__.replace("darts_", "darts."))


class DartsDatasetZarr(Dataset):
    def __init__(
        self,
        data_dir: Path | str,
        augment: list[Augmentation] | None = None,
        indices: list[int] | None = None,
        bands: list[int] | None = None,
    ):
        data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir

        store = zarr.storage.LocalStore(data_dir)
        self.zroot = zarr.group(store=store)

        assert "x" in self.zroot and "y" in self.zroot, (
            f"Dataset corrupted! {self.zroot.info=} must contain 'x' or 'y' arrays!"
        )

        self.indices = indices if indices is not None else list(range(self.zroot["x"].shape[0]))
        self.bands = bands

        self.transform = get_augmentation(augment)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]

        x = self.zroot["x"][i, self.bands] if self.bands else self.zroot["x"][i]
        y = self.zroot["y"][i]

        # Apply augmentations
        if self.transform is not None:
            augmented = self.transform(image=x.transpose(1, 2, 0), mask=y)
            x = augmented["image"].transpose(2, 0, 1)
            y = augmented["mask"]

        return x, y


class DartsDatasetInMemory(Dataset):
    def __init__(
        self,
        data_dir: Path | str,
        augment: list[Augmentation] | None = None,
        indices: list[int] | None = None,
        bands: list[int] | None = None,
    ):
        data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir

        store = zarr.storage.LocalStore(data_dir)
        self.zroot = zarr.group(store=store)

        assert "x" in self.zroot and "y" in self.zroot, (
            f"Dataset corrupted! {self.zroot.info=} must contain 'x' or 'y' arrays!"
        )

        self.x = []
        self.y = []
        indices = indices or list(range(self.zroot["x"].shape[0]))
        for i in indices:
            x = self.zroot["x"][i, bands] if bands else self.zroot["x"][i]
            y = self.zroot["y"][i]
            self.x.append(x)
            self.y.append(y)

        self.transform = get_augmentation(augment)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]

        # Apply augmentations
        if self.transform is not None:
            augmented = self.transform(image=x.transpose(1, 2, 0), mask=y)
            x = augmented["image"].transpose(2, 0, 1)
            y = augmented["mask"]

        return x, y


def _split_metadata(
    metadata: gpd.GeoDataFrame,
    data_split_method: Literal["random", "region", "sample", "none"] | None,
    data_split_by: list[str | float] | None,
):
    # Match statement doesn't like None
    data_split_method = data_split_method or "none"

    match data_split_method:
        case "none":
            return metadata, metadata
        case "random":
            assert isinstance(data_split_by, list) and len(data_split_by) == 1
            data_split_by = data_split_by[0]
            assert isinstance(data_split_by, float)
            for seed in range(100):
                train_metadata = metadata.sample(frac=data_split_by, random_state=seed)
                test_metadata = metadata.drop(train_metadata.index)
                if (~test_metadata["empty"]).sum() == 0:
                    logger.warning("Test set is empty, retrying with another random seed...")
                    continue
                return train_metadata, test_metadata
            else:
                raise ValueError("Could not split data randomly, please check your data.")
        case "region":
            assert isinstance(data_split_by, list) and len(data_split_by) > 0
            train_metadata = metadata[~metadata["region"].isin(data_split_by)]
            test_metadata = metadata[metadata["region"].isin(data_split_by)]
            return train_metadata, test_metadata
        case "sample":
            assert isinstance(data_split_by, list) and len(data_split_by) > 0
            train_metadata = metadata[~metadata["sample_id"].isin(data_split_by)]
            test_metadata = metadata[metadata["sample_id"].isin(data_split_by)]
            return train_metadata, test_metadata
        case _:
            raise ValueError(f"Invalid data split method: {data_split_method}")


def _get_fold(
    metadata: gpd.GeoDataFrame,
    fold_method: Literal["kfold", "shuffle", "stratified", "region", "region-stratified", "none"] | None,
    n_folds: int,
    fold: int,
) -> tuple[list[int], list[int]]:
    fold = fold if fold_method is not None else 0
    fold_method = fold_method or "none"
    match fold_method:
        case "none":
            foldgen = [(metadata.index.tolist(), metadata.index.tolist())]
        case "kfold":
            foldgen = KFold(n_folds).split(metadata)
        case "shuffle":
            foldgen = StratifiedShuffleSplit(n_splits=n_folds, random_state=42).split(metadata, ~metadata["empty"])
        case "stratified":
            foldgen = StratifiedKFold(n_folds, random_state=42, shuffle=True).split(metadata, ~metadata["empty"])
        case "region":
            foldgen = GroupShuffleSplit(n_folds).split(metadata, groups=metadata["region"])
        case "region-stratified":
            foldgen = StratifiedGroupKFold(n_folds, random_state=42, shuffle=True).split(
                metadata, ~metadata["empty"], groups=metadata["region"]
            )
        case _:
            raise ValueError(f"Unknown fold method: {fold_method}")

    for i, (train_index, val_index) in enumerate(foldgen):
        if i != fold:
            continue
        # Turn index into metadata index
        train_index = metadata.index[train_index].tolist()
        val_index = metadata.index[val_index].tolist()
        return train_index, val_index

    raise ValueError(f"Fold {fold} not found")


def _log_stats(metadata: gpd.GeoDataFrame, mode: str):
    n_pos = (~metadata["empty"]).sum()
    n_neg = metadata["empty"].sum()
    logger.debug(
        f"{mode} dataset: {n_pos} positive, {n_neg} negative ({len(metadata)} total)"
        f" with {metadata['region'].nunique()} unique regions and {metadata['sample_id'].nunique()} unique sample ids"
    )


class DartsDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        batch_size: int,
        # data_split is for the test split
        data_split_method: Literal["random", "region", "sample"] | None = None,
        data_split_by: list[str | float] | None = None,
        # fold is for cross-validation split (train/val)
        fold_method: Literal["kfold", "shuffle", "stratified", "region", "region-stratified"] | None = "kfold",
        total_folds: int = 5,
        fold: int = 0,
        bands: Bands | list[str] | None = None,
        augment: list[Augmentation] | None = None,  # Not used for val or test
        num_workers: int = 0,
        in_memory: bool = False,
    ):
        """Initialize the data module.

        Supports spliting the data into train and test set while also defining cv-folds.
        Folding only applies to the non-test set and splits this into a train and validation set.

        Example:
            1. Normal train-validate. (Can also be used for testing on the complete dataset)
            ```py
            dm = DartsDataModule(data_dir, batch_size)
            ```

            2. Specifying a test split by random (20% of the data will be used for testing)
            ```py
            dm = DartsDataModule(data_dir, batch_size, data_split_method="random")
            ```

            3. Specific fold for cross-validation (On the complete dataset, because data_split_method is "none").
            This will be take the third of a total of7 folds to determine the validation set.
            ```py
            dm = DartsDataModule(data_dir, batch_size, fold_method="region-stratified", fold=2, total_folds=7)
            ```

            In general this should be used in combination with a cross-validation loop.
            ```py
            for fold in range(total_folds):
                dm = DartsDataModule(
                    data_dir,
                    batch_size,
                    fold_method="region-stratified",
                    fold=fold,
                    total_folds=total_folds)
                ...
            ```

            4. Don't split anything -> only train
            ```py
            dm = DartsDataModule(data_dir, batch_size, fold_method=None)
            ```

        Args:
            data_dir (Path): The path to the data to be used for training.
                Expects a directory containing:
                1. a zarr group called "data.zarr" containing a "x" and "y" array
                2. a geoparquet file called "metadata.parquet" containing the metadata for the data.
                    This metadata should contain at least the following columns:
                    - "sample_id": The id of the sample
                    - "region": The region the sample belongs to
                    - "empty": Whether the image is empty
                    The index should refer to the index of the sample in the zarr data.
                This directory should be created by a preprocessing script.
            batch_size (int): Batch size for training and validation.
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
            fold_method (Literal["kfold", "shuffle", "stratified", "region", "region-stratified"] | None, optional):
                Method for cross-validation split. Defaults to "kfold".
            total_folds (int, optional): Total number of folds in cross-validation. Defaults to 5.
            fold (int, optional): Index of the current fold. Defaults to 0.
            bands (Bands | list[str] | None, optional): List of bands to use.
                Expects the data_dir to contain a config.toml with a "darts.bands" key,
                with which the indices of the bands will be mapped to.
                Defaults to None.
            augment (bool, optional): Whether to augment the data. Does nothing for testing. Defaults to True.
            num_workers (int, optional): Number of workers for data loading. See torch.utils.data.DataLoader.
                Defaults to 0.
            in_memory (bool, optional): Whether to load the data into memory. Defaults to False.

        """
        super().__init__()
        self.save_hyperparameters(ignore=["num_workers", "in_memory"])
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.fold = fold
        self.data_split_method = data_split_method
        self.data_split_by = data_split_by
        self.fold_method = fold_method
        self.total_folds = total_folds

        self.augment = augment
        self.num_workers = num_workers
        self.in_memory = in_memory

        data_dir = Path(data_dir)

        metadata_file = data_dir / "metadata.parquet"
        assert metadata_file.exists(), f"Metadata file {metadata_file} not found!"

        config_file = data_dir / "config.toml"
        assert config_file.exists(), f"Config file {config_file} not found!"
        data_bands = toml.load(config_file)["darts"]["bands"]
        bands = bands.names if isinstance(bands, Bands) else bands
        self.bands = [data_bands.index(b) for b in bands] if bands else None

        zdir = data_dir / "data.zarr"
        assert zdir.exists(), f"Data directory {zdir} not found!"
        zroot = zarr.group(store=LocalStore(data_dir / "data.zarr"))
        self.nsamples = zroot["x"].shape[0]

    def setup(self, stage: Literal["fit", "validate", "test", "predict"] | None = None):
        if stage == "predict" or stage is None:
            return

        metadata = gpd.read_parquet(self.data_dir / "metadata.parquet")
        train_metadata, test_metadata = _split_metadata(metadata, self.data_split_method, self.data_split_by)

        _log_stats(train_metadata, "train-split")
        _log_stats(test_metadata, "test-split")

        # Log stats about the data

        if stage in ["fit", "validate"]:
            train_index, val_index = _get_fold(train_metadata, self.fold_method, self.total_folds, self.fold)
            _log_stats(metadata.loc[train_index], "train-fold")
            _log_stats(metadata.loc[val_index], "val-fold")

            dsclass = DartsDatasetInMemory if self.in_memory else DartsDatasetZarr
            self.train = dsclass(self.data_dir / "data.zarr", self.augment, train_index, self.bands)
            self.val = dsclass(self.data_dir / "data.zarr", None, val_index, self.bands)
        if stage == "test":
            test_index = test_metadata.index.tolist()
            dsclass = DartsDatasetInMemory if self.in_memory else DartsDatasetZarr
            self.test = dsclass(self.data_dir / "data.zarr", None, test_index, self.bands)

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
