# ruff: noqa: D101
# ruff: noqa: D102
# ruff: noqa: D105
# ruff: noqa: D107
"""Training script for DARTS segmentation."""

import logging
from pathlib import Path
from typing import Literal

import albumentations as A  # noqa: N812
import geopandas as gpd
import lightning as L  # noqa: N812
import torch
import zarr
from sklearn.model_selection import (
    GroupShuffleSplit,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
)
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__.replace("darts_", "darts."))


class DartsDataset(Dataset):
    def __init__(self, data_dir: Path | str, augment: bool, indices: list[int] | None = None):
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)

        self.x_files = sorted((data_dir / "x").glob("*.pt"))
        self.y_files = sorted((data_dir / "y").glob("*.pt"))
        assert len(self.x_files) == len(self.y_files), (
            f"Dataset corrupted! Got {len(self.x_files)=} and {len(self.y_files)=}!"
        )
        if indices is not None:
            self.x_files = [self.x_files[i] for i in indices]
            self.y_files = [self.y_files[i] for i in indices]

        self.transform = (
            A.Compose(
                [
                    A.HorizontalFlip(),
                    A.VerticalFlip(),
                    A.RandomRotate90(),
                    # A.Blur(),
                    A.RandomBrightnessContrast(),
                    A.MultiplicativeNoise(per_channel=True, elementwise=True),
                    # ToTensorV2(),
                ]
            )
            if augment
            else None
        )

    def __len__(self):
        return len(self.x_files)

    def __getitem__(self, idx):
        xfile = self.x_files[idx]
        yfile = self.y_files[idx]
        assert xfile.stem == yfile.stem, f"Dataset corrupted! Files must have the same name, but got {xfile=} {yfile=}!"

        x = torch.load(xfile).numpy()
        y = torch.load(yfile).int().numpy()

        # Apply augmentations
        if self.transform is not None:
            augmented = self.transform(image=x.transpose(1, 2, 0), mask=y)
            x = augmented["image"].transpose(2, 0, 1)
            y = augmented["mask"]

        return x, y


class DartsDatasetZarr(Dataset):
    def __init__(self, data_dir: Path | str, augment: bool, indices: list[int] | None = None):
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)

        store = zarr.storage.LocalStore(data_dir)
        self.zroot = zarr.group(store=store)

        assert "x" in self.zroot and "y" in self.zroot, (
            f"Dataset corrupted! {self.zroot.info=} must contain 'x' or 'y' arrays!"
        )

        self.indices = indices if indices is not None else list(range(self.zroot["x"].shape[0]))

        self.transform = (
            A.Compose(
                [
                    A.HorizontalFlip(),
                    A.VerticalFlip(),
                    A.RandomRotate90(),
                    # A.Blur(),
                    A.RandomBrightnessContrast(),
                    A.MultiplicativeNoise(per_channel=True, elementwise=True),
                    # ToTensorV2(),
                ]
            )
            if augment
            else None
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]

        x = self.zroot["x"][i]
        y = self.zroot["y"][i]

        # Apply augmentations
        if self.transform is not None:
            augmented = self.transform(image=x.transpose(1, 2, 0), mask=y)
            x = augmented["image"].transpose(2, 0, 1)
            y = augmented["mask"]

        return x, y


class DartsDatasetInMemory(Dataset):
    def __init__(self, data_dir: Path | str, augment: bool, indices: list[int] | None = None):
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)

        x_files = sorted((data_dir / "x").glob("*.pt"))
        y_files = sorted((data_dir / "y").glob("*.pt"))
        assert len(x_files) == len(y_files), f"Dataset corrupted! Got {len(x_files)=} and {len(y_files)=}!"
        if indices is not None:
            x_files = [x_files[i] for i in indices]
            y_files = [y_files[i] for i in indices]

        self.x = []
        self.y = []
        for xfile, yfile in zip(x_files, y_files):
            assert xfile.stem == yfile.stem, (
                f"Dataset corrupted! Files must have the same name, but got {xfile=} {yfile=}!"
            )
            x = torch.load(xfile).numpy()
            y = torch.load(yfile).int().numpy()
            self.x.append(x)
            self.y.append(y)

        # TODO: make this compositable
        self.transform = (
            A.Compose(
                [
                    A.HorizontalFlip(),
                    A.VerticalFlip(),
                    A.RandomRotate90(),
                    # A.Blur(),
                    A.RandomBrightnessContrast(),
                    A.MultiplicativeNoise(per_channel=True, elementwise=True),
                    # ToTensorV2(),
                ]
            )
            if augment
            else None
        )

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
    data_split_method: Literal["random", "region", "sample"] | None,
    data_split_by: list[str] | str | float | None,
):
    # Match statement doesn't like None
    data_split_method = data_split_method or "none"

    # TODO: Assert that for random method the test data is not empty
    match data_split_method:
        case "none":
            return metadata, metadata
        case "random":
            data_split_by = data_split_by or 0.8
            assert isinstance(data_split_by, float)
            train_metadata = metadata.sample(frac=data_split_by, random_state=42)
            test_metadata = metadata.drop(train_metadata.index)
            return train_metadata, test_metadata
        case "region":
            if isinstance(data_split_by, str):
                data_split_by = [data_split_by]
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
    fold_method: Literal["kfold", "shuffle", "stratified", "region", "region-stratified", None],
    n_folds: int,
    fold: int,
) -> tuple[list[int], list[int]]:
    fold = fold if fold_method is not None else 0
    fold_method = fold_method or "none"
    match fold_method:
        case "none":
            foldgen = [(metadata.index.tolist(), metadata.index.tolist())]
        case "kfold":
            foldgen = KFold(n_folds, random_state=42).split(metadata)
        case "shuffle":
            foldgen = StratifiedShuffleSplit(n_splits=n_folds, random_state=42).split(metadata, ~metadata["empty"])
        case "stratified":
            foldgen = StratifiedKFold(n_folds, random_state=42).split(metadata, ~metadata["empty"])
        case "region":
            foldgen = GroupShuffleSplit(n_folds, random_state=42).split(metadata, groups=metadata["region"])
        case "region-stratified":
            foldgen = StratifiedGroupKFold(n_folds, random_state=42).split(
                metadata, ~metadata["empty"], groups=metadata["region"]
            )
        case _:
            raise ValueError(f"Unknown fold method: {fold_method}")

    for i, (train_index, val_index) in enumerate(foldgen):
        if i != fold:
            continue
        # Turn index into metadata index
        train_index = metadata.index.iloc[train_index].tolist()
        val_index = metadata.index.iloc[val_index].tolist()
        return train_index, val_index

    raise ValueError(f"Fold {fold} not found")


class DartsDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        batch_size: int,
        # data_split is for the test split
        data_split_method: Literal["random", "region", "sample"] | None = None,
        data_split_by: list[str] | str | float | None = None,
        # fold is for cross-validation split (train/val)
        fold_method: Literal["kfold", "shuffle", "stratified", "region", "region-stratified"] | None = "kfold",
        total_folds: int = 5,
        fold: int = 0,
        augment: bool = True,  # Not used for test
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
                "random" will split the data randomly, the seed is always 42 and the size of the test set can be
                specified by providing a float between 0 and 1 to data_split_by.
                "region" will split the data by one or multiple regions,
                which can be specified by providing a str or list of str to data_split_by.
                "sample" will split the data by sample ids, which can also be specified similar to "region".
                If None, no split is done and the complete dataset is used for both training and testing.
                The train split will further be split in the cross validation process.
                Defaults to None.
            data_split_by (list[str] | str | float | None, optional): Select by which seed/regions/samples split.
                Defaults to None.
            fold_method (Literal["kfold", "shuffle", "stratified", "region", "region-stratified"] | None, optional):
                Method for cross-validation split. Defaults to "kfold".
            total_folds (int, optional): Total number of folds in cross-validation. Defaults to 5.
            fold (int, optional): Index of the current fold. Defaults to 0.
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

        store = zarr.storage.DirectoryStore(data_dir / "data.zarr")
        zroot = zarr.group(store=store)
        self.nsamples = len(zroot["x"])

    def setup(self, stage: Literal["fit", "validate", "test", "predict"] | None = None):
        if stage == "predict" or stage is None:
            return

        metadata = gpd.read_file(self.data_dir / "metadata.parquet")
        train_metadata, test_metadata = _split_metadata(metadata, self.data_split_method, self.data_split_by)

        if stage in ["fit", "validate"]:
            train_index, val_index = _get_fold(train_metadata, self.fold_method, self.total_folds, self.fold)
            dsclass = DartsDatasetInMemory if self.in_memory else DartsDatasetZarr
            self.train = dsclass(self.data_dir, self.augment, train_index)
            self.val = dsclass(self.data_dir, False, val_index)
        if stage == "test":
            test_index = test_metadata.index.tolist()
            dsclass = DartsDatasetInMemory if self.in_memory else DartsDatasetZarr
            self.test = dsclass(self.data_dir, False, test_index)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers)
