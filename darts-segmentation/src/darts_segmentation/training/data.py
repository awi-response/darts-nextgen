# ruff: noqa: D101
# ruff: noqa: D102
# ruff: noqa: D105
# ruff: noqa: D107
"""Training script for DARTS segmentation."""

import logging
from pathlib import Path
from typing import Literal

import albumentations as A  # noqa: N812
import lightning as L  # noqa: N812
import torch
import zarr
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__.replace("darts_", "darts."))


class DartsDataset(Dataset):
    def __init__(self, data_dir: Path | str, augment: bool, indices: list[int] | None = None):
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)

        self.x_files = sorted((data_dir / "x").glob("*.pt"))
        self.y_files = sorted((data_dir / "y").glob("*.pt"))
        assert len(self.x_files) == len(
            self.y_files
        ), f"Dataset corrupted! Got {len(self.x_files)=} and {len(self.y_files)=}!"
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

        store = zarr.storage.DirectoryStore(data_dir)
        self.zroot = zarr.group(store=store)

        assert (
            "x" in self.zroot and "y" in self.zroot
        ), f"Dataset corrupted! {self.zroot.info=} must contain 'x' or 'y' arrays!"

        self.indices = indices if indices is not None else list(range(len(self.zroot["x"])))

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
            assert (
                xfile.stem == yfile.stem
            ), f"Dataset corrupted! Files must have the same name, but got {xfile=} {yfile=}!"
            x = torch.load(xfile).numpy()
            y = torch.load(yfile).int().numpy()
            self.x.append(x)
            self.y.append(y)

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


class DartsDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        batch_size: int,
        fold: int = 0,  # Not used for test
        augment: bool = True,  # Not used for test
        num_workers: int = 0,
        in_memory: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.fold = fold
        self.augment = augment
        self.num_workers = num_workers
        self.in_memory = in_memory

        data_dir = Path(data_dir)

        store = zarr.storage.DirectoryStore(data_dir)
        zroot = zarr.group(store=store)
        self.nsamples = len(zroot["x"])

    def setup(self, stage: Literal["fit", "validate", "test", "predict"] | None = None):
        if stage in ["fit", "validate"]:
            kf = KFold(n_splits=5)
            train_idx, val_idx = list(kf.split(range(self.nsamples)))[self.fold]

            dsclass = DartsDatasetInMemory if self.in_memory else DartsDatasetZarr
            self.train = dsclass(self.data_dir, self.augment, train_idx)
            self.val = dsclass(self.data_dir, False, val_idx)
        if stage == "test":
            dsclass = DartsDatasetInMemory if self.in_memory else DartsDatasetZarr
            self.test = dsclass(self.data_dir, False)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers)
