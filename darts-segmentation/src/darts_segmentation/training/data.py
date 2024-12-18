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
from torch.utils.data import DataLoader, Dataset, random_split

logger = logging.getLogger(__name__.replace("darts_", "darts."))


class DartsDataset(Dataset):
    def __init__(self, data_dir: Path | str, augment: bool):
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
        self.x_files = sorted((data_dir / "x").glob("*.pt"))
        self.y_files = sorted((data_dir / "y").glob("*.pt"))

        assert len(self.x_files) == len(
            self.y_files
        ), f"Dataset corrupted! Got {len(self.x_files)=} and {len(self.y_files)=}!"

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


class DartsDatasetInMemory(Dataset):
    def __init__(self, data_dir: Path | str, augment: bool):
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
        x_files = sorted((data_dir / "x").glob("*.pt"))
        y_files = sorted((data_dir / "y").glob("*.pt"))

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

        assert len(x_files) == len(y_files), f"Dataset corrupted! Got {len(x_files)=} and {len(y_files)=}!"

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
        augment: bool = True,
        num_workers: int = 0,
        in_memory: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.augment = augment
        self.num_workers = num_workers
        self.in_memory = in_memory

    def setup(self, stage: Literal["fit", "validate", "test", "predict"] | None = None):
        dataset = (
            DartsDataset(self.data_dir, self.augment)
            if not self.in_memory
            else DartsDatasetInMemory(self.data_dir, self.augment)
        )
        # self.dataset = dataset
        splits = [0.8, 0.1, 0.1]
        generator = torch.Generator().manual_seed(42)
        self.train, self.val, self.test = random_split(dataset, splits, generator)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers)
