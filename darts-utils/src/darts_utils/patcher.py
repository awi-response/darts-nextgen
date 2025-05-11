"""Patch a dataset into smaller patches."""

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import xarray as xr


@dataclass
class Patch:
    """Class representing a patch of a dataset."""

    i: int
    patch_idx_y: int
    patch_idx_x: int
    y: slice
    x: slice
    data: xr.Dataset | xr.DataArray

    @property
    def patch_idx(self) -> tuple[int, int]:
        """Return the patch index as a tuple."""
        return self.patch_idx_y, self.patch_idx_x

    def __repr__(self) -> str:  # noqa: D105
        return f"Patch {self.i} ({self.patch_idx_y}, {self.patch_idx_x})"


class PatchedDataset:
    """Class representing a dataset that has been patched into smaller patches.

    Example:
        Via getter/setter:

        ```python
        tile: xr.Dataset
        patches = PatchedDataset(tile, patch_size, overlap)
        print(len(patches))
        grey = (patches["blue"] + patches["green"] + patches["red"]) / 3 # grey is a numpy array
        patches[None] = grey # Replace the data in the patches with the gray data
        new_tile = patches.combine_patches()
        new_tile # This is now a DataArray containing the gray data
        ```

        Via loop:

        ```python
        tile: xr.Dataset
        patches = PatchedDataset(tile, patch_size, overlap)
        # Calculate gray area for each patch
        for patch in patches:
            patch.data = (patch.data.blue + patch.data.green + patch.data.red) / 3

        new_tile = patches.combine_patches()
        new_tile # This is now a DataArray containing the gray data
        ```

    """

    def __init__(self, ds: xr.Dataset | xr.DataArray, patch_size: int, overlap: int) -> list["Patch"]:
        """Initialize the PatchedDataset.

        Args:
            ds (xr.Dataset | xr.DataArray): The dataset to patch.
            patch_size (int): The size of the patches.
            overlap (int): The size of the overlap between patches.

        """
        self.patch_size = patch_size
        self.overlap = overlap
        self.coords = ds.coords
        self._patches = []
        h, w = ds.sizes["y"], ds.sizes["x"]
        step_size = patch_size - overlap
        # Substract the overlap from h and w so that an exact match of the last patch won't create a duplicate
        for patch_idx_y, y in enumerate(range(0, h - overlap, step_size)):
            for patch_idx_x, x in enumerate(range(0, w - overlap, step_size)):
                if y + patch_size > h:
                    y = h - patch_size
                if x + patch_size > w:
                    x = w - patch_size
                ys = slice(y, y + patch_size)
                xs = slice(x, x + patch_size)
                self._patches.append(
                    Patch(
                        i=len(self._patches),
                        patch_idx_y=patch_idx_y,
                        patch_idx_x=patch_idx_x,
                        y=ys,
                        x=xs,
                        data=ds.isel(y=ys, x=xs),
                    )
                )

        # Create a soft margin for the patches (NumPy version)
        margin_ramp = np.concatenate(
            [
                np.linspace(0, 1, overlap),
                np.ones(patch_size - 2 * overlap),
                np.linspace(1, 0, overlap),
            ]
        )
        self.soft_margin = margin_ramp.reshape(1, patch_size) * margin_ramp.reshape(patch_size, 1)

    def __getitem__(self, key: str | None) -> np.ndarray:  # noqa: D105
        is_dataarray = all(isinstance(patch.data, xr.DataArray) for patch in self._patches)
        is_dataset = all(isinstance(patch.data, xr.Dataset) for patch in self._patches)
        if is_dataset:
            assert key is not None, "Key must be provided for Dataset"
            return np.array([patch.data[key].data for patch in self._patches])
        elif is_dataarray:
            assert key is None, "Key must be None for DataArray"
            return np.array([patch.data.data for patch in self._patches])

    def __setitem__(self, key: str | None, a: np.array):  # noqa: D105
        for i, patch in enumerate(self._patches):
            if key is None:
                patch.data = xr.DataArray(a[i], dims=("y", "x"))
            else:
                patch.data[key] = xr.DataArray(a[i], dims=("y", "x"))
        return self

    def combine_patches(self) -> xr.DataArray | xr.Dataset:
        """Combine patches into a single dataarray.

        Returns:
            xr.DataArray | xr.Dataset: The combined dataarray or dataset.

        """
        is_dataarray = all(isinstance(patch.data, xr.DataArray) for patch in self._patches)
        is_dataset = all(isinstance(patch.data, xr.Dataset) for patch in self._patches)

        if is_dataarray:
            combined = xr.DataArray(0.0, dims=("y", "x"), coords=self.coords)
        elif is_dataset:
            combined = xr.Dataset(coords=self.coords)
            for var in self._patches[0].data.data_vars:
                combined[var] = xr.DataArray(0.0, dims=("y", "x"), coords=self.coords)

        weights = xr.DataArray(0.0, dims=("y", "x"), coords=self.coords)
        for patch in self._patches:
            weights[patch.y, patch.x] += self.soft_margin
            if is_dataarray:
                combined[patch.y, patch.x] += patch.data * self.soft_margin
            elif is_dataset:
                for var in patch.data.data_vars:
                    combined[var][patch.y, patch.x] += patch.data[var] * self.soft_margin
        # Normalize the combined data by the weights
        combined /= weights
        return combined

    def __len__(self) -> int:  # noqa: D105
        return len(self._patches)

    def __iter__(self) -> Iterator[Patch]:  # noqa: D105
        return iter(self._patches)
