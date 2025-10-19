"""Raw Data Store for Sentinel 2 data."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar

import xarray as xr

logger = logging.getLogger(__name__.replace("darts_", "darts."))


# Idea use a procedurally approach when downloading different bands -> easier to extend later on

SceneItem = TypeVar("SceneItem")


class StoreManager(ABC, Generic[SceneItem]):
    """Manager for storing raw sentinel 2 data.

    This class is an abstract base class and should be extended to implement the respective downloading methods.

    Usage:

        1. "Normal" usage:

        ```python
            store_manager = StoreManager(store_path)
            ds_s2 = store_manager.load(identifier, bands)
        ```

        2. Force download:

        ```python
            store_manager = StoreManager(store_path)
            ds_s2 = store_manager.load(identifier, force=True)
        ```

        3. Download only (and only if missing) and store the scene:

        ```python
            store_manager = StoreManager(store_path)
            store_manager.download(identifier) # store_path must be not None
        ```

        4. Offline mode:

        ```python
            store_manager = StoreManager(store_path)
            store_manager.open(identifier) # store_path must be not None, bands must be complete
        ```
    """

    def __init__(self, bands: list[str], store: str | Path | None = None):
        """Initialize the store manager.

        Args:
            bands (list[str]): List of bands to manage
            store (str | Path | None): Directory path for storing raw sentinel 2 data

        """
        self.bands = bands
        self.store = Path(store) if isinstance(store, str) else store

    def exists(self, identifier: str) -> bool:
        """Check if a scene already exists in the local raw data store.

        Args:
            identifier (str): Unique identifier for the scene

        Returns:
            bool: True if the scene exists in the store, False otherwise

        """
        if not self.store:
            return False

        scene_path = self.store / f"{identifier}.zarr"
        return scene_path.exists()

    def missing_bands(self, identifier: str) -> list[str]:
        """Get the list of missing bands for a scene in the store.

        Args:
            identifier (str): Unique identifier for the scene

        Returns:
            list[str]: List of missing bands

        """
        if not self.store:
            return self.bands

        scene_path = self.store / f"{identifier}.zarr"
        if not scene_path.exists():
            return self.bands

        dataset = xr.open_zarr(scene_path, consolidated=False)
        required_bands = set(self.bands)
        present_bands = set(dataset.data_vars)
        missing = required_bands - present_bands
        return list(missing)

    def complete(self, identifier: str) -> bool:
        """Check if a scene in the store contains all requested bands.

        Args:
            identifier (str): Unique identifier for the scene
            bands (list[str]): List of requested bands

        Returns:
            bool: True if all requested bands are present, False otherwise

        """
        return len(self.missing_bands(identifier)) == 0

    def save_to_store(self, dataset: xr.Dataset, identifier: str) -> None:
        """Save a scene dataset to the local raw data store.

        Will append new bands to existing store if scene already exists.
        Will overwrite existing bands in an existing store if scene already exists.

        Args:
            dataset (xr.Dataset): Dataset to save
            identifier (str): Unique identifier for the scene

        """
        assert self.store is not None, "Store must be provided to save scenes!"
        scene_path = self.store / f"{identifier}.zarr"
        encoding = self.encodings(list(dataset.data_vars))
        if not scene_path.exists():
            dataset.to_zarr(scene_path, encoding=encoding, consolidated=False, mode="w")
        else:
            # Assert that the coordinates match
            existing_dataset = xr.open_zarr(scene_path, consolidated=False)
            xr.testing.assert_allclose(existing_dataset.coords, dataset.coords)
            # Overwrite dataset coords to avoid conflicts by floating point precision issues
            dataset["x"] = existing_dataset.x
            dataset["y"] = existing_dataset.y
            dataset.to_zarr(scene_path, encoding=encoding, consolidated=False, mode="a")

    def open(self, item: str | SceneItem) -> xr.Dataset:
        """Open a scene from local store.

        Store must be provided and the scene must be present in store!

        Args:
            item (str | SceneItem): Item or scene-id to open

        Returns:
            xr.Dataset: The opened scene as xarray Dataset

        """
        identifier = self.identifier(item)
        assert self.complete(identifier), f"Scene {identifier} is incomplete in store!"
        scene_path = self.store / f"{identifier}.zarr"
        return xr.open_zarr(scene_path, consolidated=False).set_coords("spatial_ref").load()

    def download_and_store(self, item: str | SceneItem):
        """Download a scene from the source and store it in the local store.

        Store must be provided!
        Will do nothing if all required bands are already present.

        Args:
            item (str | SceneItem): Item or scene-id to open.

        """
        assert self.store is not None, "Store must be provided to download and store scenes!"
        identifier = self.identifier(item)
        missing_bands = self.missing_bands(identifier)
        if not missing_bands:
            return
        dataset = self.download_scene_from_source(item, missing_bands)
        self.save_to_store(dataset, identifier)

    def load(self, item: str | SceneItem, force: bool = False) -> xr.Dataset:
        """Load a scene.

        If `force==True` will download the scene from source even if present in store.
        Else, will try to open the scene from store first and only download missing bands.
        Will always store the downloaded scene in local store if store is provided, potentially overwriting existing.

        Args:
            item (str | SceneItem): Item or scene-id to open.
            force (bool, optional): If True, will download the scene even if present. Defaults to False.

        Returns:
            xr.Dataset: The loaded scene as xarray Dataset

        """
        identifier = self.identifier(item)
        if force:
            logger.debug(f"Force downloading scene {identifier} from source.")
            dataset = self.download_scene_from_source(item, self.bands)
            if self.store:
                self.save_to_store(dataset, identifier)
            return dataset

        missing_bands = self.missing_bands(identifier)
        if not missing_bands:
            logger.debug(f"Scene {identifier} is complete, opening from store.")
            return self.open(item)
        logger.debug(f"Scene {identifier} is missing bands {missing_bands}, downloading from source.")
        dataset = self.download_scene_from_source(item, missing_bands)
        if self.store:
            self.save_to_store(dataset, identifier)
        return dataset

    @abstractmethod
    def identifier(self, item: str | SceneItem) -> str: ...  # noqa: D102

    @abstractmethod
    def encodings(self, bands: list[str]) -> dict[str, dict[str, str]]: ...  # noqa: D102

    @abstractmethod
    def download_scene_from_source(self, item: str | SceneItem, bands: list[str]) -> xr.Dataset: ...  # noqa: D102
