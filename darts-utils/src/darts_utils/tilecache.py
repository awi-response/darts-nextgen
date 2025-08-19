"""Caching functionality for xarray datasets."""

import logging
from pathlib import Path
from typing import Any

import xarray as xr

logger = logging.getLogger(__name__.replace("darts_", "darts."))


class XarrayCacheManager:
    """Manager for caching xarray datasets.

    Example:
        ```python
            def process_tile(tile_id: str):
                # Initialize cache manager
                preprocess_cache = Path("preprocess_cache")
                cache_manager = XarrayCacheManager(preprocess_cache)

                def create_tile():
                    # Your existing tile creation logic goes here
                    return create_tile(...)  # Replace with actual implementation

                # Get cached tile or create and cache it
                tile = cache_manager.get_or_create(
                    identifier=tile_id,
                    creation_func=create_tile
                )

                return tile
        ```

    """

    def __init__(self, cache_dir: str | Path | None = None):
        """Initialize the cache manager.

        Args:
            cache_dir (str | Path | None): Directory path for caching files

        """
        self.cache_dir = Path(cache_dir) if isinstance(cache_dir, str) else cache_dir

    def exists(self, identifier: str) -> bool:
        """Check if a cached Dataset exists.

        Args:
            identifier (str): Unique identifier for the cached file

        Returns:
            bool: True if the Dataset exists in cache, False otherwise

        """
        if not self.cache_dir:
            return False

        cache_path = self.cache_dir / f"{identifier}.nc"
        return cache_path.exists()

    def load_from_cache(self, identifier: str) -> xr.Dataset | None:
        """Load a Dataset from cache if it exists.

        Args:
            identifier (str): Unique identifier for the cached file

        Returns:
            xr.Dataset | None: Dataset if found in cache, otherwise None

        """
        if not self.cache_dir:
            return None

        cache_path = self.cache_dir / f"{identifier}.nc"
        if not cache_path.exists():
            return None
        dataset = xr.open_dataset(cache_path, engine="h5netcdf").set_coords("spatial_ref")
        return dataset

    def save_to_cache(self, dataset: xr.Dataset, identifier: str) -> bool:
        """Save a Dataset to cache.

        Args:
            dataset (xr.Dataset): Dataset to cache
            identifier (str): Unique identifier for the cached file

        Returns:
            bool: Success of operation

        """
        if not self.cache_dir:
            return False

        self.cache_dir.mkdir(exist_ok=True, parents=True)
        cache_path = self.cache_dir / f"{identifier}.nc"
        logger.debug(f"Caching {identifier=} to {cache_path}")
        dataset.to_netcdf(cache_path, engine="h5netcdf")
        return True

    def get_or_create(
        self,
        identifier: str,
        creation_func: callable,
        force: bool,
        *args: tuple[Any, ...],
        **kwargs: dict[str, Any],
    ) -> xr.Dataset:
        """Get cached Dataset or create and cache it if it doesn't exist.

        Args:
            identifier (str): Unique identifier for the cached file
            creation_func (callable): Function to create the Dataset if not cached
            force (bool): If True, forces reprocessing even if cached
            *args: Arguments to pass to creation_func
            **kwargs: Keyword arguments to pass to creation_func

        Returns:
            xr.Dataset: The Dataset (either loaded from cache or newly created)

        """
        cached_dataset = None if force else self.load_from_cache(identifier)
        if not force:
            logger.debug(f"Cache hit for '{identifier}': {cached_dataset is not None}")

        if cached_dataset is not None:
            return cached_dataset

        dataset = creation_func(*args, **kwargs)
        if cached_dataset is None:
            self.save_to_cache(dataset, identifier)
        return dataset
