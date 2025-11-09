"""Functions to prepare the training data for the segmentation model training."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import geopandas as gpd
import lovely_tensors
import toml
import torch
import xarray as xr
import zarr
from darts_utils.bands import manager
from darts_utils.cuda import free_torch
from geocube.api.core import make_geocube
from zarr.codecs import BloscCodec
from zarr.storage import LocalStore

from darts_segmentation.inference import create_patches

logger = logging.getLogger(__name__.replace("darts_", "darts."))


@dataclass
class PatchCoords:
    """Wrapper which stores the coordinate information of a patch in the original image."""

    i: int
    patch_idx_y: int
    patch_idx_x: int
    y: slice
    x: slice

    @classmethod
    def from_tensor(cls, coords: torch.Tensor, patch_size: int) -> "PatchCoords":
        """Create a PatchCoords object from the returned coord tensor of `create_patches`.

        Args:
            coords (torch.Tensor): The coordinates of the patch in the original image, from `create_patches`.
            patch_size (int): The size of the patch.

        Returns:
            PatchCoords: The coordinates of the patch in the original image.

        """
        i, y, x, h, w = coords.int().numpy()
        return cls(
            i=i,
            patch_idx_y=h.item(),
            patch_idx_x=w.item(),
            y=slice(y.item(), y.item() + patch_size),
            x=slice(x.item(), x.item() + patch_size),
        )


def create_labels(
    tile: xr.Dataset,
    labels: gpd.GeoDataFrame,
    extent: gpd.GeoDataFrame | None = None,
) -> xr.DataArray:
    """Create rasterized labels from vector labels and quality mask.

    This function rasterizes the provided labels and applies quality filtering based on
    the tile's quality_data_mask. Areas outside the extent or with low quality are marked
    as invalid (class 2).

    Label encoding:
        - 0: Negative (no RTS)
        - 1: Positive (RTS present)
        - 2: Invalid/masked (outside extent, low quality, or no data)

    Args:
        tile (xr.Dataset): The input tile, containing preprocessed, harmonized data.
            Must contain a 'quality_data_mask' variable where value 2 indicates best quality.
        labels (gpd.GeoDataFrame): The labels to be used for training.
            Geometries will be rasterized as positive samples (class 1).
        extent (gpd.GeoDataFrame | None): The extent of the valid training area.
            Pixels outside this extent will be marked as invalid (class 2).
            If None, no extent masking is applied.

    Returns:
        xr.DataArray: The rasterized labels with shape (y, x) and values 0, 1, or 2.
            Pixels with quality_data_mask != 2 are automatically marked as invalid (class 2).

    """
    # Rasterize the labels
    if len(labels) > 0:
        labels["id"] = labels.index
        labels_rasterized = 1 - make_geocube(labels, measurements=["id"], like=tile["quality_data_mask"]).id.isnull()
    else:
        labels_rasterized = xr.zeros_like(tile["quality_data_mask"])

    # Rasterize the extent if provided
    if extent is not None:
        extent["id"] = extent.index
        extent_rasterized = 1 - make_geocube(extent, measurements=["id"], like=tile["quality_data_mask"]).id.isnull()
        labels_rasterized = labels_rasterized.where(extent_rasterized, 2)

    # Because rasterio use different floats, it can happen that the axes are not properly aligned
    labels_rasterized["x"] = tile.x
    labels_rasterized["y"] = tile.y

    # Filter out low-quality and no-data values (class 2 -> best quality)
    quality_mask = tile["quality_data_mask"] == 2
    # quality_mask = erode_mask(tile["quality_data_mask"] == 2, mask_erosion_size, device)
    labels_rasterized = labels_rasterized.where(quality_mask, 2)

    return labels_rasterized


def create_training_patches(
    tile: xr.Dataset,
    labels: gpd.GeoDataFrame,
    extent: gpd.GeoDataFrame | None,
    bands: list[str],
    patch_size: int,
    overlap: int,
    exclude_nopositive: bool,
    exclude_nan: bool,
    device: Literal["cuda", "cpu"] | int,
) -> tuple[torch.tensor, torch.tensor, list[PatchCoords]]:
    """Create training patches from a tile and labels with quality filtering.

    This function creates overlapping patches from the input tile and rasterized labels,
    applying several filtering criteria to ensure high-quality training data. Pixels with
    quality_data_mask == 0 are set to NaN in the input data.

    Patch filtering:
        - Excludes patches with < 10% visible pixels (> 90% invalid/masked)
        - Excludes patches where all bands are NaN
        - Optionally excludes patches without positive labels (if exclude_nopositive=True)
        - Optionally excludes patches with any NaN values (if exclude_nan=True)

    Args:
        tile (xr.Dataset): The input tile, containing preprocessed, harmonized data.
            Must contain a 'quality_data_mask' variable where 0=invalid and 2=best quality.
        labels (gpd.GeoDataFrame): The labels to be used for training.
            Geometries will be rasterized as positive samples.
        extent (gpd.GeoDataFrame | None): The extent of the valid training area.
            Pixels outside this extent will be marked as invalid in labels.
            If None, no extent masking is applied.
        bands (list[str]): The bands to extract and use for training.
            Will be normalized using the band manager.
        patch_size (int): The size of each patch in pixels (height and width).
        overlap (int): The overlap between adjacent patches in pixels.
        exclude_nopositive (bool): Whether to exclude patches where the labels do not contain
            any positive samples (class 1).
        exclude_nan (bool): Whether to exclude patches where the input data has any NaN values.
        device (Literal["cuda", "cpu"] | int): The device to use for tensor operations.
            Can be "cuda", "cpu", or an integer GPU index.

    Returns:
        tuple[torch.tensor, torch.tensor, list[PatchCoords]]: A tuple containing:
            - Input patches with shape (N, C, H, W), NaN values replaced with 0
            - Label patches with shape (N, H, W), values 0/1/2
            - List of PatchCoords objects with coordinate information

    """
    if len(labels) == 0 and exclude_nopositive:
        logger.warning("No labels found in the labels GeoDataFrame. Skipping.")
        return

    # Rasterize the labels
    labels_rasterized = create_labels(tile, labels, extent)
    tensor_labels = torch.tensor(labels_rasterized.values, device=device).float()

    invalid_mask = (tile["quality_data_mask"] == 0).data
    tile = tile[bands].transpose("y", "x")
    tile = manager.normalize(tile)
    tensor_tile = torch.as_tensor(tile.to_dataarray().data, device=device).float()
    tensor_tile[:, invalid_mask] = float("nan")  # Set invalid pixels to NaN

    assert tensor_tile.dim() == 3, f"Expects tensor_tile to has shape (C, H, W), got {tensor_tile.shape}"
    assert tensor_labels.dim() == 2, f"Expects tensor_labels to has shape (H, W), got {tensor_labels.shape}"

    # Create patches
    n_bands = len(bands)
    tensor_patches = create_patches(tensor_tile.unsqueeze(0), patch_size, overlap)
    tensor_patches = tensor_patches.reshape(-1, n_bands, patch_size, patch_size)
    tensor_labels, tensor_coords = create_patches(
        tensor_labels.unsqueeze(0).unsqueeze(0), patch_size, overlap, return_coords=True
    )
    tensor_labels = tensor_labels.reshape(-1, patch_size, patch_size)
    tensor_coords = tensor_coords.reshape(-1, 5).to(device=device)

    # Filter out patches based on settings
    few_visible = ((tensor_labels != 2).sum(dim=(1, 2)) / tensor_labels[0].numel()) < 0.1
    logger.debug(f"Excluding {few_visible.sum().item()} patches with less than 10% visible pixels")
    all_nans = torch.isnan(tensor_patches).all(dim=(2, 3)).any(dim=1)
    logger.debug(f"Excluding {all_nans.sum().item()} patches where everything is nan")
    filter_mask = few_visible | all_nans
    if exclude_nopositive:
        nopositives = (tensor_labels == 1).any(dim=(1, 2))
        logger.debug(f"Excluding {nopositives.sum().item()} patches with no positive labels")
        filter_mask |= ~nopositives
    if exclude_nan:
        has_nans = torch.isnan(tensor_patches).any(dim=(1, 2, 3))
        logger.debug(f"Excluding {has_nans.sum().item()} patches with nan values")
        filter_mask |= has_nans

    n_patches = tensor_patches.shape[0]
    logger.debug(f"Using {n_patches - filter_mask.sum().item()} patches out of {n_patches} total patches")

    tensor_patches = tensor_patches[~filter_mask].cpu()
    tensor_labels = tensor_labels[~filter_mask].cpu()
    tensor_coords = tensor_coords[~filter_mask].cpu()
    free_torch()
    coords = [PatchCoords.from_tensor(tensor_coords[i], patch_size) for i in range(tensor_coords.shape[0])]
    # Fill nan with 0, since we don't want to have NaNs in the patches
    tensor_patches = tensor_patches.nan_to_num(0.0)
    return tensor_patches, tensor_labels, coords


@dataclass
class TrainDatasetBuilder:
    """Helper class to create all necessary files for a DARTS training dataset.

    This class manages the creation of a training dataset stored in Zarr format with
    associated metadata. It handles patch creation, quality filtering, and metadata tracking.

    The dataset structure:
        - data.zarr/x: Input patches (N, C, H, W) as float32
        - data.zarr/y: Label patches (N, H, W) as uint8 with values 0/1/2
        - metadata.parquet: Patch metadata including coordinates and geometry
        - config.toml: Dataset configuration and parameters

    Attributes:
        train_data_dir (Path): Directory where the dataset will be saved.
        patch_size (int): Size of each patch in pixels.
        overlap (int): Overlap between adjacent patches in pixels.
        bands (list[str]): List of band names to include in the dataset.
        exclude_nopositive (bool): Exclude patches without positive labels.
        exclude_nan (bool): Exclude patches with any NaN values.
        device (Literal["cuda", "cpu"] | int): Device for patch creation operations.
        append (bool): If True, append to existing dataset. Defaults to False.

    """

    train_data_dir: Path
    patch_size: int
    overlap: int
    bands: list[str]
    exclude_nopositive: bool
    exclude_nan: bool
    device: Literal["cuda", "cpu"] | int
    append: bool = False

    def __post_init__(self):
        """Initialize the TrainDatasetBuilder class based on provided dataclass params.

        This will setup everything needed to add patches to the dataset:

        - Create the train_data_dir if it does not exist
        - Create an emtpy zarr store
        - Initialize the metadata list
        """
        lovely_tensors.monkey_patch()
        lovely_tensors.set_config(color=False)
        self._metadata = []
        if self.append and (self.train_data_dir / "metadata.parquet").exists():
            self._metadata = gpd.read_parquet(self.train_data_dir / "metadata.parquet").to_dict(orient="records")

        self.train_data_dir.mkdir(exist_ok=True, parents=True)

        self._zroot = zarr.group(store=LocalStore(self.train_data_dir / "data.zarr"), overwrite=not self.append)
        # We need do declare the number of patches to 0, because we can't know the final number of patches

        if not self.append:
            self._zroot.create(
                name="x",
                shape=(0, len(self.bands), self.patch_size, self.patch_size),
                # shards=(100, len(bands), patch_size, patch_size),
                chunks=(1, 1, self.patch_size, self.patch_size),
                dtype="float32",
                compressors=BloscCodec(cname="lz4", clevel=9),
            )
            self._zroot.create(
                name="y",
                shape=(0, self.patch_size, self.patch_size),
                # shards=(100, patch_size, patch_size),
                chunks=(1, self.patch_size, self.patch_size),
                dtype="uint8",
                compressors=BloscCodec(cname="lz4", clevel=9),
            )
        else:
            assert "x" in self._zroot and "y" in self._zroot, (
                "When appending to an existing dataset, the 'x' and 'y' arrays must already exist."
                "Did you set append=True by accident?"
            )

    def __len__(self):  # noqa: D105
        return len(self._metadata)

    def add_tile(
        self,
        tile: xr.Dataset,
        labels: gpd.GeoDataFrame,
        region: str,
        sample_id: str,
        extent: gpd.GeoDataFrame | None = None,
        metadata: dict[str, str] | None = None,
    ):
        """Add a tile to the dataset by creating and appending patches.

        This method processes a single tile by creating training patches and appending them
        to the Zarr arrays. Patch metadata including coordinates, geometry, and custom fields
        are tracked for later use.

        Args:
            tile (xr.Dataset): The input tile, containing preprocessed, harmonized data.
                Must contain the specified bands and a 'quality_data_mask' variable.
            labels (gpd.GeoDataFrame): The labels to be used for training.
                Geometries will be rasterized as positive samples (class 1).
            region (str): The region identifier for this tile (e.g., "Alaska", "Canada").
                Stored in metadata for tracking and filtering.
            sample_id (str): A unique identifier for this tile/sample.
                Stored in metadata for tracking and filtering.
            extent (gpd.GeoDataFrame | None, optional): The extent of the valid training area.
                Pixels outside this extent will be marked as invalid (class 2) in labels.
                If None, no extent masking is applied.
            metadata (dict[str, str], optional): Additional metadata to be added to the metadata file.
                Will not be used for training, but can be used for debugging or reproducibility.
                Path values will be automatically converted to strings.

        """
        metadata = metadata or {}
        # Convert all paths of metadata to strings
        metadata = {k: str(v) if isinstance(v, Path) else v for k, v in metadata.items()}

        x, y, stacked_coords = create_training_patches(
            tile=tile,
            labels=labels,
            extent=extent,
            bands=self.bands,
            patch_size=self.patch_size,
            overlap=self.overlap,
            exclude_nopositive=self.exclude_nopositive,
            exclude_nan=self.exclude_nan,
            device=self.device,
        )

        self._zroot["x"].append(x.numpy().astype("float32"))
        self._zroot["y"].append(y.numpy().astype("uint8"))

        for patch_id, coords in enumerate(stacked_coords):
            geometry = tile.isel(x=coords.x, y=coords.y).odc.geobox.geographic_extent.geom
            self._metadata.append(
                {
                    "z_idx": len(self._metadata),
                    "patch_id": patch_id,
                    "region": region,
                    "sample_id": sample_id,
                    "empty": not (y[patch_id] == 1).any(),
                    "x": coords.x.start,
                    "y": coords.y.start,
                    "patch_idx_x": coords.patch_idx_x,
                    "patch_idx_y": coords.patch_idx_y,
                    "geometry": geometry,
                    **metadata,
                }
            )

    def finalize(self, data_config: dict[str, str] | None = None):
        """Finalize the dataset by saving the metadata and the config file.

        Args:
            data_config (dict[str, str], optional): The data config to be saved in the config file.
                This should contain all the information needed to recreate the dataset.
                It will be saved as a toml file, along with the configuration provided in this dataclass.

        Raises:
            ValueError: If no patches were found in the dataset.

        """
        if len(self._metadata) == 0:
            logger.error("No patches found in the dataset.", exc_info=True)
            raise ValueError("No patches found in the dataset.")

        # Save the metadata
        metadata = gpd.GeoDataFrame(self._metadata, crs="EPSG:4326")
        metadata.to_parquet(self.train_data_dir / "metadata.parquet")

        data_config = data_config or {}
        # Convert the data_config paths to strings
        data_config = {k: str(v) if isinstance(v, Path) else v for k, v in data_config.items()}

        # Save a config file as toml
        config = {
            "darts": {
                "train_data_dir": str(self.train_data_dir),
                "patch_size": self.patch_size,
                "overlap": self.overlap,
                "n_bands": len(self.bands),
                "exclude_nopositive": self.exclude_nopositive,
                "exclude_nan": self.exclude_nan,
                "n_patches": len(metadata),
                "device": self.device,
                "bands": self.bands,  # keys: bands, band_factors, band_offsets
                **data_config,
            }
        }
        with open(self.train_data_dir / "config.toml", "w") as f:
            toml.dump(config, f)

        logger.info(f"Saved {len(metadata)} patches to {self.train_data_dir}")
