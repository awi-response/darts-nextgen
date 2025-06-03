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

# TODO: move erode_mask to darts_utils, since uasge is not limited to prepare_export
from geocube.api.core import make_geocube
from zarr.codecs import BloscCodec
from zarr.storage import LocalStore

from darts_segmentation.utils import Bands, create_patches

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


def create_training_patches(
    tile: xr.Dataset,
    labels: gpd.GeoDataFrame,
    bands: Bands,
    patch_size: int,
    overlap: int,
    exclude_nopositive: bool,
    exclude_nan: bool,
    device: Literal["cuda", "cpu"] | int,
    mask_erosion_size: int,
) -> tuple[torch.tensor, torch.tensor, list[PatchCoords]]:
    """Create training patches from a tile and labels.

    Args:
        tile (xr.Dataset): The input tile, containing preprocessed, harmonized data.
        labels (gpd.GeoDataFrame): The labels to be used for training.
        bands (Bands): The bands to be used for training.
        patch_size (int): The size of the patches.
        overlap (int): The size of the overlap.
        exclude_nopositive (bool): Whether to exclude patches where the labels do not contain positives.
        exclude_nan (bool): Whether to exclude patches where the input data has nan values.
        device (Literal["cuda", "cpu"] | int): The device to use for the erosion.
        mask_erosion_size (int): The size of the disk to use for erosion.

    Returns:
        tuple[torch.tensor, torch.tensor, list[PatchCoords]]: A tuple containing the input, the labels and the coords.
            The input has the format (C, H, W), the labels (H, W).

    Raises:
        ValueError: If a band is not found in the preprocessed data.

    """
    if len(labels) == 0 and exclude_nopositive:
        logger.warning("No labels found in the labels GeoDataFrame. Skipping.")
        return

    # Rasterize the labels
    if len(labels) > 0:
        labels_rasterized = 1 - make_geocube(labels, measurements=["id"], like=tile).id.isnull()
    else:
        labels_rasterized = xr.zeros_like(tile["quality_data_mask"])

    # Filter out the nodata values (class 2 -> invalid data)
    # quality_mask = erode_mask(tile["quality_data_mask"] == 2, mask_erosion_size, device)
    # labels_rasterized = xr.where(quality_mask, labels_rasterized, 2)

    # Transpose to (H, W)
    tile = tile.transpose("y", "x")
    n_bands = len(bands)
    tensor_labels = torch.tensor(labels_rasterized.values, device=device).float()
    tensor_tile = torch.zeros((n_bands, tile.dims["y"], tile.dims["x"]), device=device)
    invalid_mask = (tile["quality_data_mask"] == 0).values
    # This will also order the data into the correct order of bands
    for i, band in enumerate(bands):
        if band.name not in tile:
            raise ValueError(f"Band '{band.name}' not found in the preprocessed data.")
        band_data = torch.tensor(tile[band.name].values, device=device).float()
        # Normalize the bands and clip the values
        band_data = band_data * band.factor + band.offset
        band_data = band_data.clip(0, 1)
        # Apply the quality mask
        band_data[invalid_mask] = float("nan")
        # Merge with the tile and move back to cpu
        tensor_tile[i] = band_data

    assert tensor_tile.dim() == 3, f"Expects tensor_tile to has shape (C, H, W), got {tensor_tile.shape}"
    assert tensor_labels.dim() == 2, f"Expects tensor_labels to has shape (H, W), got {tensor_labels.shape}"

    # Create patches
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
    coords = [PatchCoords.from_tensor(tensor_coords[i], patch_size) for i in range(tensor_coords.shape[0])]
    return tensor_patches, tensor_labels, coords


@dataclass
class TrainDatasetBuilder:
    """Helper class to create all necessary files for a DARTS training dataset."""

    train_data_dir: Path
    patch_size: int
    overlap: int
    bands: Bands
    exclude_nopositive: bool
    exclude_nan: bool
    mask_erosion_size: int
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

    def add_tile(
        self,
        tile: xr.Dataset,
        labels: gpd.GeoDataFrame,
        region: str,
        sample_id: str,
        metadata: dict[str, str] | None = None,
    ):
        """Add a tile to the dataset.

        Args:
            tile (xr.Dataset): The input tile, containing preprocessed, harmonized data.
            labels (gpd.GeoDataFrame): The labels to be used for training.
            region (str): The region of the tile.
            sample_id (str): The sample id of the tile.
            metadata (dict[str, str], optional): Any metadata to be added to the metadata file.
                Will not be used for the training, but can be used for better debugging or reproducibility.

        """
        metadata = metadata or {}
        # Convert all paths of metadata to strings
        metadata = {k: str(v) if isinstance(v, Path) else v for k, v in metadata.items()}

        x, y, stacked_coords = create_training_patches(
            tile=tile,
            labels=labels,
            bands=self.bands,
            patch_size=self.patch_size,
            overlap=self.overlap,
            exclude_nopositive=self.exclude_nopositive,
            exclude_nan=self.exclude_nan,
            device=self.device,
            mask_erosion_size=self.mask_erosion_size,
        )

        self._zroot["x"].append(x.numpy().astype("float32"))
        self._zroot["y"].append(y.numpy().astype("uint8"))

        for patch_id, coords in enumerate(stacked_coords):
            geometry = tile.isel(x=coords.x, y=coords.y).odc.geobox.geographic_extent.geom
            self._metadata.append(
                {
                    "patch_id": patch_id,
                    "region": region,
                    "sample_id": sample_id,
                    "empty": not (y == 1).any(),
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
                "mask_erosion_size": self.mask_erosion_size,
                "n_patches": len(metadata),
                "device": self.device,
                **self.bands.to_config(),  # keys: bands, band_factors, band_offsets
                **data_config,
            }
        }
        with open(self.train_data_dir / "config.toml", "w") as f:
            toml.dump(config, f)

        logger.info(f"Saved {len(metadata)} patches to {self.train_data_dir}")
