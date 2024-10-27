"""Functionality for segmenting tiles."""

import logging
from pathlib import Path
from typing import Any, TypedDict

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import xarray as xr

from darts_segmentation.utils import predict_in_patches

logger = logging.getLogger(__name__)


class SMPSegmenterConfig(TypedDict):
    """Configuration for the segmentor."""

    input_combination: list[str]
    model: dict[str, Any]
    norm_factors: dict[str, float]
    # patch_size: int


def validate_config(config: dict[str, Any]) -> SMPSegmenterConfig:
    """Validate the config for the segmentor.

    Args:
        config: The configuration to validate.

    Returns:
        The validated configuration.

    Raises:
        KeyError: in case the config is missing required keys.

    """
    if "input_combination" not in config:
        raise KeyError("input_combination not found in the config!")
    if "model" not in config:
        raise KeyError("model not found in the config!")
    if "norm_factors" not in config:
        raise KeyError("norm_factors not found in the config!")
    return config


class SMPSegmenter:
    """An actor that keeps a model as its state and segments tiles."""

    config: SMPSegmenterConfig
    model: nn.Module
    device: torch.device

    def __init__(self, model_checkpoint: Path | str):
        """Initialize the segmenter.

        Args:
            model_checkpoint (Path): The path to the model checkpoint.

        """
        self.device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
        ckpt = torch.load(model_checkpoint, map_location=self.device)
        self.config = validate_config(ckpt["config"])
        self.model = smp.create_model(**self.config["model"], encoder_weights=None)
        self.model.to(self.device)
        self.model.load_state_dict(ckpt["statedict"])
        self.model.eval()

    def tile2tensor(self, tile: xr.Dataset) -> torch.Tensor:
        """Take a tile and convert it to a pytorch tensor.

        Respects the input combination from the config.

        Returns:
            A torch tensor for the full tile consisting of the bands specified in `self.band_combination`.

        """
        bands = []
        # e.g. input_combination: ["red", "green", "blue", "relative_elevation", ...]
        # tile.data_vars: ["red", "green", "blue", "relative_elevation", ...]

        for feature_name in self.config["input_combination"]:
            norm = self.config["norm_factors"][feature_name]
            band_data = tile[feature_name]
            # Normalize the band data
            band_data = band_data * norm
            bands.append(torch.from_numpy(band_data.values))

        return torch.stack(bands, dim=0)

    def tile2tensor_batched(self, tiles: list[xr.Dataset]) -> torch.Tensor:
        """Take a list of tiles and convert them to a pytorch tensor.

        Respects the the input combination from the config.

        Returns:
            A torch tensor for the full tile consisting of the bands specified in `self.band_combination`.

        """
        bands = []
        for feature_name in self.config["input_combination"]:
            norm = self.config["norm_factors"][feature_name]
            for tile in tiles:
                band_data = tile[feature_name]
                # Normalize the band data
                band_data = band_data * norm
                bands.append(torch.from_numpy(band_data.values))
        # TODO: Test this
        return torch.stack(bands, dim=0).reshape(len(tiles), len(self.config["input_combination"]), *bands[0].shape)

    def segment_tile(
        self, tile: xr.Dataset, patch_size: int = 1024, overlap: int = 16, batch_size: int = 8, reflection: int = 0
    ) -> xr.Dataset:
        """Run inference on a tile.

        Args:
            tile: The input tile, containing preprocessed, harmonized data.
            patch_size (int): The size of the patches. Defaults to 1024.
            overlap (int): The size of the overlap. Defaults to 16.
            batch_size (int): The batch size for the prediction, NOT the batch_size of input tiles.
            Tensor will be sliced into patches and these again will be infered in batches. Defaults to 8.
            reflection (int): Reflection-Padding which will be applied to the edges of the tensor. Defaults to 0.

        Returns:
            Input tile augmented by a predicted `probabilities` layer with type float32 and range [0, 1].

        """
        # Convert the tile to a tensor
        tensor_tile = self.tile2tensor(tile)

        # Create a batch dimension, because predict expects it
        tensor_tile = tensor_tile.unsqueeze(0)

        probabilities = predict_in_patches(
            self.model, tensor_tile, patch_size, overlap, batch_size, reflection, self.device
        ).squeeze(0)

        # Highly sophisticated DL-based predictor
        # TODO: is there a better way to pass metadata?
        tile["probabilities"] = tile["red"].copy(data=probabilities.cpu().numpy())
        tile["probabilities"].attrs = {
            "long_name": "Probabilities",
        }
        tile["probabilities"] = tile["probabilities"].fillna(float("nan")).rio.write_nodata(float("nan"))
        return tile

    def segment_tile_batched(
        self,
        tiles: list[xr.Dataset],
        patch_size: int = 1024,
        overlap: int = 16,
        batch_size: int = 8,
        reflection: int = 0,
    ) -> list[xr.Dataset]:
        """Run inference on a list of tiles.

        Args:
            tiles: The input tiles, containing preprocessed, harmonized data.
            patch_size (int): The size of the patches. Defaults to 1024.
            overlap (int): The size of the overlap. Defaults to 16.
            batch_size (int): The batch size for the prediction, NOT the batch_size of input tiles.
            Tensor will be sliced into patches and these again will be infered in batches. Defaults to 8.
            reflection (int): Reflection-Padding which will be applied to the edges of the tensor. Defaults to 0.

        Returns:
            A list of input tiles augmented by a predicted `probabilities` layer with type float32 and range [0, 1].

        """
        # Convert the tiles to tensors
        # TODO: maybe create a batched tile2tensor function?
        # tensor_tiles = [self.tile2tensor(tile).to(self.dev) for tile in tiles]
        tensor_tiles = self.tile2tensor_batched(tiles)

        # Create a batch dimension, because predict expects it
        tensor_tiles = torch.stack(tensor_tiles, dim=0)

        probabilities = predict_in_patches(
            self.model, tensor_tiles, patch_size, overlap, batch_size, reflection, self.device
        )

        # Highly sophisticated DL-based predictor
        for tile, probs in zip(tiles, probabilities):
            # TODO: is there a better way to pass metadata?
            tile["probabilities"] = tile["red"].copy(data=probs.cpu().numpy())
            tile["probabilities"].attrs = {
                "long_name": "Probabilities",
            }
            tile["probabilities"] = tile["probabilities"].fillna(float("nan")).rio.write_nodata(float("nan"))
        return tiles

    def __call__(
        self,
        input: xr.Dataset | list[xr.Dataset],
        patch_size: int = 1024,
        overlap: int = 16,
        batch_size: int = 8,
        reflection: int = 0,
    ) -> xr.Dataset | list[xr.Dataset]:
        """Run inference on a single tile or a list of tiles.

        Args:
            input (xr.Dataset | list[xr.Dataset]): A single tile or a list of tiles.
            patch_size (int): The size of the patches. Defaults to 1024.
            overlap (int): The size of the overlap. Defaults to 16.
            batch_size (int): The batch size for the prediction, NOT the batch_size of input tiles.
            Tensor will be sliced into patches and these again will be infered in batches. Defaults to 8.
            reflection (int): Reflection-Padding which will be applied to the edges of the tensor. Defaults to 0.

        Returns:
            A single tile or a list of tiles augmented by a predicted `probabilities` layer, depending on the input.
            Each `probability` has type float32 and range [0, 1].

        Raises:
            ValueError: in case the input is not an xr.Dataset or a list of xr.Dataset

        """
        if isinstance(input, xr.Dataset):
            return self.segment_tile(
                input, patch_size=patch_size, overlap=overlap, batch_size=batch_size, reflection=reflection
            )
        elif isinstance(input, list):
            return self.segment_tile_batched(
                input, patch_size=patch_size, overlap=overlap, batch_size=batch_size, reflection=reflection
            )
        else:
            raise ValueError(f"Expected xr.Dataset or list of xr.Dataset, got {type(input)}")
