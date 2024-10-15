"""Functionality for segmenting tiles."""

from pathlib import Path
from typing import Any, TypedDict

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import xarray as xr

from darts_segmentation.utils import predict_in_patches


class SMPSegmenterConfig(TypedDict):
    """Configuration for the segmentor."""

    input_combination: list[str]
    model: dict[str, Any]
    # patch_size: int


class SMPSegmenter:
    """An actor that keeps a model as its state and segments tiles."""

    config: SMPSegmenterConfig
    model: nn.Module

    def __init__(self, model_checkpoint: Path | str):
        """Initialize the segmenter.

        Args:
            model_checkpoint (Path): The path to the model checkpoint.
            device (str, optional): PyTorch device. Defaults to "cpu".

        """
        self.dev = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
        ckpt = torch.load(model_checkpoint, map_location=self.dev)
        assert "input_combination" in ckpt["config"], "input_combination not found in the checkpoint!"
        assert "model" in ckpt["config"], "model not found in the checkpoint!"
        assert "norm_factors" in ckpt["config"], "norm_factors not found in the checkpoint!"
        self.config = ckpt["config"]
        self.model = smp.create_model(**self.config["model"], encoder_weights=None)
        self.model.load_state_dict(ckpt["statedict"])
        self.model.eval()

    def tile2tensor(self, tile: xr.Dataset) -> torch.Tensor:
        """Take a tile and convert it to a pytorch tensor, according to the input combination from the config.

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

    def segment_tile(self, tile: xr.Dataset) -> xr.Dataset:
        """Run inference on a tile.

        Args:
          tile: The input tile, containing preprocessed, harmonized data.

        Returns:
          Input tile augmented by a predicted `probabilities` layer with type float32 and range [0, 1].

        """
        # Convert the tile to a tensor
        tensor_tile = self.tile2tensor(tile).to(self.dev)

        # Create a batch dimension, because predict expects it
        tensor_tile = tensor_tile.unsqueeze(0)

        probabilities = predict_in_patches(self.model, tensor_tile).squeeze(0)

        # Highly sophisticated DL-based predictor
        # TODO: is there a better way to pass metadata?
        tile["probabilities"] = tile["red"].copy(data=probabilities.cpu().numpy())
        tile["probabilities"].attrs = {}
        return tile

    def segment_tile_batched(self, tiles: list[xr.Dataset]) -> list[xr.Dataset]:
        """Run inference on a list of tiles.

        Args:
          tiles: The input tiles, containing preprocessed, harmonized data.

        Returns:
          A list of input tiles augmented by a predicted `probabilities` layer with type float32 and range [0, 1].

        """
        # Convert the tiles to tensors
        # TODO: maybe create a batched tile2tensor function?
        tensor_tiles = [self.tile2tensor(tile).to(self.dev) for tile in tiles]

        # Create a batch dimension, because predict expects it
        tensor_tiles = torch.stack(tensor_tiles, dim=0)

        probabilities = predict_in_patches(self.model, tensor_tiles)

        # Highly sophisticated DL-based predictor
        for tile, probs in zip(tiles, probabilities):
            # TODO: is there a better way to pass metadata?
            tile["probabilities"] = tile["red"].copy(data=probs.cpu().numpy())
            tile["probabilities"].attrs = {}
        return tiles

    def __call__(self, input: xr.Dataset | list[xr.Dataset]) -> xr.Dataset | list[xr.Dataset]:
        """Run inference on a single tile or a list of tiles.

        Args:
          input: A single tile or a list of tiles.

        Returns:
          A single tile or a list of tiles augmented by a predicted `probabilities` layer, depending on the input.
          Each `probability` has type float32 and range [0, 1].

        Raises:
            ValueError: in case the input is not an xr.Dataset or a list of xr.Dataset

        """
        if isinstance(input, xr.Dataset):
            return self.segment_tile(input)
        elif isinstance(input, list):
            return self.segment_tile_batched(input)
        else:
            raise ValueError(f"Expected xr.Dataset or list of xr.Dataset, got {type(input)}")
