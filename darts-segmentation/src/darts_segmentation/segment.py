"""Functionality for segmenting tiles."""

from collections.abc import Generator
from pathlib import Path

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import xarray as xr

from darts_segmentation.hardcoded_stuff import NORMALIZATION_FACTORS


def patch_generator(
    h: int, w: int, patch_size: int, margin_size: int
) -> Generator[tuple[int, int, int, int], None, None]:
    """Yield patch coordinates based on height, width, patch size and margin size.

    Args:
        h (int): Height of the image.
        w (int): Width of the image.
        patch_size (int): Patch size.
        margin_size (int): Margin size.

    Yields:
        tuple[int, int, int, int]: The patch coordinates y, x, patch_idx_h and patch_idx_w.

    """
    step_size = patch_size - margin_size
    for y in range(0, h, step_size):
        for x in range(0, w, step_size):
            if y + patch_size > h:
                y = h - patch_size
            if x + patch_size > w:
                x = w - patch_size
            patch_idx_h = y // step_size
            patch_idx_w = x // step_size
            yield y, x, patch_idx_h, patch_idx_w


def create_patches(tensor_tiles: torch.Tensor, patch_size: int, margin_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Create patches from an input tensor, as well as a trapazoidal soft margin.

    Args:
        tensor_tiles (torch.Tensor): The input tensor. Shape: (BS, C, H, W).
        patch_size (int): The size of the patches.
        margin_size (int): The size of the margin.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The patches tensor and the soft margin tensor.
            The patches tensor has shape (N_h, N_w, BS, C, patch_size, patch_size),
            the soft margin tensor has shape (patch_size, patch_size).

    """
    assert tensor_tiles.dim() == 4, f"Expects tensor_tiles to has shape (BS, C, H, W), got {tensor_tiles.shape}"
    bs, c, h, w = tensor_tiles.shape

    step_size = patch_size - margin_size
    patches = torch.zeros((bs, h // step_size, w // step_size, c, patch_size, patch_size), device=tensor_tiles.device)

    for y, x, patch_idx_h, patch_idx_w in patch_generator(h, w, patch_size, margin_size):
        patches[:, patch_idx_h, patch_idx_w, :] = tensor_tiles[:, :, y : y + patch_size, x : x + patch_size]

    margin_ramp = torch.cat(
        [
            torch.linspace(0, 1, margin_size),
            torch.ones(patch_size - 2 * margin_size),
            torch.linspace(1, 0, margin_size),
        ]
    )

    soft_margin = margin_ramp.reshape(1, 1, patch_size) * margin_ramp.reshape(1, patch_size, 1)
    return patches, soft_margin


class Segmenter:
    """An actor that keeps a model as its state and segments tiles."""

    config: dict
    model: nn.Module

    def __init__(self, model_checkpoint: Path | str):
        """Initialize the segmenter.

        Args:
            model_checkpoint (Path): The path to the model checkpoint.
            device (str, optional): PyTorch device. Defaults to "cpu".

        """
        self.dev = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
        ckpt = torch.load(model_checkpoint, map_location=self.dev)
        self.config = ckpt["config"]
        self.model = smp.create_model(**self.config["model"], encoder_weights=None)
        self.model.load_state_dict(ckpt["statedict"])
        self.model.eval()

    def tile2tensor(self, tile: xr.Dataset) -> torch.Tensor:
        """Take a tile and convert it to a pytorch tensor, according to the input combination from the config.

        Returns:
          A torch tensor for the full tile consisting of the bands specified in `self.band_combination`.

        Raises:
          ValueError: in case a specified band is not found in the input tile.

        """
        bands = []
        for band_name in self.config["input_combination"]:
            for var in tile.data_vars:
                assert isinstance(var, str)
                dim_name = f"{var}_band"
                if band_name in tile[dim_name]:
                    band_data = tile[var].loc[{dim_name: band_name}]
                    band_data = band_data / NORMALIZATION_FACTORS[var]
                    bands.append(torch.from_numpy(band_data.values))
                    break
            else:
                raise ValueError(f"Band {band_name} not found in the input!")
        return torch.stack(bands, dim=0)

    @torch.no_grad()
    def predict(self, tensor_tile: torch.Tensor, patch_size: int = 1024, margin_size: int = 16) -> torch.Tensor:
        """Predict on a tensor.

        Args:
            tensor_tile: The input tensor. Shape: (BS, C, H, W).
            patch_size (int): The size of the patches. Defaults to 1024.
            margin_size (int): The size of the margin. Defaults to 16.

        Returns:
          The predicted tensor.

        """
        # TODO: Maybe move patch_size to the model config?
        bs, c, h, w = tensor_tile.shape

        patches, soft_margin = create_patches(tensor_tile, patch_size, margin_size)

        _, nh, nw, _, _, _ = patches.shape

        # Flatten the patches so they fit to the model
        # (BS, N_h, N_w, C, patch_size, patch_size) -> (BS * N_h * N_w, C, patch_size, patch_size)
        patches = patches.view(bs * nh * nw, c, patch_size, patch_size)

        # Infer logits with model and turn into probabilities with sigmoid
        patched_logits = self.model(patches)
        patched_probabilities = torch.sigmoid(patched_logits)  # TODO: check if this is the correct function

        prediction = torch.zeros(bs, h, w, device=tensor_tile.device)
        weights = torch.zeros(bs, h, w, device=tensor_tile.device)

        for y, x, patch_idx_h, patch_idx_w in patch_generator(h, w, patch_size, margin_size):
            patch = patched_probabilities[patch_idx_h, patch_idx_w]
            prediction[:, y : y + patch_size, x : x + patch_size] += patch * soft_margin
            weights[:, y : y + patch_size, x : x + patch_size] += soft_margin

        # Avoid division by zero
        weights = torch.where(weights == 0, torch.ones_like(weights), weights)
        return prediction / weights

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

        probabilities = self.predict(tensor_tile)

        # Highly sophisticated DL-based predictor
        tile["probabilities"] = tile["ndvi"].copy(data=probabilities.cpu().numpy())
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

        probabilities = self.predict(tensor_tiles)

        # Highly sophisticated DL-based predictor
        for tile, probs in zip(tiles, probabilities):
            tile["probabilities"] = tile["ndvi"].copy(data=probs.cpu().numpy())
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
