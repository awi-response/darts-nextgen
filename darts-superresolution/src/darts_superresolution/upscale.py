"""AI Superresolution Upscaling of satellite imagery."""

import logging
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import torch
import xarray as xr

from darts_superresolution.data_processing.patching import create_patches_from_tile
from darts_superresolution.model import GaussianDiffusion, define_net

logger = logging.getLogger(__name__)


class Sentinel2UpscalerConfig(TypedDict):
    """Configuration for the segmentor."""

    model: dict[str, Any]
    phase: str


def validate_config(config: dict[str, Any]) -> Sentinel2UpscalerConfig:
    """Validate the config for the segmentor.

    Args:
        config: The configuration to validate.

    Returns:
        The validated configuration.

    Raises:
        KeyError: in case the config is missing required keys.

    """
    if "model" not in config:
        raise KeyError("model not found in the config!")
    return config


class Sentinel2Upscaler:
    """AI Superresolution Upscaling of satellite imagery."""

    config: Sentinel2UpscalerConfig
    model: GaussianDiffusion  # nn.Module
    device: torch.device

    def __init__(self, model_checkpoint: Path | str) -> None:
        """Initialize the Sentinel2Upscaler.

        Args:
            model_checkpoint (Path | str): The path to the model checkpoint.

        """
        logger.debug(f"Loading model from {model_checkpoint}")
        self.device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
        ckpt = torch.load(model_checkpoint, map_location=self.device)
        self.config = validate_config(ckpt["config"])
        self.config["phase"] = "eval"
        logger.debug(f"Loaded config: {self.config}")
        schedule_opt = self.config["model"]["beta_schedule"]["val"]
        self.model = define_net(self.config)
        self.model.set_new_noise_schedule(schedule_opt, device=self.device)
        self.model.load_state_dict(ckpt["statedict"], strict=False)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def upscale_s2_to_planet(self, tile: xr.Dataset) -> xr.Dataset:
        """Upscale a sentinel 2 satellite imagery from 10m resolution to the 3.125 Planet OrthoTile resolution.

        Args:
            tile (xr.Dataset): The input sentinel 2 tile.

        Returns:
            xr.Dataset: The upscaled sentinel 2 tile.

        """
        # Convert the tile to a tensor
        bands = []
        for feature_name in ["red", "green", "blue", "nir"]:
            band_data = tile[feature_name]
            bands.append(torch.from_numpy(band_data.values))

        tensor_tile = torch.stack(bands, dim=0)

        # Create a batch dimension, because predict expects it
        tensor_tile = tensor_tile.unsqueeze(0)

        upsampled_data = create_patches_from_tile(tensor_tile).to(self.device)
        output = self.model.super_resolution(upsampled_data, continous=True)[-1]  # c, h, w
        output = output.cpu().detach()

        # This rescales the output to 0 to 255 8-bit, from -1 to 1 float
        # This will most likely not be necessary if the image is not saved.
        output = (output - output.min()) / (output.max() - output.min())

        output[0, :, :] = (output[0, :, :] * (1583 - 31)) + 31
        output[1, :, :] = (output[1, :, :] * (1973 - 43)) + 43
        output[2, :, :] = (output[2, :, :] * (2225 - 25)) + 25
        output[3, :, :] = (output[3, :, :] * (4553 - 83)) + 83

        output = output.numpy().astype(np.uint16)
        for i, feature_name in enumerate(["red", "green", "blue", "nir"]):
            tile[feature_name] = tile[feature_name].copy(data=output[i, :, :])

        return output
