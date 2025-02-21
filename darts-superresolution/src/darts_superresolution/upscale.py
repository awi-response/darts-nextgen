"""AI Superresolution Upscaling of satellite imagery."""

import logging
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import torch
import xarray as xr

from darts_superresolution.data_processing.patching import create_patches_from_tile
from darts_superresolution.model import GaussianDiffusion, ModelConfig, define_net

logger = logging.getLogger(__name__)


DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Sentinel2UpscalerCheckpoint(TypedDict):
    """Custom generated checkpoint for the network."""

    config: ModelConfig
    statedict: dict[str, Any]


class Sentinel2Upscaler:
    """AI Superresolution Upscaling of satellite imagery."""

    config: ModelConfig
    model: GaussianDiffusion  # nn.Module
    device: torch.device

    def __init__(
        self,
        model_checkpoint: Path | str,
        distributed: bool = False,
        device: torch.device = DEFAULT_DEVICE,
    ) -> None:
        """Initialize the Sentinel2Upscaler.

        Args:
            model_checkpoint (Path | str): The path to the model checkpoint.

        """
        logger.debug(f"Loading model from {model_checkpoint}")
        self.device = device
        ckpt: Sentinel2UpscalerCheckpoint = torch.load(model_checkpoint, map_location=self.device)
        self.config = ckpt["config"]
        logger.debug(f"Loaded config for superresolution model: {self.config}")

        schedule_opt = self.config["beta_schedule"]["val"]
        self.model = define_net(self.config)
        self.model.set_new_noise_schedule(schedule_opt, device=self.device)

        statedict = ckpt["statedict"]
        if distributed:
            self.model.module.load_state_dict(statedict, strict=False)
        else:
            self.model.load_state_dict(statedict, strict=False)

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
        # Rescale from uint16 (0-approx.5000) to float32 (0-1)
        tile = tile.copy(deep=True)

        tile["red"] = tile["red"].astype("float32") / 7248
        tile["green"] = tile["green"].astype("float32") / 7352
        tile["blue"] = tile["blue"].astype("float32") / 7280
        tile["nir"] = tile["nir"].astype("float32") / 6416

        # Convert the tile to a tensor
        bands = []
        for feature_name in ["red", "green", "blue", "nir"]:
            band_data = tile[feature_name]
            bands.append(torch.from_numpy(band_data.astype("float32").values))

        tensor_tile = torch.stack(bands, dim=0)

        # Create a batch dimension, because predict expects it
        tensor_tile = tensor_tile.unsqueeze(0)

        upsampled_data = create_patches_from_tile(tensor_tile).to(self.device)

        output = self.model.super_resolution(upsampled_data, continous=True)[-1]  # c, h, w
        output = output.cpu().detach()

        # This rescales the output to 0 to 255 8-bit, from -1 to 1 float
        # This will most likely not be necessary if the image is not saved.
        output = (output - output.min()) / (output.max() - output.min())

        output[0, :, :] = output[0, :, :] * 7248  # (1583 - 31)) + 31
        output[1, :, :] = output[1, :, :] * 7352  # (1973 - 43)) + 43
        output[2, :, :] = output[2, :, :] * 7280  # (2225 - 25)) + 25
        output[3, :, :] = output[3, :, :] * 6416  # (4553 - 83)) + 83

        output = output.numpy().astype(np.uint16)

        # This bit still throws an error because lr and sr do not match in shape, should be easy to fix.

        # for i, feature_name in enumerate(["red", "green", "blue", "nir"]):
        #     tile[feature_name] = tile[feature_name].copy(data=output[i, :, :])

        return output
