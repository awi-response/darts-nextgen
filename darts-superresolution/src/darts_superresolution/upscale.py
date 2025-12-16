"""AI Superresolution Upscaling of satellite imagery."""

import logging
from pathlib import Path
from typing import Any, TypedDict
import numpy as np
import torch
import xarray as xr
import tifffile
import matplotlib.pyplot as plt
from math import floor
from darts_superresolution.data_processing.patching import create_patches_from_tile
from darts_superresolution.model import GaussianDiffusion, ModelConfig, define_net
from darts_superresolution.data_processing.util import adaptive_instance_normalization, wavelet_color_fix

logger = logging.getLogger(__name__)

BATCH_SIZE = 24
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_MODEL_CONFIG: ModelConfig = {
    "which_model_G": "wave",
    "finetune_norm": False,
    "unet": {
        "in_channel": 32,
        "out_channel": 16,
        "inner_channel": 128,
        "norm_groups": 16,
        "channel_multiplier": [1, 2, 4, 4, 8, 8, 16],
        "attn_res": [16],
        "res_blocks": 3,
        "dropout": 0,
    },
    "beta_schedule": {
        "train": {"schedule": "linear", "n_timestep": 2000, "linear_start": 1e-6, "linear_end": 1e-2},
        "val": {"schedule": "linear", "n_timestep": 100, "linear_start": 1e-6, "linear_end": 1e-2},
    },
    "diffusion": {"image_size": 384, "channels": 4, "conditional": True},#192
}



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
        self.config = DEFAULT_MODEL_CONFIG
        # self.config = ckpt["config"]
        # print("Config: ", self.config)
        logger.debug(f"Loaded config for superresolution model: {self.config}")

        schedule_opt = self.config["beta_schedule"]["val"]
        self.model = define_net(self.config)
        print("config: ", self.config["diffusion"]["image_size"])
        # self.model.set_new_noise_schedule(schedule_opt, device=self.device)

        #print("checkpoint: ", ckpt.keys())
        #statedict = ckpt["statedict"]
        statedict = ckpt
        # print("Statedict: ", statedict)
        if distributed:
            self.model.module.load_state_dict(statedict, strict=False)
        else:
            self.model.load_state_dict(statedict, strict=False)
        self.model.set_new_noise_schedule(schedule_opt, device=self.device)
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

        print("16 bit")

        tile["red"] = tile["red"].astype("float32") / 7248
        tile["green"] = tile["green"].astype("float32") / 7352
        tile["blue"] = tile["blue"].astype("float32") / 7280
        tile["nir"] = tile["nir"].astype("float32") / 6416

        bands = []

        for feature_name in ["red", "green", "blue", "nir"]:
            band_data = tile[feature_name]
            band_data_numpy = band_data.fillna(np.float32(0)).values
            # tifffile.imwrite("/isipd/projects/p_lucas_chamier/192_60/192_60/Input_Channels_{}.tif".format(feature_name), band_data_numpy)
            band_data_torch = torch.from_numpy(band_data_numpy)
            bands.append(band_data_torch)
            # bands.append(torch.from_numpy(band_data.astype("float32").values))
        
        tensor_tile = torch.stack(bands, dim=0)

        # output_tensor = tensor_tile.clone()

        # Create a batch dimension, because predict expects it
        tensor_tile = tensor_tile.unsqueeze(0)

        patches_up, non_zero_upsampled_data, zero_upsampled_data, non_zero_indices, zero_indices = create_patches_from_tile(tensor_tile, stride=120, input_patch_size = 120, output_patch_size=384)
        non_zero_upsampled_data = non_zero_upsampled_data.to(self.device) # Must be inplace?

        ## Here, I find the areas in the patches that are zero across all channels. I want these values to remain zero later on.
        # zero_spatial_mask = (non_zero_upsampled_data == 0).all(dim=1)
        non_zero_upsampled_data.to(self.device)

        output = []

        print("Num batches: ", non_zero_upsampled_data.shape[0])
        print("Full shape: ", non_zero_upsampled_data.shape)
        
        for batch_index in range(0, non_zero_upsampled_data.shape[0], BATCH_SIZE):
            
            with torch.autocast(device_type=str(self.device), dtype=torch.float16):#or torch.float16

                if batch_index + BATCH_SIZE > non_zero_upsampled_data.shape[0]:
                    input_batch = non_zero_upsampled_data[batch_index:non_zero_upsampled_data.shape[0]]
                    #output_batch = [self.model.super_resolution(input_batch, continous=False)]#[-(non_zero_upsampled_data.shape[0]-batch_index):]]
                else:
                    test_input_batch = non_zero_upsampled_data[batch_index:batch_index+BATCH_SIZE].cpu().detach().numpy()
                    # tifffile.imwrite("/isipd/projects/p_lucas_chamier/192_60/192_60/input_batch_"+str(batch_index)+".tif", test_input_batch)
                    input_batch = non_zero_upsampled_data[batch_index:batch_index+BATCH_SIZE]
                
                output_batch = [self.model.super_resolution(input_batch, continous=False)]
                    
                # tifffile.imwrite("/isipd/projects/p_lucas_chamier/192_60/192_60/output_batch_"+str(batch_index)+".tif", output_batch[0].cpu().detach().numpy())

                    # output_batch = output_batch_full[0][-BATCH_SIZE:]
            
            # output_batch = adaptive_instance_normalization(output_batch, input_batch)
            # print(output_batch[0].shape, type(input_batch))
            # print(input_batch.shape)
            output_batch = wavelet_color_fix(output_batch[0], input_batch)
            # tifffile.imwrite("/isipd/projects/p_lucas_chamier/192_60/192_60/output_batch_"+str(batch_index)+".tif", output_batch.cpu().detach().numpy())

            output_batch = [output_batch.cpu().detach()] # with color_fix
            # output_batch = [output_batch[0].cpu().detach()] # without color_fix
            output += output_batch

        # print(len(output), output[0].shape)
        output = torch.cat(output, dim=0)

        print("output shape after cat: ", output.shape)

        output = (output - output.min()) / (output.max() - output.min())
        print("output after cpu: ", output.min(), output.max())

        # to make it 16-bit, uncomment below
        output[:, 0, :, :] = output[:, 0, :, :] * 7248  # (1583 - 31)) + 31
        output[:, 1, :, :] = output[:, 1, :, :] * 7352  # (1973 - 43)) + 43
        output[:, 2, :, :] = output[:, 2, :, :] * 7280  # (2225 - 25)) + 25
        output[:, 3, :, :] = output[:, 3, :, :] * 6416  # (4553 - 83)) + 83

### Ignore
        # zero_mask_broadcasted = zero_spatial_mask.unsqueeze(1).expand_as(output)  # [N1, C, W, H]
        # output[zero_mask_broadcasted] = 0
### Ignore

        patches_up[non_zero_indices] = output

        # to make it 16-bit, uncomment below

        patches_up = patches_up.numpy().astype(np.uint16)
        # patches_up = patches_up.numpy().astype(np.float32)
        return patches_up# output
