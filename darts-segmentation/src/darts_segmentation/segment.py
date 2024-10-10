"""Functionality for segmenting tiles."""

from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import xarray as xr

from .hardcoded_stuff import NORMALIZATION_FACTORS


class Segmenter:
    """An actor that keeps a model as its state and segments tiles."""

    config: dict
    model: nn.Module

    def __init__(self, model_checkpoint: Path):
        """Initialize the segmenter."""
        self.dev = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
        ckpt = torch.load(model_checkpoint, map_location=self.dev)
        self.config = ckpt["config"]
        self.model = smp.create_model(**self.config["model"], encoder_weights=None)
        self.model.load_state_dict(ckpt["statedict"])

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

    def segment_tile(self, tile: xr.Dataset) -> xr.Dataset:
        """Run inference on a tile.

        Args:
          tile: The input tile, containing preprocessed, harmonized data.

        Returns:
          Input tile augmented by a predicted `probabilities` layer of type uint8 and a `binarized` layer of type bool.

        """
        tensor_tile = self.tile2tensor(tile)

        predictions = tile["ndvi"].copy(data=tensor_tile[:1].numpy())
        # TODO: Missing implementation
        tile["probabilities"] = (predictions / 255).astype(np.uint8)  # Highly sophisticated DL-based predictor
        tile["binarized"] = tile["ndvi"] > 0.5  # Highly sophisticated DL-based predictor
        return tile
