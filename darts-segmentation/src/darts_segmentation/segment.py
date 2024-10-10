"""Functionality for segmenting tiles."""

from pathlib import Path

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import xarray as xr
import yaml


class Segmenter:
    """An actor that keeps a model as its state and segments tiles."""

    config: dict
    model: nn.Module

    def __init__(self, model_dir: Path):
        """Initialize the segmenter."""
        self.config = yaml.safe_load((model_dir / "config.yaml").open())
        self.model = smp.create_model(**self.config)
        self.dev = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")

        ckpt = "latest"
        if ckpt == "latest":
            ckpt_nums = [int(ckpt.stem) for ckpt in model_dir.glob("checkpoints/*.pt")]
            last_ckpt = max(ckpt_nums)
        else:
            last_ckpt = int(ckpt)
        ckpt = model_dir / "checkpoints" / f"{last_ckpt:02d}.pt"

        # Parallelized Model needs to be declared before loading
        try:
            self.model.load_state_dict(torch.load(ckpt, map_location=self.dev))
        except Exception:
            self.model = nn.DataParallel(self.model)
            self.model.load_state_dict(torch.load(ckpt, map_location=self.dev))

    def segment_tile(self, tile: xr.Dataset) -> xr.Dataset:
        """Run inference on a tile.

        Args:
          tile: The input tile, containing preprocessed, harmonized data.

        Returns:
          Input tile augmented by a predicted `probabilities` layer.

        """
        # TODO: Missing implementation
        tile["probabilities"] = tile["ndvi"]  # Highly sophisticated DL-based predictor
        return tile
