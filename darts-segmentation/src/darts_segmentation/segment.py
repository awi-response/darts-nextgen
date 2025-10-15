"""Functionality for segmenting tiles."""

import logging
from pathlib import Path
from typing import Any, TypedDict

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import xarray as xr
from darts_utils.bands import manager
from darts_utils.cuda import free_torch
from stopuhr import stopwatch

from darts_segmentation.inference import predict_in_patches

logger = logging.getLogger(__name__.replace("darts_", "darts."))

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SMPSegmenterConfig(TypedDict):
    """Configuration for the segmentor."""

    model: dict[str, Any]
    bands: list[str]

    @classmethod
    def from_ckpt(cls, ckpt: dict[str, Any]) -> "SMPSegmenterConfig":
        """Load and validate the config from a checkpoint for the segmentor.

        Args:
            ckpt: The checkpoint to load.

        Returns:
            The configuration.

        """
        # Legacy version: config and directly in ckpt
        if "config" in ckpt:
            config = ckpt["config"]
            # Handling legacy case that the config contains the old keys
            if "input_combination" in config and "norm_factors" in config:
                # Check if all input_combination features are in norm_factors
                config["bands"] = config["input_combination"]
                config.pop("norm_factors")
                config.pop("input_combination")
            # Another legacy case uses a deprecated "Bands" class, which is pickled into the config as dict
            if isinstance(config["bands"], dict):
                config["bands"] = config["bands"]["bands"]
        # New version: load directly from lightning checkpoint
        else:
            config = ckpt["hyper_parameters"]["config"]

        assert "model" in config, "Model config is missing!"
        assert "bands" in config, "Bands config is missing!"
        return config


class SMPSegmenter:
    """An actor that keeps a model as its state and segments tiles."""

    config: SMPSegmenterConfig
    model: nn.Module
    device: torch.device

    def __init__(self, model_checkpoint: Path | str, device: torch.device = DEFAULT_DEVICE):
        """Initialize the segmenter.

        Args:
            model_checkpoint (Path): The path to the model checkpoint.
            device (torch.device): The device to run the model on.
                Defaults to torch.device("cuda") if cuda is available, else torch.device("cpu").

        """
        if isinstance(model_checkpoint, str):
            model_checkpoint = Path(model_checkpoint)
        self.device = device
        ckpt = torch.load(model_checkpoint, map_location=self.device, weights_only=False)
        self.config = SMPSegmenterConfig.from_ckpt(ckpt)
        # Overwrite the encoder weights with None, because we load our own
        self.config["model"] |= {"encoder_weights": None}
        self.model = smp.create_model(**self.config["model"])
        self.model.to(self.device)

        # Legacy version
        if "statedict" in ckpt.keys():
            statedict = ckpt["statedict"]
        else:
            statedict = ckpt["state_dict"]
            # Lightning Checkpoints are prefixed with "model." -> we need to remove them. This is an in-place function
            torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(statedict, "model.")
        self.model.load_state_dict(statedict)
        self.model.eval()

        logger.debug(f"Successfully loaded model from {model_checkpoint.resolve()} with inputs: {self.config['bands']}")

    @property
    def required_bands(self) -> set[str]:
        """The bands required by this model."""
        return set(self.config["bands"])

    @stopwatch.f(
        "Segmenting tile",
        printer=logger.debug,
        print_kwargs=["patch_size", "overlap", "batch_size", "reflection"],
    )
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
        tile = tile[self.config["bands"]].transpose("y", "x")
        tile = manager.normalize(tile)
        # ? The heavy operation is .to_dataarray()
        tensor_tile = torch.as_tensor(tile.to_dataarray().data)

        # Create a batch dimension, because predict expects it
        tensor_tile = tensor_tile.unsqueeze(0)

        probabilities = predict_in_patches(
            self.model, tensor_tile, patch_size, overlap, batch_size, reflection, self.device
        ).squeeze(0)

        # Highly sophisticated DL-based predictor
        tile["probabilities"] = (("y", "x"), probabilities.cpu().numpy())
        tile["probabilities"].attrs = {"long_name": "Probabilities"}

        # Cleanup cuda memory
        del tensor_tile, probabilities
        free_torch()

        return tile

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
            return NotImplementedError("Currently passing multiple datasets at once is not supported.")
        else:
            raise ValueError(f"Expected xr.Dataset or list of xr.Dataset, got {type(input)}")
