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
    """Semantic segmentation model wrapper for RTS detection using Segmentation Models PyTorch.

    This class provides a stateful inference interface for semantic segmentation models trained
    with the DARTS pipeline. It handles model loading, normalization, patch-based inference,
    and memory management.

    Attributes:
        config (SMPSegmenterConfig): Model configuration including architecture and required bands.
        model (nn.Module): The loaded PyTorch segmentation model.
        device (torch.device): Device where the model is loaded (CPU or GPU).

    Note:
        The segmenter automatically:
        - Loads model weights from PyTorch Lightning or legacy checkpoints
        - Normalizes input data using band-specific statistics from darts_utils.bands
        - Handles memory cleanup after inference to prevent GPU memory leaks

    Example:
        Basic segmentation workflow:

        ```python
        from darts_segmentation import SMPSegmenter
        import torch

        # Initialize segmenter
        segmenter = SMPSegmenter(
            model_checkpoint="path/to/model.ckpt",
            device=torch.device("cuda")
        )

        # Check required bands
        print(segmenter.required_bands)
        # {'blue', 'green', 'red', 'nir', 'ndvi', 'slope', 'hillshade', ...}

        # Run inference on preprocessed tile
        result = segmenter.segment_tile(
            tile=preprocessed_tile,
            patch_size=1024,
            overlap=16,
            batch_size=8
        )

        # Access predictions
        probabilities = result["probabilities"]  # float32, range [0, 1]
        ```

    """

    config: SMPSegmenterConfig
    model: nn.Module
    device: torch.device

    def __init__(self, model_checkpoint: Path | str, device: torch.device = DEFAULT_DEVICE):
        """Initialize the segmenter with a trained model checkpoint.

        Args:
            model_checkpoint (Path | str): Path to the model checkpoint file (.ckpt).
                Supports both PyTorch Lightning checkpoints and legacy formats.
            device (torch.device, optional): Device to load the model on.
                Defaults to CUDA if available, else CPU.

        Note:
            The checkpoint must contain:
            - Model architecture configuration (config or hyper_parameters)
            - Trained weights (state_dict or statedict)
            - Required input bands list
            Using lightning checkpoints from our training pipeline is recommended.

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
        """Run semantic segmentation inference on a single tile.

        This method performs patch-based inference with optional overlap and reflection padding
        to handle edge artifacts. The tile is automatically normalized using band-specific
        statistics before inference.

        Args:
            tile (xr.Dataset): Input tile containing preprocessed data. Must include all bands
                specified in `self.required_bands`. Variables should be float32 reflectance
                or normalized feature values.
            patch_size (int, optional): Size of square patches for inference in pixels.
                Larger patches use more memory but may be faster. Defaults to 1024.
            overlap (int, optional): Overlap between adjacent patches in pixels. Helps reduce
                edge artifacts. Defaults to 16.
            batch_size (int, optional): Number of patches to process simultaneously. Higher
                values use more GPU memory but may be faster. Defaults to 8.
            reflection (int, optional): Reflection padding applied to tile edges in pixels.
                Reduces edge effects. Defaults to 0.

        Returns:
            xr.Dataset: Input tile augmented with a new data variable:
                - probabilities (float32): Segmentation probabilities in range [0, 1].
                  Attributes: long_name="Probabilities"

        Note:
            Processing pipeline:
            1. Extract and reorder bands according to model requirements
            2. Normalize using darts_utils.bands.manager
            3. Convert to torch tensor
            4. Run patch-based inference with overlap blending
            5. Convert predictions back to xarray

            Memory management:
            - Automatically frees GPU memory after inference
            - Predictions are moved to CPU before returning

        Example:
            Run inference with custom parameters:

            ```python
            result = segmenter.segment_tile(
                tile=preprocessed_tile,
                patch_size=512,  # Smaller patches for limited GPU memory
                overlap=32,      # More overlap for smoother predictions
                batch_size=4,    # Smaller batches for memory constraints
                reflection=16    # Add padding to reduce edge artifacts
            )

            # Extract probabilities
            probs = result["probabilities"]
            ```

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
