"""DARTS v1 ensemble based on two models, one trained with TCVIS data and the other without."""

import logging
from pathlib import Path

import torch
import xarray as xr
from darts_segmentation.segment import SMPSegmenter

logger = logging.getLogger(__name__.replace("darts_", "darts."))

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EnsembleV1:
    """DARTS v1 ensemble based on a list of models."""

    def __init__(
        self,
        model_dict,
        device: torch.device = DEFAULT_DEVICE,
    ):
        """Initialize the ensemble.

        Args:
            model_dict (dict): The paths to model checkpoints to ensemble, the key is should be a model identifier
                to be written to outputs.
            device (torch.device): The device to run the model on.
                Defaults to torch.device("cuda") if cuda is available, else torch.device("cpu").

        """
        model_paths = {k: Path(v) for k, v in model_dict.items()}
        logger.debug(
            "Loading models:\n" + "\n".join([f" - {k.upper()} model: {v.resolve()}" for k, v in model_paths.items()])
        )
        self.models = {k: SMPSegmenter(v, device=device) for k, v in model_paths.items()}

    def segment_tile(
        self,
        tile: xr.Dataset,
        patch_size: int = 1024,
        overlap: int = 16,
        batch_size: int = 8,
        reflection: int = 0,
        keep_inputs: bool = False,
    ) -> xr.Dataset:
        """Run inference on a tile.

        Args:
            tile: The input tile, containing preprocessed, harmonized data.
            patch_size (int): The size of the patches. Defaults to 1024.
            overlap (int): The size of the overlap. Defaults to 16.
            batch_size (int): The batch size for the prediction, NOT the batch_size of input tiles.
                Tensor will be sliced into patches and these again will be infered in batches. Defaults to 8.
            reflection (int): Reflection-Padding which will be applied to the edges of the tensor. Defaults to 0.
            keep_inputs (bool, optional): Whether to keep the input probabilities in the output. Defaults to False.

        Returns:
            Input tile augmented by a predicted `probabilities` layer with type float32 and range [0, 1].

        """
        probabilities = {}
        for model_name, model in self.models.items():
            probabilities[model_name] = model.segment_tile(
                tile, patch_size=patch_size, overlap=overlap, batch_size=batch_size, reflection=reflection
            )["probabilities"].copy()

        # calculate the mean
        tile["probabilities"] = xr.concat(probabilities.values(), dim="model_probs").mean(dim="model_probs")

        if keep_inputs:
            for k, v in probabilities.items():
                tile[f"probabilities-{k}"] = v

        return tile

    def segment_tile_batched(
        self,
        tiles: list[xr.Dataset],
        patch_size: int = 1024,
        overlap: int = 16,
        batch_size: int = 8,
        reflection: int = 0,
        keep_inputs: bool = False,
    ) -> list[xr.Dataset]:
        """Run inference on a list of tiles.

        Args:
            tiles: The input tiles, containing preprocessed, harmonized data.
            patch_size (int): The size of the patches. Defaults to 1024.
            overlap (int): The size of the overlap. Defaults to 16.
            batch_size (int): The batch size for the prediction, NOT the batch_size of input tiles.
                Tensor will be sliced into patches and these again will be infered in batches. Defaults to 8.
            reflection (int): Reflection-Padding which will be applied to the edges of the tensor. Defaults to 0.
            keep_inputs (bool, optional): Whether to keep the input probabilities in the output. Defaults to False.

        Returns:
            A list of input tiles augmented by a predicted `probabilities` layer with type float32 and range [0, 1].

        """
        return [
            self.segment_tile(
                tile,
                patch_size=patch_size,
                overlap=overlap,
                batch_size=batch_size,
                reflection=reflection,
                keep_inputs=keep_inputs,
            )
            for tile in tiles
        ]

    def __call__(
        self,
        input: xr.Dataset | list[xr.Dataset],
        patch_size: int = 1024,
        overlap: int = 16,
        batch_size: int = 8,
        reflection: int = 0,
        keep_inputs: bool = False,
    ) -> xr.Dataset:
        """Run the ensemble on the given tile.

        Args:
            input (xr.Dataset | list[xr.Dataset]): A single tile or a list of tiles.
            patch_size (int): The size of the patches. Defaults to 1024.
            overlap (int): The size of the overlap. Defaults to 16.
            batch_size (int): The batch size for the prediction, NOT the batch_size of input tiles.
                Tensor will be sliced into patches and these again will be infered in batches. Defaults to 8.
            reflection (int): Reflection-Padding which will be applied to the edges of the tensor. Defaults to 0.
            keep_inputs (bool, optional): Whether to keep the input probabilities in the output. Defaults to False.

        Returns:
            xr.Dataset: Output tile with the ensemble applied.

        Raises:
            ValueError: in case the input is not an xr.Dataset or a list of xr.Dataset

        """
        if isinstance(input, xr.Dataset):
            return self.segment_tile(
                input,
                patch_size=patch_size,
                overlap=overlap,
                batch_size=batch_size,
                reflection=reflection,
                keep_inputs=keep_inputs,
            )
        elif isinstance(input, list):
            return self.segment_tile_batched(
                input,
                patch_size=patch_size,
                overlap=overlap,
                batch_size=batch_size,
                reflection=reflection,
                keep_inputs=keep_inputs,
            )
        else:
            raise ValueError("Input must be an xr.Dataset or a list of xr.Dataset.")
