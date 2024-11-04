"""DARTS v1 ensemble based on two models, one trained with TCVIS data and the other without."""

from pathlib import Path

import xarray as xr
from darts_segmentation.segment import SMPSegmenter


class EnsembleV1:
    """DARTS v1 ensemble based on two models, one trained with TCVIS data and the other without."""

    def __init__(
        self,
        rts_v6_tcvis_model_path: str | Path,
        rts_v6_notcvis_model_path: str | Path,
    ):
        """Initialize the ensemble.

        Args:
            rts_v6_tcvis_model_path (str | Path): Path to the model trained with TCVIS data.
            rts_v6_notcvis_model_path (str | Path): Path to the model trained without TCVIS data.

        """
        self.rts_v6_tcvis_model = SMPSegmenter(rts_v6_tcvis_model_path)
        self.rts_v6_notcvis_model = SMPSegmenter(rts_v6_notcvis_model_path)

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
        tcvis_probabilities = self.rts_v6_tcvis_model.segment_tile(
            tile, patch_size=patch_size, overlap=overlap, batch_size=batch_size, reflection=reflection
        )["probabilities"].copy()
        notcvis_probabilities = self.rts_v6_notcvis_model.segment_tile(
            tile, patch_size=patch_size, overlap=overlap, batch_size=batch_size, reflection=reflection
        )["probabilities"].copy()

        tile["probabilities"] = (tcvis_probabilities + notcvis_probabilities) / 2

        if keep_inputs:
            tile["probabilities-tcvis"] = tcvis_probabilities
            tile["probabilities-notcvis"] = notcvis_probabilities

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
        for tile in tiles:  # Note that tile is still a reference -> tiles will be changed!
            tcvis_probabilities = self.rts_v6_tcvis_model.segment_tile(
                tile, patch_size=patch_size, overlap=overlap, batch_size=batch_size, reflection=reflection
            )["probabilities"].copy()
            notcvis_propabilities = self.rts_v6_notcvis_model.segment_tile(
                tile, patch_size=patch_size, overlap=overlap, batch_size=batch_size, reflection=reflection
            )["probabilities"].copy()

            tile["probabilities"] = (tcvis_probabilities + notcvis_propabilities) / 2

            if keep_inputs:
                tile["probabilities-tcvis"] = tcvis_probabilities
                tile["probabilities-notcvis"] = notcvis_propabilities

        return tiles

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
            tile (xr.Dataset): Input tile from preprocessing.
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
