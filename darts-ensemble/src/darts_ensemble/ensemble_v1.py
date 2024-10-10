"""DARTS v1 ensemble based on two models, one trained with TCVIS data and the other without."""

from pathlib import Path

import xarray as xr
from darts_segmentation.segment import Segmenter


class EnsembleV1:
    """DARTS v1 ensemble based on two models, one trained with TCVIS data and the other without."""

    def __init__(
        self,
        rts_v6_tcvis_model_path: str | Path,
        rts_v6_notcvis_model_path: str | Path,
        binarize_threshold: float = 0.5,
    ):
        """Initialize the ensemble.

        Args:
            rts_v6_tcvis_model_path (str | Path): Path to the model trained with TCVIS data.
            rts_v6_notcvis_model_path (str | Path): Path to the model trained without TCVIS data.
            binarize_threshold (float, optional): Threshold to binarize the ensemble output. Defaults to 0.5.

        """
        self.rts_v6_tcvis_model = Segmenter(rts_v6_tcvis_model_path)
        self.rts_v6_notcvis_model = Segmenter(rts_v6_notcvis_model_path)
        self.threshold = binarize_threshold

    def __call__(self, tile: xr.Dataset, keep_inputs: bool = False) -> xr.Dataset:
        """Run the ensemble on the given tile.

        Args:
            tile (xr.Dataset): Input tile from preprocessing.
            keep_inputs (bool, optional): Whether to keep the input probabilities in the output. Defaults to False.

        Returns:
            xr.Dataset: Output tile with the ensemble applied.

        """
        tcvis_tile = self.rts_v6_tcvis_model.segment_tile(tile)
        notcvis_tile = self.rts_v6_notcvis_model.segment_tile(tile)

        tile["probabilities"] = (tcvis_tile["probabilities"] + notcvis_tile["probabilities"]) / 2

        if keep_inputs:
            tile["probabilities-tcvis"] = tcvis_tile["probabilities"]
            tile["probabilities-notcvis"] = notcvis_tile["probabilities"]

        return tile
