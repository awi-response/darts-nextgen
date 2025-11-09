"""DARTS v1 ensemble based on two models, one trained with TCVIS data and the other without."""

import logging
from pathlib import Path

import torch
import xarray as xr
from darts_segmentation.segment import SMPSegmenter
from stopuhr import stopwatch

logger = logging.getLogger(__name__.replace("darts_", "darts."))

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EnsembleV1:
    """Model ensemble that averages predictions from multiple segmentation models.

    This class manages multiple trained segmentation models and combines their predictions
    by averaging, providing more robust and stable predictions than any single model.
    It's particularly useful for combining models trained with different data sources
    (e.g., with and without TCVIS data).

    Attributes:
        models (dict[str, SMPSegmenter]): Dictionary mapping model names to loaded segmenters.

    Note:
        The ensemble automatically:
        - Manages multiple model instances with separate configurations
        - Handles band requirements across all models
        - Averages probability predictions (simple arithmetic mean)
        - Optionally preserves individual model outputs for analysis

    Example:
        Create and use an ensemble:

        ```python
        from darts_ensemble import EnsembleV1
        import torch

        # Initialize ensemble with multiple models
        ensemble = EnsembleV1(
            model_dict={
                "with_tcvis": "path/to/model_with_tcvis.ckpt",
                "without_tcvis": "path/to/model_without_tcvis.ckpt",
            },
            device=torch.device("cuda")
        )

        # Check combined band requirements
        print(ensemble.required_bands)
        # {'blue', 'green', 'red', 'nir', 'ndvi', 'tc_brightness', ...}

        # Run ensemble inference
        result = ensemble.segment_tile(
            tile=preprocessed_tile,
            keep_inputs=True  # Keep individual model predictions
        )

        # Access predictions
        ensemble_probs = result["probabilities"]  # Averaged
        model1_probs = result["probabilities-with_tcvis"]  # Individual
        model2_probs = result["probabilities-without_tcvis"]  # Individual
        ```

    """

    def __init__(
        self,
        model_dict,
        device: torch.device = DEFAULT_DEVICE,
    ):
        """Initialize the ensemble with multiple model checkpoints.

        Args:
            model_dict (dict[str, str | Path]): Mapping of model identifiers to checkpoint paths.
                Keys are used to name individual model outputs (e.g., "with_tcvis", "without_tcvis").
                Values are paths to model checkpoint files.
            device (torch.device, optional): Device to load all models on.
                Defaults to CUDA if available, else CPU.

        Note:
            All models are loaded on the same device. For multi-GPU ensembles, instantiate
            separate EnsembleV1 objects per device.

        """
        model_paths = {k: Path(v) for k, v in model_dict.items()}
        logger.debug(
            "Loading models:\n" + "\n".join([f" - {k.upper()} model: {v.resolve()}" for k, v in model_paths.items()])
        )
        self.models = {k: SMPSegmenter(v, device=device) for k, v in model_paths.items()}

    @property
    def model_names(self) -> list[str]:
        """The names of the models in this ensemble."""
        return list(self.models.keys())

    @property
    def required_bands(self) -> set[str]:
        """The combined bands required by all models in this ensemble."""
        bands = set()
        for model in self.models.values():
            bands.update(model.required_bands)
        return bands

    @stopwatch.f(
        "Ensemble inference",
        printer=logger.debug,
        print_kwargs=["patch_size", "overlap", "batch_size", "reflection", "keep_inputs"],
    )
    def segment_tile(
        self,
        tile: xr.Dataset,
        patch_size: int = 1024,
        overlap: int = 16,
        batch_size: int = 8,
        reflection: int = 0,
        keep_inputs: bool = False,
    ) -> xr.Dataset:
        """Run ensemble inference on a single tile by averaging multiple model predictions.

        Each model in the ensemble processes the tile independently, then predictions are
        combined by simple arithmetic averaging to produce the final ensemble prediction.

        Args:
            tile (xr.Dataset): Input tile containing preprocessed data. Must include all bands
                required by any model in the ensemble (union of all `required_bands`).
            patch_size (int, optional): Size of square patches for inference in pixels.
                Defaults to 1024.
            overlap (int, optional): Overlap between adjacent patches in pixels. Defaults to 16.
            batch_size (int, optional): Number of patches to process simultaneously per model.
                Defaults to 8.
            reflection (int, optional): Reflection padding applied to tile edges in pixels.
                Defaults to 0.
            keep_inputs (bool, optional): If True, preserves individual model predictions as
                separate variables (e.g., "probabilities-with_tcvis"). Defaults to False.

        Returns:
            xr.Dataset: Input tile augmented with:
                - probabilities (float32): Ensemble-averaged predictions in range [0, 1].
                  Attributes: long_name="Probabilities"
                - probabilities-{model_name} (float32): Individual model predictions
                  (only if keep_inputs=True)

        Note:
            Averaging method: Simple arithmetic mean across all models. For N models:
            ensemble_prob = (prob_1 + prob_2 + ... + prob_N) / N

            This approach assumes equal confidence in all models. Consider weighted averaging
            if models have different validation performances.

        Example:
            Run ensemble with analysis of individual models:

            ```python
            result = ensemble.segment_tile(
                tile=preprocessed_tile,
                patch_size=1024,
                overlap=16,
                keep_inputs=True  # Keep individual predictions
            )

            # Compare ensemble vs individual models
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 3)
            result["probabilities"].plot(ax=axes[0], title="Ensemble")
            result["probabilities-with_tcvis"].plot(ax=axes[1], title="Model 1")
            result["probabilities-without_tcvis"].plot(ax=axes[2], title="Model 2")
            ```

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
