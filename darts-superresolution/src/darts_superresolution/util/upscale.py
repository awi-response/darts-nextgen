"""AI Superresolution Upscaling of satellite imagery."""

import logging
import math
import sys
from pathlib import Path
from typing import Any, Literal, TypedDict

import numpy as np
import torch
import xarray as xr

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    tqdm = None

from darts_superresolution.config.model_parameters import (
    ConsistencyConfig,
    DEFAULT_MODEL_CONFIG,
    ModelConfig,
)
from darts_superresolution.util.patching import create_patches_from_tile
from darts_superresolution.util.util import wavelet_color_fix
from darts_superresolution.model import define_net

logger = logging.getLogger(__name__)

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Sentinel2UpscalerCheckpoint(TypedDict):
    """Custom generated checkpoint for the network."""

    config: ModelConfig
    statedict: dict[str, Any]


class Sentinel2Upscaler:
    """AI Superresolution Upscaling of satellite imagery."""

    config: ModelConfig
    model: Any
    device: torch.device

    def __init__(
        self,
        model_checkpoint: Path | str,
        backend: Literal["diffusion", "consistency"] = "consistency",
        distributed: bool = False,
        device: torch.device = DEFAULT_DEVICE,
        consistency_repo_root: Path | str | None = None,
        consistency_steps: int = 1,
        consistency_use_ema: bool = True,
        consistency_ensemble_runs: int = 8,
        input_patch_size: int = 120,
        output_patch_size: int = 384,
        patch_stride: int = 120,
        inference_input_min_max: tuple[float, float] | None = (-1.0, 1.0),
        inference_batch_size: int = 24,
        diffusion_use_ddim: bool = False,
        diffusion_ddim_steps: int = 50,
        diffusion_ddim_eta: float = 0.0,
    ) -> None:
        """Initialize the Sentinel2Upscaler."""
        logger.debug("Loading model from %s", model_checkpoint)
        self.device = device
        self.backend = backend
        self.consistency_steps = consistency_steps
        self.consistency_use_ema = consistency_use_ema
        self.consistency_ensemble_runs = max(1, consistency_ensemble_runs)
        self.input_patch_size = input_patch_size
        self.output_patch_size = output_patch_size
        self.patch_stride = patch_stride
        self.inference_input_min_max = inference_input_min_max
        self.inference_batch_size = max(1, int(inference_batch_size))
        self.diffusion_use_ddim = bool(diffusion_use_ddim)
        self.diffusion_ddim_steps = max(1, int(diffusion_ddim_steps))
        self.diffusion_ddim_eta = float(diffusion_ddim_eta)
        self.config = DEFAULT_MODEL_CONFIG
        logger.debug("Using backend: %s", self.backend)

        if self.backend == "diffusion":
            self.model = self._load_diffusion_model(model_checkpoint, distributed)
        elif self.backend == "consistency":
            self.model = self._load_consistency_model(model_checkpoint, consistency_repo_root)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def _finalize_model(self, model: Any) -> Any:
        """Move model to target device and set eval mode."""
        model.eval()
        model.to(self.device)
        return model

    def _load_diffusion_model(self, model_checkpoint: Path | str, distributed: bool):
        """Load diffusion checkpoint and configure scheduler for inference."""
        ckpt: Sentinel2UpscalerCheckpoint = torch.load(model_checkpoint, map_location=self.device)
        logger.debug("Loaded shared diffusion config for inference")

        schedule_opt = self.config["beta_schedule"]["val"]
        model = define_net(self.config)

        statedict = ckpt
        if distributed:
            model.module.load_state_dict(statedict, strict=False)
        else:
            model.load_state_dict(statedict, strict=False)
        model.set_new_noise_schedule(schedule_opt, device=self.device)

        return self._finalize_model(model)

    def _resolve_consistency_repo_root(self, consistency_repo_root: Path | str | None) -> None:
        """Optionally prepend consistency repo root to sys.path for legacy module imports."""
        if consistency_repo_root is None:
            return

        repo_root = str(Path(consistency_repo_root).resolve())
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

    def _import_consistency_class(self):
        """Import consistency model lazily so diffusion-only installs still work."""
        try:
            from darts_superresolution.model.consistency import ConsistencyWavelet as Consistency
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Could not import consistency model modules. Set consistency_repo_root to the "
                "consistency repository root so imports like src.consistency_model.* resolve."
            ) from exc

        return Consistency

    def _instantiate_consistency_model(self, Consistency: Any):
        """Create a fresh consistency model instance with inference defaults."""
        return Consistency(
            config=ConsistencyConfig,
            bins_min=10,
            bins_max=150,
            loss_func="HybridWavelet",
            use_ema=True,
        )

    def _extract_state_dict_from_checkpoint(self, ckpt_obj: Any) -> dict[str, Any] | None:
        """Extract state_dict from converted checkpoints or raw state_dict wrappers."""
        if not isinstance(ckpt_obj, dict):
            return None

        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            return ckpt_obj["state_dict"]

        return ckpt_obj

    def _load_consistency_converted(
        self,
        Consistency: Any,
        checkpoint_path: Path,
    ) -> tuple[Any | None, Exception | None]:
        """Try loading tensor-only converted checkpoint. Returns (model, error)."""
        try:
            ckpt_obj = torch.load(str(checkpoint_path), map_location=self.device, weights_only=True)
            state_dict = self._extract_state_dict_from_checkpoint(ckpt_obj)
            if not isinstance(state_dict, dict):
                return None, None

            logger.info("Loading converted consistency checkpoint from %s", checkpoint_path)
            model = self._instantiate_consistency_model(Consistency)
            model.load_state_dict(state_dict, strict=False)
            return self._finalize_model(model), None
        except Exception as exc:  # noqa: BLE001
            logger.info("Could not load as converted checkpoint: %s", exc)
            return None, exc

    def _load_consistency_lightning(self, Consistency: Any, checkpoint_path: Path):
        """Load original PyTorch Lightning consistency checkpoint."""
        torch.serialization.add_safe_globals([ConsistencyConfig])

        logger.info("Loading PyTorch Lightning consistency checkpoint from %s", checkpoint_path)
        model = Consistency.load_from_checkpoint(
            str(checkpoint_path),
            config=ConsistencyConfig,
            bins_min=10,
            bins_max=150,
            use_ema=True,
            map_location=self.device,
            strict=False,
            weights_only=False,
        )

        logger.info(
            "Loaded original checkpoint format; consider converting with "
            "src/darts_superresolution/util/convert_checkpoint.sh"
        )
        return self._finalize_model(model)

    def _load_consistency_model(self, model_checkpoint: Path | str, consistency_repo_root: Path | str | None):
        """Load consistency model from converted or original Lightning checkpoint."""
        self._resolve_consistency_repo_root(consistency_repo_root)
        Consistency = self._import_consistency_class()
        checkpoint_path = Path(model_checkpoint)

        model, converted_load_error = self._load_consistency_converted(Consistency, checkpoint_path)
        if model is not None:
            return model

        if checkpoint_path.name.endswith(".converted.ckpt"):
            raise RuntimeError(
                f"Converted checkpoint could not be loaded: {checkpoint_path}.\n"
                f"Reason: {converted_load_error}\n\n"
                "This usually means the file was converted with the older helper that still "
                "serialized non-tensor metadata. Re-run conversion with the updated helper so "
                "the output contains tensors only."
            ) from converted_load_error

        try:
            return self._load_consistency_lightning(Consistency, checkpoint_path)
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                f"Failed to load checkpoint from {checkpoint_path}.\n"
                f"Original error: {exc}\n\n"
                f"The checkpoint contains references to 'src.*' modules that don't exist in DARTS.\n"
                f"Please convert it using:\n"
                f"  ./src/darts_superresolution/util/convert_checkpoint.sh "
                f"{checkpoint_path} {checkpoint_path}.converted.ckpt\n"
                f"Then use the converted checkpoint."
            ) from exc
        except KeyError as exc:
            raise RuntimeError(
                f"Checkpoint at {checkpoint_path} is not a valid PyTorch Lightning checkpoint.\n"
                f"Original error: {exc}\n\n"
                "If this is a converted checkpoint, re-run conversion with the updated helper so "
                "it is saved as a tensor-only state_dict wrapper."
            ) from exc

    def _infer_batch_consistency_wavelet_ensemble(self, input_batch: torch.Tensor) -> torch.Tensor:
        """Run multiple stochastic consistency samples and average in wavelet space."""
        conditioning_dwt = self.model.apply_dwt(input_batch)

        model_fn = self.model.model_ema if self.consistency_use_ema else self.model.model
        dwsr_fn = self.model.dwsr_ema if self.consistency_use_ema else self.model.dwsr

        dwt_outputs = []
        for run_idx in range(self.consistency_ensemble_runs):
            torch.manual_seed(run_idx)

            x_sr = dwsr_fn(conditioning_dwt)

            model_time = torch.tensor(
                [[self.model.time_max]], device=self.device, dtype=conditioning_dwt.dtype
            ).expand(input_batch.shape[0], 1)

            noise_shape = (
                input_batch.shape[0],
                16,
                conditioning_dwt.shape[2],
                conditioning_dwt.shape[3],
            )
            noise = torch.randn(noise_shape, device=self.device, dtype=conditioning_dwt.dtype)
            noisy_residual = self.model.image_time_product(noise, model_time.squeeze(-1))
            output_dwt = noisy_residual + x_sr

            output_dwt = self.model._forward_conditional(
                model_fn, dwsr_fn, output_dwt, model_time, conditioning_dwt
            )

            if self.consistency_steps > 1:
                _timesteps = torch.linspace(
                    self.model.bins_max - 1,
                    0,
                    self.consistency_steps,
                    device=self.device,
                ).long()
                times_seq = self.model.timesteps_to_times(_timesteps, self.model.bins_max)

                for step_time in times_seq[1:]:
                    step_time_batch = step_time.expand(input_batch.shape[0], 1)
                    noise = torch.randn_like(output_dwt)
                    re_noised = output_dwt + self.model.image_time_product(
                        noise, step_time_batch.squeeze(-1)
                    )
                    output_dwt = self.model._forward_conditional(
                        model_fn, dwsr_fn, re_noised, step_time_batch, conditioning_dwt
                    )

            dwt_outputs.append(output_dwt)

        mean_dwt = torch.stack(dwt_outputs).mean(dim=0)
        _, _, orig_h, orig_w = input_batch.shape
        return self.model.apply_idwt(mean_dwt, orig_h, orig_w)

    @torch.no_grad()
    def _infer_batch(self, input_batch: torch.Tensor) -> torch.Tensor:
        if self.backend == "diffusion":
            return self.model.super_resolution(
                input_batch,
                continous=False,
                use_ddim=self.diffusion_use_ddim,
                ddim_steps=self.diffusion_ddim_steps,
                ddim_eta=self.diffusion_ddim_eta,
            )

        if self.consistency_ensemble_runs > 1:
            return self._infer_batch_consistency_wavelet_ensemble(input_batch)

        _, _, h, w = input_batch.shape
        sr, _ = self.model.sample_conditional(
            conditioning=input_batch,
            x_image_size=h,
            y_image_size=w,
            steps=self.consistency_steps,
            use_ema=self.consistency_use_ema,
        )
        return sr

    @torch.no_grad()
    def upscale_s2_to_planet(self, tile: xr.Dataset) -> xr.Dataset:
        """Upscale a sentinel 2 satellite imagery from 10m resolution to the 3.125 Planet OrthoTile resolution."""
        tile = tile.copy(deep=True)
        logger.debug("Preparing Sentinel-2 bands as normalized float32 inputs")

        tile["red"] = tile["red"].astype("float32") / 7248
        tile["green"] = tile["green"].astype("float32") / 7352
        tile["blue"] = tile["blue"].astype("float32") / 7280
        tile["nir"] = tile["nir"].astype("float32") / 6416

        bands = []
        for feature_name in ["red", "green", "blue", "nir"]:
            band_data = tile[feature_name]
            band_data_numpy = band_data.fillna(np.float32(0)).values
            band_data_torch = torch.from_numpy(band_data_numpy)
            bands.append(band_data_torch)

        tensor_tile = torch.stack(bands, dim=0).unsqueeze(0)

        patches_up, non_zero_upsampled_data, zero_upsampled_data, non_zero_indices, zero_indices = create_patches_from_tile(
            tensor_tile,
            stride=self.patch_stride,
            input_patch_size=self.input_patch_size,
            output_patch_size=self.output_patch_size,
            input_min_max=self.inference_input_min_max,
        )
        non_zero_upsampled_data = non_zero_upsampled_data.to(self.device)

        # Track pure background pixels (all channels == 0) inside non-zero patches.
        zero_spatial_mask = (patches_up[non_zero_indices] == 0).all(dim=1, keepdim=True)

        output = []
        logger.debug("Non-zero upsampled batch count: %s", non_zero_upsampled_data.shape[0])
        logger.debug("Non-zero upsampled tensor shape: %s", non_zero_upsampled_data.shape)
        total_patches = non_zero_upsampled_data.shape[0]
        total_batches = math.ceil(total_patches / self.inference_batch_size) if total_patches > 0 else 0
        batch_starts = range(0, total_patches, self.inference_batch_size)
        if tqdm is not None:
            batch_iterator = tqdm(
                batch_starts,
                total=total_batches,
                desc="Inference",
                unit="batch",
                leave=False,
                disable=total_batches <= 1,
            )
        else:
            batch_iterator = batch_starts

        for batch_index in batch_iterator:
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
                if batch_index + self.inference_batch_size > non_zero_upsampled_data.shape[0]:
                    input_batch = non_zero_upsampled_data[batch_index:non_zero_upsampled_data.shape[0]]
                else:
                    input_batch = non_zero_upsampled_data[batch_index:batch_index + self.inference_batch_size]

                output_batch = [self._infer_batch(input_batch)]

            output_batch = wavelet_color_fix(output_batch[0], input_batch)
            output_batch = [output_batch.cpu().detach()]
            output += output_batch

        output = torch.cat(output, dim=0)
        logger.debug("Output shape after concatenation: %s", output.shape)

        output = (output - output.min()) / (output.max() - output.min())
        logger.debug("Output range after normalization: min=%s max=%s", output.min(), output.max())

        output[:, 0, :, :] = output[:, 0, :, :] * 7248
        output[:, 1, :, :] = output[:, 1, :, :] * 7352
        output[:, 2, :, :] = output[:, 2, :, :] * 7280
        output[:, 3, :, :] = output[:, 3, :, :] * 6416

        output[zero_spatial_mask.expand_as(output)] = 0
        patches_up[non_zero_indices] = output

        return patches_up.numpy().astype(np.uint16)
