from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from darts_superresolution.config.model_parameters import DEFAULT_INFERENCE_BATCH_SIZE


@dataclass
class InferencePaths:
    model_checkpoint: Path = Path(
        "/p/scratch/hai_earth_04/lucas/Diffusion_Model/checkpoint/"
        "DiffusionWeightedWavelets_bs16_1.5_2.0_2.0_1_cosine_750full_T0_1.6AMP_best_gen.pth"
    )
    test_scene_dir: Path = Path(
        "/p/scratch/hai_earth_04/lucas/sentinel2/20220826T200911_20220826T200905_T17XMJ/"
    )
    output_path: Path = Path("/p/scratch/hai_earth_04/lucas/test_consistency_darts_recon.tif")


@dataclass
class PatchingConfig:
    input_patch_size: int = 120
    output_patch_size: int = 384
    patch_stride: int = 110


@dataclass
class DiffusionInferenceConfig:
    use_ddim: bool = True
    ddim_steps: int = 50
    ddim_eta: float = 0.0


@dataclass
class ConsistencyInferenceConfig:
    steps: int = 1
    use_ema: bool = True
    ensemble_runs: int = 8
    repo_root: Path | None = None


@dataclass
class RuntimeConfig:
    inference_batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE
    inference_input_min_max: tuple[float, float] | None = (-1.0, 1.0)


@dataclass
class InferenceConfig:
    """Single source of truth for runtime inference choices and paths."""

    backend: Literal["diffusion", "consistency"] = "diffusion"
    paths: InferencePaths = field(default_factory=InferencePaths)
    patching: PatchingConfig = field(default_factory=PatchingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    diffusion: DiffusionInferenceConfig = field(default_factory=DiffusionInferenceConfig)
    consistency: ConsistencyInferenceConfig = field(default_factory=ConsistencyInferenceConfig)
