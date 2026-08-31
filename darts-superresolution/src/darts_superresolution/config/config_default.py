from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class InferencePaths:
    model_checkpoint: Path = Path(
        "/home/hgf_gfz/hgf_fkh8398/Model_Checkpoints/DiffusionWeightedWavelets_bs16_1.5_2.0_2.0_1_cosine_full_T0_1.6AMP_best_continued_3_best_gen.pth"
        # "/p/scratch/hai_earth_04/lucas/Consistency_Model/checkpoint/consistency_wavelet_converted.ckpt"
    )
    test_scene_dir: Path = Path(
        "/home/hgf_gfz/hgf_fkh8398/Datasets/SR_Data/sentinel2/20220826T200911_20220826T200905_T17XMJ"
    )
    output_path: Path = Path("/home/hgf_gfz/hgf_fkh8398/test_ddim_darts_recon.tif")


@dataclass
class PatchingConfig:
    input_patch_size: int = 120
    output_patch_size: int = 384
    patch_stride: int = 100


@dataclass
class DiffusionInferenceConfig:
    # Number of inference sampling steps for diffusion backend.
    # When DDIM is enabled, this maps to DDIM steps.
    diffusion_steps: int = 2000
    use_ddim: bool = True
    # Optional legacy alias; if set, it takes precedence over diffusion_steps.
    ddim_steps: int | None = 100
    ddim_eta: float = 0.0
    # Run seeded diffusion/DDIM ensemble and average predictions.
    diffusion_ensemble: bool = False
    diffusion_ensemble_runs: int = 3
    diffusion_ensemble_seed_offset: int = 1
    diffusion_ensemble_space: Literal["image", "wavelet"] = "wavelet"
    # Diffusion output postprocess controls (DDIM-safe defaults).
    diffusion_output_scaling: Literal["fixed", "global_minmax"] = "fixed"
    colorfix: Literal["none", "wavelet", "adain"] = "wavelet"

@dataclass
class ConsistencyInferenceConfig:
    steps: int = 1
    use_ema: bool = True
    ensemble_runs: int = 8
    repo_root: Path | None = None

@dataclass
class RuntimeConfig:
    # Runtime-only inference batch size (single source of truth).
    inference_batch_size: int = 24
    # Single user-facing place to control value normalization before model inference.
    # Passed from infer.py -> upscale.py -> util/patching.py.
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

