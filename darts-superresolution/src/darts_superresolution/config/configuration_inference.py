from dataclasses import dataclass, field
from typing import List

@dataclass
class InferenceConfig:

    # ── Paths ──────────────────────────────────────────────────────────────────

    # Consistency model checkpoint to evaluate (required)
    consistency_ckpt: str = "/p/scratch/hai_earth_04/lucas/Consistency_Model/checkpoint/consistency_wavelet0.3_image0.7_10steps_no_l1_1.0lpips_bins1.0_continued-epoch=169-val_loss=0.0014.ckpt"

    # Diffusion model checkpoint for comparison (set to None to skip)
    # diffusion_ckpt: str = "/p/scratch/hai_earth_04/lucas/Diffusion_Model/checkpoint/DiffusionWeightedWavelets_bs16_1.5_2.0_2.0_1_cosine_full_T0_1.6AMP_best_continued_3_best_gen.pth"
    # diffusion_ckpt: Optional[str]
    diffusion_ckpt: str = None

    # Test data directories
    lr_path: str = "/p/scratch/hai_earth_04/original_data_from_lucas/Diffusion/Test/sr_60_200"
    hr_path: str = "/p/scratch/hai_earth_04/original_data_from_lucas/Diffusion/Test/hr_200"

    # Where to write SR images and metric CSVs
    output_dir: str = "/p/scratch/hai_earth_04/original_data_from_lucas/Diffusion/Test/results"

    # ── Sampling ───────────────────────────────────────────────────────────────

    # Consistency sampling steps: 1 = single-step (fastest), >1 = multi-step
    n_consistency_steps: int = 1

    # Whether to use EMA weights for consistency model inference
    use_ema: bool = True

    # Number of DDPM steps for the diffusion model (only used if diffusion_ckpt is set)
    n_diffusion_steps: int = 200
    use_ddim: bool = True        # set True to use DDIM instead of DDPM
    ddim_eta: float = 0.0         # 0.0 = deterministic, 1.0 = stochastic

    # ── Data ───────────────────────────────────────────────────────────────────

    # Upsample LR images by this integer scale factor before feeding to the model
    # (bicubic). Set to None if LR and HR are already at the same spatial size.
    lr_scale: int = None

    batch_size: int = 1
    num_workers: int = 1

    # ── Output ─────────────────────────────────────────────────────────────────

    # Save SR .tif outputs to output_dir/consistency_sr and output_dir/diffusion_sr
    save_images: bool = True
