"""Core inference script for Sentinel-2 superresolution.

This is the practical end-to-end entrypoint for running the current inference
pipeline, keeping the code close to the deployment behavior while remaining
simple enough to rename or split later.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from fractions import Fraction
import logging
import time
from math import floor
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
import xarray as xr

from darts_superresolution.config.config import InferenceConfig
from darts_superresolution.util.upscale import Sentinel2Upscaler
from darts_superresolution.util.patching import create_tile_from_patches

logger = logging.getLogger(__name__)


def load_s2_scene(cfg: InferenceConfig, fpath: str | Path) -> tuple[int, int, int, int, xr.Dataset]:
    """Load a Sentinel-2 tile and return patch-grid dimensions plus dataset."""
    start_time = time.time()
    logger.debug("Loading Sentinel-2 scene from %s", fpath)
    fpath = Path(fpath)

    try:
        s2_image = next(fpath.glob("*_SR_clip.tif"))
    except StopIteration as exc:
        raise FileNotFoundError(f"No matching TIFF files found in {fpath} (.glob('*_SR_clip.tif'))") from exc

    s2_da = xr.open_dataarray(s2_image)
    bands = {1: "blue", 2: "green", 3: "red", 4: "nir"}

    datasets = [
        s2_da.sel(band=index)
        .assign_attrs({"data_source": "s2", "long_name": f"Sentinel 2 {name.capitalize()}"})
        .to_dataset(name=name)
        .drop_vars("band")
        for index, name in bands.items()
    ]
    ds_s2 = xr.merge(datasets)

    size_x = int(ds_s2.sizes["x"])
    size_y = int(ds_s2.sizes["y"])
    pad_x = (
        cfg.patching.patch_stride - (size_x - cfg.patching.input_patch_size) % cfg.patching.patch_stride
    ) % cfg.patching.patch_stride
    pad_y = (
        cfg.patching.patch_stride - (size_y - cfg.patching.input_patch_size) % cfg.patching.patch_stride
    ) % cfg.patching.patch_stride

    num_patches_x = floor((size_x + pad_x - cfg.patching.input_patch_size) / cfg.patching.patch_stride) + 1
    num_patches_y = floor((size_y + pad_y - cfg.patching.input_patch_size) / cfg.patching.patch_stride) + 1

    logger.debug("Loaded Sentinel-2 scene in %.3f s", time.time() - start_time)
    return num_patches_x, num_patches_y, pad_x, pad_y, ds_s2


def _round_scaled(value: int, scale: Fraction) -> int:
    """Round value * scale using exact rational arithmetic."""
    num = value * scale.numerator
    den = scale.denominator
    if num >= 0:
        return (num + den // 2) // den
    return -(((-num) + den // 2) // den)


def _apply_overrides(cfg: InferenceConfig, overrides: dict[str, Any]) -> InferenceConfig:
    cfg = deepcopy(cfg)

    if "backend" in overrides:
        cfg.backend = overrides["backend"]

    if "model_checkpoint" in overrides:
        cfg.paths.model_checkpoint = Path(overrides["model_checkpoint"])
    if "test_scene_dir" in overrides:
        cfg.paths.test_scene_dir = Path(overrides["test_scene_dir"])
    if "output_path" in overrides:
        cfg.paths.output_path = Path(overrides["output_path"])

    if "input_patch_size" in overrides:
        cfg.patching.input_patch_size = int(overrides["input_patch_size"])
    if "output_patch_size" in overrides:
        cfg.patching.output_patch_size = int(overrides["output_patch_size"])
    if "patch_stride" in overrides:
        cfg.patching.patch_stride = int(overrides["patch_stride"])

    if "batch_size" in overrides:
        cfg.runtime.inference_batch_size = int(overrides["batch_size"])
    if "inference_batch_size" in overrides:
        cfg.runtime.inference_batch_size = int(overrides["inference_batch_size"])
    if "input_min_max" in overrides:
        cfg.runtime.inference_input_min_max = overrides["input_min_max"]

    if "use_ddim" in overrides:
        cfg.diffusion.use_ddim = bool(overrides["use_ddim"])
    if "diffusion_steps" in overrides:
        cfg.diffusion.diffusion_steps = int(overrides["diffusion_steps"])
    if "ddim_steps" in overrides:
        cfg.diffusion.ddim_steps = int(overrides["ddim_steps"])
        cfg.diffusion.diffusion_steps = int(overrides["ddim_steps"])
    if "ddim_eta" in overrides:
        cfg.diffusion.ddim_eta = float(overrides["ddim_eta"])

    if "consistency_steps" in overrides:
        cfg.consistency.steps = int(overrides["consistency_steps"])
    if "consistency_use_ema" in overrides:
        cfg.consistency.use_ema = bool(overrides["consistency_use_ema"])
    if "consistency_ensemble_runs" in overrides:
        cfg.consistency.ensemble_runs = int(overrides["consistency_ensemble_runs"])
    if "consistency_repo_root" in overrides:
        repo_root = overrides["consistency_repo_root"]
        cfg.consistency.repo_root = None if repo_root is None else Path(repo_root)

    return cfg


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DARTS superresolution inference")
    parser.add_argument("--backend", choices=["diffusion", "consistency"])

    parser.add_argument("--model-checkpoint", "--model_checkpoint", dest="model_checkpoint")
    parser.add_argument("--test-scene-dir", "--test_scene_dir", dest="test_scene_dir")
    parser.add_argument("--output-path", "--output_path", dest="output_path")

    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int)
    parser.add_argument("--input-patch-size", "--input_patch_size", dest="input_patch_size", type=int)
    parser.add_argument("--output-patch-size", "--output_patch_size", dest="output_patch_size", type=int)
    parser.add_argument("--patch-stride", "--patch_stride", dest="patch_stride", type=int)

    parser.add_argument("--input-min-max", nargs=2, type=float, metavar=("MIN", "MAX"))
    parser.add_argument("--no-input-min-max", action="store_true")

    parser.add_argument("--use-ddim", "--use_ddim", dest="use_ddim", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--diffusion-steps", "--diffusion_steps", dest="diffusion_steps", type=int)
    parser.add_argument("--ddim-steps", "--ddim_steps", dest="ddim_steps", type=int)
    parser.add_argument("--ddim-eta", "--ddim_eta", dest="ddim_eta", type=float)

    parser.add_argument("--consistency-steps", "--consistency_steps", dest="consistency_steps", type=int)
    parser.add_argument(
        "--consistency-use-ema",
        "--consistency_use_ema",
        dest="consistency_use_ema",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--consistency-ensemble-runs",
        "--consistency_ensemble_runs",
        dest="consistency_ensemble_runs",
        type=int,
    )
    parser.add_argument(
        "--consistency-repo-root",
        "--consistency_repo_root",
        dest="consistency_repo_root",
    )
    return parser


def run_inference(cfg: InferenceConfig | None = None, **overrides: Any) -> Path:
    """Run the current inference pipeline and write the reconstructed image."""

    cfg = InferenceConfig() if cfg is None else cfg
    cfg = _apply_overrides(cfg, overrides)

    start_time = time.time()
    num_patches_x, num_patches_y, pad_x, pad_y, img = load_s2_scene(cfg, cfg.paths.test_scene_dir)
    # Stitching uses num_patches_y for output height and num_patches_x for output width.
    # Therefore, height should follow the source y-dimension and width the source x-dimension.
    orig_height = int(img.sizes["y"])
    orig_width = int(img.sizes["x"])
    scale = Fraction(cfg.patching.output_patch_size, cfg.patching.input_patch_size)
    crop_height = _round_scaled(orig_height, scale)
    crop_width = _round_scaled(orig_width, scale)

    logger.info("Patch grid: %s x %s", num_patches_x, num_patches_y)
    logger.info("Original scene size: %s x %s", orig_height, orig_width)

    model = Sentinel2Upscaler(
        cfg.paths.model_checkpoint,
        backend=cfg.backend,
        consistency_steps=cfg.consistency.steps,
        consistency_use_ema=cfg.consistency.use_ema,
        consistency_ensemble_runs=cfg.consistency.ensemble_runs,
        consistency_repo_root=cfg.consistency.repo_root,
        input_patch_size=cfg.patching.input_patch_size,
        output_patch_size=cfg.patching.output_patch_size,
        patch_stride=cfg.patching.patch_stride,
        inference_batch_size=cfg.runtime.inference_batch_size,
        inference_input_min_max=cfg.runtime.inference_input_min_max,
        diffusion_use_ddim=cfg.diffusion.use_ddim,
        diffusion_ddim_steps=(
            cfg.diffusion.ddim_steps
            if cfg.diffusion.ddim_steps is not None
            else cfg.diffusion.diffusion_steps
        ),
        diffusion_ddim_eta=cfg.diffusion.ddim_eta,
    )

    upscaled_image = model.upscale_s2_to_planet(img)
    upscaled_image_no_overlap = create_tile_from_patches(
        upscaled_image,
        4,
        input_patch_size=cfg.patching.input_patch_size,
        output_patch_size=cfg.patching.output_patch_size,
        stride=cfg.patching.patch_stride,
        num_patches_x=num_patches_x,
        num_patches_y=num_patches_y,
        method="crop",
    )

    # Remove mirrored padding introduced in create_patches_from_tile.
    # With method="crop", each patch contributes only its central stride-sized area,
    # equivalent to trimming crop_margin_input pixels from each padded patch border.
    crop_margin_input = (cfg.patching.input_patch_size - cfg.patching.patch_stride) // 2
    # Keep pad-axis mapping consistent with the height/width mapping above.
    top_pad = pad_y // 2
    left_pad = pad_x // 2

    start_h = _round_scaled(top_pad - crop_margin_input, scale)
    start_w = _round_scaled(left_pad - crop_margin_input, scale)

    # Clamp by preserving requested crop size whenever possible.
    max_start_h = max(0, upscaled_image_no_overlap.shape[2] - crop_height)
    max_start_w = max(0, upscaled_image_no_overlap.shape[3] - crop_width)
    start_h = max(0, min(start_h, max_start_h))
    start_w = max(0, min(start_w, max_start_w))
    end_h = start_h + crop_height
    end_w = start_w + crop_width

    upscaled_image_no_overlap = upscaled_image_no_overlap[:, :, start_h:end_h, start_w:end_w]

    logger.info(
        "Pad-aware crop: pad_x=%s pad_y=%s top_pad=%s left_pad=%s start_h=%s start_w=%s end_h=%s end_w=%s",
        pad_x,
        pad_y,
        top_pad,
        left_pad,
        start_h,
        start_w,
        end_h,
        end_w,
    )

    logger.info(
        "Reconstructed image: shape=%s dtype=%s min=%s max=%s",
        upscaled_image_no_overlap.shape,
        upscaled_image_no_overlap.dtype,
        upscaled_image_no_overlap.min(),
        upscaled_image_no_overlap.max(),
    )

    output_u16 = np.clip(np.asarray(upscaled_image_no_overlap), 0, np.iinfo(np.uint16).max).astype(np.uint16)
    end_time = time.time()
    logger.info("Inference and reconstruction completed in %.3f s", end_time - start_time)
    tifffile.imwrite(str(cfg.paths.output_path), output_u16)
    return cfg.paths.output_path


def run_inference_cli(argv: list[str] | None = None) -> Path:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    overrides = {k: v for k, v in vars(args).items() if v is not None and k != "no_input_min_max"}

    if args.no_input_min_max:
        overrides["input_min_max"] = None
    elif args.input_min_max is not None:
        overrides["input_min_max"] = tuple(args.input_min_max)

    return run_inference(**overrides)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    output_path = run_inference_cli()
    print(f"Wrote reconstructed image to {output_path}")
