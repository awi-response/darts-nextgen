"""Core inference script for Sentinel-2 superresolution.

This is the practical end-to-end entrypoint for running the current inference
pipeline, keeping the code close to the deployment behavior while remaining
simple enough to rename or split later.
"""

from __future__ import annotations

from fractions import Fraction
import logging
import time
from math import floor
from pathlib import Path

import numpy as np
import tifffile
import xarray as xr

from darts_superresolution.config.model_defaults import DEFAULT_INFERENCE_BATCH_SIZE
from upscale import Sentinel2Upscaler
from darts_superresolution.data_processing.patching import create_tile_from_patches

logger = logging.getLogger(__name__)

INPUT_PATCH_SIZE = 120
OUTPUT_PATCH_SIZE = 384
PATCH_STRIDE = 100
INFERENCE_BATCH_SIZE = DEFAULT_INFERENCE_BATCH_SIZE
INFERENCE_INPUT_MIN_MAX = None#[-1.0, 1.0]

MODEL_PATH = Path(
    # "/p/scratch/hai_earth_04/lucas/Diffusion_Model/checkpoint/DiffusionWeightedWavelets_bs16_1.5_2.0_2.0_1_cosine_750full_T0_1.6AMP_best_gen.pth"
    "/p/scratch/hai_earth_04/lucas/Consistency_Model/checkpoint/consistency_wavelet_converted.ckpt"
)
TEST_IMG_PATH = Path(
    "/p/scratch/hai_earth_04/lucas/sentinel2/20220826T200911_20220826T200905_T17XMJ/"
)
OUTPUT_PATH = Path("/p/scratch/hai_earth_04/lucas/test_consistency_darts_recon.tif")


def load_s2_scene(fpath: str | Path) -> tuple[int, int, int, int, xr.Dataset]:
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
    pad_x = (PATCH_STRIDE - (size_x - INPUT_PATCH_SIZE) % PATCH_STRIDE) % PATCH_STRIDE
    pad_y = (PATCH_STRIDE - (size_y - INPUT_PATCH_SIZE) % PATCH_STRIDE) % PATCH_STRIDE

    num_patches_x = floor((size_x + pad_x - INPUT_PATCH_SIZE) / PATCH_STRIDE) + 1
    num_patches_y = floor((size_y + pad_y - INPUT_PATCH_SIZE) / PATCH_STRIDE) + 1

    logger.debug("Loaded Sentinel-2 scene in %.3f s", time.time() - start_time)
    return num_patches_x, num_patches_y, pad_x, pad_y, ds_s2


def _round_scaled(value: int, scale: Fraction) -> int:
    """Round value * scale using exact rational arithmetic."""
    num = value * scale.numerator
    den = scale.denominator
    if num >= 0:
        return (num + den // 2) // den
    return -(((-num) + den // 2) // den)


def run_inference() -> Path:
    """Run the current inference pipeline and write the reconstructed image."""

    start_time = time.time()
    num_patches_x, num_patches_y, pad_x, pad_y, img = load_s2_scene(TEST_IMG_PATH)
    # Stitching uses num_patches_y for output height and num_patches_x for output width.
    # Therefore, height should follow the source y-dimension and width the source x-dimension.
    orig_height = int(img.sizes["y"])
    orig_width = int(img.sizes["x"])
    scale = Fraction(OUTPUT_PATCH_SIZE, INPUT_PATCH_SIZE)
    crop_height = _round_scaled(orig_height, scale)
    crop_width = _round_scaled(orig_width, scale)

    logger.info("Patch grid: %s x %s", num_patches_x, num_patches_y)
    logger.info("Original scene size: %s x %s", orig_height, orig_width)

    model = Sentinel2Upscaler(
        MODEL_PATH,
        backend="consistency",
        input_patch_size=INPUT_PATCH_SIZE,
        output_patch_size=OUTPUT_PATCH_SIZE,
        patch_stride=PATCH_STRIDE,
        inference_batch_size=INFERENCE_BATCH_SIZE,
        inference_input_min_max=INFERENCE_INPUT_MIN_MAX,
    )

    upscaled_image = model.upscale_s2_to_planet(img)
    upscaled_image_no_overlap = create_tile_from_patches(
        upscaled_image,
        4,
        input_patch_size=INPUT_PATCH_SIZE,
        output_patch_size=OUTPUT_PATCH_SIZE,
        stride=PATCH_STRIDE,
        num_patches_x=num_patches_x,
        num_patches_y=num_patches_y,
        method="crop",
    )

    # Remove mirrored padding introduced in create_patches_from_tile.
    # With method="crop", each patch contributes only its central stride-sized area,
    # equivalent to trimming crop_margin_input pixels from each padded patch border.
    crop_margin_input = (INPUT_PATCH_SIZE - PATCH_STRIDE) // 2
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
    tifffile.imwrite(str(OUTPUT_PATH), output_u16)
    return OUTPUT_PATH


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    output_path = run_inference()
    print(f"Wrote reconstructed image to {output_path}")
