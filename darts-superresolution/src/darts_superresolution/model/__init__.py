"""Superresolution model package.

This module exposes the modern inference network definition helpers directly,
so callers can import them from `darts_superresolution.model` without going
through a file named `inference.py`.
"""

import logging

import torch
import torch.nn as nn

from config.model_parameters import DEFAULT_MODEL_CONFIG, ModelConfig
from model import unet
from model.diffusion import GaussianDiffusion as GaussianDiffusion

logger = logging.getLogger(__name__)


def define_net(model_opt: ModelConfig = DEFAULT_MODEL_CONFIG, distributed: bool = False):
    """Build the diffusion model used by the current inference pipeline."""

    if ("norm_groups" not in model_opt["unet"]) or model_opt["unet"]["norm_groups"] is None:
        model_opt["unet"]["norm_groups"] = 32

    model = unet.UNet(
        in_channel=model_opt["unet"]["in_channel"],
        out_channel=model_opt["unet"]["out_channel"],
        norm_groups=model_opt["unet"]["norm_groups"],
        inner_channel=model_opt["unet"]["inner_channel"],
        channel_mults=model_opt["unet"]["channel_multiplier"],
        attn_res=model_opt["unet"]["attn_res"],
        res_blocks=model_opt["unet"]["res_blocks"],
        dropout=model_opt["unet"]["dropout"],
        image_size=model_opt["diffusion"]["image_size"],
    )
    net = GaussianDiffusion(
        model,
        image_size=model_opt["diffusion"]["image_size"],
        channels=model_opt["diffusion"]["channels"],
        loss_type="l1",  # L1 or L2
        conditional=model_opt["diffusion"]["conditional"],
        schedule_opt=model_opt["beta_schedule"]["val"],
    )

    if distributed:
        logger.debug("Using nn.DataParallel for superresolution model.")
        assert torch.cuda.is_available()
        net = nn.DataParallel(net)

    return net
