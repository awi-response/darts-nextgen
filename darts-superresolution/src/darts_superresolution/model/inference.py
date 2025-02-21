import logging
from typing import TypedDict

import torch
import torch.nn as nn

from darts_superresolution.model.wave_modules import diffusion, unet

logger = logging.getLogger(__name__)


class BetaScheduleTrain(TypedDict):
    schedule: str
    n_timestep: int
    linear_start: float
    linear_end: float


class BetaScheduleVal(TypedDict):
    schedule: str
    n_timestep: int
    linear_start: float
    linear_end: float


class BetaSchedule(TypedDict):
    train: BetaScheduleTrain
    val: BetaScheduleVal


class UNET(TypedDict):
    in_channel: int
    out_channel: int
    inner_channel: int
    norm_groups: int
    channel_multiplier: list[int]
    attn_res: list[int]
    res_blocks: int
    dropout: int


class Diffusion(TypedDict):
    image_size: int
    channels: int
    conditional: bool


class ModelConfig(TypedDict):
    which_model_G: str
    finetune_norm: bool
    unet: UNET
    beta_schedule: BetaSchedule
    diffusion: Diffusion


DEFAULT_MODEL_CONFIG: ModelConfig = {
    "which_model_G": "wave",
    "finetune_norm": False,
    "unet": {
        "in_channel": 32,
        "out_channel": 16,
        "inner_channel": 128,
        "norm_groups": 16,
        "channel_multiplier": [1, 2, 4, 4, 8, 8, 16],
        "attn_res": [16],
        "res_blocks": 3,
        "dropout": 0,
    },
    "beta_schedule": {
        "train": {"schedule": "linear", "n_timestep": 2000, "linear_start": 1e-6, "linear_end": 1e-2},
        "val": {"schedule": "linear", "n_timestep": 2000, "linear_start": 1e-6, "linear_end": 1e-2},
    },
    "diffusion": {"image_size": 192, "channels": 4, "conditional": True},
}


def define_net(model_opt: ModelConfig = DEFAULT_MODEL_CONFIG, distributed: bool = False):
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
    net = diffusion.GaussianDiffusion(
        model,
        image_size=model_opt["diffusion"]["image_size"],
        channels=model_opt["diffusion"]["channels"],
        loss_type="l1",  # L1 or L2
        conditional=model_opt["diffusion"]["conditional"],
        schedule_opt=model_opt["beta_schedule"]["train"],
    )

    if distributed:
        logger.debug("Using nn.DataParallel for superresolution model.")
        assert torch.cuda.is_available()
        net = nn.DataParallel(net)

    return net
