from __future__ import annotations

from dataclasses import field, make_dataclass
from typing import Any, TypedDict

# Runtime inference defaults
DEFAULT_INFERENCE_BATCH_SIZE = 24


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
        "val": {"schedule": "linear", "n_timestep": 200, "linear_start": 1e-6, "linear_end": 1e-2},
    },
    "diffusion": {"image_size": 384, "channels": 4, "conditional": True},
}


def build_consistency_config_class():
    """Build the lightweight config object expected by the consistency checkpoint loader."""
    unet_config = {
        **DEFAULT_MODEL_CONFIG["unet"],
        "beta_schedule": DEFAULT_MODEL_CONFIG["beta_schedule"],
        "diffusion": DEFAULT_MODEL_CONFIG["diffusion"],
    }

    return make_dataclass(
        "ConsistencyConfig",
        [
            ("sample_dimension", tuple[int | None, int | None], field(default=(None, None))),
            ("unet", dict[str, Any], field(default_factory=lambda: dict(unet_config))),
            ("use_regularization", bool, field(default=False)),
            ("data_std", float, field(default=0.5)),
            ("time_min", float, field(default=0.002)),
            ("time_max", float, field(default=80.0)),
            ("clip_output", bool, field(default=False)),
            ("in_channels", int, field(default=4)),
            ("lr", float, field(default=1e-4)),
        ],
    )


ConsistencyConfig = build_consistency_config_class()