"""Utility functions for legacy training."""

import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def convert_lightning_checkpoint(
    *,
    lightning_checkpoint: Path,
    out_directory: Path,
    checkpoint_name: str,
    framework: str = "smp",
):
    """Convert a lightning checkpoint to our own format.

    The final checkpoint will contain the model configuration and the state dict.
    It will be saved to:

    ```python
        out_directory / f"{checkpoint_name}_{formatted_date}.ckpt"
    ```

    Args:
        lightning_checkpoint (Path): Path to the lightning checkpoint.
        out_directory (Path): Output directory for the converted checkpoint.
        checkpoint_name (str): A unique name of the new checkpoint.
        framework (str, optional): The framework used for the model. Defaults to "smp".

    """
    import torch

    logger.debug(f"Loading checkpoint from {lightning_checkpoint.resolve()}")
    lckpt = torch.load(lightning_checkpoint, weights_only=False)

    now = datetime.now()
    formatted_date = now.strftime("%Y-%m-%d")
    config = lckpt["hyper_parameters"]
    config["time"] = formatted_date
    config["name"] = checkpoint_name
    config["model_framework"] = framework

    own_ckpt = {
        "config": config,
        "statedict": lckpt["state_dict"],
    }

    out_directory.mkdir(exist_ok=True, parents=True)

    out_checkpoint = out_directory / f"{checkpoint_name}_{formatted_date}.ckpt"

    torch.save(own_ckpt, out_checkpoint)

    logger.info(f"Saved converted checkpoint to {out_checkpoint.resolve()}")
