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
    lckpt = torch.load(lightning_checkpoint, weights_only=False, map_location=torch.device("cpu"))

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


def suggest_optuna_params_from_wandb_config(trial, config: dict):
    """Get optuna parameters from a wandb sweep config.

    This functions translate a wandb sweep config to a dict of values, suggested from optuna.

    Args:
        trial (optuna.Trial): The optuna trial
        config (dict): The wandb sweep config

    Raises:
        ValueError: If the distribution is not recognized.

    Returns:
        dict: A dict of parameters with the values suggested from optuna.

    Example:
        Assume a wandb config which looks like this:

        ```yaml
            parameters:
                learning_rate:
                    max: !!float 1e-3
                    min: !!float 1e-7
                    distribution: log_uniform_values
                batch_size:
                    value: 8
                gamma:
                    value: 0.9
                augment:
                    value: True
                model_arch:
                    values:
                        - UnetPlusPlus
                        - Unet
                model_encoder:
                    values:
                        - resnext101_32x8d
                        - resnet101
                        - dpn98

        ```

        This function will return a dict like this:

        ```python
            {
                "learning_rate": trial.suggest_loguniform("learning_rate", 1e-7, 1e-3),
                "batch_size": 8,
                "gamma": 0.9,
                "augment": True,
                "model_arch": trial.suggest_categorical("model_arch", ["UnetPlusPlus", "Unet"]),
                "model_encoder": trial.suggest_categorical(
                    "model_encoder", ["resnext101_32x8d", "resnet101", "dpn98"]
                ),
            }
        ```

        See https://docs.wandb.ai/guides/sweeps/sweep-config-keys for more information on the sweep config.

        Note: Not all distribution types are supported.

    """
    import optuna

    trial: optuna.Trial = trial

    wandb_params: dict[str, dict] = config["parameters"]

    conv = {}
    for param, constrains in wandb_params.items():
        # Handle bad case first: user didn't specified the "distribution key"
        if "distribution" not in constrains.keys():
            if "value" in constrains.keys():
                conv[param] = constrains["value"]
            elif "values" in constrains.keys():
                conv[param] = trial.suggest_categorical(param, constrains["values"])
            elif "min" in constrains.keys() and "max" in constrains.keys():
                conv[param] = trial.suggest_float(param, constrains["min"], constrains["max"])
            else:
                raise ValueError(f"Unknown constrains for parameter {param}")
            continue

        # Now handle the good case where the user specified the distribution
        distribution = constrains["distribution"]
        if distribution == "categorical":
            conv[param] = trial.suggest_categorical(param, constrains["values"])
        elif distribution == "int_uniform":
            conv[param] = trial.suggest_int(param, constrains["min"], constrains["max"])
        elif distribution == "uniform":
            conv[param] = trial.suggest_float(param, constrains["min"], constrains["max"])
        elif distribution == "q_uniform":
            conv[param] = trial.suggest_float(param, constrains["min"], constrains["max"], step=constrains["q"])
        elif distribution == "log_uniform_values":
            conv[param] = trial.suggest_float(param, constrains["min"], constrains["max"], log=True)
        else:
            raise ValueError(f"Unknown distribution {distribution}")

    return conv
