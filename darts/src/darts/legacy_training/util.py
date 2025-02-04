"""Utility functions for legacy training."""

import logging
import secrets
import string
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
    config = lckpt["hyper_parameters"]["config"]
    del config["model"]["encoder_weights"]
    config["time"] = formatted_date
    config["name"] = checkpoint_name
    config["model_framework"] = framework

    statedict = lckpt["state_dict"]
    # Statedict has model. prefix before every weight. We need to remove them. This is an in-place function
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(statedict, "model.")

    own_ckpt = {
        "config": config,
        "statedict": lckpt["state_dict"],
    }

    out_directory.mkdir(exist_ok=True, parents=True)

    out_checkpoint = out_directory / f"{checkpoint_name}_{formatted_date}.ckpt"

    torch.save(own_ckpt, out_checkpoint)

    logger.info(f"Saved converted checkpoint to {out_checkpoint.resolve()}")


def get_value_from_trial(trial, constrains, param: str):
    """Get a value from an optuna trial based on the constrains.

    Args:
        trial (optuna.Trial): The optuna trial
        constrains (dict): The constrains for the parameter
        param (str): The parameter name

    Raises:
        ValueError: Unknown distribution
        ValueError: Unknown constrains

    Returns:
        str | float | int: The value suggested by optuna

    """
    # Handle bad case first: user didn't specified the "distribution key"
    if "distribution" not in constrains.keys():
        if "value" in constrains.keys():
            res = constrains["value"]
        elif "values" in constrains.keys():
            res = trial.suggest_categorical(param, constrains["values"])
        elif "min" in constrains.keys() and "max" in constrains.keys():
            res = trial.suggest_float(param, constrains["min"], constrains["max"])
        else:
            raise ValueError(f"Unknown constrains for parameter {param}")

        return res

    # Now handle the good case where the user specified the distribution
    distribution = constrains["distribution"]
    match distribution:
        case "categorical":
            res = trial.suggest_categorical(param, constrains["values"])
        case "int_uniform":
            res = trial.suggest_int(param, constrains["min"], constrains["max"])
        case "uniform":
            res = trial.suggest_float(param, constrains["min"], constrains["max"])
        case "q_uniform":
            res = trial.suggest_float(param, constrains["min"], constrains["max"], step=constrains["q"])
        case "log_uniform_values":
            res = trial.suggest_float(param, constrains["min"], constrains["max"], log=True)
        case _:
            raise ValueError(f"Unknown distribution {distribution}")

    return res


def suggest_optuna_params_from_wandb_config(trial, config: dict):
    """Get optuna parameters from a wandb sweep config.

    This functions translate a wandb sweep config to a dict of values, suggested from optuna.

    Args:
        trial (optuna.Trial): The optuna trial
        config (dict): The wandb sweep config

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
        conv[param] = get_value_from_trial(trial, constrains, param)
    return conv


def generate_id(length: int = 8) -> str:
    """Generate a random base-36 string of `length` digits.

    This method is taken from the wandb SDK.

    There are ~2.8T base-36 8-digit strings. Generating 210k ids will have a ~1% chance of collision.

    Args:
        length (int, optional): The length of the string. Defaults to 8.

    Returns:
        str: A random base-36 string of `length` digits.

    """
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))
