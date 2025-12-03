"""Hyperparameters for training."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import cyclopts
import toml
import yaml

from darts_segmentation.training.augmentations import Augmentation

if TYPE_CHECKING:
    import scipy.stats

logger = logging.getLogger(__name__.replace("darts_", "darts."))


@cyclopts.Parameter(name="*")
@dataclass(frozen=True)
class Hyperparameters:
    """Hyperparameters for Cyclopts CLI.

    Attributes:
        model_arch (str): Architecture of the model to use.
        model_encoder (str): Encoder type for the model.
        model_encoder_weights (str | None): Weights for the encoder, if any.
        augment (list[Augmentation] | None): List of augmentations to apply.
        learning_rate (float): Learning rate for training.
        gamma (float): Decay factor for learning rate.
        focal_loss_alpha (float | None): Alpha parameter for focal loss, if using.
        focal_loss_gamma (float): Gamma parameter for focal loss.
        batch_size (int): Batch size for training.
        bands (list[str] | None, optional): List of bands to use. Defaults to None.

    """

    # ! Only, single values or lists are supported here.
    # Other values, e.g. dicts would mess with the tuning script, since it appends the hparams to a dataframe.
    model_arch: str = "Unet"
    model_encoder: str = "dpn107"
    model_encoder_weights: str | None = None
    augment: list[Augmentation] | None = None
    learning_rate: float = 1e-3
    gamma: float = 0.9
    focal_loss_alpha: float | None = None
    focal_loss_gamma: float = 2.0
    batch_size: int = 8
    bands: list[str] | None = None  # Maybe this should also be a hyperparameter?


HP_NAMES = [field.name for field in Hyperparameters.__dataclass_fields__.values()]


def parse_hyperparameters(  # noqa: C901
    hpconfig_file: Path,
) -> dict[str, "list | scipy.stats.rv_discrete | scipy.stats.rv_continuous"]:
    """Parse hyperparameter configuration file to a valid dictionary for sklearn parameter search.

    Can be YAML or TOML.
    Must contain a key called "hyperparameters" containing a list of hyperparameters distributions.
    These distributions can either be explicit defined by another dictionary containing a "distribution" key,
    or they can be implicit defined by a single value, a list or a dictionary containing a "low" and "high" key.

    The following distributions are supported:
        - "uniform": Uniform distribution - must have a "low" and "high" value
        - "loguniform": Log-uniform distribution - must have a "low" and "high" value
        - "intuniform": Integer uniform distribution - must have a "low" and "high" value (both are inclusive)
        - "choice": Choice distribution - must have a list of "choices" for explicit case, else just pass a list
        - "value": Fixed value distribution - must have a "value" key for explicit case, else just pass a value

    Examples:
        Explicit Toml:

        ```toml
        [hyperparameters]
        learning_rate = {distribution = "loguniform", low = 1.0e-5, high = 1.0e-2}
        batch_size = {distribution = "choice", choices = [32, 64, 128]}
        gamma = {distribution = "uniform", low = 0.9, high = 2.5}
        dropout = {distribution = "uniform", low = 0.0, high = 0.5}
        layers = {distribution = "intuniform", low = 1, high = 10}
        architecture = {distribution = "constant", value = "resnet"}
        ```

        Explicit YAML:

        ```yaml
        hyperparameters:
            learning_rate:
                distribution: loguniform
                low: 1.0e-5
                high: 1.0e-2
            batch_size:
                distribution: choice
                choices: [32, 64, 128]
            gamma:
                distribution: uniform
                low: 0.9
                high: 2.5
            dropout:
                distribution: uniform
                low: 0.0
                high: 0.5
            layers:
                distribution: intuniform
                low: 1
                high: 10
            architecture:
                distribution: constant
                value: "resnet"
        ```

        Implicit YAML:

        ```yaml
        hyperparameters:
            learning_rate:
                distribution: loguniform
                low: 1.0e-5
                high: 1.0e-2
            batch_size: [32, 64, 128]
            gamma:
                low: 0.9
                high: 2.5
            dropout:
                low: 0.0
                high: 0.5
            layers:
                low: 1
                high: 10
            architecture: "resnet"
        ```

        Will all result in the following dictionary:

        ```py
        {
            "learning_rate": scipy.stats.loguniform(1.0e-5, 1.0e-2),
            "batch_size": [32, 64, 128],
            "gamma": scipy.stats.uniform(0.9, 1.6),
            "dropout": scipy.stats.uniform(0.0, 0.5),
            "layers": scipy.stats.randint(1, 11),
            "architecture": ["resnet"]
        }
        ```

    Args:
        hpconfig_file (Path): Path to the hyperparameter configuration file.

    Returns:
        dict: Dictionary of hyperparameters to tune and their distributions.

    Raises:
        ValueError: If the hyperparameter configuration file is not a valid YAML or TOML file.

    """
    import scipy.stats

    from darts_segmentation.training.reversed_loguniform import reversed_loguniform

    # Read yaml
    if hpconfig_file.suffix == ".yaml" or hpconfig_file.suffix == ".yml":
        with hpconfig_file.open() as f:
            hpconfig = yaml.safe_load(f)["hyperparameters"]
    # Read toml
    elif hpconfig_file.suffix == ".toml":
        with hpconfig_file.open() as f:
            hpconfig = toml.load(f)["hyperparameters"]
    else:
        raise ValueError(f"Invalid hyperparameter configuration file format: {hpconfig.suffix}")

    hpdistributions = {}
    for hparam, config in hpconfig.items():
        if "-" in hparam:
            logger.debug(f"Hyphen in hyperparameter name {hparam} is not supported. Replacing with underscore.")
            hparam = hparam.replace("-", "_")
        # Assume implicit case
        if isinstance(config, list):
            # Choice
            hpdistributions[hparam] = config
            continue
        elif not isinstance(config, dict):
            # Constant
            hpdistributions[hparam] = [config]
            continue
        else:
            if "low" in config.keys() and "high" in config.keys():
                if isinstance(config["low"], int) and isinstance(config["high"], int):
                    # Randint
                    hpdistributions[hparam] = scipy.stats.randint(config["low"], config["high"] + 1)
                    continue
                elif isinstance(config["low"], float) and isinstance(config["high"], float):
                    # Randfloat
                    hpdistributions[hparam] = scipy.stats.uniform(config["low"], config["high"] - config["low"])
                    continue
                else:
                    raise ValueError(
                        f"Invalid hyperparameter configuration for {hparam}: low and high must be of the same type."
                        f" Got {type(config['low'])=} and {type(config['high'])=}."
                    )

        # Now Explicit
        assert isinstance(config, dict), f"Invalid hyperparameter configuration for {hparam}"
        assert "distribution" in config, (
            f"Could not implicitly define distribution for {hparam}."
            " Please provide a distribution type via a 'distribution' key."
        )

        match config["distribution"]:
            case "uniform":
                assert "low" in config, f"Missing 'low' key in hyperparameter configuration for uniform {hparam}"
                assert "high" in config, f"Missing 'high' key in hyperparameter configuration for uniform {hparam}"
                assert isinstance(config["low"], (int, float)), (
                    f"Invalid 'low' value in hyperparameter configuration for uniform {hparam}:"
                    f" got {config['low']} of type {type(config['low'])}"
                )
                assert isinstance(config["high"], (int, float)), (
                    f"Invalid 'high' value in hyperparameter configuration for uniform {hparam}:"
                    f" got {config['high']} of type {type(config['high'])}"
                )
                assert config["low"] < config["high"], (
                    f"'low' value must be less than 'high' value in hyperparameter configuration for uniform {hparam}:"
                    f" got low={config['low']}, high={config['high']}"
                )
                hpdistributions[hparam] = scipy.stats.uniform(config["low"], config["high"] - config["low"])
            case "loguniform":
                assert "low" in config, f"Missing 'low' key in hyperparameter configuration for loguniform {hparam}"
                assert "high" in config, f"Missing 'high' key in hyperparameter configuration for loguniform {hparam}"
                assert isinstance(config["low"], (int, float)), (
                    f"Invalid 'low' value in hyperparameter configuration for loguniform {hparam}:"
                    f" got {config['low']} of type {type(config['low'])}"
                )
                assert isinstance(config["high"], (int, float)), (
                    f"Invalid 'high' value in hyperparameter configuration for loguniform {hparam}:"
                    f" got {config['high']} of type {type(config['high'])}"
                )
                assert config["low"] < config["high"], (
                    f"'low' must be less than 'high' in hyperparameter configuration for loguniform {hparam}:"
                    f" got low={config['low']}, high={config['high']}"
                )
                hpdistributions[hparam] = scipy.stats.loguniform(config["low"], config["high"])
            case "reversed_loguniform":
                assert "low" in config, f"Missing 'low' key in hyperparameter configuration for loguniform {hparam}"
                assert "high" in config, f"Missing 'high' key in hyperparameter configuration for loguniform {hparam}"
                assert isinstance(config["low"], (int, float)), (
                    f"Invalid 'low' value in hyperparameter configuration for loguniform {hparam}:"
                    f" got {config['low']} of type {type(config['low'])}"
                )
                assert isinstance(config["high"], (int, float)), (
                    f"Invalid 'high' value in hyperparameter configuration for loguniform {hparam}:"
                    f" got {config['high']} of type {type(config['high'])}"
                )
                assert config["low"] < config["high"], (
                    f"'low' must be less than 'high' in hyperparameter configuration for loguniform {hparam}:"
                    f" got low={config['low']}, high={config['high']}"
                )
                if "n" in config:
                    assert isinstance(config["n"], (int, float)), (
                        f"Invalid 'n' value in hyperparameter configuration for loguniform {hparam}:"
                        f" got {config['n']} of type {type(config['n'])}"
                    )
                else:
                    config["n"] = 1
                assert config["high"] < config["n"], (
                    f"'high' must be less than 'n' in hyperparameter configuration for loguniform {hparam}:"
                    f" got high={config['high']}, n={config['n']}"
                )
                hpdistributions[hparam] = reversed_loguniform(config["low"], config["high"], config["n"])
            case "intuniform":
                assert "low" in config, f"Missing 'low' key in hyperparameter configuration for int_uniform {hparam}"
                assert "high" in config, f"Missing 'high' key in hyperparameter configuration for int_uniform {hparam}"
                assert isinstance(config["low"], int), (
                    f"'low' must be an integer in hyperparameter configuration for intuniform {hparam}:"
                    f" got {config['low']} of type {type(config['low'])}"
                )
                assert isinstance(config["high"], int), (
                    f"'high' must be an integer in hyperparameter configuration for intuniform {hparam}:"
                    f" got {config['high']} of type {type(config['high'])}"
                )
                assert config["low"] < config["high"], (
                    f"'low' must be less than 'high' in hyperparameter configuration for intuniform {hparam}:"
                    f" got low={config['low']}, high={config['high']}"
                )
                hpdistributions[hparam] = scipy.stats.randint(config["low"], config["high"] + 1)
            case "choice":
                assert "choices" in config, f"Missing 'choices' key in hyperparameter configuration for choice {hparam}"
                hpdistributions[hparam] = config["choices"]
            case "constant":
                assert "value" in config, f"Missing 'value' key in hyperparameter configuration for constant {hparam}"
                hpdistributions[hparam] = [config["value"]]
            case _:
                raise ValueError(f"Invalid hyperparameter type: {config['distribution']}")

    return hpdistributions


def sample_hyperparameters(
    param_grid: dict[str, "list | scipy.stats.rv_discrete | scipy.stats.rv_continuous"],
    n_trials: int | Literal["grid"] = 100,
) -> list[Hyperparameters]:
    """Sample hyperparameters from a parameter grid.

    This function samples a list of hyperparameter combinations from a parameter grid.
    It supports both random sampling and grid search.

    Args:
        param_grid (dict): Dictionary of hyperparameters to tune and their distributions.
            Values can be lists of values or scipy.stats distribution objects.
        n_trials (int | Literal["grid"], optional): Number of hyperparameter combinations to sample.
            If set to "grid", will perform a grid search over all possible combinations.
            Defaults to 100.

    Returns:
        list: List of dictionaries, where each dictionary represents a hyperparameter combination.

    Raises:
        ValueError: If n_trials is not an integer (saying a random search) or 'grid'.

    """
    from sklearn.model_selection import ParameterSampler

    # Check if the parameter of the grid are valid (part of Hyperparameters cclass)
    for hparam in param_grid.keys():
        assert hparam in HP_NAMES, f"Invalid hyperparameter: {hparam} in config but not part of valid {HP_NAMES=}"

    # Random search
    if isinstance(n_trials, int):
        param_list = list(ParameterSampler(param_grid, n_iter=n_trials, random_state=42))
    elif n_trials == "grid":
        n_combinations = 1
        for hparam, choices in param_grid.items():
            assert isinstance(choices, list), (
                f"In a grid search, each parameter must be a list of choices. Got {type(choices)} for {hparam}."
            )
            n_combinations *= len(choices)
        param_list = list(ParameterSampler(param_grid, n_iter=n_combinations, random_state=42))
    else:
        raise ValueError(
            f"Invalid value for n_trials: {n_trials}. Must be an integer to perform a random search or 'grid'."
        )

    # Convert to Hyperparameters objects
    param_list = [Hyperparameters(**params) for params in param_list]

    return param_list
