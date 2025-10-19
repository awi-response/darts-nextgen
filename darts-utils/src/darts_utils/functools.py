"""Function helpers."""

import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path

import toml
from stopuhr import stopwatch

logger = logging.getLogger(__name__.replace("darts_utils", "darts.shared_utils"))


@stopwatch.f("Save function arguments to config file", printer=logger.debug, print_kwargs=["fpath"])
def write_function_args_to_config_file(
    fpath: Path,
    function: callable,
    locals_: dict,
):
    """Write the arguments of a function.

    Args:
        fpath (Path): Path to the config file
        function (callable): function to get the arguments from
        locals_ (dict): locals() dictionary. Needs to be called in parent function

    """
    nargs = function.__code__.co_argcount + function.__code__.co_kwonlyargcount
    args_ = function.__code__.co_varnames[:nargs]
    config = {k: locals_[k] for k in args_ if k in locals_}
    # Convert everything to toml serializable
    for key, value in config.items():
        if isinstance(value, Path):
            config[key] = str(value.resolve())
        elif isinstance(value, list):
            config[key] = [str(v.resolve()) if isinstance(v, Path) else v for v in value]
        elif is_dataclass(value):
            config[key] = asdict(value)
    with open(fpath, "w") as f:
        toml.dump(config, f)
