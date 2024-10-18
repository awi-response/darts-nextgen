"""Utility functions for parsing and handling configuration files."""

import logging
import tomllib
from contextlib import suppress

import cyclopts

logger = logging.getLogger(__name__)


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict[str, dict[str, str]]:
    """Flatten a nested dictionary.

    Args:
        d (dict): The dictionary to flatten.
        parent_key (str, optional): The parent key. Defaults to "".
        sep (str, optional): The separator. Defaults to ".".

    Returns:
        dict[str, dict[str, str]]: The flattened dictionary.
            Key is the original key, value is a dictionary with the value and a concatenated key to save parents.

    Examples:
    ```python
    >>> d = {
    >>>     "a": 1,
    >>>     "b": {
    >>>         "c": 2,
    >>>     },
    >>> }
    >>> print(flatten_dict(d))
    {
        "a": {"value": 1, "key": "a"},
        "b.c": {"value": 2, "key": "b.c"},
    }
    ```

    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((k, {"value": v, "key": new_key}))
    return dict(items)


def config_parser(
    apps: list[cyclopts.App], commands: tuple[str, ...], mapping: dict[str, cyclopts.config.Unset | list[str]]
):
    """Parser for cyclopts config. An own implementation is needed to select our own toml structure.

    First, the configuration file at "config.toml" is loaded.
    Then, this config is flattened and then mapped to the input arguments of the called function.
    Hence parent keys are not considered.

    Args:
        apps (list[cyclopts.App]): The cyclopts apps.
        commands (tuple[str, ...]): The commands.
        mapping (dict[str, cyclopts.config.Unset | list[str]]): The mapping of the arguments.

    Examples:
        Config file `./config.toml`:

        ```toml
        [darts.hello] # The parent key is completely ignored
        name = "Tobias"
        ```

        Function signature which is called:

        ```python
        # ... setup code for cyclopts
        @app.command()
        def hello(name: str):
            print(f"Hello {name}")
        ```

        Calling the function from CLI:

        ```sh
        $ darts hello
        Hello Tobias

        $ darts hello --name=Max
        Hello Max
        ```

    """
    with open("config.toml", "rb") as f:
        config_data: dict = tomllib.load(f)["darts"]

    # Flatten the config data ()
    flat_config = flatten_dict(config_data)

    for key, value in mapping.items():
        if not isinstance(value, cyclopts.config.Unset) or value.related_set(mapping):
            continue

        with suppress(KeyError):
            new_value = flat_config[key]["value"]
            parent_key = flat_config[key]["key"]
            if not isinstance(new_value, list):
                new_value = [new_value]
            mapping[key] = [str(x) for x in new_value]
            logger.debug(f"Set cyclopts parameter '{key}' to {new_value} from 'config:{parent_key}' ")
