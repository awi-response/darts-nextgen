"""Utility functions for parsing and handling configuration files."""

import logging
import tomllib
from contextlib import suppress
from pathlib import Path

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


class ConfigParser:
    """Parser for cyclopts config.

    An own implementation is needed to select our own toml structure and source.
    Implemented as a class to be able to provide the config-file as a parameter of the CLI.
    """

    def __init__(self) -> None:
        """Initialize the ConfigParser (no-op)."""
        self._config = None

    def open_config(self, file_path: str | Path) -> None:
        """Open the config file, takes the 'darts' key, flattens the resulting dict and saves as config.

        Args:
            file_path (str | Path): The path to the config file.

        Raises:
            FileNotFoundError: If the file does not exist.

        """
        if isinstance(file_path, str):
            file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Config file '{file_path}' not found.")

        with file_path.open("rb") as f:
            config = tomllib.load(f)["darts"]

        # Flatten the config data ()
        self._config = flatten_dict(config)

    def apply_config(self, mapping: dict[str, cyclopts.config.Unset | list[str]]):
        """Apply the loaded config to the cyclopts mapping.

        Args:
            mapping (dict[str, cyclopts.config.Unset  |  list[str]]): The mapping of the arguments.

        """
        for key, value in mapping.items():
            if not isinstance(value, cyclopts.config.Unset) or value.related_set(mapping):
                continue

            with suppress(KeyError):
                new_value = self._config[key]["value"]
                parent_key = self._config[key]["key"]
                if not isinstance(new_value, list):
                    new_value = [new_value]
                mapping[key] = [str(x) for x in new_value]
                logger.debug(f"Set cyclopts parameter '{key}' to {new_value} from 'config:{parent_key}' ")

    def __call__(
        self, apps: list[cyclopts.App], commands: tuple[str, ...], mapping: dict[str, cyclopts.config.Unset | list[str]]
    ):
        """Parser for cyclopts config. An own implementation is needed to select our own toml structure.

        First, the configuration file at "config.toml" is loaded.
        Then, this config is flattened and then mapped to the input arguments of the called function.
        Hence parent keys are not considered.

        Args:
            apps (list[cyclopts.App]): The cyclopts apps. Unused, but must be provided for the cyclopts hook.
            commands (tuple[str, ...]): The commands. Unused, but must be provided for the cyclopts hook.
            mapping (dict[str, cyclopts.config.Unset | list[str]]): The mapping of the arguments.

        Examples:
            ### Setup the cyclopts App

            ```python
            import cyclopts
            from darts.utils.config import ConfigParser

            config_parser = ConfigParser()
            app = cyclopts.App(config=config_parser)

            # Intercept the logging behavior to add a file handler
            @app.meta.default
            def launcher(
                *tokens: Annotated[str, cyclopts.Parameter(show=False, allow_leading_hyphen=True)],
                log_dir: Path = Path("logs"),
                config_file: Path = Path("config.toml"),
            ):
                command, bound = app.parse_args(tokens)
                add_logging_handlers(command.__name__, console, log_dir)
                return command(*bound.args, **bound.kwargs)

            if __name__ == "__main__":
                app.meta()
            ```


            ### Usage

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

        Raises:
            ValueError: If no config file is specified. Should not occur if the cyclopts App is setup correctly.

        """
        if self._config is None:
            config_param = mapping.get("config-file", None)
            if not config_param:
                raise ValueError("No config file (--config-file) specified.")
            if isinstance(config_param, list):
                config_file = config_param[0]
            elif isinstance(config_param, cyclopts.config.Unset):
                config_file = config_param.iparam.default
            # else never happens
            self.open_config(config_file)

        self.apply_config(mapping)
