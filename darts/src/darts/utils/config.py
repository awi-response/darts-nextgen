"""Utility functions for parsing and handling configuration files."""

import logging
import tomllib
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
        "c": {"value": 2, "key": "b.c"},
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

        """
        file_path = file_path if isinstance(file_path, Path) else Path(file_path)

        if not file_path.exists():
            logger.warning(f"No config file found at {file_path.resolve()}")
            self._config = {}
            return

        with file_path.open("rb") as f:
            config = tomllib.load(f)["darts"]

        # Flatten the config data ()
        self._config = flatten_dict(config)
        logger.info(f"loaded config from '{file_path.resolve()}'")

    def apply_config(self, arguments: cyclopts.ArgumentCollection):
        """Apply the loaded config to the cyclopts mapping.

        Args:
            arguments (cyclopts.ArgumentCollection): The arguments to apply the config to.

        """
        to_add = []
        for k in self._config.keys():
            value = self._config[k]["value"]

            try:
                argument, remaining_keys, _ = arguments.match(f"--{k}")
            except ValueError:
                # Config key not found in arguments - ignore
                continue

            # Skip if the argument is not bound to a parameter
            if argument.tokens or argument.field_info.kind is argument.field_info.VAR_KEYWORD:
                continue

            # Skip if the argument is from the config file
            if any(x.source != "config-file" for x in argument.tokens):
                continue

            # Parse value to tuple of strings
            if not isinstance(value, list):
                value = (value,)
            value = tuple(str(x) for x in value)
            # Add the new tokens to the list
            for i, v in enumerate(value):
                to_add.append(
                    (
                        argument,
                        cyclopts.Token(keyword=k, value=v, source="config-file", index=i, keys=remaining_keys),
                    )
                )
        # Add here after all "arguments.match" calls, to avoid changing the list while iterating
        for argument, token in to_add:
            argument.append(token)

    def __call__(self, apps: list[cyclopts.App], commands: tuple[str, ...], arguments: cyclopts.ArgumentCollection):
        """Parser for cyclopts config. An own implementation is needed to select our own toml structure.

        First, the configuration file at "config.toml" is loaded.
        Then, this config is flattened and then mapped to the input arguments of the called function.
        Hence parent keys are not considered.

        Args:
            apps (list[cyclopts.App]): The cyclopts apps. Unused, but must be provided for the cyclopts hook.
            commands (tuple[str, ...]): The commands. Unused, but must be provided for the cyclopts hook.
            arguments (cyclopts.ArgumentCollection): The arguments to apply the config to.

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
                command, bound, _ = app.parse_args(tokens)
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

        """
        if self._config is None:
            config_arg, _, _ = arguments.match("--config-file")
            config_file = config_arg.convert_and_validate()
            # Use default config file if not specified
            if not config_file:
                config_file = config_arg.field_info.default
            # else never happens
            self.open_config(config_file)

        self.apply_config(arguments)
