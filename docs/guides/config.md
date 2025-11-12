# Config Files

The `darts` CLI support passing parameters via a config file in TOML format.
This can be useful to reduce the amount of parameters you need to pass or to safe different configurations.
In general, the CLI tries to match all parameters under the `darts` key of the config file, skipping not needed ones.

## Example usage

Let's take a closer look with the example command `darts hello`.
This command has the following function signature:

```python
def hello(name: str, n: int = 1):
    """Say hello to someone.

    Args:
        name (str): The name of the person to say hello to
        n (int, optional): The number of times to say hello. Defaults to 1.

    Raises:
        ValueError: If n is 3.

    """
    for i in range(n):
        logger.debug(f"Currently at {i=}")
        if n == 3:
            raise ValueError("I don't like 3")
        logger.info(f"Hello {name}")
```

Let's run the command without making a config file:

```sh
$ uv run darts hello Alice
DEBUG Currently at i=0
INFO Hello Alice
```

Now specify a config file `config.toml`:

```toml
[darts]
name = "Not Alice"
n = 2
```

And run the same command:

```sh
$ uv run darts hello Alice
DEBUG Currently at i=0
INFO Hello Alice
DEBUG Currently at i=1
INFO Hello Alice
```

The `name` parameter is still taken from the CLI, while the `n` parameter is taken from the config file.

Because the CLI utilized a custom TOML parser to parse the config file and pass it to the CLI tool cyclopts, only parameters under the `darts` key are considered.
Subheading keys are not considered, but can be used to structure the config file:

```toml
[darts]
name = "Not Alice"

[darts.numbers]
n = 2
```

The `numbers` key is ignored by the CLI, hence `n` will be add to the command as before.

!!! warning

    The only parameters not passed from the config file are the `--config-file`, `--log-dir`, `--log-plain` and the verbosity parameters.
    These parameters are evaluated before the config file is parsed, hence it is not possible to specify the logging directory via the config file.

## Real world example with Sentinel 2 processing

Sentinel 2 processing via. Area of Interest file:

```toml
[darts]
ee-project = "your-ee-project"
dask-worker = 4

[darts.paths]
input-cache = "./data/cache/s2gee"
output-data-dir = "./data/out"
arcticdem-dir = "./data/datacubes/arcticdem"
tcvis-dir = "./data/datacubes/tcvis"
model-file = "./models/s2-tcvis-final-large_2025-02-12.ckpt"
```

Running the command:

```sh
uv run darts inference sentinel2-sequential --aoi-shapefile path/to/your/aoi.geojson --start-date 2024-07 --end-date 2024-09
```
