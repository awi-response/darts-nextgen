from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import ray
import typer
import xarray as xr
from typing_extensions import Annotated

app = typer.Typer()


@dataclass
class Tile:
    data: xr.Dataset
    path: str

    def __repr__(self):
        return f"Tile({self.path}, {self.data.dims})"


def open_dataset_ray(row: dict[str, Any]) -> dict[str, Any]:
    data = xr.open_dataset(row["path"])
    tile = Tile(data, row["path"])
    return {
        "tile": tile,
    }


def batch_process(batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    print(batch["tile"])
    # Here we could call our darts-* functions
    return batch


@app.command()
def run(
    data: Annotated[Path, typer.Option(help="Path to the input data directory")] = Path(
        "data"
    ),
):
    ray.init()

    if not data.exists():
        raise FileNotFoundError(f"Data directory {data} not found")

    files = data.glob("*.nc")
    file_list = [f"local:////{file.resolve().absolute()}" for file in files]

    if len(file_list) == 0:
        raise FileNotFoundError(f"No NetCDF files found in {data}")

    ds = ray.data.read_binary_files(file_list, include_paths=True)

    print(ds.schema())
    ds = ds.limit(10)
    print(ds.schema())
    # Opening the files
    ds = ds.map(open_dataset_ray)
    print(ds.schema())
    ds = ds.map_batches(batch_process, batch_size=5)
    print(ds.schema())

    ds = ds.materialize()
    print(ds.schema())
    print(ds)

    # Preprocess the data
    # ds = ds.map(lambda x: preprocess(x))


if __name__ == "__main__":
    app()
