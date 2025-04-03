
# Architecture describtion

This repository is a workspace repository, managed by [uv](https://docs.astral.sh/uv/).
Read more about workspaces at the [uv docs](https://docs.astral.sh/uv/concepts/projects/workspaces/).
Each workspace-member starts with `darts-*` and can be seen as an own package or module, except the `darts` directory which is the top-level package.
Each package has it's own internal functions and it's public facing API.
The public facing API of each package MUST follow the [API paradigms](#api-paradigms).

[TOC]

## Package overview

!!! tip "Main design priciple"
    Each package should provide _components_ - stateless functions or stateful classes - which should be then combined either by the top-level `darts` package or by the user.
    Each component should take a Xarray Dataset as input and return a Xarray Dataset as output, with the exception of components of the `darts-aquisition` and `darts-export` packages.
    This way it should be easy to combine different components to build a custom pipeline for different parallelization frameworks and workflows.

| Package Name            | Type     | Description                                                                                       | (Major) Dependencies - all need Xarray |
| ----------------------- | -------- | ------------------------------------------------------------------------------------------------- | -------------------------------------- |
| `darts-acquisition`     | Data     | Fetches data from the data sources and created Xarray Datasets                                    | GEE, rasterio, ODC-Geo                 |
| `darts-preprocessing`   | Data     | Combines Xarray Datasets from different acquisition sources and do some preprocessing on the data | Cupy, Xarray-Spatial                   |
| `darts-superresolution` | Train    | Run a super-resolution model to scale Sentinel 2 images from 10m to 3m resolution                 | PyTorch                                |
| `darts-segmentation`    | Train    | Run the segmentation model                                                                        | PyTorch, segmentation_models_pytorch   |
| `darts-ensemble`        | Ensemble | Ensembles the different models and run the multi-stage inference pipeline.                        | PyTorch                                |
| `darts-postprocessing`  | Data     | Further refines the output from an ensemble or segmentaion and binarizes the probs                | Scipy, Cucim                           |
| `darts-export`          | Data     | Saves the results from inference and combines the result to the final DARTS dataset               | GeoPandas, rasterio                    |
| `darts-utils`           | Data     | Shared utilities for data processing                                                              |                                        |

The packages are currently designed around the [v2 Pipeline](../guides/pipeline-v2.md).

### Conceptual migration from thaw-slump-segmentation

- The `darts-ensemble` and `darts-postprocessing` packages is the successor of the `process-02-inference` and `process-03-ensemble` scripts.
- The `darts-preprocessing` and `darts-acquisition` packages are the successors of the `setup-raw-data` script and manual work of obtaining data.
- The `darts-export` package is splitted from the  `inference` script, should include the previous manual works of combining everything into the final dataset.
- The `darts-superresolution` package is the successor of the `superresolution` repository.
- The `darts-segmentation` package is the successor of the `train` and `prepare_data` script.

The following diagram visualizes how the new packages are meant to work together.
![DARTS nextgen architecture](../assets/darts_nextgen_architecture.png){ loading=lazy }

!!! warning "This is a mock"

    This diagram is not realised in any form. It just exists for demonstrational purposes.
    To see an example of a realized pipeline based on this architecture please see the [Pipeline v2 Guide](../guides/pipeline-v2.md)

### Create a new package

A new package can easily created with:

```py
uv init darts-packagename
```

uv creates a minimal project structure for us.

The following things needs to be done updated and created:

1. The `pyproject.toml` file inside the new package:

   Add to the `pyproject.toml` file inside the new package is the following to enable Ruff:

    ```toml
    [tool.ruff]
    # Extend the `pyproject.toml` file in the parent directory...
    extend = "../pyproject.toml"
    ```

    Please also provide a description and a list of authors to the file.

2. The docs:
    By updating the `notebooks/create_api_docs.ipynb`, running it and updating the `nav` section of the `mkdocs.yml` with the generated text.
    To enable code detection, also add the package directory under `plugins` in the `mkdocs.yml`.

3. The Readme of the package

### Versioning

All packages have at all time the same version.
The versioning is done via git-tags and the [uv dynamic versioning tool](https://github.com/ninoseki/uv-dynamic-versioning).
Hence, the version of the `pyproject.toml` of each subpackage is ignored and has no meaning.

## PyTorch Model checkpoints

Each checkpoint is stored as a torch `.pt` tensor file. The checkpoint MUST have the following structure:

```py
{
    "config": {
        "model_framework": "smp", # Identifier which framework or model was used
        "model": { ... }, # Model specific hyperparameter which are needed to create the model
        "input_combination": [ ... ], # List of strings of the names with which the model was trained, order is important
        "patch_size": 1024, # Patch size on which the model was trained
        ... # More model-framework specific parameter, e.g. normalization method and factors
    },
    "statedict": model.module.state_dict(),
}
```

!!! tip "Pre-Deprecation warning"

    It is planned to switch from our custom structure to huggingface model accessors.

## API paradigms

The packages should pass the data as Xarray Datasets between each other. Datasets can hold coordinate information aswell as other metadata (like CRS) in a single self-describing object.
Since different `tiles` do not share the same coordinates or metadata, each `tile` should be represented by a single Xarray `Dataset`.

- Each public facing API function which in some way **transforms** data should accept a Xarray Dataset as input and return an Xarray Dataset.
  - Data can also be accepted as a list of Xarray Dataset as input and returned as a list of Xarray Datasets for batched processing.
    In this case, concattenation should happend internally and on `numpy` or `pytorch` level, NOT on `xarray` abstraction level.
    The reason behind this it that the tiles don't share their coordinates, resulting in a lot of empty spaces between the tiles and high memory usage.
    The name of the function should then be `function_batched`.
- Each public facing API function which **loads** data should return a single Xarray Dataset for each `tile`.
- Data should NOT be saved to file internally, with `darts-export` as the only exception. Instead, data should returned in-memory as a Xarray Dataset, so the user / pipeline can decide what to save and when.
- Function names should be verbs, e.g. `process`, `ensemble`, `do_inference`.
- If a function is stateless it should NOT be part of a class or wrapper
- If a function is stateful it should be part of a class or wrapper, this is important for Ray
- Each Tile should be represented as a single `xr.Dataset` with each feature / band as `DataVariable`.
- Each DataVariable should have their `data_source` documented in the `attrs`, aswell as `long_name` and `units` if any for plotting.
- A `_FillValue` should also be set for no-data with `.rio.write_nodata("no-data-value")`.

!!! tip "Components"

    The goal of these paradigms is to write functions which work as [Components](../guides/components.md).
    Potential users can then later pick their components and put them together in their custom pipeline, utilizing their own parallelization framework.

### Examples

Here are some examples, how these API paradigms should look like.

!!! warning "This is a mock"

    Even if some real packages are shown, these examples use mock-functions / non-existing functions and will not work .

1. Single transformation

    ```py
    import darts-package
    import xarray as xr

    # User loads / creates the dataset (a single tile) by themself
    ds = xr.open_dataset("...")

    # User calls the function to transform the dataset
    ds = darts-package.transform(ds, **kwargs)

    # User can decide by themself what to do next, e.g. save
    ds.to_netcdf("...")
    ```

2. Batched transformation

    ```py
    import darts_package
    import xarray as xr

    # User loads / creates multiple datasets (hence, multiple tiles) by themself
    data = [xr.open_dataset("..."), xr.open_dataset("..."), ...]

    # User calls the function to transform the dataset
    data = darts_package.transform_batched(data, **kwargs)

    # User can decide by themself what to do next
    data[0].whatever()
    ```

3. Load & preprocess some data

    ```py
    import darts_package

    # User calls the function to transform the dataset
    ds = darts_package.load("path/to/data", **kwargs)

    # User can decide by themself what to do next
    ds.whatever()
    ```

4. Custom pipeline example

    ```py
    from pathlib import Path
    import darts_preprocessing
    import darts_ensemble

    DATA_DIR = Path("./data/")
    MODEL_FILE = Path("./models/model.pt")
    OUT_DIR = Path("./out/")

    # Inference is a stateful transformation, because it needs to load the model
    # Hence, the 
    ensemble = darts_ensemble.EnsembleV1(MODEL_FILE)

    # The data directory contains subfolders which then hold the input data
    for dir in DATA_DIR:
        name = dir.name
        
        # Load the files from the processing directory
        ds = darts_preprocessing.load_and_preprocess(dir)

        # Do the inferencce
        ds = ensemble.inference(ds)

        # Save the results
        ds.to_netcdf(OUT_DIR / f"{name}-result.nc")
    ```

5. Pipeline with Ray

    ```py
    from dataclasses import dataclass
    from pathlib import Path
    import ray
    import darts_preprocess
    import darts_inference
    import darts_export

    DATA_DIR = Path("./data/")
    MODEL_DIR = Path("./models/")
    OUT_DIR = Path("./out/")

    ray.init()

    # We need to wrap the Xarray dataset in a class, so that Ray can serialize it
    @dataclass
    class Tile:
        ds: xr.Dataset

    # Wrapper for ray
    def open_dataset_ray(row: dict[str, Any]) -> dict[str, Any]:
        data = xr.open_dataset(row["path"])
        tile = Tile(data)
        return {
            "input": tile,
        }
    
    # Wrapper for the preprocessing -> Stateless
    def preprocess_tile_ray(row: dict[str, Tile]) -> dict[str, Tile]:
        ds = darts_preprocess.preprocess(row["input"].ds)
        return {
            "preprocessed": Tile(ds),
            "input": row["input"]
        }

    # Wrapper for the inference -> Statefull
    class EnsembleRay:
        def __init__(self):
            self.ensemble = darts_inference.Ensemble.load(MODEL_DIR)

        def __call__(self, row: dict[str, Tile]) -> dict[str, Tile]:
            ds = self.ensemble.inference(row["preprocessed"].ds)
            return {
                "output": Tile(ds),
                "preprocessed": row["preprocessed"],
                "input": row["input"],
            }

    # We need to add 'local:///' to tell ray that we want to use the local filesystem
    files = data.glob("*.nc")
    file_list = [f"local:////{file.resolve().absolute()}" for file in files]

    ds = ray.data.read_binary_files(file_list, include_paths=True)
    ds = ds.map(open_dataset_ray) # Lazy open
    ds = ds.map(preprocess_tile_ray) # Lazy preprocess
    ds = ds.map(EnsembleRay) # Lazy inference

    # Save the results
    for row in ds.iter_rows():
        darts_export.save(row["output"].ds, OUT_DIR / f"{row['input'].ds.name}-result.nc")
    
    ```

### About the Xarray overhead with Ray

Ray expects batched data to be in either numpy or pandas format and can't work with Xarray datasets directly.
Hence, a wrapper with custom stacking functions is needed.
This tradeoff is not small, however, the benefits in terms of maintainability and readability are worth it.

![Xarray overhead with Ray](../assets/xarray_ray_overhead.png){ loading=lazy }
