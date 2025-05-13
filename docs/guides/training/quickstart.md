# Quickstart Training

In this tutorial, you should be able to quickly setup the training of a segmentation model on the PLANET data.

## 0. Prereq

- Make sure you have installed the package and all dependencies. See the [installation guide](../installation.md) for more information.
- Clone [this repository](https://github.com/initze/ML_training_labels) to obtain the labels for the training data.
- Ask a maintainer for access to the PLANET training data.

## 1. Setup the configuration file

Copy this configuration file to your local machine, e.g. under `configs/planet-training-quickstart.toml`, and adjust

- the paths to your needs
- the account settings of earth engine and wandb

???+ abstract "Configuration file"
    ```toml title="configs/planet-training-quickstart.toml"
    [darts.wandb] # (5)
    wandb-project = "..."
    wandb-entity = "..."
    ee-project = "..."

    [darts.paths] # (3)
    data-dir = "/path/to/planet_data"
    arcticdem-dir = "/path/to/data/datacubes/arcticdem2m.icechunk"
    tcvis-dir = "/path/to/data/datacubes/tcvis.icechunk"
    admin-dir = "/path/to/data/aux/admin"
    preprocess-cache = "/path/to/data/cache"
    artifact-dir = "/path/to/artifacts"

    [darts.training.paths] # (4)
    labels-dir = "/path/to/ML_training_labels/retrogressive_thaw_slumps" # (1)
    train-data-dir = "/path/to/data/training/planet_quickstart" # (2)

    [darts.preprocess]
    tpi-outer-radius = 100
    tpi-inner-radius = 0
    mask-erosion-size = 3

    [darts.training]
    device = "auto"
    num-workers = 16
    max-epochs = 6
    log-every-n-steps = 2
    check-val-every-n-epoch = 5
    plot-every-n-val-epochs = 4 # == 20 epochs
    early-stopping-patience = 0
    bands = [
        'blue',
        'green',
        'red',
        'nir',
        'ndvi',
        'tc_brightness',
        'tc_greenness',
        'tc_wetness',
        'relative_elevation',
        'slope',
    ]
    fold = 0

    [darts.test]
    data-split-method = "region"
    data-split-by = ['Taymyrsky Dolgano-Nenetsky District']

    [darts.training.preprocessing]
    patch-size = 896
    overlap = 224 # increase to 64 if exclude-nan = True
    exclude-nopositive = false
    exclude-nan = false
    force-preprocess = false

    # Only used in cross-validation and tuning
    [darts.cross-validation]
    fold-method = "region-stratified"
    total-folds = 5
    n-folds = 2
    n-randoms = 1
    scoring-metric = ["val/JaccardIndex", "val/Recall"]
    multi-score-strategy = "geometric"

    # Only used in training or cross-validation, not tuning
    [darts.training.hyperparameters]
    learning-rate = 4e-4
    batch-size = 6
    gamma = 0.999
    focal-loss-alpha = 0.92
    focal-loss-gamma = 1.6
    model-arch = "UPerNet"
    model-encoder = "tu-maxvit_tiny_rw_224"
    augment = [
        "HorizontalFlip",
        "VerticalFlip",
        "RandomRotate90",
        "Blur",
        "RandomBrightnessContrast",
        "MultiplicativeNoise"
    ]

    # Only used for tuning
    [darts.tuning]
    hpconfig = "configs/planet-training-quickstart.toml" # link to this file for convinience
    n-trials = 10

    # Only used for tuning
    [hyperparameters]
    learning-rate = {distribution = "loguniform", low = 1.0e-5, high = 1.0e-3}
    batch-size = 6
    gamma = 0.997
    focal-loss-alpha = {low = 0.8, high = 0.99}
    focal-loss-gamma = {low = 0.0, high = 2.0}
    model-arch = ["Unet", "MAnet", "UPerNet", "Segformer"]
    model-encoder = ["resnet50", "resnext50_32x4d", "mit_b2", "tu-convnextv2_tiny", "tu-maxvit_tiny_rw_224"]
    augment = {distribution = "constant", value = [
        "HorizontalFlip",
        "VerticalFlip",
        "RandomRotate90",
        "Blur",
        "RandomBrightnessContrast",
        "MultiplicativeNoise"
    ]}
    ```
    
    1. This should point to the directory of [the repository](https://github.com/initze/ML_training_labels) you cloned in step 2.
    2. The `train-data-dir` should point to a fast read-access storage, like a local mounted SSD to speed up the training process.
    3. Change these paths to your needs. I recommend to just change the "/path/to/" part to have everything in one place.
    4. These paths aswell.
    5. Change these to your account settings.

## 2. Preprocess the data

```sh
[uv run] darts preprocess-planet-train-data --config-file configs/planet-training-quickstart.toml
```

This will create the training data in the `train-data-dir` specified in the configuration file.

??? tip "Take a look at the data"

    If the `preprocess-cache` directory is specified, the preprocessing will automatically cache the preprocessed data before it is turned into the training data format.
    You can visualize the data with xarray:

    ```python
    import xarray as xr
    from pathlib import Path

    fpath = list(Path("/path/to/data/cache/planet_v2").glob("*.nc"))[0]
    tile = xr.open_zarr(fpath, decode_coords="all")
    tile
    ```

    ```python
    # Visualize the data (reduce the resolution for faster plotting)
    tile.red[::10, ::10].plot.imshow(cmap="Reds")
    ```

    To have a look at how the training data looks like, you can use `zarr` and `geopandas`:

    ```python
    import zarr

    zroot = zarr.open("/path/to/data/training/planet_quickstart/data.zarr")
    zroot.tree()
    ```

    ```python
    print(zroot["x"].shape)
    ```

    ```python
    import geopandas as gpd

    metadata = gpd.read_parquet("/path/to/data/training/planet_quickstart/metadata.parquet")
    metadata.head()
    ```

    ```python
    metadata.explore()
    ```

## 3. Train the model

```sh
[uv run] darts train-smp --config-file configs/planet-training-quickstart.toml
```

## 4. Test the model

```sh
[uv run] darts test-smp --config-file configs/planet-training-quickstart.toml
```

## 5. Do a cross-validation

!!! warning "This will take a while"

```sh
[uv run] darts cross-validation-smp --config-file configs/planet-training-quickstart.toml
```

# 6. Hyperparameter tuning

!!! warning "This will take a while"

```sh
[uv run] darts tune-smp --config-file configs/planet-training-quickstart.toml
```
