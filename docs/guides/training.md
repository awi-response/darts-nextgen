# Training

[TOC]

???+ note "Preprocessed data"
    All training and sweeps expect data to be present in preprocessed form.
    This means that the `train_data_dir` should look like this:

    ```sh
    train_data_dir/
    ├── config.toml
    ├── cross-val.zarr/
    ├── test.zarr/
    ├── val-test.zarr/
    └── labels.geojson
    ```

    With each zarr group containing a `x` and `y` dataarray.

    Ideally, use the preprocessing functions explained below to create this structure.

## Preprocess the data

The train, validation and test flow ist best descriped in the following image:
![DARTS training process](../assets/training-process.png){ loading=lazy }

To split your sentinel 2 data into the three different datasets and preprocess it, you can use the following command:

```sh
[uv run] darts preprocess-s2-train-data --your-args-here ... 
```

!!! info "PLANET data"

    If you are using PLANET data, you can use the following command instead:

    ```sh
    [uv run] darts preprocess-planet-train-data --your-args-here ...
    ```

This will create three data splits:

- `cross-val`, used for train and validation
- `val-test` 5% random leave-out for testing the randomness distribution shift of the data
- `test` leave-out region for testing the spatial distribution shift of the data

The final train data is saved to disk in form of zarr arrays with dimensions `[n, c, h, w]` and `[n, h, w]` for the labels respectivly, with chunksizes of `n=1`.
Hence, every sample is saved in a separate chunk and therefore in a seperate file on disk, but all managed by zarr.

The preprocessing is done with the same components used in the segmentation pipeline.
Hence, the same configuration options are available.
In addition, this preprocessing splits larger images into smaller patches of a fixed size.
Size and overlap can be configured in the configuration file or via the arguments of the CLI.

??? tip "You can also use the underlying functions directly:"

    ::: darts.legacy_training.preprocess_s2_train_data
        options:
            heading_level: 3

    ::: darts.legacy_training.preprocess_planet_train_data
        options:
            heading_level: 3

## Simple SMP train and test

To train a simple SMP (Segmentation Model Pytorch) model you can use the command:

```sh
[uv run] darts train-smp --your-args-here ...
```

Configurations for the architecture and encoder can be found in the [SMP documentation](https://smp.readthedocs.io/en/latest/index.html) for model configurations.

!!! warning "Change defaults"
    Even though the defaults from the CLI are somewhat useful, it is recommended to create a config file and change the behavior of the training there.

This will train a model with the `cross-val` data and save the model to disk.
You don't need to specify the concrete path to the `cross-val` split, the training script expects that the `--train-data-dir` points to the root directory of the splits, hence, the same path used in the preprocessing should be specified.
The training relies on PyTorch Lightning, which is a high-level interface for PyTorch.
It is recommended to use Weights and Biases (wandb) for the logging, because the training script is heavily influenced by how the organization of wandb works.

Each training run is assigned a unique name and id pair and optionally a trial name.
The name, which the user _can_ provide, should be used as a grouping mechanism of equal hyperparameter and code.
Hence, different versions of the same name should only differ by random state or run settings parameter, like logs.
Each version is assigned a unique id.
Artifacts (metrics & checkpoints) are then stored under `{artifact_dir}/{run_name}/{run_id}` in no-crossval runs.
If `trial_name` is specified, the artifacts are stored under `{artifact_dir}/{trial_name}/{run_name}-{run_id}`.
Wandb logs are always stored under `{wandb_entity}/{wandb_project}/{run_name}`, regardless of `trial_name`.
However, they are further grouped by the `trial_name` (via job_type), if specified.
Both `run_name` and `run_id` are also stored in the hparams of each checkpoint.


You can now test the model on the other two splits (`val-test` and `test`) with the following command:

```sh
[uv run] darts test-smp --your-args-here ...
```

The checkpoint stored is not usable for the pipeline yet, since it is stored in a different format.
To convert the model to a format, you need to convert is first:

```sh
[uv run] darts convert-lightning-checkpoint --your-args-here ...
```

??? tip "You can also use the underlying functions directly:"

    ::: darts.legacy_training.train_smp
        options:
            heading_level: 3

    ::: darts.legacy_training.test_smp
        options:
            heading_level: 3

    ::: darts.legacy_training.convert_lightning_checkpoint
        options:
            heading_level: 3

## Run a cross-validation hyperparameter sweep

!!! tip "Terminal Multiplexers"

    It is recommended to use a terminal multiplexer like `tmux`, `screen` or `zellij` to run multiple training runs in parallel.
    This way there is no need to have multiple terminal open over the span of multiple days.

To sweep over a certrain set of hyperparameters, some preparations are necessary:

1. Create a sweep configuration file in YAML format. This file should contain the hyperparameters to sweep over and the search space for each hyperparameter.
2. Setup a PostgreSQL database to store the results of the sweep, so we can run multiple runs in parallel with Optuna.

The sweep configuration file should look like a [wandb sweep configuration](https://docs.wandb.ai/guides/sweeps/sweep-config-keys/).
All values will be parsed and transformed to fit to an optuna sweep.

To setup the PostgreSQL database, search for an appropriate guide on how to setup a PostgreSQL database.
There are many ways to do this, depending on your environment.
The only important thing is that the database is reachable from the machine you are running the sweep on.

Now you can setup the sweep with the following command:

```sh
uv run darts optuna-sweep-smp --your-args-here ... --device 0
```

This will output some information about the sweep, especially the sweep id.
In addition, it will start running trials on the CUDA:0 device.

!!! note "Starting and continuing sweeps"

    Starting and continuing sweeps is done via the same `optuna-sweep-smp` command.
    Depending on the two arguments `-sweep-id` and `device`, the command will decide what to do.
    If the `sweep-id` is not specified, a new sweep will be started.
    If the `sweep-id` is specified, the sweep will continue from the last run.
    If the `device` is specified, `n-trials` will be started on the specified device (sequentially).
    If the `device` is not specified, but `sweep-id` is, then an error will be raised.
    If neither `device` nor `sweep-id` is specified, then a new sweep will be created without starting trials.

To start a second runner, you must open a new terminal (or panel/window in a terminal multiplexer) and run the following command:

```sh
uv run darts optuna-sweep-smp --your-args-here ... --device 1 --sweep-id <sweep-id>
```

!!! info "Multiple runners"

    You can run as many runners as you have devices available.
    Each runner will start n trials sequentially, specified by `n-trials`, which each request a new hyperparameter-combination from optuna.
    Each trial further creates multiple runs, depending on the `n_folds` and `n_randoms` parameters.
    This is the cross-validation part: Each trial, hence same hyperparameter-combination, is run `n_folds` times with `n_randoms` different random seeds.
    Therefore, the total number of runs done by a runner is `n-trials * n_folds * n_randoms`.
    This should ensure that a single random good (or bad) run does not influence the overall result of a hyperparameter-combination.

## Example config and sweep-config files

For better readability, the example config file uses different sub-headings which are not necessary and could be named differently or even removed.
The only important heading is the `[darts]` heading, which is the root of the configuration file.
Every value which is not under a `darts` top-level heading is ignored, as descriped in the [Configuration Guide](config.md).

The following `config.toml` expects that the labels are cloned from the [ML_training_labels repository](https://github.com/initze/ML_training_labels) and that PLANET scenes and tiles are downloaded into the `/large-storage/planet_data` directory.
The resulting file structure would look like this:

```sh title="File structure under cd ."
./
├── ../ML_training_labels/retrogressive_thaw_slumps/
├── darts/
├── logs/
└── configs/
    ├── planet-sweep-config.toml
    └── planet-tcvis-sweep.yaml
```

```sh title="File structure under /large-storage/"
/large-storage/
├── planet_data/
└── darts-nextgen/
    ├── artifacts/
    └── data/
        ├── training/
        │   └── planet_native_tcvis_896_partial/
        ├── cache/
        ├── datacubes/
        │   ├── arcticdem/
        │   └── tcvis/
        └── aux/admin/
```

```sh title="File structure under /fast-storage/"
/fast-storage/
└── darts-nextgen/
    └── data/
        └── training/
            └── planet_native_tcvis_896_partial/
```


```toml title="configs/planet-sweep-config.toml"
[darts.wandb]
wandb-project = "darts"
wandb-entity = "your-wandb-username"

[darts.sweep]
n-trials = 100
sweep-db = "postgresql://pguser@localhost:5432/sweeps"
n_folds = 3
n_randoms = 3
sweep-id = "sweep-cv-large-planet"

[darts.training]
num-workers = 16
max-epochs = 60
log-every-n-steps = 100
check-val-every-n-epoch = 5
plot-every-n-val-epochs = 4 # == 20 epochs
early-stopping-patience = 0

# These are the default one, if not specified in the sweep-config
[darts.hyperparameters]
batch-size = 6
augment = true

[darts.training_preprocess]
ee-project = "your-ee-project"
tpi-outer-radius = 100
tpi-inner-radius = 0
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
patch-size = 896
overlap = 0 # increase to 64 if exclude-nan = True
exclude-nopositive = false
exclude-nan = false
test-val-split = 0.05
test-regions = ['Taymyrsky Dolgano-Nenetsky District']

[darts.paths]
data-dir = "/large-storage/planet_data"
labels-dir = "../ML_training_labels/retrogressive_thaw_slumps" # (1)
arcticdem-dir = "/large-storage/darts-nextgen/data/datacubes/arcticdem"
tcvis-dir = "/large-storage/darts-nextgen/data/datacubes/tcvis"
admin-dir = "/large-storage/darts-nextgen/data/aux/admin"
train-data-dir = "/fast-storage/darts-nextgen/data/training/planet_native_tcvis_896_partial" # (2)
preprocess-cache = "/large-storage/darts-nextgen/data/cache"
sweep-config = "configs/planet-tcvis-sweep.yaml"
artifact-dir = "/large-storage/darts-nextgen/artifacts"
```

1. Clone [this repository](https://github.com/initze/ML_training_labels) to obtain the labels for the training data.
2. The `train-data-dir` should point to a fast read-access storage, like a local mounted SSD to speed up the training process.

```yaml title="configs/planet-tcvis-sweep.yaml"	
name: planet-tcvis-large
method: random
metric:
  goal: maximize
  name: val0/JaccardIndex
parameters:
  learning_rate:
    max: !!float 1e-3
    min: !!float 1e-5
    distribution: log_uniform_values
  gamma: # How fast the learning rate will decrease
    value: 0.997
  focal_loss_alpha: # How much the positive class is weighted
    min: 0.8
    max: 0.99
  focal_loss_gamma: # How much focus should be given to "bad" predictions
    min: 0.0
    max: 2.0
  model_arch:
    values:
      - Unet
      - MAnet
      - UPerNet
      - Segformer
  model_encoder:
    values:
      - resnet50
      - resnext50_32x4d
      - mit_b2
      - tu-convnextv2_tiny
      - tu-maxvit_tiny_rw_224
```
