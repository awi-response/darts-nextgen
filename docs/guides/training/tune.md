# Hyperparameter tuning

With the tuning script hyperparameters can be tuned by running a sweep.
The sweep uses cross-validation to evaluate the performance of a single hyperparameter configuration.

```sh
[uv run] darts training tune-smp ...
```

???+ info "Use the function"

    [darts_segmentation.training.tune.tune_smp][]

## Configuration file

How the hyperparameters should be sweeped can be configured in a YAML or Toml file, specified by the `hpconfig` parameter.
This file must contain a key called `"hyperparameters"` containing a list of hyperparameters distributions.
This file can also be the same one as your `config-file`, here it is important that you do NOT prefix the `"hyperparameters"` key with `"darts."`.

??? question "Why using a separate configuration file / different parent key?"

    - It makes creating different sweeps easier
    - It separates the sweep configuration from the normal configuration
    - It allows for using dicts in the config - this is not possible right now due to the way we handle the main configuration file.

## Suppoorted Distrubtions

The distributions from which the hyperparameters should be sampled can either be explicit defined by a dictionary containing a `"distribution"` key,
or they can be implicit defined by a single value, a list or a dictionary containing a `"low"` and `"high"` key.

The following distributions are supported:

- `"uniform"`: Uniform distribution - must have a `"low"` and `"high"` value
- `"loguniform"`: Log-uniform distribution - must have a `"low"` and `"high"` value
- `"reversed-loguniform"`: N - Log-uniform distribution - must have a `"low"` and `"high"` value, optionally a `"n"` value which default to `1`.
- `"intuniform"`: Integer uniform distribution - must have a `"low"` and `"high"` value (both are inclusive)
- `"choice"`: Choice distribution - must have a list of `"choices"` for explicit case, else just pass a list
- `"value"`: Fixed value distribution - must have a `"value"` key for explicit case, else just pass a value

And the following hyperparameters can be configured:

| Hyperparameter        | Type          | Default  |
| --------------------- | ------------- | -------- |
| model_arch            | str           | "Unet"   |
| model_encoder         | str           | "dpn107" |
| model_encoder_weights | str or None   | None     |
| augment               | bool          | True     |
| learning_rate         | float         | 1e-3     |
| gamma                 | float         | 0.9      |
| focal_loss_alpha      | float or None | None     |
| focal_loss_gamma      | float         | 2.0      |
| batch_size            | int           | 8        |

## Number of trials and Grid-Search

Per default, a random search is performed, where the number of samples can be specified by `n_trials`.
If `n_trials` is set to "grid", a grid search is performed instead.
However, this expects to be every hyperparameter to be configured as either constant value or a choice / list.

Optionally it is possible to retrain and test with the best hyperparameter configuration by setting `retrain_and_test` to `True`.
This will retrain the model on the complete train split without folding and test the data on the test split.

## Parallel execution with multiprocessing

The tuning script supports parallel execution of cross-validation runs across multiple devices using multiprocessing.
This can significantly speed up hyperparameter tuning when you have multiple GPUs available.

To enable parallel execution, use the `--strategy tune-parallel` flag along with specifying multiple devices:

```sh
[uv run] darts training tune-smp \
    --strategy tune-parallel \
    --devices 0 1 2 3 \
    --hpconfig configs/hyperparameters.yaml \
    ...
```

### How it works

When using `tune-parallel`:

- Multiple cross-validation runs (each with a different hyperparameter configuration) are executed in parallel
- Each cross-validation run is assigned to an available GPU from the device pool
- Within each cross-validation, the individual folds are executed sequentially (not in parallel)
- Once a cross-validation completes, the GPU is returned to the pool and assigned to the next pending run

This approach maximizes GPU utilization when running many hyperparameter configurations, as the number of parallel workers equals the number of specified devices.

### Example

If you have 4 GPUs and want to tune 100 hyperparameter configurations with 5-fold cross-validation:

```sh
[uv run] darts training tune-smp \
    --strategy tune-parallel \
    --devices 0 1 2 3 \
    --n-trials 100 \
    --n-folds 5 \
    --hpconfig configs/hyperparameters.yaml \
    --train-data-dir data/preprocessed
```

This will run 4 cross-validations in parallel (one per GPU), and each cross-validation will sequentially train 5 models (one per fold). As cross-validations complete, new ones are started until all 100 hyperparameter configurations have been evaluated.

!!! note "Strategy comparison"

    - **Serial execution** (default): Cross-validations run one after another. Within each cross-validation, you can optionally use `--strategy cv-parallel` to parallelize the fold training.
    - **`tune-parallel`**: Multiple cross-validations run in parallel across GPUs. Within each cross-validation, folds are trained sequentially.
    - You cannot combine `tune-parallel` with `cv-parallel` - choose one level of parallelization based on your workload.

!!! warning "DDP compatibility"

    When using `tune-parallel`, distributed data parallel (DDP) strategies are automatically disabled for the cross-validation runs to prevent conflicts with multiprocessing. Each training run will use a single device only.
