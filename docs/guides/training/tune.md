# Hyperparameter tuning

With the tuning script hyperparameters can be tuned by running a sweep.
The sweep uses cross-validation to evaluate the performance of a single hyperparameter configuration.

```sh
[uv run] darts tune-smp ...
```

???+ info "Use the function"

    ::: darts_segmentation.training.tune.tune_smp

How the hyperparameters should be sweeped can be configured in a YAML or Toml file, specified by the `hpconfig` parameter.
This file must contain a key called `"hyperparameters"` containing a list of hyperparameters distributions.
These distributions can either be explicit defined by another dictionary containing a `"distribution"` key,
or they can be implicit defined by a single value, a list or a dictionary containing a `"low"` and `"high"` key.

The following distributions are supported:

- `"uniform"`: Uniform distribution - must have a `"low"` and `"high"` value
- `"loguniform"`: Log-uniform distribution - must have a `"low"` and `"high"` value
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

Because the configuration file doesn't use the `darts` key, it can also be merged into the normal configuration file and specified by the `hpconfig` parameter to also use that file.

??? question "Why using a separate configuration file?"

    - It makes creating different sweeps easier
    - It separates the sweep configuration from the normal configuration
    - It allows for using dicts in the config - this is not possible right now due to the way we handle the main configuration file.

Per default, a random search is performed, where the number of samples can be specified by `n_trials`.
If `n_trials` is set to "grid", a grid search is performed instead.
However, this expects to be every hyperparameter to be configured as either constant value or a choice / list.

Optionally it is possible to retrain and test with the best hyperparameter configuration by setting `retrain_and_test` to `True`.
This will retrain the model on the complete train split without folding and test the data on the test split.
