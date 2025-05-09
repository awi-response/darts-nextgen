"""Wrapper for SMP training scripts.

These are necessary for the CLI to get the necessary function definitions without importing the
darts_segmentation.training module which needs a lot of heavy imports like torch, lightning, or torchvision.
Thus, calling commands which don't need these imports can be executed way faster, e.g. --help.

This must include the complete docstring as well as the necessary imports.
I recommend using a LLM to generate this based on the original docstring.

TODO: Make this unnecessary. It creates another source of documentation which needs to be updated.
It is in general ugly.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    pass


def train_smp(
    *,
    # Data config
    train_data_dir: Path,
    data_split_method: Literal["random", "region", "sample"] | None = None,
    data_split_by: list[str] | None = None,
    fold_method: Literal["kfold", "shuffle", "stratified", "region", "region-stratified"] = "kfold",
    total_folds: int = 5,
    fold: int = 0,
    # Run config
    run_name: str | None = None,
    cv_name: str | None = None,
    tune_name: str | None = None,
    artifact_dir: Path = Path("artifacts"),
    continue_from_checkpoint: Path | None = None,
    # Hyperparameters
    model_arch: str = "Unet",
    model_encoder: str = "dpn107",
    model_encoder_weights: str | None = None,
    augment: bool = True,
    learning_rate: float = 1e-3,
    gamma: float = 0.9,
    focal_loss_alpha: float | None = None,
    focal_loss_gamma: float = 2.0,
    batch_size: int = 8,
    # Epoch and Logging config
    max_epochs: int = 100,
    log_every_n_steps: int = 10,
    check_val_every_n_epoch: int = 3,
    early_stopping_patience: int = 5,
    plot_every_n_val_epochs: int = 5,
    # Device and Manager config
    random_seed: int = 42,
    num_workers: int = 0,
    device: int | str = "auto",
    # Wandb config
    wandb_entity: str | None = None,
    wandb_project: str | None = None,
):
    """Run the training of the SMP model, specifically binary segmentation.

    Please see https://smp.readthedocs.io/en/latest/index.html for model configurations of architecture and encoder.

    Please also consider reading our training guide (docs/guides/training.md).

    This training function is meant for single training runs but is also used for cross-validation and hyperparameter
    tuning by cv.py and tune.py.
    This strongly affects where artifacts are stored:

    - Run was created by a tune: `{artifact_dir}/{tune_name}/{cv_name}/{run_name}-{run_id}`
    - Run was created by a cross-validation: `{artifact_dir}/_cross_validations/{cv_name}/{run_name}-{run_id}`
    - Single runs: `{artifact_dir}/_runs/{run_name}-{run_id}`

    `run_name` can be specified by the user, else it is generated automatically.
    In case of cross-validation, the run name is generated automatically by the cross-validation.
    `run_id` is generated automatically by the training function.
    Both are saved to the final checkpoint.

    You can specify the frequency on how often logs will be written and validation will be performed.
        - `log_every_n_steps` specifies how often train-logs will be written. This does not affect validation.
        - `check_val_every_n_epoch` specifies how often validation will be performed.
            This will also affect early stopping.
        - `early_stopping_patience` specifies how many epochs to wait for improvement before stopping.
            In epochs, this would be `check_val_every_n_epoch * early_stopping_patience`.
        - `plot_every_n_val_epochs` specifies how often validation samples will be plotted.
            Since plotting is quite costly, you can reduce the frequency. Works similar like early stopping.
            In epochs, this would be `check_val_every_n_epoch * plot_every_n_val_epochs`.
    Example: There are 400 training samples and the batch size is 2, resulting in 200 training steps per epoch.
    If `log_every_n_steps` is set to 50 then the training logs and metrics will be logged 4 times per epoch.
    If `check_val_every_n_epoch` is set to 5 then validation will be performed every 5 epochs.
    If `plot_every_n_val_epochs` is set to 2 then validation samples will be plotted every 10 epochs.
    If `early_stopping_patience` is set to 3 then early stopping will be performed after 15 epochs without improvement.

    The data structure of the training data expects the "preprocessing" step to be done beforehand,
    which results in the following data structure:

    ```sh
    preprocessed-data/ # the top-level directory
    ├── config.toml
    ├── data.zarr/ # this zarr group contains the dataarrays x and y
    ├── metadata.parquet # this contains information necessary to split the data into train, val, and test sets.
    └── labels.geojson
    ```

    Args:
        train_data_dir (Path): The path (top-level) to the data to be used for training.
            Expects a directory containing:
            1. a zarr group called "data.zarr" containing a "x" and "y" array
            2. a geoparquet file called "metadata.parquet" containing the metadata for the data.
                This metadata should contain at least the following columns:
                - "sample_id": The id of the sample
                - "region": The region the sample belongs to
                - "empty": Whether the image is empty
                The index should refer to the index of the sample in the zarr data.
            This directory should be created by a preprocessing script.
        batch_size (int): Batch size for training and validation.
        data_split_method (Literal["random", "region", "sample"] | None, optional):
            The method to use for splitting the data into a train and a test set.
            "random" will split the data randomly, the seed is always 42 and the test size is 20%.
            "region" will split the data by one or multiple regions,
            which can be specified by providing a str or list of str to data_split_by.
            "sample" will split the data by sample ids, which can also be specified similar to "region".
            If None, no split is done and the complete dataset is used for both training and testing.
            The train split will further be split in the cross validation process.
            Defaults to None.
        data_split_by (list[str] | None, optional): Select by which regions/samples split. Defaults to None.
        fold_method (Literal["kfold", "shuffle", "stratified", "region", "region-stratified"], optional):
            Method for cross-validation split. Defaults to "kfold".
        total_folds (int, optional): Total number of folds in cross-validation. Defaults to 5.
        fold (int, optional): Index of the current fold. Defaults to 0.
        run_name (str | None, optional): Name of the run. If None is generated automatically. Defaults to None.
        cv_name (str | None, optional): Name of the cross-validation.
            Should only be specified by a cross-validation script.
            Defaults to None.
        tune_name (str | None, optional): Name of the tuning.
            Should only be specified by a tuning script.
            Defaults to None.
        artifact_dir (Path, optional): Top-level path to the training output directory.
            Will contain checkpoints and metrics. Defaults to Path("lightning_logs").
        continue_from_checkpoint (Path | None, optional): Path to a checkpoint to continue training from.
            Defaults to None.
        model_arch (str, optional): Model architecture to use. Defaults to "Unet".
        model_encoder (str, optional): Encoder to use. Defaults to "dpn107".
        model_encoder_weights (str | None, optional): Path to the encoder weights. Defaults to None.
        augment (bool, optional): Weather to apply augments or not. Defaults to True.
        learning_rate (float, optional): Learning Rate. Defaults to 1e-3.
        gamma (float, optional): Multiplicative factor of learning rate decay. Defaults to 0.9.
        focal_loss_alpha (float, optional): Weight factor to balance positive and negative samples.
            Alpha must be in [0...1] range, high values will give more weight to positive class.
            None will not weight samples. Defaults to None.
        focal_loss_gamma (float, optional): Focal loss power factor. Defaults to 2.0.
        batch_size (int, optional): Batch Size. Defaults to 8.
        max_epochs (int, optional): Maximum number of epochs to train. Defaults to 100.
        log_every_n_steps (int, optional): Log every n steps. Defaults to 10.
        check_val_every_n_epoch (int, optional): Check validation every n epochs. Defaults to 3.
        early_stopping_patience (int, optional): Number of epochs to wait for improvement before stopping.
            Defaults to 5.
        plot_every_n_val_epochs (int, optional): Plot validation samples every n epochs. Defaults to 5.
        random_seed (int, optional): Random seed for deterministic training. Defaults to 42.
        num_workers (int, optional): Number of Dataloader workers. Defaults to 0.
        device (int | str, optional): The device to run the model on. Defaults to "auto".
        wandb_entity (str | None, optional): Weights and Biases Entity. Defaults to None.
        wandb_project (str | None, optional): Weights and Biases Project. Defaults to None.

    """
    from darts_segmentation.training import train_smp as _train_smp

    _train_smp(
        train_data_dir=train_data_dir,
        data_split_method=data_split_method,
        data_split_by=data_split_by,
        fold_method=fold_method,
        total_folds=total_folds,
        fold=fold,
        run_name=run_name,
        cv_name=cv_name,
        tune_name=tune_name,
        artifact_dir=artifact_dir,
        continue_from_checkpoint=continue_from_checkpoint,
        model_arch=model_arch,
        model_encoder=model_encoder,
        model_encoder_weights=model_encoder_weights,
        augment=augment,
        learning_rate=learning_rate,
        gamma=gamma,
        focal_loss_alpha=focal_loss_alpha,
        focal_loss_gamma=focal_loss_gamma,
        batch_size=batch_size,
        max_epochs=max_epochs,
        log_every_n_steps=log_every_n_steps,
        check_val_every_n_epoch=check_val_every_n_epoch,
        early_stopping_patience=early_stopping_patience,
        plot_every_n_val_epochs=plot_every_n_val_epochs,
        random_seed=random_seed,
        num_workers=num_workers,
        device=device,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
    )


def test_smp(
    *,
    train_data_dir: Path,
    run_id: str,
    run_name: str,
    model_ckp: Path | None = None,
    batch_size: int = 8,
    data_split_method: Literal["random", "region", "sample"] | None = None,
    data_split_by: list[str] | str | float | None = None,
    artifact_dir: Path = Path("artifacts"),
    num_workers: int = 0,
    device: int | str = "auto",
    wandb_entity: str | None = None,
    wandb_project: str | None = None,
):
    """Run the testing of the SMP model.

    The data structure of the training data expects the "preprocessing" step to be done beforehand,
    which results in the following data structure:

    ```sh
    preprocessed-data/ # the top-level directory
    ├── config.toml
    ├── data.zarr/ # this zarr group contains the dataarrays x and y
    ├── metadata.parquet # this contains information necessary to split the data into train, val, and test sets.
    └── labels.geojson
    ```

    Args:
        train_data_dir (Path): The path (top-level) to the data to be used for training.
            Expects a directory containing:
            1. a zarr group called "data.zarr" containing a "x" and "y" array
            2. a geoparquet file called "metadata.parquet" containing the metadata for the data.
                This metadata should contain at least the following columns:
                - "sample_id": The id of the sample
                - "region": The region the sample belongs to
                - "empty": Whether the image is empty
                The index should refer to the index of the sample in the zarr data.
            This directory should be created by a preprocessing script.
        run_id (str): ID of the run.
        run_name (str): Name of the run.
        model_ckp (Path | None): Path to the model checkpoint.
            If None, try to find the latest checkpoint in `artifact_dir / run_name / run_id / checkpoints`.
            Defaults to None.
        batch_size (int): Batch size for training and validation.
        data_split_method (Literal["random", "region", "sample"] | None, optional):
            The method to use for splitting the data into a train and a test set.
            "random" will split the data randomly, the seed is always 42 and the size of the test set can be
            specified by providing a float between 0 and 1 to data_split_by.
            "region" will split the data by one or multiple regions,
            which can be specified by providing a str or list of str to data_split_by.
            "sample" will split the data by sample ids, which can also be specified similar to "region".
            If None, no split is done and the complete dataset is used for both training and testing.
            The train split will further be split in the cross validation process.
            Defaults to None.
        data_split_by (list[str] | str | float | None, optional): Select by which seed/regions/samples split.
            Defaults to None.
        artifact_dir (Path, optional): Directory to save artifacts. Defaults to Path("artifacts").
        num_workers (int, optional): Number of workers for the DataLoader. Defaults to 0.
        device (int | str, optional): Device to use. Defaults to "auto".
        wandb_entity (str | None, optional): WandB entity. Defaults to None.
        wandb_project (str | None, optional): WandB project. Defaults to None.

    """
    from darts_segmentation.training import test_smp as _test_smp

    _test_smp(
        train_data_dir=train_data_dir,
        run_id=run_id,
        run_name=run_name,
        model_ckp=model_ckp,
        batch_size=batch_size,
        data_split_method=data_split_method,
        data_split_by=data_split_by,
        artifact_dir=artifact_dir,
        num_workers=num_workers,
        device=device,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
    )


def convert_lightning_checkpoint(
    *,
    lightning_checkpoint: Path,
    out_directory: Path,
    checkpoint_name: str,
    framework: str = "smp",
):
    """Convert a lightning checkpoint to our own format.

    The final checkpoint will contain the model configuration and the state dict.
    It will be saved to:

    ```python
        out_directory / f"{checkpoint_name}_{formatted_date}.ckpt"
    ```

    Args:
        lightning_checkpoint (Path): Path to the lightning checkpoint.
        out_directory (Path): Output directory for the converted checkpoint.
        checkpoint_name (str): A unique name of the new checkpoint.
        framework (str, optional): The framework used for the model. Defaults to "smp".

    """
    from darts_segmentation.training import convert_lightning_checkpoint as _convert_lightning_checkpoint

    _convert_lightning_checkpoint(
        lightning_checkpoint=lightning_checkpoint,
        out_directory=out_directory,
        checkpoint_name=checkpoint_name,
        framework=framework,
    )


def tune_smp(
    *,
    # Data
    hpconfig: Path,
    train_data_dir: Path,
    data_split_method: Literal["random", "region", "sample"] | None = None,
    data_split_by: list[str] | None = None,
    fold_method: Literal["kfold", "shuffle", "stratified", "region", "region-stratified"] = "kfold",
    total_folds: int = 5,
    # Tune config
    n_folds: int | None = None,
    n_randoms: int = 3,
    n_trials: int | Literal["grid"] = 100,
    retrain_and_test: bool = True,
    tune_name: str | None = None,
    artifact_dir: Path = Path("artifacts"),
    # Scoring
    scoring_metric: list[str] = ["val/JaccardIndex", "val/Recall"],
    multi_score_strategy: Literal["harmonic", "arithmetic", "geometric", "min"] = "harmonic",
    # Epoch and Logging config
    max_epochs: int = 100,
    log_every_n_steps: int = 10,
    check_val_every_n_epoch: int = 3,
    early_stopping_patience: int = 5,
    plot_every_n_val_epochs: int = 5,
    # Device and Manager config
    num_workers: int = 0,
    device: int | str = "auto",
    # Wandb config
    wandb_entity: str | None = None,
    wandb_project: str | None = None,
):
    """Tune the hyper-parameters of the model using cross-validation and random states.

    Please see https://smp.readthedocs.io/en/latest/index.html for model configurations of architecture and encoder.

    Please also consider reading our training guide (docs/guides/training.md).

    This tuning script is designed to sweep over hyperparameters with a cross-validation
    used to evaluate each hyperparameter configuration.
    Optionally, by setting `retrain_and_test` to True, the best hyperparameters are then selected based on the
    cross-validation scores and a new model is trained on the entire train-split and tested on the test-split.

    Hyperparameters can be configured using a `hpconfig` file (YAML or Toml).
    Please consult the training guide or the documentation of
    `darts_segmentation.training.hparams.parse_hyperparameters` to learn how such a file should be structured.
    Per default, a random search is performed, where the number of samples can be specified by `n_trials`.
    If `n_trials` is set to "grid", a grid search is performed instead.
    However, this expects to be every hyperparameter to be configured as either constant value or a choice / list.

    To specify on which metric(s) the cv score is calculated, the `scoring_metric` parameter can be specified.
    Each score can be provided by either ":higher" or ":lower" to indicate the direction of the metrics.
    This allows to correctly combine multiple metrics by doing 1/metric before calculation if a metric is ":lower".
    If no direction is provided, it is assumed to be ":higher".
    Has no real effect on the single score calculation, since only the mean is calculated there.

    In a multi-score setting, the score is calculated by combine-then-reduce the metrics.
    Meaning that first for each fold the metrics are combined using the specified strategy,
    and then the results are reduced via mean.
    Please refer to the documentation to understand the different multi-score strategies.

    If one of the metrics of any of the runs contains NaN, Inf, -Inf or is 0 the score is reported to be "unstable".
    In such cases, the configuration is not considered for further evaluation.

    Artifacts are stored under `{artifact_dir}/{tune_name}`.

    You can specify the frequency on how often logs will be written and validation will be performed.
        - `log_every_n_steps` specifies how often train-logs will be written. This does not affect validation.
        - `check_val_every_n_epoch` specifies how often validation will be performed.
            This will also affect early stopping.
        - `early_stopping_patience` specifies how many epochs to wait for improvement before stopping.
            In epochs, this would be `check_val_every_n_epoch * early_stopping_patience`.
        - `plot_every_n_val_epochs` specifies how often validation samples will be plotted.
            Since plotting is quite costly, you can reduce the frequency. Works similar like early stopping.
            In epochs, this would be `check_val_every_n_epoch * plot_every_n_val_epochs`.
    Example: There are 400 training samples and the batch size is 2, resulting in 200 training steps per epoch.
    If `log_every_n_steps` is set to 50 then the training logs and metrics will be logged 4 times per epoch.
    If `check_val_every_n_epoch` is set to 5 then validation will be performed every 5 epochs.
    If `plot_every_n_val_epochs` is set to 2 then validation samples will be plotted every 10 epochs.
    If `early_stopping_patience` is set to 3 then early stopping will be performed after 15 epochs without improvement.

    The data structure of the training data expects the "preprocessing" step to be done beforehand,
    which results in the following data structure:

    ```sh
    preprocessed-data/ # the top-level directory
    ├── config.toml
    ├── data.zarr/ # this zarr group contains the dataarrays x and y
    ├── metadata.parquet # this contains information necessary to split the data into train, val, and test sets.
    └── labels.geojson
    ```

    Args:
        hpconfig (Path): The path to the hyperparameter configuration file.
            Please see the documentation of `hyperparameters` for more information.
        train_data_dir (Path): The path (top-level) to the data to be used for training.
            Expects a directory containing:
            1. a zarr group called "data.zarr" containing a "x" and "y" array
            2. a geoparquet file called "metadata.parquet" containing the metadata for the data.
                This metadata should contain at least the following columns:
                - "sample_id": The id of the sample
                - "region": The region the sample belongs to
                - "empty": Whether the image is empty
                The index should refer to the index of the sample in the zarr data.
            This directory should be created by a preprocessing script.
        data_split_method (Literal["random", "region", "sample"] | None, optional):
            The method to use for splitting the data into a train and a test set.
            "random" will split the data randomly, the seed is always 42 and the test size is 20%.
            "region" will split the data by one or multiple regions,
            which can be specified by providing a str or list of str to data_split_by.
            "sample" will split the data by sample ids, which can also be specified similar to "region".
            If None, no split is done and the complete dataset is used for both training and testing.
            The train split will further be split in the cross validation process.
            Defaults to None.
        data_split_by (list[str] | None, optional): Select by which regions/samples split. Defaults to None.
        fold_method (Literal["kfold", "shuffle", "stratified", "region", "region-stratified"], optional):
            Method for cross-validation split. Defaults to "kfold".
        total_folds (int, optional): Total number of folds in cross-validation. Defaults to 5.
        n_folds (int | None, optional): Number of folds to perform in cross-validation.
            If None, all folds (total_folds) will be used.
            Defaults to None.
        n_randoms (int, optional): Number of random seeds to perform in cross-validation.
            First three seeds are always 42, 21, 69, further seeds are deterministic generated.
            Defaults to 3.
        n_trials (int | Literal["grid"], optional): Number of trials to perform in hyperparameter tuning.
            If "grid", span a grid search over all configured hyperparameters.
            In a grid search, only constant or choice hyperparameters are allowed.
            Defaults to 100.
        retrain_and_test (bool, optional): Whether to retrain the model with the best hyperparameters and test it.
            Defaults to True.
        tune_name (str | None, optional): Name of the tuning run.
            Will be generated based on the number of existing directories in the artifact directory if None.
            Defaults to None.
        artifact_dir (Path, optional): Top-level path to the training output directory.
            Will contain checkpoints and metrics. Defaults to Path("lightning_logs").
        scoring_metric (list[str]): Metric(s) to use for scoring.
        multi_score_strategy (Literal["harmonic", "arithmetic", "geometric", "min"], optional):
            Strategy for combining multiple metrics. Defaults to "harmonic".
        max_epochs (int, optional): Maximum number of epochs to train. Defaults to 100.
        log_every_n_steps (int, optional): Log every n steps. Defaults to 10.
        check_val_every_n_epoch (int, optional): Check validation every n epochs. Defaults to 3.
        early_stopping_patience (int, optional): Number of epochs to wait for improvement before stopping.
            Defaults to 5.
        plot_every_n_val_epochs (int, optional): Plot validation samples every n epochs. Defaults to 5.
        num_workers (int, optional): Number of Dataloader workers. Defaults to 0.
        device (int | str, optional): The device to run the model on. Defaults to "auto".
        wandb_entity (str | None, optional): Weights and Biases Entity. Defaults to None.
        wandb_project (str | None, optional): Weights and Biases Project. Defaults to None.

    """
    from darts_segmentation.training import tune_smp as _tune_smp

    _tune_smp(
        hpconfig=hpconfig,
        train_data_dir=train_data_dir,
        data_split_method=data_split_method,
        data_split_by=data_split_by,
        fold_method=fold_method,
        total_folds=total_folds,
        n_folds=n_folds,
        n_randoms=n_randoms,
        n_trials=n_trials,
        retrain_and_test=retrain_and_test,
        tune_name=tune_name,
        artifact_dir=artifact_dir,
        scoring_metric=scoring_metric,
        multi_score_strategy=multi_score_strategy,
        max_epochs=max_epochs,
        log_every_n_steps=log_every_n_steps,
        check_val_every_n_epoch=check_val_every_n_epoch,
        early_stopping_patience=early_stopping_patience,
        plot_every_n_val_epochs=plot_every_n_val_epochs,
        num_workers=num_workers,
        device=device,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
    )


def cross_validation_smp(
    *,
    # Data
    train_data_dir: Path,
    data_split_method: Literal["random", "region", "sample"] | None = None,
    data_split_by: list[str] | None = None,
    fold_method: Literal["kfold", "shuffle", "stratified", "region", "region-stratified"] = "kfold",
    total_folds: int = 5,
    # CV config
    n_folds: int | None = None,
    n_randoms: int = 3,
    cv_name: str | None = None,
    tune_name: str | None = None,
    artifact_dir: Path = Path("artifacts"),
    # Hyperparameters
    model_arch: str = "Unet",
    model_encoder: str = "dpn107",
    model_encoder_weights: str | None = None,
    augment: bool = True,
    learning_rate: float = 1e-3,
    gamma: float = 0.9,
    focal_loss_alpha: float | None = None,
    focal_loss_gamma: float = 2.0,
    batch_size: int = 8,
    # Scoring
    scoring_metric: list[str] = ["val/JaccardIndex", "val/Recall"],
    multi_score_strategy: Literal["harmonic", "arithmetic", "geometric", "min"] = "harmonic",
    # Epoch and Logging config
    max_epochs: int = 100,
    log_every_n_steps: int = 10,
    check_val_every_n_epoch: int = 3,
    early_stopping_patience: int = 5,
    plot_every_n_val_epochs: int = 5,
    # Device and Manager config
    num_workers: int = 0,
    device: int | str = "auto",
    # Wandb config
    wandb_entity: str | None = None,
    wandb_project: str | None = None,
):
    """Perform cross-validation for a model with given hyperparameters.

    Please see https://smp.readthedocs.io/en/latest/index.html for model configurations of architecture and encoder.

    Please also consider reading our training guide (docs/guides/training.md).

    This cross-validation function is designed to evaluate the performance of a single model configuration.
    It can be used by a tuning script to tune hyperparameters.
    It calls the training function, hence most functionality is the same as the training function.
    In general, it does perform this:

    ```py
    for seed in seeds:
        for fold in folds:
            train_model(seed=seed, fold=fold, ...)
    ```

    and calculates a score from the results.

    To specify on which metric(s) the score is calculated, the `scoring_metric` parameter can be specified.
    Each score can be provided by either ":higher" or ":lower" to indicate the direction of the metrics.
    This allows to correctly combine multiple metrics by doing 1/metric before calculation if a metric is ":lower".
    If no direction is provided, it is assumed to be ":higher".
    Has no real effect on the single score calculation, since only the mean is calculated there.

    In a multi-score setting, the score is calculated by combine-then-reduce the metrics.
    Meaning that first for each fold the metrics are combined using the specified strategy,
    and then the results are reduced via mean.
    Please refer to the documentation to understand the different multi-score strategies.

    If one of the metrics of any of the runs contains NaN, Inf, -Inf or is 0 the score is reported to be "unstable".

    Artifacts are stored under `{artifact_dir}/{tune_name}` for tunes (meaning if `tune_name` is not None)
    else `{artifact_dir}/_cross_validation`.

    You can specify the frequency on how often logs will be written and validation will be performed.
        - `log_every_n_steps` specifies how often train-logs will be written. This does not affect validation.
        - `check_val_every_n_epoch` specifies how often validation will be performed.
            This will also affect early stopping.
        - `early_stopping_patience` specifies how many epochs to wait for improvement before stopping.
            In epochs, this would be `check_val_every_n_epoch * early_stopping_patience`.
        - `plot_every_n_val_epochs` specifies how often validation samples will be plotted.
            Since plotting is quite costly, you can reduce the frequency. Works similar like early stopping.
            In epochs, this would be `check_val_every_n_epoch * plot_every_n_val_epochs`.
    Example: There are 400 training samples and the batch size is 2, resulting in 200 training steps per epoch.
    If `log_every_n_steps` is set to 50 then the training logs and metrics will be logged 4 times per epoch.
    If `check_val_every_n_epoch` is set to 5 then validation will be performed every 5 epochs.
    If `plot_every_n_val_epochs` is set to 2 then validation samples will be plotted every 10 epochs.
    If `early_stopping_patience` is set to 3 then early stopping will be performed after 15 epochs without improvement.

    The data structure of the training data expects the "preprocessing" step to be done beforehand,
    which results in the following data structure:

    ```sh
    preprocessed-data/ # the top-level directory
    ├── config.toml
    ├── data.zarr/ # this zarr group contains the dataarrays x and y
    ├── metadata.parquet # this contains information necessary to split the data into train, val, and test sets.
    └── labels.geojson
    ```

    Args:
        train_data_dir (Path): The path (top-level) to the data to be used for training.
            Expects a directory containing:
            1. a zarr group called "data.zarr" containing a "x" and "y" array
            2. a geoparquet file called "metadata.parquet" containing the metadata for the data.
                This metadata should contain at least the following columns:
                - "sample_id": The id of the sample
                - "region": The region the sample belongs to
                - "empty": Whether the image is empty
                The index should refer to the index of the sample in the zarr data.
            This directory should be created by a preprocessing script.
        batch_size (int): Batch size for training and validation.
        data_split_method (Literal["random", "region", "sample"] | None, optional):
            The method to use for splitting the data into a train and a test set.
            "random" will split the data randomly, the seed is always 42 and the test size is 20%.
            "region" will split the data by one or multiple regions,
            which can be specified by providing a str or list of str to data_split_by.
            "sample" will split the data by sample ids, which can also be specified similar to "region".
            If None, no split is done and the complete dataset is used for both training and testing.
            The train split will further be split in the cross validation process.
            Defaults to None.
        data_split_by (list[str] | None, optional): Select by which regions/samples split. Defaults to None.
        fold_method (Literal["kfold", "shuffle", "stratified", "region", "region-stratified"], optional):
            Method for cross-validation split. Defaults to "kfold".
        total_folds (int, optional): Total number of folds in cross-validation. Defaults to 5.
        n_folds (int | None, optional): Number of folds to perform in cross-validation.
            If None, all folds (total_folds) will be used.
            Defaults to None.
        n_randoms (int, optional): Number of random seeds to perform in cross-validation.
            First three seeds are always 42, 21, 69, further seeds are deterministic generated.
            Defaults to 3.
        cv_name (str | None, optional): Name of the cross-validation.
            If None, a name is generated automatically.
            Defaults to None.
        tune_name (str | None, optional): Name of the tuning.
            Should only be specified by a tuning script.
            Defaults to None.
        artifact_dir (Path, optional): Top-level path to the training output directory.
            Will contain checkpoints and metrics. Defaults to Path("lightning_logs").
        model_arch (str, optional): Model architecture to use. Defaults to "Unet".
        model_encoder (str, optional): Encoder to use. Defaults to "dpn107".
        model_encoder_weights (str | None, optional): Path to the encoder weights. Defaults to None.
        augment (bool, optional): Weather to apply augments or not. Defaults to True.
        learning_rate (float, optional): Learning Rate. Defaults to 1e-3.
        gamma (float, optional): Multiplicative factor of learning rate decay. Defaults to 0.9.
        focal_loss_alpha (float, optional): Weight factor to balance positive and negative samples.
            Alpha must be in [0...1] range, high values will give more weight to positive class.
            None will not weight samples. Defaults to None.
        focal_loss_gamma (float, optional): Focal loss power factor. Defaults to 2.0.
        batch_size (int, optional): Batch Size. Defaults to 8.
        scoring_metric (list[str]): Metric(s) to use for scoring.
        multi_score_strategy (Literal["harmonic", "arithmetic", "geometric", "min"], optional):
            Strategy for combining multiple metrics. Defaults to "harmonic".
        max_epochs (int, optional): Maximum number of epochs to train. Defaults to 100.
        log_every_n_steps (int, optional): Log every n steps. Defaults to 10.
        check_val_every_n_epoch (int, optional): Check validation every n epochs. Defaults to 3.
        early_stopping_patience (int, optional): Number of epochs to wait for improvement before stopping.
            Defaults to 5.
        plot_every_n_val_epochs (int, optional): Plot validation samples every n epochs. Defaults to 5.
        num_workers (int, optional): Number of Dataloader workers. Defaults to 0.
        device (int | str, optional): The device to run the model on. Defaults to "auto".
        wandb_entity (str | None, optional): Weights and Biases Entity. Defaults to None.
        wandb_project (str | None, optional): Weights and Biases Project. Defaults to None.

    """
    from darts_segmentation.training.cv import cross_validation_smp as _cross_validation_smp

    _cross_validation_smp(
        train_data_dir=train_data_dir,
        data_split_method=data_split_method,
        data_split_by=data_split_by,
        fold_method=fold_method,
        total_folds=total_folds,
        n_folds=n_folds,
        n_randoms=n_randoms,
        cv_name=cv_name,
        tune_name=tune_name,
        artifact_dir=artifact_dir,
        model_arch=model_arch,
        model_encoder=model_encoder,
        model_encoder_weights=model_encoder_weights,
        augment=augment,
        learning_rate=learning_rate,
        gamma=gamma,
        focal_loss_alpha=focal_loss_alpha,
        focal_loss_gamma=focal_loss_gamma,
        batch_size=batch_size,
        scoring_metric=scoring_metric,
        multi_score_strategy=multi_score_strategy,
        max_epochs=max_epochs,
        log_every_n_steps=log_every_n_steps,
        check_val_every_n_epoch=check_val_every_n_epoch,
        early_stopping_patience=early_stopping_patience,
        plot_every_n_val_epochs=plot_every_n_val_epochs,
        num_workers=num_workers,
        device=device,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
    )
