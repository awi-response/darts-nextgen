"""More advanced hyper-parameter tuning."""

# TODO: Concurrent training:
# Can't parallelize cv only because for tunes without cv this would result in only one concurrent training job.
# Parallelizing tune only is also bad, because then cv is not parallelized at all.
# Concept:
# Implement concurrency for both tune and cv, but don't parallelize cv in a tuning situation.
# Reminder: Set the device for final training and test

import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from darts_segmentation.training.scoring import check_score_is_unstable

if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__.replace("darts_", "darts."))


def tune_smp(
    hpconfig: Path,
    # Data
    train_data_dir: Path,
    data_split_method: Literal["random", "region", "sample"] | None = None,
    data_split_by: list[str] | None = None,
    fold_method: Literal["kfold", "shuffle", "stratified", "region", "region-stratified"] = "kfold",
    total_folds: int = 5,
    bands: list[str] | None = None,
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
        fold_method (Literal["kfold", "shuffle", "stratified", "region", "region-stratified"], optional):
            Method for cross-validation split. Defaults to "kfold".
        total_folds (int, optional): Total number of folds in cross-validation. Defaults to 5.
        bands (list[str] | None, optional): List of bands to use. Defaults to None.
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

    Returns:
        tuple[float, pd.DataFrame]: The best score (if retrained and tested) and the run infos of all runs.

    """
    import pandas as pd
    from darts_utils.namegen import generate_counted_name

    from darts_segmentation.training.cv import cross_validation_smp
    from darts_segmentation.training.hparams import parse_hyperparameters, sample_hyperparameters
    from darts_segmentation.training.scoring import score_from_single_run
    from darts_segmentation.training.train import test_smp, train_smp

    tick_fstart = time.perf_counter()

    tune_name = tune_name or generate_counted_name(artifact_dir)
    artifact_dir = artifact_dir / tune_name
    run_infos_file = artifact_dir / f"{tune_name}.parquet"

    # Check if the artifact directory is empty
    assert not artifact_dir.exists(), f"{artifact_dir} already exists."

    param_grid = parse_hyperparameters(hpconfig)
    param_list = sample_hyperparameters(param_grid, n_trials)

    logger.info(
        f"Starting tune '{tune_name}' with data from {train_data_dir.resolve()}."
        f" Artifacts will be saved to {artifact_dir.resolve()}."
        f" Will run n_trials*n_randoms*n_folds ="
        f" {len(param_list)}*{n_randoms}*{n_folds} = {len(param_list) * n_randoms * n_folds} experiments."
    )

    run_infos: list[pd.DataFrame] = []
    best_score = 0
    best_hp = None
    for i, hp in enumerate(param_list):
        cv_name = f"{tune_name}-cv{i}"
        logger.debug(f"Starting cv {cv_name} ({i}/{len(param_list)})")
        score, is_unstable, cv_run_infos = cross_validation_smp(
            # Data
            train_data_dir=train_data_dir,
            data_split_method=data_split_method,
            data_split_by=data_split_by,
            fold_method=fold_method,
            total_folds=total_folds,
            bands=bands,
            # CV config
            n_folds=n_folds,
            n_randoms=n_randoms,
            cv_name=cv_name,
            tune_name=tune_name,
            artifact_dir=artifact_dir,
            # Hyperparameters
            model_arch=hp.model_arch,
            model_encoder=hp.model_encoder,
            model_encoder_weights=hp.model_encoder_weights,
            augment=hp.augment,
            learning_rate=hp.learning_rate,
            gamma=hp.gamma,
            focal_loss_alpha=hp.focal_loss_alpha,
            focal_loss_gamma=hp.focal_loss_gamma,
            batch_size=hp.batch_size,
            # Scoring
            scoring_metric=scoring_metric,
            multi_score_strategy=multi_score_strategy,
            # Epoch and Logging config
            max_epochs=max_epochs,
            log_every_n_steps=log_every_n_steps,
            check_val_every_n_epoch=check_val_every_n_epoch,
            early_stopping_patience=early_stopping_patience,
            plot_every_n_val_epochs=plot_every_n_val_epochs,
            # Device and Manager config
            num_workers=num_workers,
            device=device,
            # Wandb config
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
        )

        if not is_unstable and score > best_score:
            best_score = score
            best_hp = hp

        hpd = asdict(hp)
        for key, value in hpd.items():
            cv_run_infos[key] = value
        cv_run_infos["cv_name"] = cv_name
        run_infos.append(cv_run_infos)

        # Save already here to prevent data loss if something goes wrong
        pd.concat(run_infos).reset_index(drop=True).to_parquet(run_infos_file)
        logger.debug(f"Saved run infos to {run_infos_file}")

    run_infos = pd.concat(run_infos).reset_index(drop=True)

    tick_fend = time.perf_counter()
    logger.info(
        f"Tuning completed in {tick_fend - tick_fstart:.2f}s. The best score was {best_score:.4f} with {best_hp}."
    )

    if not retrain_and_test:
        return 0, run_infos

    if best_hp is None:
        logger.error("No hyperparameters resulted in a valid score. Please check the logs for more information.")
        return 0, run_infos

    logger.info("Starting retraining with the best hyperparameters.")

    tick_fstart = time.perf_counter()
    trainer = train_smp(
        # Data
        train_data_dir=train_data_dir,
        data_split_method=data_split_method,
        data_split_by=data_split_by,
        fold_method=None,
        # Run config
        run_name=f"{tune_name}-retrain",
        artifact_dir=artifact_dir,
        # Hyperparameters
        model_arch=best_hp.model_arch,
        model_encoder=best_hp.model_encoder,
        model_encoder_weights=best_hp.model_encoder_weights,
        augment=best_hp.augment,
        learning_rate=best_hp.learning_rate,
        gamma=best_hp.gamma,
        focal_loss_alpha=best_hp.focal_loss_alpha,
        focal_loss_gamma=best_hp.focal_loss_gamma,
        batch_size=best_hp.batch_size,
        # Epoch and log config
        max_epochs=max_epochs,
        log_every_n_steps=log_every_n_steps,
        check_val_every_n_epoch=check_val_every_n_epoch,
        early_stopping_patience=early_stopping_patience,
        plot_every_n_val_epochs=plot_every_n_val_epochs,
        # Device and Manager config
        random_seed=42,
        num_workers=num_workers,
        device=device,
        # Wandb
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
    )
    run_id = trainer.lightning_module.hparams["run_id"]
    trainer = test_smp(
        train_data_dir=train_data_dir,
        run_id=run_id,
        run_name=f"{tune_name}-retrain",
        model_ckp=trainer.checkpoint_callback.best_model_path,
        batch_size=best_hp.batch_size,
        data_split_method=data_split_method,
        data_split_by=data_split_by,
        artifact_dir=artifact_dir,
        num_workers=num_workers,
        device=device,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
    )

    run_info = {k: v.item() for k, v in trainer.callback_metrics.items()}
    test_scoring_metric = (
        scoring_metric.replace("val/", "test/")
        if isinstance(scoring_metric, str)
        else [sm.replace("val/", "test/") for sm in scoring_metric]
    )
    score = score_from_single_run(run_info, test_scoring_metric, multi_score_strategy)
    is_unstable = check_score_is_unstable(run_info, scoring_metric)
    tick_fend = time.perf_counter()
    logger.info(
        f"Retraining and testing completed successfully in {tick_fend - tick_fstart:.2f}s"
        f" with {score=:.4f} ({'stable' if not is_unstable else 'unstable'})."
    )

    return score, run_infos
