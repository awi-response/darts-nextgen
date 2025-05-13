"""Cross-validation implementation for binary segmentation."""

import logging
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from darts_segmentation.training.augmentations import Augmentation

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__.replace("darts_", "darts."))


def _gen_seed(n_randoms: int) -> list[int]:
    # Custom generator for deterministic seeds
    rng = random.Random(42)
    seeds = [42, 21, 69]  # First three seeds should be known
    if n_randoms > 3:
        seeds += rng.sample(range(9999), n_randoms - 3)
    elif n_randoms < 3:
        seeds = seeds[:n_randoms]
    return seeds


def cross_validation_smp(
    # Data
    train_data_dir: Path,
    data_split_method: Literal["random", "region", "sample"] | None = None,
    data_split_by: list[str | float] | None = None,
    fold_method: Literal["kfold", "shuffle", "stratified", "region", "region-stratified"] = "kfold",
    total_folds: int = 5,
    bands: list[str] | None = None,
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
    augment: list[Augmentation] | None = None,
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
            "random" will split the data randomly, the seed is always 42 and the test size can be specified
            by providing a list with a single a float between 0 and 1 to data_split_by
            This will be the fraction of the data to be used for testing.
            E.g. [0.2] will use 20% of the data for testing.
            "region" will split the data by one or multiple regions,
            which can be specified by providing a str or list of str to data_split_by.
            "sample" will split the data by sample ids, which can also be specified similar to "region".
            If None, no split is done and the complete dataset is used for both training and testing.
            The train split will further be split in the cross validation process.
            Defaults to None.
        data_split_by (list[str | float] | None, optional): Select by which regions/samples to split or
            the size of test set. Defaults to None.
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

    Returns:
        tuple[float, bool, pd.DataFrame]: A single score, a boolean indicating if the score is unstable,
            and a DataFrame containing run info (seed, fold, metrics, duration, checkpoint)

    """
    import pandas as pd
    import torch
    from darts_utils.namegen import generate_counted_name

    from darts_segmentation.training.scoring import check_score_is_unstable, score_from_runs
    from darts_segmentation.training.train import train_smp

    tick_fstart = time.perf_counter()

    # Expects artifact_dir to be already nested by tune, similar to train
    artifact_dir = artifact_dir if tune_name else artifact_dir / "_cross_validations"
    cv_name = cv_name or generate_counted_name(artifact_dir)
    artifact_dir = artifact_dir / cv_name

    n_folds = n_folds or total_folds

    logger.info(
        f"Starting cross-validation '{cv_name}' with data from {train_data_dir.resolve()}."
        f" Artifacts will be saved to {artifact_dir.resolve()}."
        f" Will run n_randoms*n_folds = {n_randoms}*{n_folds} = {n_randoms * n_folds} experiments."
    )

    seeds = _gen_seed(n_randoms)
    logger.debug(f"Using seeds: {seeds}")

    run_infos = []
    for i, seed in enumerate(seeds):
        for fold in range(n_folds):
            tick_rstart = time.time()
            run_name = f"{cv_name}-run-f{fold}s{seed}"
            current = i * len(seeds) + fold
            logger.debug(f"Starting run {run_name} ({current}/{n_folds * len(seeds)})")
            trainer = train_smp(
                # Data
                train_data_dir=train_data_dir,
                data_split_method=data_split_method,
                data_split_by=data_split_by,
                fold_method=fold_method,
                total_folds=total_folds,
                fold=fold,
                bands=bands,
                # Run config
                run_name=run_name,
                cv_name=cv_name,
                tune_name=tune_name,
                artifact_dir=artifact_dir,
                continue_from_checkpoint=None,  # ?: Support later?
                # Hyperparameters
                model_arch=model_arch,
                model_encoder=model_encoder,
                model_encoder_weights=model_encoder_weights,
                augment=augment,
                learning_rate=learning_rate,
                gamma=gamma,
                focal_loss_alpha=focal_loss_alpha,
                focal_loss_gamma=focal_loss_gamma,
                batch_size=batch_size,
                # Epoch and log config
                max_epochs=max_epochs,
                log_every_n_steps=log_every_n_steps,
                check_val_every_n_epoch=check_val_every_n_epoch,
                early_stopping_patience=early_stopping_patience,
                plot_every_n_val_epochs=plot_every_n_val_epochs,
                # Device and Manager config
                random_seed=seed,
                num_workers=num_workers,
                device=device,
                # Wandb
                wandb_entity=wandb_entity,
                wandb_project=wandb_project,
            )
            tick_rend = time.time()

            run_info = {
                "run_name": run_name,
                "run_id": trainer.lightning_module.hparams["run_id"],
                "seed": seed,
                "fold": fold,
                "duration": tick_rend - tick_rstart,
            }
            for metric, value in trainer.logged_metrics.items():
                run_info[metric] = value.item() if isinstance(value, torch.Tensor) else value
            if trainer.checkpoint_callback:
                run_info["checkpoint"] = trainer.checkpoint_callback.best_model_path
            run_info["is_unstable"] = check_score_is_unstable(run_info, scoring_metric)

            logger.debug(f"{run_info=}")
            run_infos.append(run_info)

    logger.debug(f"{run_infos=}")
    score = score_from_runs(run_infos, scoring_metric, multi_score_strategy)

    run_infos = pd.DataFrame(run_infos)
    run_infos["score"] = score
    is_unstable = run_infos["is_unstable"].any()
    run_infos["score_is_unstable"] = is_unstable
    if is_unstable:
        logger.warning("Score is unstable, meaning at least one of the metrics is NaN, Inf, -Inf or 0.")
    run_infos.to_parquet(artifact_dir / "run_infos.parquet")
    logger.debug(f"Saved run infos to {artifact_dir / 'run_infos.parquet'}")

    tick_fend = time.perf_counter()
    logger.info(
        f"Finished cross-validation '{cv_name}' in {tick_fend - tick_fstart:.2f}s"
        f" with {score=:.4f} ({'stable' if not is_unstable else 'unstable'})."
    )

    return score, is_unstable, run_infos
