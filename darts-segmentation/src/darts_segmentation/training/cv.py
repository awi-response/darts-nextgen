"""Cross-validation implementation for binary segmentation."""

import logging
import random
import time
from dataclasses import dataclass, field
from multiprocessing import Queue
from typing import TYPE_CHECKING, Literal

import cyclopts

from darts_segmentation.training.hparams import Hyperparameters
from darts_segmentation.training.train import DataConfig, DeviceConfig, LoggingConfig, TrainingConfig, TrainRunConfig

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__.replace("darts_", "darts."))

available_devices = Queue()


@cyclopts.Parameter(name="*")
@dataclass(frozen=True)
class CrossValidationConfig:
    """Configuration for cross-validation.

    This is used to configure the cross-validation process.
    It is used by the `cross_validation_smp` function.

    Attributes:
        n_folds (int | None, optional): Number of folds to perform in cross-validation.
            If None, all folds (total_folds) will be used. Defaults to None.
        n_randoms (int, optional): Number of random seeds to perform in cross-validation.
            First three seeds are always 42, 21, 69, further seeds are deterministic generated.
            Defaults to 3.
        tune_name (str | None, optional): Name of the tuning. Should only be specified by a tuning script.
            Defaults to None.
        scoring_metric (list[str]): Metric(s) to use for scoring. Defaults to ["val/JaccardIndex", "val/Recall"].
        multi_score_strategy (Literal["harmonic", "arithmetic", "geometric", "min"], optional):
            Strategy for combining multiple metrics. Defaults to "harmonic".

    """

    n_folds: int | None = None
    n_randoms: int = 3
    tune_name: str | None = None
    scoring_metric: list[str] = field(default_factory=lambda: ["val/JaccardIndex", "val/Recall"])
    multi_score_strategy: Literal["harmonic", "arithmetic", "geometric", "min"] = "harmonic"

    @property
    def rng_seeds(self) -> list[int]:
        """Generate a list of seeds for cross-validation.

        Returns:
            list[int]: A list of seeds for cross-validation.
            The first three seeds are always 42, 21, 69, further seeds are deterministically generated.

        """
        # Custom generator for deterministic seeds
        rng = random.Random(42)
        seeds = [42, 21, 69]  # First three seeds should be known
        if self.n_randoms > 3:
            seeds += rng.sample(range(9999), self.n_randoms - 3)
        elif self.n_randoms < 3:
            seeds = seeds[: self.n_randoms]
        return seeds


@dataclass
class _ProcessInputs:
    current: int
    total: int
    seed: int
    fold: int
    cv: CrossValidationConfig
    run: TrainRunConfig
    training_config: TrainingConfig
    logging_config: LoggingConfig
    data_config: DataConfig
    device_config: DeviceConfig
    hparams: Hyperparameters


@dataclass
class _ProcessOutputs:
    run_info: dict


def _run_training(inp: _ProcessInputs):
    # Wrapper function for handling parallel multiprocessing training runs.
    import torch

    from darts_segmentation.training.scoring import check_score_is_unstable
    from darts_segmentation.training.train import train_smp

    # Setup device configuration: If strategy is "cv-parallel" expect a mp scenario:
    # Wait for a device to become available.
    # Otherwise, expect a serial scenario, where the devices and strategy are set by the user.
    is_parallel = inp.device_config.strategy == "cv-parallel"
    if is_parallel:
        device = available_devices.get()
        device_config = inp.device_config.in_parallel(device)
        logger.debug(f"Starting run {inp.run.name} ({inp.current}/{inp.total}) on device {device}.")
    else:
        device = None
        device_config = inp.device_config
        logger.debug(f"Starting run {inp.run.name} ({inp.current}/{inp.total}).")

    try:
        tick_rstart = time.time()
        trainer = train_smp(
            run=inp.run,
            training_config=inp.training_config,
            data_config=inp.data_config,
            device_config=device_config,
            hparams=inp.hparams,
            logging_config=inp.logging_config,
        )
        tick_rend = time.time()

        run_info = {
            "run_name": inp.run.name,
            "run_id": trainer.lightning_module.hparams["run_id"],
            "seed": inp.seed,
            "fold": inp.fold,
            "duration": tick_rend - tick_rstart,
        }
        for metric, value in trainer.logged_metrics.items():
            run_info[metric] = value.item() if isinstance(value, torch.Tensor) else value
        if trainer.checkpoint_callback:
            run_info["checkpoint"] = trainer.checkpoint_callback.best_model_path
        run_info["is_unstable"] = check_score_is_unstable(run_info, inp.cv.scoring_metric)

        logger.debug(f"{run_info=}")
        output = _ProcessOutputs(run_info=run_info)
    finally:
        # If we are in parallel mode, we need to return the device to the queue.
        if is_parallel:
            logger.debug(f"Free device {device} for cv {inp.run.name}")
            available_devices.put(device)
    return output


def cross_validation_smp(
    *,
    name: str | None = None,
    cv: CrossValidationConfig = CrossValidationConfig(),
    training_config: TrainingConfig = TrainingConfig(),
    data_config: DataConfig = DataConfig(),
    device_config: DeviceConfig = DeviceConfig(),
    hparams: Hyperparameters = Hyperparameters(),
    logging_config: LoggingConfig = LoggingConfig(),
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
        name (str | None, optional): Name of the cross-validation. If None, a name is generated automatically.
            Defaults to None.
        cv (CrossValidationConfig): Configuration for cross-validation.
        training_config (TrainingConfig): Configuration for the training.
        data_config (DataConfig): Configuration for the data.
        device_config (DeviceConfig): Configuration for the devices to use.
        hparams (Hyperparameters): Hyperparameters for the training.
        logging_config (LoggingConfig): Logging configuration.

    Returns:
        tuple[float, bool, pd.DataFrame]: A single score, a boolean indicating if the score is unstable,
            and a DataFrame containing run info (seed, fold, metrics, duration, checkpoint)

    """
    import pandas as pd
    from darts_utils.namegen import generate_counted_name

    from darts_segmentation.training.adp import _adp
    from darts_segmentation.training.scoring import score_from_runs

    tick_fstart = time.perf_counter()

    artifact_dir = logging_config.artifact_dir_at_cv(cv.tune_name)
    cv_name = name or generate_counted_name(artifact_dir)
    artifact_dir = artifact_dir / cv_name
    artifact_dir.mkdir(parents=True, exist_ok=True)

    n_folds = cv.n_folds or data_config.total_folds

    logger.info(
        f"Starting cross-validation '{cv_name}' with data from {data_config.train_data_dir.resolve()}."
        f" Artifacts will be saved to {artifact_dir.resolve()}."
        f" Will run n_randoms*n_folds = {cv.n_randoms}*{n_folds} = {cv.n_randoms * n_folds} experiments."
    )

    seeds = cv.rng_seeds
    logger.debug(f"Using seeds: {seeds}")

    # Plan which runs to perform. These are later consumed based on the parallelization strategy.
    process_inputs: list[_ProcessInputs] = []
    for i, seed in enumerate(seeds):
        for fold in range(n_folds):
            current = i * len(seeds) + fold
            total = n_folds * len(seeds)
            run = TrainRunConfig(
                name=f"{cv_name}-run-f{fold}s{seed}",
                cv_name=cv_name,
                tune_name=cv.tune_name,
                fold=fold,
                random_seed=seed,
            )
            process_inputs.append(
                _ProcessInputs(
                    current=current,
                    total=total,
                    seed=seed,
                    fold=fold,
                    cv=cv,
                    run=run,
                    training_config=training_config,
                    logging_config=logging_config,
                    data_config=data_config,
                    device_config=device_config,
                    hparams=hparams,
                )
            )

    run_infos = []
    # This function abstracts away common logic for running multiprocessing
    for inp, output in _adp(
        process_inputs=process_inputs,
        device_config=device_config,
        available_devices=available_devices,
        _run=_run_training,
    ):
        run_infos.append(output.run_info)

    logger.debug(f"{run_infos=}")
    score = score_from_runs(run_infos, cv.scoring_metric, cv.multi_score_strategy)

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
