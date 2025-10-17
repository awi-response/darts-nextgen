"""More advanced hyper-parameter tuning."""

import logging
import time
from dataclasses import asdict, dataclass
from multiprocessing import Queue
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import cyclopts
from darts_utils.paths import DefaultPaths, paths

from darts_segmentation.training.cv import CrossValidationConfig
from darts_segmentation.training.hparams import Hyperparameters
from darts_segmentation.training.scoring import check_score_is_unstable
from darts_segmentation.training.train import DataConfig, DeviceConfig, LoggingConfig, TrainingConfig, TrainRunConfig

if TYPE_CHECKING:
    import pandas as pd


logger = logging.getLogger(__name__.replace("darts_", "darts."))

available_devices = Queue()


@dataclass
class _ProcessInputs:
    current: int
    total: int
    tune_name: str
    cv: CrossValidationConfig
    training_config: TrainingConfig
    logging_config: LoggingConfig
    data_config: DataConfig
    device_config: DeviceConfig
    hparams: Hyperparameters


@dataclass
class _ProcessOutputs:
    run_infos: "pd.DataFrame"
    score: float
    is_unstable: bool


def _run_cv(inp: _ProcessInputs):
    # Wrapper function for handling parallel multiprocessing training runs.
    import pandas as pd

    from darts_segmentation.training.cv import cross_validation_smp

    cv_name = f"{inp.tune_name}-cv{inp.current}"

    # Setup device configuration: If strategy is "tune-parallel" expect a mp scenario:
    # Wait for a device to become available.
    # Otherwise, expect a serial scenario, where the devices and strategy are set by the user.
    is_parallel = inp.device_config.strategy == "tune-parallel"
    if is_parallel:
        device = available_devices.get()
        device_config = inp.device_config.in_parallel(device)
        logger.info(f"Starting cv '{cv_name}' ({inp.current + 1}/{inp.total}) on device {device}.")
    else:
        device = None
        device_config = inp.device_config.in_parallel()
        logger.info(f"Starting cv '{cv_name}' ({inp.current + 1}/{inp.total}).")

    try:
        score, is_unstable, cv_run_infos = cross_validation_smp(
            name=cv_name,
            tune_name=inp.tune_name,
            cv=inp.cv,
            training_config=inp.training_config,
            data_config=inp.data_config,
            logging_config=inp.logging_config,
            hparams=inp.hparams,
            device_config=device_config,
        )

        for key, value in asdict(inp.hparams).items():
            cv_run_infos[key] = value if not isinstance(value, list) else pd.Series([value] * len(cv_run_infos))

        cv_run_infos["cv_name"] = cv_name
        output = _ProcessOutputs(
            run_infos=cv_run_infos,
            score=score,
            is_unstable=is_unstable,
        )
    finally:
        # If we are in parallel mode, we need to return the device to the queue.
        if is_parallel:
            logger.debug(f"Free device {device} for cv {cv_name}")
            available_devices.put(device)
    return output


def tune_smp(
    *,
    name: str | None = None,
    n_trials: int | Literal["grid"] = 100,
    retrain_and_test: bool = False,
    default_dirs: DefaultPaths = DefaultPaths(),
    cv_config: CrossValidationConfig = CrossValidationConfig(),
    training_config: TrainingConfig = TrainingConfig(),
    data_config: DataConfig = DataConfig(),
    device_config: DeviceConfig = DeviceConfig(),
    logging_config: LoggingConfig = LoggingConfig(),
    hpconfig: Path | None = None,
    config_file: Annotated[Path | None, cyclopts.Parameter(parse=False)] = None,
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
        name (str | None, optional): Name of the tuning run.
            Will be generated based on the number of existing directories in the artifact directory if None.
            Defaults to None.
        n_trials (int | Literal["grid"], optional): Number of trials to perform in hyperparameter tuning.
            If "grid", span a grid search over all configured hyperparameters.
            In a grid search, only constant or choice hyperparameters are allowed.
            Defaults to 100.
        retrain_and_test (bool, optional): Whether to retrain the model with the best hyperparameters and test it.
            Defaults to False.
        default_dirs (DefaultPaths, optional): The default directories for DARTS. Defaults to a config filled with None.
        cv_config (CrossValidationConfig, optional): Configuration for cross-validation.
            Defaults to CrossValidationConfig().
        training_config (TrainingConfig, optional): Configuration for training.
            Defaults to TrainingConfig().
        data_config (DataConfig, optional): Configuration for data.
            Defaults to DataConfig().
        device_config (DeviceConfig, optional): Configuration for device.
            Defaults to DeviceConfig().
        logging_config (LoggingConfig, optional): Configuration for logging.
            Defaults to LoggingConfig().
        hpconfig (Path | None, optional): Path to the hyperparameter configuration file.
            Please see the documentation of `hyperparameters` for more information.
            Defaults to None.
        config_file (Path | None, optional): Path to the configuration file. If provided,
            it will be used instead of `hpconfig` if `hpconfig` is None. Defaults to None.

    Returns:
        tuple[float, pd.DataFrame]: The best score (if retrained and tested) and the run infos of all runs.

    Raises:
        ValueError: If no hyperparameter configuration file is provided.

    """
    import pandas as pd
    from darts_utils.namegen import generate_counted_name

    from darts_segmentation.training.adp import _adp
    from darts_segmentation.training.hparams import parse_hyperparameters, sample_hyperparameters
    from darts_segmentation.training.scoring import score_from_single_run
    from darts_segmentation.training.train import test_smp, train_smp

    tick_fstart = time.perf_counter()

    paths.set_defaults(default_dirs)

    tune_name = name or generate_counted_name(logging_config.artifact_dir)
    artifact_dir = logging_config.artifact_dir / tune_name
    run_infos_file = artifact_dir / f"{tune_name}.parquet"

    # Check if the artifact directory is empty
    assert not artifact_dir.exists(), f"{artifact_dir} already exists."
    artifact_dir.mkdir(parents=True, exist_ok=True)

    hpconfig = hpconfig or config_file
    if hpconfig is None:
        raise ValueError(
            "No hyperparameter configuration file provided. Please provide a valid file via the `--hpconfig` flag."
        )
    param_grid = parse_hyperparameters(hpconfig)
    logger.debug(f"Parsed hyperparameter grid: {param_grid}")
    param_list = sample_hyperparameters(param_grid, n_trials)

    logger.info(
        f"Starting tune '{tune_name}' with data from {data_config.train_data_dir.resolve()}."
        f" Artifacts will be saved to {artifact_dir.resolve()}."
        f" Will run n_trials*n_randoms*n_folds ="
        f" {len(param_list)}*{cv_config.n_randoms}*{cv_config.n_folds} ="
        f" {len(param_list) * cv_config.n_randoms * cv_config.n_folds} experiments."
    )

    # Plan which runs to perform. These are later consumed based on the parallelization strategy.
    process_inputs = [
        _ProcessInputs(
            current=i,
            total=len(param_list),
            tune_name=tune_name,
            cv=cv_config,
            training_config=training_config,
            logging_config=logging_config,
            data_config=data_config,
            device_config=device_config,
            hparams=hparams,
        )
        for i, hparams in enumerate(param_list)
    ]

    run_infos: list[pd.DataFrame] = []
    best_score = 0
    best_hp = None

    # This function abstracts away common logic for running multiprocessing
    for inp, output in _adp(
        process_inputs=process_inputs,
        is_parallel=device_config.strategy == "tune-parallel",
        devices=device_config.devices,
        available_devices=available_devices,
        _run=_run_cv,
    ):
        run_infos.append(output.run_infos)
        if not output.is_unstable and output.score > best_score:
            best_score = output.score
            best_hp = inp.hparams

        # Save already here to prevent data loss if something goes wrong
        pd.concat(run_infos).reset_index(drop=True).to_parquet(run_infos_file)
        logger.debug(f"Saved run infos to {run_infos_file}")

    if len(run_infos) == 0:
        logger.error("No hyperparameters resulted in a valid score. Please check the logs for more information.")
        return 0, run_infos

    run_infos = pd.concat(run_infos).reset_index(drop=True)

    tick_fend = time.perf_counter()

    if best_hp is None:
        logger.warning(
            f"Tuning completed in {tick_fend - tick_fstart:.2f}s."
            " No hyperparameters resulted in a valid score. Please check the logs for more information."
        )
        return 0, run_infos
    logger.info(
        f"Tuning completed in {tick_fend - tick_fstart:.2f}s. The best score was {best_score:.4f} with {best_hp}."
    )

    # =====================
    # === End of tuning ===
    # =====================

    if not retrain_and_test:
        return 0, run_infos

    logger.info("Starting retraining with the best hyperparameters.")

    tick_fstart = time.perf_counter()
    trainer = train_smp(
        run=TrainRunConfig(name=f"{tune_name}-retrain"),
        training_config=training_config,  # TODO: device and strategy
        data_config=DataConfig(
            train_data_dir=data_config.train_data_dir,
            data_split_method=data_config.data_split_method,
            data_split_by=data_config.data_split_by,
            fold_method=None,  # No fold method for retraining
            total_folds=None,  # No folds for retraining
        ),
        logging_config=LoggingConfig(
            artifact_dir=artifact_dir,
            log_every_n_steps=logging_config.log_every_n_steps,
            check_val_every_n_epoch=logging_config.check_val_every_n_epoch,
            plot_every_n_val_epochs=logging_config.plot_every_n_val_epochs,
            wandb_entity=logging_config.wandb_entity,
            wandb_project=logging_config.wandb_project,
        ),
        hparams=best_hp,
    )
    run_id = trainer.lightning_module.hparams["run_id"]
    trainer = test_smp(
        train_data_dir=data_config.train_data_dir,
        run_id=run_id,
        run_name=f"{tune_name}-retrain",
        model_ckp=trainer.checkpoint_callback.best_model_path,
        batch_size=best_hp.batch_size,
        data_split_method=data_config.data_split_method,
        data_split_by=data_config.data_split_by,
        artifact_dir=artifact_dir,
        num_workers=training_config.num_workers,
        device_config=device_config,
        wandb_entity=logging_config.wandb_entity,
        wandb_project=logging_config.wandb_project,
    )

    run_info = {k: v.item() for k, v in trainer.callback_metrics.items()}
    test_scoring_metric = (
        cv_config.scoring_metric.replace("val/", "test/")
        if isinstance(cv_config.scoring_metric, str)
        else [sm.replace("val/", "test/") for sm in cv_config.scoring_metric]
    )
    score = score_from_single_run(run_info, test_scoring_metric, cv_config.multi_score_strategy)
    is_unstable = check_score_is_unstable(run_info, cv_config.scoring_metric)
    tick_fend = time.perf_counter()
    logger.info(
        f"Retraining and testing completed successfully in {tick_fend - tick_fstart:.2f}s"
        f" with {score=:.4f} ({'stable' if not is_unstable else 'unstable'})."
    )

    return score, run_infos
