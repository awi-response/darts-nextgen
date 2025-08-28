"""Training scripts for DARTS."""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import cyclopts
import toml

from darts_segmentation.training.hparams import Hyperparameters

if TYPE_CHECKING:
    import pytorch_lightning as pl

logger = logging.getLogger(__name__.replace("darts_", "darts."))

# Quick note about the different Config classes:
# These contain the input parameters for the training script.
# They can be shared by the cross-validation and tuning scripts, which reduces duplication of code and documentation.
# Except the TrainRunConfig, which is only used by the training script and must be created by the other scripts.
# That means for a tune or cross-validation, all settings are the same between different runs,
# except the ones in TrainRunConfig.


@cyclopts.Parameter(name="*")
@dataclass(frozen=True)
class DataConfig:
    """Data related parameters for training.

    Defines the script inputs for the training script and can be propagated by the cross-validation and tuning scripts.

    Attributes:
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
            Defaults to "train".
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
        subsample (int | None, optional): If set, will subsample the dataset to this number of samples.
            This is useful for debugging and testing. Defaults to None.
        in_memory (bool, optional): If True, the dataset will be loaded into memory.

    """

    train_data_dir: Path = Path("train")
    data_split_method: Literal["random", "region", "sample"] | None = None
    data_split_by: list[str | float] | None = None
    fold_method: Literal["kfold", "shuffle", "stratified", "region", "region-stratified"] = "kfold"
    total_folds: int = 5
    subsample: int | None = None
    in_memory: bool = False
    # fold is only used in the training function, and fulfills a similar purpose as the random seed in terms of cv.
    # Hence it is moves to the run config.


@cyclopts.Parameter(name="*")
@dataclass(frozen=True)
class TrainRunConfig:
    """Run related parameters for training.

    Defines the script inputs for the training script. Must be build by the cross-validation and tuning scripts.

    Attributes:
        name (str | None, optional): Name of the run. If None is generated automatically. Defaults to None.
        cv_name (str | None, optional): Name of the cross-validation.
            Should only be specified by a cross-validation script.
            Defaults to None.
        tune_name (str | None, optional): Name of the tuning.
            Should only be specified by a tuning script.
            Defaults to None.
        fold (int, optional): Index of the current fold. Defaults to 0.
        random_seed (int, optional): Random seed for deterministic training. Defaults to 42.

    """

    name: str | None = None
    cv_name: str | None = None
    tune_name: str | None = None
    fold: int = 0
    random_seed: int = 42


@cyclopts.Parameter(name="*")
@dataclass(frozen=True)
class TrainingConfig:
    """Training related parameters for training.

    Defines the script inputs for the training script and can be propagated by the cross-validation and tuning scripts.

    Attributes:
        weights_from_checkpoint (Path | None, optional): Path to the lightning checkpoint to load the model from.
            If None, the model will be trained from scratch. Defaults to None.
        continue_from_checkpoint (Path | None, optional): Path to a checkpoint to continue training from.
            Differs from `weights_from_checkpoint` in that it will continue training from this training state,
            hence all optimizer states, learning rate schedulers, etc. will be continued.
            Defaults to None.
        max_epochs (int, optional): Maximum number of epochs to train. Defaults to 100.
        early_stopping_patience (int, optional): Number of epochs to wait for improvement before stopping.
            Defaults to 5.
        num_workers (int, optional): Number of Dataloader workers. Defaults to 0.
        save_top_k (int, optional): Number of best checkpoints to save.
            Set to 0 to disable saving checkpoints.
            Set to -1 to save all checkpoints.
            Defaults to 1.

    """

    weights_from_checkpoint: Path | None = None
    continue_from_checkpoint: Path | None = None
    max_epochs: int = 100
    early_stopping_patience: int = 5
    num_workers: int = 0
    save_top_k: int = 1


@cyclopts.Parameter(name="*")
@dataclass(frozen=True)
class LoggingConfig:
    """Logging related parameters for training.

    Defines the script inputs for the training script and can be propagated by the cross-validation and tuning scripts.

    Attributes:
        artifact_dir (Path, optional): Top-level path to the training output directory.
            Will contain checkpoints and metrics. Defaults to Path("artifacts").
        log_every_n_steps (int, optional): Log every n steps. Defaults to 10.
        check_val_every_n_epoch (int, optional): Check validation every n epochs. Defaults to 3.
        plot_every_n_val_epochs (int, optional): Plot validation samples every n epochs. Defaults to 5.
        wandb_entity (str | None, optional): Weights and Biases Entity. Defaults to None.
        wandb_project (str | None, optional): Weights and Biases Project. Defaults to None.

    """

    artifact_dir: Path = Path("artifacts")
    log_every_n_steps: int = 10
    check_val_every_n_epoch: int = 3
    plot_every_n_val_epochs: int = 5
    wandb_entity: str | None = None
    wandb_project: str | None = None

    def artifact_dir_at_run(self, cv_name: str | None, tune_name: str | None) -> Path:
        """Nest the artifact directory to avoid cluttering the root directory.

        For cv it is expected that the cv function already nests the artifact directory
        Meaning for cv the artifact_dir of this function should be either
        {artifact_dir}/_cross_validations/{cv_name} or {artifact_dir}/{tune_name}/{cv_name}

        Also creates the directory if it does not exist.

        Args:
            cv_name (str | None): Name of the cross-validation.
            tune_name (str | None): Name of the tuning.

        Raises:
            ValueError: If tune_name is specified, but cv_name is not, which is invalid.

        Returns:
            Path: The nested artifact directory path.

        """
        # Run only
        if cv_name is None and tune_name is None:
            artifact_dir = self.artifact_dir / "_runs"
        # Cross-validation only
        elif cv_name is not None and tune_name is None:
            artifact_dir = self.artifact_dir / "_cross_validations" / cv_name
        # Cross-validation and tuning
        elif cv_name is not None and tune_name is not None:
            artifact_dir = self.artifact_dir / tune_name / cv_name
        # Tuning only (invalid)
        else:
            raise ValueError(
                "Cannot parse artifact directory for cross-validation and tuning. "
                "Please specify either cv_name or tune_name, but not both."
            )
        artifact_dir.mkdir(parents=True, exist_ok=True)
        return artifact_dir

    def artifact_dir_at_cv(self, tune_name: str | None) -> Path:
        """Nest the artifact directory for cross-validation runs.

        Similar to `parse_artifact_dir_for_run`, but meant to be used by the cross-validation script.

        Also creates the directory if it does not exist.

        Args:
            tune_name (str | None): Name of the tuning, if applicable.

        Returns:
            Path: The nested artifact directory path for cross-validation runs.

        """
        artifact_dir = self.artifact_dir / tune_name if tune_name else self.artifact_dir / "_cross_validations"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        return artifact_dir


@cyclopts.Parameter(name="*")
@dataclass(frozen=True)
class DeviceConfig:
    """Device and Distributed Strategy related parameters.

    Attributes:
        accelerator (Literal["auto", "cpu", "gpu", "mps", "tpu"], optional): Accelerator to use.
            Defaults to "auto".
        strategy (Literal["auto", "ddp", "ddp_fork", "ddp_notebook", "fsdp", "cv-parallel", "tune-parallel", "cv-parallel", "tune-parallel"], optional):
            Distributed strategy to use. Defaults to "auto".
        devices (list[int | str], optional): List of devices to use. Defaults to ["auto"].
        num_nodes (int): Number of nodes to use for distributed training. Defaults to 1.

    """  # noqa: E501

    accelerator: Literal["auto", "cpu", "gpu", "mps", "tpu"] = "auto"
    strategy: Literal["auto", "ddp", "ddp_fork", "ddp_notebook", "fsdp", "cv-parallel", "tune-parallel"] = "auto"
    devices: list[int | str] = field(default_factory=lambda: ["auto"])
    num_nodes: int = 1

    def in_parallel(self, device: int | str | None = None) -> "DeviceConfig":
        """Turn the current configuration into a suitable configuration for parallel training.

        Args:
            device (int | str | None, optional): The device to use for parallel training.
                If None, assumes non-multiprocessing parallel training and propagate all devices.
                Defaults to None.

        Returns:
            DeviceConfig: A new DeviceConfig instance that is suitable for parallel training.

        """
        # In case of parallel training via multiprocessing, only few strategies are allowed.
        if self.strategy in ["ddp", "ddp_fork", "ddp_notebook", "fsdp"]:
            logger.warning("Using 'ddp_fork' instead of 'ddp' for multiprocessing.")
            return DeviceConfig(
                accelerator=self.accelerator,
                strategy="ddp_fork",  # Fork is the only supported strategy for multiprocessing
                devices=self.devices,
                num_nodes=self.num_nodes,
            )
        elif device is not None:
            return DeviceConfig(
                accelerator=self.accelerator,
                strategy=self.strategy,
                # If a device is specified, we assume that we want to run on a single device
                devices=[device],
                num_nodes=1,
            )
        else:
            return self

    @property
    def lightning_strategy(self) -> str:
        """Get the Lightning strategy for the current configuration.

        Returns:
            str: The Lightning strategy to use.

        """
        # custom strategy for cross-validation and tuning - disabling intra-training parallelism
        if self.strategy == "cv-parallel" or self.strategy == "tune-parallel":
            return "auto"
        # print warning if ddp
        if self.strategy == "ddp":
            logger.warning(
                "Using 'ddp' strategy can have unknown side effects, since it will call the CLI multiple times."
                " Use 'ddp_fork' for slower, but more reliable multiprocessing."
            )
        # print warning about untested fsdp strategy
        if self.strategy == "fsdp":
            logger.warning(
                "Using 'fsdp' strategy is untested and may not work as expected. "
                "Please report any issues to the DARTS team."
            )
        # Add find unused parameters to the strategy for ddp and fsdp
        if self.strategy in ["ddp", "ddp_fork", "ddp_notebook", "fsdp"]:
            return f"{self.strategy}_find_unused_parameters_true"
        return self.strategy


def train_smp(
    *,
    run: TrainRunConfig = TrainRunConfig(),
    training_config: TrainingConfig = TrainingConfig(),
    data_config: DataConfig = DataConfig(),
    logging_config: LoggingConfig = LoggingConfig(),
    device_config: DeviceConfig = DeviceConfig(),
    hparams: Hyperparameters = Hyperparameters(),
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
        data_config (DataConfig): Data related parameters for training.
        run (TrainRunConfig): Run related parameters for training.
        logging_config (LoggingConfig): Logging related parameters for training.
        device_config (DeviceConfig): Device and distributed strategy related parameters.
        training_config (TrainingConfig): Training related parameters for training.
        hparams (Hyperparameters): Hyperparameters for the model.

    Returns:
        pl.Trainer: The trainer object used for training. Contains also metrics.

    """
    import lightning as L  # noqa: N812
    import lovely_tensors
    import torch
    from darts.utils.logging import LoggingManager
    from darts_utils.namegen import generate_counted_name, generate_id
    from lightning.pytorch import seed_everything
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger, WandbLogger

    from darts_segmentation.segment import SMPSegmenterConfig
    from darts_segmentation.training.callbacks import BinarySegmentationMetrics, BinarySegmentationPreview
    from darts_segmentation.training.data import DartsDataModule
    from darts_segmentation.training.module import LitSMP

    LoggingManager.apply_logging_handlers("lightning.pytorch", level=logging.INFO)

    tick_fstart = time.perf_counter()

    # Get the right nesting of the artifact directory
    artifact_dir = logging_config.artifact_dir_at_run(run.cv_name, run.tune_name)

    # Create unique run identification (name can be specified by user, id can be interpreded as a 'version')
    run_name = run.name or generate_counted_name(artifact_dir)
    run_id = generate_id()  # Needed for wandb

    logger.info(
        f"Starting training '{run_name}' ('{run_id}') with data from {data_config.train_data_dir.resolve()}."
        f" Artifacts will be saved to {(artifact_dir / f'{run_name}-{run_id}').resolve()}."
    )
    logger.debug(
        f"Using config:\n\t{run}\n\t{training_config}\n\t{data_config}\n\t{logging_config}\n\t"
        f"{device_config}\n\t{hparams}"
    )
    if training_config.continue_from_checkpoint:
        logger.debug(f"Continuing from checkpoint '{training_config.continue_from_checkpoint.resolve()}'")

    lovely_tensors.monkey_patch()
    lovely_tensors.set_config(color=False)
    torch.set_float32_matmul_precision("medium")
    seed_everything(run.random_seed, workers=True, verbose=False)

    dataset_config = toml.load(data_config.train_data_dir / "config.toml")["darts"]
    bands: list[str] = dataset_config["bands"]
    if hparams.bands:
        # Filter bands by specified
        bands = [b for b in bands if b in hparams.bands]

    config = SMPSegmenterConfig(
        bands=bands,
        model={
            "arch": hparams.model_arch,
            "encoder_name": hparams.model_encoder,
            "encoder_weights": hparams.model_encoder_weights,
            "in_channels": len(bands),
            "classes": 1,
        },
    )

    # Data and model
    datamodule = DartsDataModule(
        data_dir=data_config.train_data_dir,
        batch_size=hparams.batch_size,
        data_split_method=data_config.data_split_method,
        data_split_by=data_config.data_split_by,
        fold_method=data_config.fold_method,
        total_folds=data_config.total_folds,
        fold=run.fold,
        subsample=data_config.subsample,
        bands=hparams.bands,
        augment=hparams.augment,
        num_workers=training_config.num_workers,
        in_memory=data_config.in_memory,
    )
    if training_config.weights_from_checkpoint:
        logger.debug(f"Loading model weights from checkpoint '{training_config.weights_from_checkpoint.resolve()}'")
        model = LitSMP.load_from_checkpoint(
            training_config.weights_from_checkpoint,
            map_location="cpu",
        )
    else:
        model = LitSMP(
            config=config,
            learning_rate=hparams.learning_rate,
            gamma=hparams.gamma,
            focal_loss_alpha=hparams.focal_loss_alpha,
            focal_loss_gamma=hparams.focal_loss_gamma,
            # Storing the model / checkpoint version in the hparams
            model_version="2",
            # These are only stored in the hparams and are only used as metadata
            run_id=run_id,
            run_name=run_name,
            cv_name=run.cv_name or "none",
            tune_name=run.tune_name or "none",
            random_seed=run.random_seed,
            datetime=datetime.now(),
            model_framework="smp",
        )

    # Loggers
    trainer_loggers = [
        CSVLogger(save_dir=artifact_dir, name=None, version=f"{run_name}-{run_id}"),
    ]
    logger.debug(f"Logging CSV to {Path(trainer_loggers[0].log_dir).resolve()}")
    if logging_config.wandb_entity and logging_config.wandb_project:
        tags = [data_config.train_data_dir.stem]
        if run.cv_name:
            tags.append(run.cv_name)
        if run.tune_name:
            tags.append(run.tune_name)
        wandb_logger = WandbLogger(
            save_dir=artifact_dir.parent.parent if run.tune_name or run.cv_name else artifact_dir.parent,
            name=run_name,
            version=run_id,
            project=logging_config.wandb_project,
            entity=logging_config.wandb_entity,
            resume="allow",
            # Using the group and job_type is a workaround for wandb's lack of support for manually sweeps
            group=run.tune_name or "none",
            job_type=run.cv_name or "none",
            # Using tags to quickly identify the run
            tags=tags,
        )
        trainer_loggers.append(wandb_logger)
        logger.debug(
            f"Logging to WandB with entity '{logging_config.wandb_entity}' and project '{logging_config.wandb_project}'"
            f"Artifacts are logged to {(Path(wandb_logger.save_dir) / 'wandb').resolve()}"
        )

    # Callbacks and profiler
    callbacks = [
        # RichProgressBar(),
        ModelCheckpoint(
            filename="epoch={epoch}-step={step}-val_iou={val/JaccardIndex:.2f}",
            auto_insert_metric_name=False,
            verbose=True,
            monitor="val/JaccardIndex",
            mode="max",
            save_last="link",
            save_top_k=training_config.save_top_k,
        ),
        BinarySegmentationMetrics(
            bands=bands,
            val_set=f"val{run.fold}",
            plot_every_n_val_epochs=logging_config.plot_every_n_val_epochs,
            is_crossval=bool(run.cv_name),
            batch_size=hparams.batch_size,
            patch_size=dataset_config["patch_size"],
        ),
        BinarySegmentationPreview(
            bands=bands,
            val_set=f"val{run.fold}",
            plot_every_n_val_epochs=logging_config.plot_every_n_val_epochs,
        ),
        # Something does not work well here...
        # ThroughputMonitor(batch_size_fn=lambda batch: batch[0].size(0), window_size=log_every_n_steps),
    ]
    # There is a bug when continuing from a checkpoint and using the RichProgressBar
    # https://github.com/Lightning-AI/pytorch-lightning/issues/20976
    # Seems like there is also another bug, so disable rich completly
    # if training_config.continue_from_checkpoint is None:
    #     callbacks.append(RichProgressBar())

    if training_config.early_stopping_patience:
        logger.debug(f"Using EarlyStopping with patience {training_config.early_stopping_patience}")
        early_stopping = EarlyStopping(
            monitor="val/JaccardIndex", mode="max", patience=training_config.early_stopping_patience
        )
        callbacks.append(early_stopping)

    # Unsupported: https://github.com/Lightning-AI/pytorch-lightning/issues/19983
    # profiler_dir = artifact_dir / f"{run_name}-{run_id}" / "profiler"
    # profiler_dir.mkdir(parents=True, exist_ok=True)
    # profiler = AdvancedProfiler(dirpath=profiler_dir, filename="perf_logs", dump_stats=True)
    # logger.debug(f"Using profiler with output to {profiler.dirpath.resolve()}")

    logger.debug(
        f"Creating lightning-trainer on {device_config.accelerator} with devices {device_config.devices}"
        f" and strategy '{device_config.lightning_strategy}'"
    )
    # Train
    trainer = L.Trainer(
        max_epochs=training_config.max_epochs,
        callbacks=callbacks,
        log_every_n_steps=logging_config.log_every_n_steps,
        logger=trainer_loggers,
        check_val_every_n_epoch=logging_config.check_val_every_n_epoch,
        accelerator=device_config.accelerator,
        devices=device_config.devices if device_config.devices[0] != "auto" else "auto",
        strategy=device_config.lightning_strategy,
        num_nodes=device_config.num_nodes,
        deterministic=False,  # True does not work for some reason
        # profiler=profiler,
    )
    trainer.fit(model, datamodule, ckpt_path=training_config.continue_from_checkpoint)

    tick_fend = time.perf_counter()
    logger.info(f"Finished training '{run_name}' in {tick_fend - tick_fstart:.2f}s.")

    if logging_config.wandb_entity and logging_config.wandb_project:
        wandb_logger.finalize("success")
        wandb_logger.experiment.finish(exit_code=0)
        logger.debug(f"Finalized WandB logging for '{run_name}'")

    return trainer


def test_smp(
    *,
    train_data_dir: Path,
    run_id: str,
    run_name: str,
    model_ckp: Path | None = None,
    batch_size: int = 8,
    data_split_method: Literal["random", "region", "sample"] | None = None,
    data_split_by: list[str] | str | float | None = None,
    bands: list[str] | None = None,
    artifact_dir: Path = Path("artifacts"),
    num_workers: int = 0,
    device_config: DeviceConfig = DeviceConfig(),
    wandb_entity: str | None = None,
    wandb_project: str | None = None,
) -> "pl.Trainer":
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
        bands (list[str] | None, optional): List of bands to use. Defaults to None.
        artifact_dir (Path, optional): Directory to save artifacts. Defaults to Path("lightning_logs").
        num_workers (int, optional): Number of workers for the DataLoader. Defaults to 0.
        device_config (DeviceConfig, optional): Device and distributed strategy related parameters.
        wandb_entity (str | None, optional): WandB entity. Defaults to None.
        wandb_project (str | None, optional): WandB project. Defaults to None.

    Returns:
        Trainer: The trainer object used for training.

    """
    import lightning as L  # noqa: N812
    import lovely_tensors
    import torch
    from darts.utils.logging import LoggingManager
    from lightning.pytorch import seed_everything
    from lightning.pytorch.callbacks import RichProgressBar, ThroughputMonitor
    from lightning.pytorch.loggers import CSVLogger, WandbLogger

    from darts_segmentation.training.callbacks import BinarySegmentationMetrics
    from darts_segmentation.training.data import DartsDataModule
    from darts_segmentation.training.module import LitSMP

    LoggingManager.apply_logging_handlers("lightning.pytorch")

    tick_fstart = time.perf_counter()

    # Further nest the artifact directory to avoid cluttering the root directory
    artifact_dir = artifact_dir / "_runs"

    logger.info(
        f"Starting testing '{run_name}' ('{run_id}') with data from {train_data_dir.resolve()}."
        f" Artifacts will be saved to {(artifact_dir / f'{run_name}-{run_id}').resolve()}."
    )
    logger.debug(f"Using config:\n\t{batch_size=}\n\t{device_config}")

    lovely_tensors.set_config(color=False)
    lovely_tensors.monkey_patch()
    torch.set_float32_matmul_precision("medium")
    seed_everything(42, workers=True)

    dataset_config = toml.load(train_data_dir / "config.toml")["darts"]

    all_bands: list[str] = dataset_config["bands"]
    bands = [b for b in all_bands if b in bands] if bands else all_bands

    # Data and model
    datamodule = DartsDataModule(
        data_dir=train_data_dir,
        batch_size=batch_size,
        data_split_method=data_split_method,
        data_split_by=data_split_by,
        bands=bands,
        num_workers=num_workers,
    )
    # Try to infer model checkpoint if not given
    if model_ckp is None:
        checkpoint_dir = artifact_dir / f"{run_name}-{run_id}" / "checkpoints"
        logger.debug(f"No checkpoint provided. Looking for model checkpoint in {checkpoint_dir.resolve()}")
        model_ckp = max(checkpoint_dir.glob("*.ckpt"), key=lambda x: x.stat().st_mtime)
    logger.debug(f"Using model checkpoint at {model_ckp.resolve()}")
    model = LitSMP.load_from_checkpoint(model_ckp)

    # Loggers
    trainer_loggers = [
        CSVLogger(save_dir=artifact_dir, version=f"{run_name}-{run_id}"),
    ]
    logger.debug(f"Logging CSV to {Path(trainer_loggers[0].log_dir).resolve()}")
    if wandb_entity and wandb_project:
        wandb_logger = WandbLogger(
            save_dir=artifact_dir.parent,
            name=run_name,
            version=run_id,
            project=wandb_project,
            entity=wandb_entity,
            resume="allow",
            # Using the group and job_type is a workaround for wandb's lack of support for manually sweeps
            group="none",
            job_type="none",
        )
        trainer_loggers.append(wandb_logger)
        logger.debug(
            f"Logging to WandB with entity '{wandb_entity}' and project '{wandb_project}'."
            f"Artifacts are logged to {(Path(wandb_logger.save_dir) / 'wandb').resolve()}"
        )

    # Callbacks
    callbacks = [
        RichProgressBar(),
        BinarySegmentationMetrics(
            bands=bands,
            batch_size=batch_size,
            patch_size=dataset_config["patch_size"],
        ),
        ThroughputMonitor(batch_size_fn=lambda batch: batch[0].size(0)),
    ]

    # Test
    trainer = L.Trainer(
        callbacks=callbacks,
        logger=trainer_loggers,
        accelerator=device_config.accelerator,
        strategy=device_config.lightning_strategy,
        num_nodes=device_config.num_nodes,
        devices=device_config.devices,
        deterministic=True,
    )

    trainer.test(model, datamodule, ckpt_path=model_ckp)

    tick_fend = time.perf_counter()
    logger.info(f"Finished testing '{run_name}' in {tick_fend - tick_fstart:.2f}s.")

    if wandb_entity and wandb_project:
        wandb_logger.finalize("success")
        wandb_logger.experiment.finish(exit_code=0)
        logger.debug(f"Finalized WandB logging for '{run_name}'")

    return trainer


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
    import torch

    logger.debug(f"Loading checkpoint from {lightning_checkpoint.resolve()}")
    lckpt = torch.load(lightning_checkpoint, weights_only=False, map_location=torch.device("cpu"))

    now = datetime.now()
    formatted_date = now.strftime("%Y-%m-%d")
    config = lckpt["hyper_parameters"]["config"]
    del config["model"]["encoder_weights"]
    config["time"] = formatted_date
    config["name"] = checkpoint_name
    config["model_framework"] = framework

    statedict = lckpt["state_dict"]
    # Statedict has model. prefix before every weight. We need to remove them. This is an in-place function
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(statedict, "model.")

    own_ckpt = {
        "config": config,
        "statedict": lckpt["state_dict"],
    }

    out_directory.mkdir(exist_ok=True, parents=True)

    out_checkpoint = out_directory / f"{checkpoint_name}_{formatted_date}.ckpt"

    torch.save(own_ckpt, out_checkpoint)

    logger.info(f"Saved converted checkpoint to {out_checkpoint.resolve()}")
