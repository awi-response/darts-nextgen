"""Training scripts for DARTS."""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import cyclopts
import toml

from darts_segmentation.training.augmentations import Augmentation

if TYPE_CHECKING:
    import pytorch_lightning as pl

logger = logging.getLogger(__name__.replace("darts_", "darts."))


@cyclopts.Parameter(name="*")
@dataclass(frozen=True)
class DataParameters:
    """Data related parameters for training.

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
        fold (int, optional): Index of the current fold. Defaults to 0.
        bands (list[str] | None, optional): List of bands to use. Defaults to None.

    """

    train_data_dir: Path
    data_split_method: Literal["random", "region", "sample"] | None = None
    data_split_by: list[str | float] | None = None
    fold_method: Literal["kfold", "shuffle", "stratified", "region", "region-stratified"] = "kfold"
    total_folds: int = 5
    fold: int = 0  # Only in train
    bands: list[str] | None = None  # Maybe this should also be a hyperparameter?

    def with_fold(self, fold: int) -> "DataParameters":
        """Return a new instance with the specified fold.

        Need to maintain the immutability of the dataclass.
        Only the fold parameter is changed by other scripts, e.g. cross-validation or tuning.

        Returns:
            DataParameters: A new instance with the specified fold.

        """
        return DataParameters(
            train_data_dir=self.train_data_dir,
            data_split_method=self.data_split_method,
            data_split_by=self.data_split_by,
            fold_method=self.fold_method,
            total_folds=self.total_folds,
            fold=fold,
            bands=self.bands,
        )


@cyclopts.Parameter(name="*")
@dataclass(frozen=True)
class TrainRunConfig:
    """Run related parameters for training.

    Attributes:
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
        max_epochs (int, optional): Maximum number of epochs to train. Defaults to 100.
        random_seed (int, optional): Random seed for deterministic training. Defaults to 42.
        num_workers (int, optional): Number of Dataloader workers. Defaults to 0.
        device (list[int | str], optional): The device(s) to run the model on. Defaults to ["auto"].

    """

    run_name: str | None = None
    cv_name: str | None = None
    tune_name: str | None = None
    artifact_dir: Path = Path("artifacts")
    continue_from_checkpoint: Path | None = None
    max_epochs: int = 100
    random_seed: int = 42
    num_workers: int = 0
    # device: int | str = "auto"
    device: list[int | str] = field(default_factory=lambda: ["auto"])


@cyclopts.Parameter(name="*")
@dataclass(frozen=True)
class HyperParameters:
    """Hyperparameters for Cyclopts CLI.

    Attributes:
        model_arch (str): Architecture of the model to use.
        model_encoder (str): Encoder type for the model.
        model_encoder_weights (str | None): Weights for the encoder, if any.
        augment (list[Augmentation] | None): List of augmentations to apply.
        learning_rate (float): Learning rate for training.
        gamma (float): Decay factor for learning rate.
        focal_loss_alpha (float | None): Alpha parameter for focal loss, if using.
        focal_loss_gamma (float): Gamma parameter for focal loss.
        batch_size (int): Batch size for training.

    """

    model_arch: str = "Unet"
    model_encoder: str = "dpn107"
    model_encoder_weights: str | None = None
    augment: list[Augmentation] | None = None
    learning_rate: float = 1e-3
    gamma: float = 0.9
    focal_loss_alpha: float | None = None
    focal_loss_gamma: float = 2.0
    batch_size: int = 8


@cyclopts.Parameter(name="*")
@dataclass(frozen=True)
class LoggingConfig:
    """Logging related parameters for training.

    Attributes:
        log_every_n_steps (int, optional): Log every n steps. Defaults to 10.
        check_val_every_n_epoch (int, optional): Check validation every n epochs. Defaults to 3.
        early_stopping_patience (int, optional): Number of epochs to wait for improvement before stopping.
            Defaults to 5.
        plot_every_n_val_epochs (int, optional): Plot validation samples every n epochs. Defaults to 5.
        wandb_entity (str | None, optional): Weights and Biases Entity. Defaults to None.
        wandb_project (str | None, optional): Weights and Biases Project. Defaults to None.

    """

    log_every_n_steps: int = 10
    check_val_every_n_epoch: int = 3
    early_stopping_patience: int = 5
    plot_every_n_val_epochs: int = 5
    wandb_entity: str | None = None
    wandb_project: str | None = None


def train_smp(
    *,
    # Data config
    train_data_dir: Path,
    data_split_method: Literal["random", "region", "sample"] | None = None,
    data_split_by: list[str | float] | None = None,
    fold_method: Literal["kfold", "shuffle", "stratified", "region", "region-stratified"] = "kfold",
    total_folds: int = 5,
    fold: int = 0,
    bands: list[str] | None = None,
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
    augment: list[Augmentation] | None = None,
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
    # device: int | str = "auto",
    device: list[int | str] = ["auto"],
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
        fold (int, optional): Index of the current fold. Defaults to 0.
        bands (list[str] | None, optional): List of bands to use. Defaults to None.
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
        device (list[int | str], optional): The device(s) to run the model on. Defaults to ["auto"].
        wandb_entity (str | None, optional): Weights and Biases Entity. Defaults to None.
        wandb_project (str | None, optional): Weights and Biases Project. Defaults to None.

    Returns:
        pl.Trainer: The trainer object used for training. Contains also metrics.

    """
    import lightning as L  # noqa: N812
    import lovely_tensors
    import torch
    from darts.utils.logging import LoggingManager
    from darts_utils.namegen import generate_counted_name, generate_id
    from lightning.pytorch import seed_everything
    from lightning.pytorch.callbacks import EarlyStopping, RichProgressBar
    from lightning.pytorch.loggers import CSVLogger, WandbLogger

    from darts_segmentation.segment import SMPSegmenterConfig
    from darts_segmentation.training.callbacks import BinarySegmentationMetrics, BinarySegmentationPreview
    from darts_segmentation.training.data import DartsDataModule
    from darts_segmentation.training.module import LitSMP
    from darts_segmentation.utils import Bands

    LoggingManager.apply_logging_handlers("lightning.pytorch", level=logging.INFO)

    tick_fstart = time.perf_counter()

    # Further nest the artifact directory to avoid cluttering the root directory
    # For cv it is expected that the cv function already nests the artifact directory
    # Meaning for cv the artifact_dir of this function should be either
    # {artifact_dir}/_cross_validations/{cv_name} or {artifact_dir}/{tune_name}/{cv_name}
    artifact_dir = artifact_dir if cv_name else artifact_dir / "_runs"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Create unique run identification (name can be specified by user, id can be interpreded as a 'version')
    run_name = run_name or generate_counted_name(artifact_dir)
    run_id = generate_id()  # Needed for wandb

    logger.info(
        f"Starting training '{run_name}' ('{run_id}') with data from {train_data_dir.resolve()}."
        f" Artifacts will be saved to {(artifact_dir / f'{run_name}-{run_id}').resolve()}."
    )
    device_str = ",".join(str(d) for d in device)  # make a string with device list
    logger.debug(
        f"Using config:\n\t"
        # Hyperparameters
        f"{model_arch=}\n\t{model_encoder=}\n\t{model_encoder_weights=}\n\t{augment=}\n\t{learning_rate=}\n\t{gamma=}\n\t{batch_size=}\n\t"
        # Data
        f"{data_split_method=}\n\t{data_split_by=}\n\t{fold_method=}\n\t{total_folds=}\n\t{fold=}\n\t{bands=}\n\t"
        # Logging config
        f"{max_epochs=}\n\t{log_every_n_steps=}\n\t{check_val_every_n_epoch=}\n\t{early_stopping_patience=}\n\t{plot_every_n_val_epochs=}\n\t"
        # Run config
        f"{num_workers=}\n\t{device_str=}\n\t{random_seed=}"
    )
    if continue_from_checkpoint:
        logger.debug(f"Continuing from checkpoint '{continue_from_checkpoint.resolve()}'")
    if wandb_entity or wandb_project:
        logger.debug(f"Using Weights & Biases:\n\t{wandb_entity=}\n\t{wandb_project=}")

    lovely_tensors.monkey_patch()
    lovely_tensors.set_config(color=False)
    torch.set_float32_matmul_precision("medium")
    seed_everything(random_seed, workers=True, verbose=False)

    data_config = toml.load(train_data_dir / "config.toml")["darts"]
    all_bands = Bands.from_config(data_config)
    bands = all_bands.filter(bands) if bands else all_bands
    config = SMPSegmenterConfig(
        bands=bands,
        model={
            "arch": model_arch,
            "encoder_name": model_encoder,
            "encoder_weights": model_encoder_weights,
            "in_channels": len(all_bands) if bands is None else len(bands),
            "classes": 1,
        },
    )

    # Data and model
    datamodule = DartsDataModule(
        data_dir=train_data_dir,
        batch_size=batch_size,
        data_split_method=data_split_method,
        data_split_by=data_split_by,
        fold_method=fold_method,
        total_folds=total_folds,
        fold=fold,
        bands=bands,
        augment=augment,
        num_workers=num_workers,
    )
    model = LitSMP(
        config=config,
        learning_rate=learning_rate,
        gamma=gamma,
        focal_loss_alpha=focal_loss_alpha,
        focal_loss_gamma=focal_loss_gamma,
        # These are only stored in the hparams and are not used
        run_id=run_id,
        run_name=run_name,
        cv_name=cv_name or "none",
        tune_name=tune_name or "none",
        random_seed=random_seed,
    )

    # Loggers
    trainer_loggers = [
        CSVLogger(save_dir=artifact_dir, name=None, version=f"{run_name}-{run_id}"),
    ]
    logger.debug(f"Logging CSV to {Path(trainer_loggers[0].log_dir).resolve()}")
    if wandb_entity and wandb_project:
        tags = [train_data_dir.stem]
        if cv_name:
            tags.append(cv_name)
        if tune_name:
            tags.append(tune_name)
        wandb_logger = WandbLogger(
            save_dir=artifact_dir.parent.parent if tune_name or cv_name else artifact_dir.parent,
            name=run_name,
            version=run_id,
            project=wandb_project,
            entity=wandb_entity,
            resume="allow",
            # Using the group and job_type is a workaround for wandb's lack of support for manually sweeps
            group=tune_name or "none",
            job_type=cv_name or "none",
            # Using tags to quickly identify the run
            tags=tags,
        )
        trainer_loggers.append(wandb_logger)
        logger.debug(
            f"Logging to WandB with entity '{wandb_entity}' and project '{wandb_project}'."
            f"Artifacts are logged to {(Path(wandb_logger.save_dir) / 'wandb').resolve()}"
        )

    # Callbacks and profiler
    callbacks = [
        RichProgressBar(),
        BinarySegmentationMetrics(
            bands=bands,
            val_set=f"val{fold}",
            plot_every_n_val_epochs=plot_every_n_val_epochs,
            is_crossval=bool(cv_name),
            batch_size=batch_size,
            patch_size=data_config["patch_size"],
        ),
        BinarySegmentationPreview(
            bands=bands,
            val_set=f"val{fold}",
            plot_every_n_val_epochs=plot_every_n_val_epochs,
        ),
        # ThroughputMonitor(batch_size_fn=lambda batch: batch[0].size(0), window_size=log_every_n_steps),
    ]
    if early_stopping_patience:
        logger.debug(f"Using EarlyStopping with patience {early_stopping_patience}")
        early_stopping = EarlyStopping(monitor="val/JaccardIndex", mode="max", patience=early_stopping_patience)
        callbacks.append(early_stopping)

    # Unsupported: https://github.com/Lightning-AI/pytorch-lightning/issues/19983
    # profiler_dir = artifact_dir / f"{run_name}-{run_id}" / "profiler"
    # profiler_dir.mkdir(parents=True, exist_ok=True)
    # profiler = AdvancedProfiler(dirpath=profiler_dir, filename="perf_logs", dump_stats=True)
    # logger.debug(f"Using profiler with output to {profiler.dirpath.resolve()}")

    # Train
    trainer = L.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        log_every_n_steps=log_every_n_steps,
        logger=trainer_loggers,
        check_val_every_n_epoch=check_val_every_n_epoch,
        accelerator="gpu" if isinstance(device, list) else device,
        devices=device if isinstance(device[0], int) else "auto",
        strategy="ddp_find_unused_parameters_true",
        deterministic=False,  # True does not work for some reason
        # profiler=profiler,
    )
    trainer.fit(model, datamodule, ckpt_path=continue_from_checkpoint)

    tick_fend = time.perf_counter()
    logger.info(f"Finished training '{run_name}' in {tick_fend - tick_fstart:.2f}s.")

    if wandb_entity and wandb_project:
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
    device: int | str = "auto",
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
        device (int | str, optional): Device to use. Defaults to "auto".
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
    from darts_segmentation.utils import Bands

    LoggingManager.apply_logging_handlers("lightning.pytorch")

    tick_fstart = time.perf_counter()

    # Further nest the artifact directory to avoid cluttering the root directory
    artifact_dir = artifact_dir / "_runs"

    logger.info(
        f"Starting testing '{run_name}' ('{run_id}') with data from {train_data_dir.resolve()}."
        f" Artifacts will be saved to {(artifact_dir / f'{run_name}-{run_id}').resolve()}."
    )
    logger.debug(f"Using config:\n\t{batch_size=}\n\t{device=}")

    lovely_tensors.set_config(color=False)
    lovely_tensors.monkey_patch()
    torch.set_float32_matmul_precision("medium")
    seed_everything(42, workers=True)

    data_config = toml.load(train_data_dir / "config.toml")["darts"]

    all_bands = Bands.from_config(data_config)
    bands = all_bands.filter(bands) if bands else all_bands

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
            patch_size=data_config["patch_size"],
        ),
        ThroughputMonitor(batch_size_fn=lambda batch: batch[0].size(0)),
    ]

    # Test
    trainer = L.Trainer(
        callbacks=callbacks,
        logger=trainer_loggers,
        accelerator="gpu" if isinstance(device, int) else device,
        devices=[device] if isinstance(device, int) else device,
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
