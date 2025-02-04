"""Testing scripts for DARTS."""

import logging
import time
from pathlib import Path

import toml

logger = logging.getLogger(__name__)


def test_smp(
    *,
    train_data_dir: Path,
    run_id: str,
    run_name: str,
    model_ckp: Path | None = None,
    batch_size: int = 8,
    artifact_dir: Path = Path("lightning_logs"),
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
    ├── cross-val/ # this directory contains the data for the training and validation
    │   ├── x/
    │   └── y/
    ├── val-test/ # this directory contains the data for the random selected validation set
    │   ├── x/
    │   └── y/
    └── test/ # this directory contains the data for the left-out-region test set
        ├── x/
        └── y/
    ```

    `x` and `y` are the directories which contain torch-tensor files (.pt) for the input and target data.

    Args:
        train_data_dir (Path): Path to the training data directory (top-level).
        run_id (str): ID of the run.
        run_name (str): Name of the run.
        model_ckp (Path | None): Path to the model checkpoint.
            If None, try to find the latest checkpoint in `artifact_dir / run_name / run_id / checkpoints`.
            Defaults to None.
        batch_size (int, optional): Batch size. Defaults to 8.
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
    from darts_segmentation.training.callbacks import BinarySegmentationMetrics
    from darts_segmentation.training.data import DartsDataModule
    from darts_segmentation.training.module import SMPSegmenter
    from lightning.pytorch import seed_everything
    from lightning.pytorch.callbacks import RichProgressBar
    from lightning.pytorch.loggers import CSVLogger, WandbLogger

    from darts.utils.logging import LoggingManager

    LoggingManager.apply_logging_handlers("lightning.pytorch")

    tick_fstart = time.perf_counter()
    logger.info(f"Starting testing '{run_name}' ('{run_id}') with data from {train_data_dir.resolve()}.")
    logger.debug(f"Using config:\n\t{batch_size=}\n\t{device=}")

    lovely_tensors.monkey_patch()

    torch.set_float32_matmul_precision("medium")
    seed_everything(42, workers=True)

    preprocess_config = toml.load(train_data_dir / "config.toml")["darts"]

    # Data and model
    datamodule_val_test = DartsDataModule(
        data_dir=train_data_dir / "val-test",
        batch_size=batch_size,
        num_workers=num_workers,
    )
    datamodule_test = DartsDataModule(
        data_dir=train_data_dir / "test",
        batch_size=batch_size,
        num_workers=num_workers,
    )
    # Try to infer model checkpoint if not given
    if model_ckp is None:
        checkpoint_dir = artifact_dir / run_name / run_id / "checkpoints"
        logger.debug(f"No checkpoint provided. Looking for model checkpoint in {checkpoint_dir.resolve()}")
        model_ckp = max(checkpoint_dir.glob("*.ckpt"), key=lambda x: x.stat().st_mtime)
    model = SMPSegmenter.load_from_checkpoint(model_ckp)

    # Loggers
    trainer_loggers = [
        CSVLogger(save_dir=artifact_dir, name=run_name, version=run_id),
    ]
    logger.debug(f"Logging CSV to {Path(trainer_loggers[0].log_dir).resolve()}")
    if wandb_entity and wandb_project:
        wandb_logger = WandbLogger(
            save_dir=artifact_dir,
            name=run_name,
            id=run_id,
            project=wandb_project,
            entity=wandb_entity,
        )
        trainer_loggers.append(wandb_logger)
        logger.debug(
            f"Logging to WandB with entity '{wandb_entity}' and project '{wandb_project}'."
            f"Artifacts are logged to {(Path(wandb_logger.save_dir) / 'wandb').resolve()}"
        )

    # Callbacks
    metrics_cb = BinarySegmentationMetrics(
        input_combination=preprocess_config["bands"],
    )
    callbacks = [
        RichProgressBar(),
        metrics_cb,
    ]

    # Test
    trainer = L.Trainer(
        callbacks=callbacks,
        logger=trainer_loggers,
        accelerator="gpu" if isinstance(device, int) else device,
        devices=[device] if isinstance(device, int) else device,
        deterministic=True,
    )
    # Overwrite the names of the test sets to test agains two separate sets
    metrics_cb.test_set = "val-test"
    model.test_set = "val-test"
    trainer.test(model, datamodule_val_test, ckpt_path=model_ckp)
    metrics_cb.test_set = "test"
    model.test_set = "test"
    trainer.test(model, datamodule_test)

    tick_fend = time.perf_counter()
    logger.info(f"Finished testing '{run_name}' in {tick_fend - tick_fstart:.2f}s.")

    if wandb_entity and wandb_project:
        wandb_logger.finalize("success")
        wandb_logger.experiment.finish(exit_code=0)
        logger.debug(f"Finalized WandB logging for '{run_name}'")

    return trainer
