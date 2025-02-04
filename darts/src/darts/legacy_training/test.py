"""Testing scripts for DARTS."""

import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def test_smp(
    *,
    train_data_dir: Path,
    model_ckp: Path,
    run_id: str,
    run_name: str,
    batch_size: int = 8,
    artifact_dir: Path = Path("lightning_logs"),
    num_workers: int = 0,
    device: int | str = "auto",
    wandb_entity: str | None = None,
    wandb_project: str | None = None,
):
    import lightning as L  # noqa: N812
    import lovely_tensors
    import torch
    from darts_segmentation.training.data import DartsDataModule
    from darts_segmentation.training.module import SMPSegmenter
    from lightning.pytorch.callbacks import RichProgressBar
    from lightning.pytorch.loggers import CSVLogger, WandbLogger

    from darts.utils.logging import LoggingManager

    LoggingManager.apply_logging_handlers("lightning.pytorch")

    tick_fstart = time.perf_counter()
    logger.info(f"Starting testing '{run_name}' ('{run_id}') with data from {train_data_dir.resolve()}.")
    logger.debug(f"Using config:\n\t{batch_size=}\n\t{device=}")

    lovely_tensors.monkey_patch()

    torch.set_float32_matmul_precision("medium")
    torch.manual_seed(42)

    # Data and model
    datamodule = DartsDataModule(
        data_dir=train_data_dir / "cross-val",
        batch_size=batch_size,
        num_workers=num_workers,
    )
    model = SMPSegmenter.load_from_checkpoint(model_ckp)

    # Loggers
    trainer_loggers = [
        CSVLogger(save_dir=artifact_dir, name=run_name),
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
    callbacks = [
        RichProgressBar(),
    ]

    # Train
    trainer = L.Trainer(
        callbacks=callbacks,
        logger=trainer_loggers,
        accelerator="gpu" if isinstance(device, int) else device,
        devices=[device] if isinstance(device, int) else device,
    )
    trainer.test(model, datamodule)

    tick_fend = time.perf_counter()
    logger.info(f"Finished testing '{run_name}' in {tick_fend - tick_fstart:.2f}s.")

    if wandb_entity and wandb_project:
        wandb_logger.finalize("success")
        wandb_logger.experiment.finish(exit_code=0)
        logger.debug(f"Finalized WandB logging for '{run_name}'")

    return trainer
