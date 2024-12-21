"""Training and sweeping scripts for DARTS."""

import logging
import time
from pathlib import Path

import toml
import yaml

logger = logging.getLogger(__name__)


def train_smp(
    *,
    # Data config
    train_data_dir: Path,
    artifact_dir: Path = Path("lightning_logs"),
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
    num_workers: int = 0,
    device: int | str = "auto",
    wandb_entity: str | None = None,
    wandb_project: str | None = None,
    run_name: str | None = None,
):
    """Run the training of the SMP model.

    Please see https://smp.readthedocs.io/en/latest/index.html for model configurations.

    You can specify the frequency on how often logs will be written and validation will be performed.
        - `log_every_n_steps` specifies how often train-logs will be written. This does not affect validation.
        - `check_val_every_n_epoch` specifies how often validation will be performed.
            This will also affect early stopping.
        - `early_stopping_patience` specifies how many epochs to wait for improvement before stopping.
            In epochs, this would be `check_val_every_n_epoch * early_stopping_patience`.
        - `plot_every_n_val_epochs` specifies how often validation samples will be plotted.
            Since plotting is quite costly, you can reduce the frequency. Works similar like early stopping.
            In epochs, this would be `check_val_every_n_epoch * plot_every_n_val_epochs`.

    Args:
        train_data_dir (Path): Path to the training data directory.
        artifact_dir (Path, optional): Path to the training output directory.
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
        run_name (str | None, optional): Name of this run, as a further grouping method for logs etc. Defaults to None.

    """
    import lightning as L  # noqa: N812
    import lovely_tensors
    import torch
    from darts_segmentation.segment import SMPSegmenterConfig
    from darts_segmentation.training.data import DartsDataModule
    from darts_segmentation.training.module import SMPSegmenter
    from lightning.pytorch.callbacks import EarlyStopping, RichProgressBar
    from lightning.pytorch.loggers import CSVLogger, WandbLogger

    from darts.utils.logging import LoggingManager

    LoggingManager.apply_logging_handlers("lightning.pytorch")

    tick_fstart = time.perf_counter()
    logger.info(f"Starting training '{run_name}' with data from {train_data_dir.resolve()}.")
    logger.debug(
        f"Using config:\n\t{model_arch=}\n\t{model_encoder=}\n\t{model_encoder_weights=}\n\t{augment=}\n\t"
        f"{learning_rate=}\n\t{gamma=}\n\t{batch_size=}\n\t{max_epochs=}\n\t{log_every_n_steps=}\n\t"
        f"{check_val_every_n_epoch=}\n\t{early_stopping_patience=}\n\t{plot_every_n_val_epochs=}\n\t{num_workers=}"
        f"\n\t{device=}"
    )

    lovely_tensors.monkey_patch()

    torch.set_float32_matmul_precision("medium")

    preprocess_config = toml.load(train_data_dir / "config.toml")["darts"]

    config = SMPSegmenterConfig(
        input_combination=preprocess_config["bands"],
        model={
            "arch": model_arch,
            "encoder_name": model_encoder,
            "encoder_weights": model_encoder_weights,
            "in_channels": len(preprocess_config["bands"]),
            "classes": 1,
        },
        norm_factors=preprocess_config["norm_factors"],
    )

    # Data and model
    datamodule = DartsDataModule(train_data_dir, batch_size, augment, num_workers)
    model = SMPSegmenter(config, learning_rate, gamma, focal_loss_alpha, focal_loss_gamma, plot_every_n_val_epochs)

    # Loggers
    trainer_loggers = [
        CSVLogger(save_dir=artifact_dir, name=run_name),
    ]
    logger.debug(f"Logging CSV to {Path(trainer_loggers[0].log_dir).resolve()}")
    if wandb_entity and wandb_project:
        wandb_logger = WandbLogger(
            save_dir=artifact_dir,
            name=run_name,
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
    if early_stopping_patience:
        logger.debug(f"Using EarlyStopping with patience {early_stopping_patience}")
        early_stopping = EarlyStopping(monitor="val/JaccardIndex", mode="max", patience=early_stopping_patience)
        callbacks.append(early_stopping)

    # Train
    trainer = L.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        log_every_n_steps=log_every_n_steps,
        logger=trainer_loggers,
        check_val_every_n_epoch=check_val_every_n_epoch,
        accelerator="gpu" if isinstance(device, int) else device,
        devices=[device] if isinstance(device, int) else device,
    )
    trainer.fit(model, datamodule)

    tick_fend = time.perf_counter()
    logger.info(f"Finished training '{run_name}' in {tick_fend - tick_fstart:.2f}s.")


def sweep_smp(
    *,
    # Data and sweep config
    train_data_dir: Path,
    sweep_config: Path,
    sweep_count: int = 10,
    sweep_id: str | None = None,
    artifact_dir: Path = Path("lightning_logs"),
    # Epoch and Logging config
    max_epochs: int = 100,
    log_every_n_steps: int = 10,
    check_val_every_n_epoch: int = 3,
    plot_every_n_val_epochs: int = 5,
    # Device and Manager config
    num_workers: int = 0,
    device: int | str | None = None,
    wandb_entity: str | None = None,
    wandb_project: str | None = None,
):
    """Create a sweep and run it on specified cuda devices, or continue an existing sweep.

    If `sweep_id` is None, a new sweep will be created. Otherwise, the sweep with the given ID will be continued.

    If a `cuda_device` is specified, run an agent on this device. If None, do nothing.

    You can specify the frequency on how often logs will be written and validation will be performed.
        - `log_every_n_steps` specifies how often train-logs will be written. This does not affect validation.
        - `check_val_every_n_epoch` specifies how often validation will be performed.
            This will also affect early stopping.
        - `plot_every_n_val_epochs` specifies how often validation samples will be plotted.
            Since plotting is quite costly, you can reduce the frequency. Works similar like early stopping.
            In epochs, this would be `check_val_every_n_epoch * plot_every_n_val_epochs`.

    Example:
        In one terminal, start a sweep:
        ```sh
            $ rye run darts sweep_smp --config-file /path/to/sweep-config.toml
            ...  # Many logs
            Created sweep with ID 123456789
            ... # More logs from spawned agent
        ```

        In another terminal, start an a second agent:
        ```sh
            $ rye run darts sweep_smp --sweep-id 123456789
            ...
        ```

    Args:
        train_data_dir (Path): Path to the training data directory.
        sweep_config (Path): Path to the sweep yaml configuration file. Must contain a valid wandb sweep configuration.
            Hyperparameters must contain the following fields: `model_arch`, `model_encoder`, `augment`, `gamma`,
            `batch_size`.
            Please read https://docs.wandb.ai/guides/sweeps/sweep-config-keys for more information.
        sweep_count (int, optional): Number of runs to execute. Defaults to 10.
        sweep_id (str | None, optional): The ID of the sweep. If None, a new sweep will be created. Defaults to None.
        artifact_dir (Path, optional): Path to the training output directory.
            Will contain checkpoints and metrics. Defaults to Path("lightning_logs").
        max_epochs (int, optional): Maximum number of epochs to train. Defaults to 100.
        log_every_n_steps (int, optional): Log every n steps. Defaults to 10.
        check_val_every_n_epoch (int, optional): Check validation every n epochs. Defaults to 3.
        plot_every_n_val_epochs (int, optional): Plot validation samples every n epochs. Defaults to 5.
        num_workers (int, optional): Number of Dataloader workers. Defaults to 0.
        device (int | str | None, optional): The device to run the model on. Defaults to None.
        wandb_entity (str | None, optional): Weights and Biases Entity. Defaults to None.
        wandb_project (str | None, optional): Weights and Biases Project. Defaults to None.
        run_name (str | None, optional): Name of this run, as a further grouping method for logs etc. Defaults to None.

    """
    import wandb

    # Wandb has a stupid way of logging (they log per default with click.echo to stdout)
    # We need to silence this and redirect all possible logs to our logger
    # wl = wandb.setup({"silent": True})
    # wandb.termsetup(wl.settings, logging.getLogger("wandb"))
    # LoggingManager.apply_logging_handlers("wandb")

    if sweep_id is not None and device is None:
        logger.warning("Continuing a sweep without specifying a device will not do anything.")

    with sweep_config.open("r") as f:
        sweep_configuration = yaml.safe_load(f)

    logger.debug(f"Loaded sweep configuration from {sweep_config.resolve()}:\n{sweep_configuration}")

    if sweep_id is None:
        sweep_id = wandb.sweep(sweep=sweep_configuration, project=wandb_project, entity=wandb_entity)
        logger.info(f"Created sweep with ID {sweep_id}")
        logger.info("To start a sweep agents, use the following command:")
        logger.info(f"$ rye run darts sweep_smp --sweep-id {sweep_id}")

    def run():
        run = wandb.init(config=sweep_configuration)
        # We need to manually log the run data since the wandb logger only logs to its own logs and click
        logger.info(f"Starting sweep run '{run.settings.run_name}'")
        logger.debug(f"Run data is saved locally in {Path(run.settings.sync_dir).resolve()}")
        logger.debug(f"View project at {run.settings.project_url}")
        logger.debug(f"View sweep at {run.settings.sweep_url}")
        logger.debug(f"View run at {run.settings.run_url}")

        # We set the default weights to None, to be able to use different architectures
        model_encoder_weights = None
        # We set early stopping to None, because wandb will handle the early stopping
        early_stopping_patience = None
        learning_rate = wandb.config["learning_rate"]
        gamma = wandb.config["gamma"]
        batch_size = wandb.config["batch_size"]
        model_arch = wandb.config["model_arch"]
        model_encoder = wandb.config["model_encoder"]
        augment = wandb.config["augment"]

        train_smp(
            # Data config
            train_data_dir=train_data_dir,
            artifact_dir=artifact_dir,
            # Hyperparameters
            model_arch=model_arch,
            model_encoder=model_encoder,
            model_encoder_weights=model_encoder_weights,
            augment=augment,
            learning_rate=learning_rate,
            gamma=gamma,
            batch_size=batch_size,
            # Epoch and Logging config
            early_stopping_patience=early_stopping_patience,
            max_epochs=max_epochs,
            log_every_n_steps=log_every_n_steps,
            check_val_every_n_epoch=check_val_every_n_epoch,
            plot_every_n_val_epochs=plot_every_n_val_epochs,
            # Device and Manager config
            num_workers=num_workers,
            device=device,
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            run_name=wandb.run.name,
        )

    if device is None:
        logger.info("No device specified, closing script...")
        return

    logger.info("Starting a default sweep agent")
    wandb.agent(sweep_id, function=run, count=sweep_count, project=wandb_project, entity=wandb_entity)
