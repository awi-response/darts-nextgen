"""Sweeping scripts for DARTS.

Unused yet!
"""

import logging
import os
import random
from itertools import product
from pathlib import Path
from statistics import mean

import yaml

from darts.legacy_training.train import train_smp

logger = logging.getLogger(__name__)


def _gather_and_reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    wandb_env = {}
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            wandb_env[k] = v
            del os.environ[k]
    return wandb_env


def _apply_wandb_env(wandb_env):
    for k, v in wandb_env.items():
        os.environ[k] = v


def wandb_cv_sweep_smp(
    *,
    # Data and sweep config
    train_data_dir: Path,
    sweep_config: Path,
    n_trials: int = 10,
    n_folds: int = 5,
    n_randoms: int = 5,
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
    """Create or continue a cross-validation sweep with wandb and run it on the specified cuda device.

    If `sweep_id` is None, a new sweep will be created. Otherwise, the sweep with the given ID will be continued.
    All artifacts are gathered under nested directory based on the sweep id: {artifact_dir}/sweep-{sweep_id}.
    Since each sweep-configuration has (currently) an own name and id, a single run can be found under:
    {artifact_dir}/sweep-{sweep_id}/{run_name}/{run_id}. Read the training-docs for more info.

    If a `cuda_device` is specified, run an agent on this device. If None, do nothing.

    You can specify the frequency on how often logs will be written and validation will be performed.
        - `log_every_n_steps` specifies how often train-logs will be written. This does not affect validation.
        - `check_val_every_n_epoch` specifies how often validation will be performed.
            This will also affect early stopping.
        - `plot_every_n_val_epochs` specifies how often validation samples will be plotted.
            Since plotting is quite costly, you can reduce the frequency. Works similar like early stopping.
            In epochs, this would be `check_val_every_n_epoch * plot_every_n_val_epochs`.

    This will NOT use cross-validation. For cross-validation, use `optuna_sweep_smp`.

    Example:
        In one terminal, start a sweep:
        ```sh
            $ rye run darts wandb-sweep-smp --config-file /path/to/sweep-config.toml
            ...  # Many logs
            Created sweep with ID 123456789
            ... # More logs from spawned agent
        ```

        In another terminal, start an a second agent:
        ```sh
            $ rye run darts wandb-sweep-smp --sweep-id 123456789
            ...
        ```

    Args:
        train_data_dir (Path): Path to the training data directory.
        sweep_config (Path): Path to the sweep yaml configuration file. Must contain a valid wandb sweep configuration.
            Hyperparameters must contain the following fields: `model_arch`, `model_encoder`, `augment`, `gamma`,
            `batch_size`.
            Please read https://docs.wandb.ai/guides/sweeps/sweep-config-keys for more information.
        n_trials (int, optional): Number of runs to execute. Only used for non-grid sweeps. Defaults to 10.
        n_folds (int, optinoal): Number of folds in cross-validation. Defaults to 5.
        n_randoms (int, optional): Number of repetitions with different random-seeds.
            First 3 are always "42", "21" and "69" for better default comparibility with rest of this pipeline.
            Rest are pseudo-random generated beforehand, hence always equal.
            Defaults to 5.
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

    score_metric = sweep_configuration["metric"]["name"]

    logger.debug(f"Loaded sweep configuration from {sweep_config.resolve()}:\n{sweep_configuration}")

    if sweep_id is None:
        sweep_id = wandb.sweep(sweep=sweep_configuration, project=wandb_project, entity=wandb_entity)
        logger.info(f"Created sweep with ID {sweep_id}")
        logger.info("To start a sweep agents, use the following command:")
        logger.info(f"$ rye run darts sweep_smp --sweep-id {sweep_id}")

    # This complete function is a super dubious hack and neither used nor tested yet:
    # Currently, wandb doesn't provide a way to have multiple runs with same configuration run on the same process
    # They designed their lib to use multiprocessing or similar. However, using mp is not really an option here,
    # since that would clash with PyTorch Lightning.
    # Hence, this function alters the environment variables which are used by the wandb.init to overwrite existing
    # instances / runs.
    # A sweep-run (hyperparameter configuration recommended by wandb) is created first. The wandb sweep algo uses this
    # run as benchmark (logging of the avg. score from all the folds)
    # Then a fold is run over this configuration, each overwriting the existing wandb-env and creating a new run.
    # I recommend reading this issue here:
    # https://github.com/wandb/wandb/issues/5119
    def _sweep_run():
        with wandb.init(config=sweep_configuration) as sweep_run:
            # We need to manually log the run data since the wandb logger only logs to its own logs and click
            logger.info(f"Starting sweep run '{sweep_run.settings.run_name}'")
            logger.debug(f"Run data is saved locally in {Path(sweep_run.settings.sync_dir).resolve()}")
            logger.debug(f"View project at {sweep_run.settings.project_url}")
            logger.debug(f"View sweep at {sweep_run.settings.sweep_url}")
            logger.debug(f"View run at {sweep_run.settings.run_url}")

            # We set the default weights to None, to be able to use different architectures
            model_encoder_weights = None
            # We set early stopping to None, because wandb will handle the early stopping
            early_stopping_patience = None
            learning_rate = sweep_run.config["learning_rate"]
            gamma = sweep_run.config["gamma"]
            batch_size = sweep_run.config["batch_size"]
            model_arch = sweep_run.config["model_arch"]
            model_encoder = sweep_run.config["model_encoder"]
            augment = sweep_run.config["augment"]
            focal_loss_alpha = sweep_run.config["focal_loss_alpha"]
            focal_loss_gamma = sweep_run.config["focal_loss_gamma"]

            folds = list(range(n_folds))
            rng = random.Random(42)
            seeds = [42, 21, 69]
            if n_randoms > 3:
                seeds += rng.sample(range(9999), n_randoms - 3)
            elif n_randoms < 3:
                seeds = seeds[:n_randoms]

            sweep_run_name = sweep_run.name
            sweep_run_env = _gather_and_reset_wandb_env()

            cvscores = []
            for fold, seed in product(folds, seeds):
                _gather_and_reset_wandb_env()
                with wandb.init(group=sweep_run_name, name=f"{sweep_run_name}-{fold=}-{seed=}") as cv_run:
                    trainer = train_smp(
                        # Data config
                        train_data_dir=train_data_dir,
                        artifact_dir=artifact_dir,
                        current_fold=fold,
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
                        # Epoch and Logging config
                        early_stopping_patience=early_stopping_patience,
                        max_epochs=max_epochs,
                        log_every_n_steps=log_every_n_steps,
                        check_val_every_n_epoch=check_val_every_n_epoch,
                        plot_every_n_val_epochs=plot_every_n_val_epochs,
                        # Device and Manager config
                        random_seed=seed,
                        num_workers=num_workers,
                        device=device,
                        wandb_entity=wandb_entity,
                        wandb_project=wandb_project,
                        run_name=cv_run.name,
                    )
                    score = trainer.logged_metrics[score_metric]
                    cvscores.append(score)

            _apply_wandb_env(sweep_run_env)
            sweep_run.log({score_metric: mean(cvscores)})  # TODO: make score-var selectable

    if device is None:
        logger.info("No device specified, closing script...")
        return

    logger.info("Starting a default sweep agent")
    wandb.agent(sweep_id, function=_sweep_run, count=n_trials, project=wandb_project, entity=wandb_entity)
